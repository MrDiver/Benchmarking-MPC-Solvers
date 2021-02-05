import numpy as np
import numdifftools as nd

from MPCBenchmark.agents.agent import Agent


class ILQR(Agent):

    def __init__(self, bounds_low, bounds_high, input_size, output_size, model, params: dict) -> None:
        super().__init__(bounds_low, bounds_high, input_size, output_size, model)

        self.max_iter = params["max_iter"]
        self.init_mu = params["init_mu"]
        self.mu = self.init_mu
        self.min_mu = params["mu_min"]
        self.max_mu = params["mu_max"]
        self.init_delta = params["init_delta"]
        self.delta = self.init_delta
        self.threshold = params["threshold"]

        # general parameters
        self.pred_len = params["T"]
        # self.input_size = config.INPUT_SIZE
        # self.dt = config.DT

        self.delta = self.init_delta
        self.bounds_high = bounds_high
        self.bounds_low = bounds_low

        
# get cost func
        def c(xugz):
            x = xugz[:,:self.input_size]
            u = xugz[:,self.input_size:(self.input_size+self.output_size)]
            z = self.model._transform(x, u)
            g_z = xugz[:,(self.input_size+self.output_size):]
            return self.model._state_cost(z, g_z)

        def ct(xgz):
            x = xgz[:,:self.input_size]
            g_z = xgz[:,self.input_size:]
            z = self.model._transform(x, np.zeros((x.shape[0],self.output_size)))
            return self.model._terminal_cost(z, g_z)

        self.Jacobian_cost = nd.Jacobian(c)
        self.Jacobian_terminal_cost = nd.Jacobian(ct)
        self.Hessian_cost = nd.Hessian(c)
        self.Hessian_terminal_cost = nd.Hessian(ct)




        self.prev_sol = np.zeros((self.pred_len,self.output_size))

    def calc_action(self, state, optimal_solution=None):
        self.prev_sol = np.zeros((self.pred_len,self.output_size))
        print("state",state.shape)
        current_state = state
        # previous solution
        solution = self.prev_sol

        # converged solution
        converged = False

        # derivatives
        derivatives = True

        self.mu = self.init_mu
        self.delta = self.init_delta

        # forward pass
        alphas = 1 ** (1 / (np.arange(1, 10) ** 2))

        for _ in range(self.max_iter):
            # forward pass
            accepted = False

            xs = None
            cost = None
            f_x = None
            f_u = None
            l_x = None
            l_uu = None
            l_ux = None

            # derivatives
            if derivatives:
                xs, cost, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux = \
                    self.forward_pass(current_state, optimal_solution, solution)
                derivatives = False

            # backward pass
            k, K = self.backward_pass(f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux)

            # forward pass
            for alpha in alphas:
                new_xs, new_solution = \
                    self.calc_input_trajectory(k, K, xs, solution, alpha)

                new_cost = - self.model.get_reward()

                if new_cost < cost:
                    if np.abs((cost - new_cost) / cost) < self.threshold:
                        converged = True

                    solution = new_solution
                    derivatives = True

                    # decrease regularization term
                    self.delta = min(1.0, self.delta) / self.init_delta
                    self.mu *= self.delta
                    if self.mu <= self.min_mu:
                        self.mu = 0.0

                    # accept the solution
                    accepted = True
                    break

            if not accepted:
                # increase regularization term.
                self.delta = max(1.0, self.delta) * self.init_delta
                self.mu = max(self.min_mu, self.mu * self.delta)

                if self.mu > self.max_mu:
                    break

            if converged:
                break

        self.prev_sol[:-1] = solution[1:]
        self.prev_sol[-1] = solution[-1]  # last use the terminal input
        return solution[0]

    def forward_pass(self, current_x, optimal_solution, solution):
        # get size
        # initialze
        x = current_x
        xs = current_x[np.newaxis, :]  # Das ist kein python

        def predict_trajectory(x,solution):
            for t in range(self.pred_len):
                print(x.shape)
                print(solution[t].shape)
                next_x = self.model.predict(x, solution[t])
                print("next x",next_x.shape)
                print("xs",xs.shape)
                # update
                xs = np.concatenate((xs, next_x), axis=0)  # das ist kein python
                x = next_x

        # check costs
        cost = - self.model.get_reward()

        f_x = nd.core.Gradient(xs[:-1])
        f_u = nd.core.Gradient(xs[:-1])

        l_x, l_xx, l_u, l_uu, l_ux = \
            self._calc_gradient_hessian_cost(xs,optimal_solution, solution)

        return xs, cost, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux

    def _calc_gradient_hessian_cost(self, pred_xs, goal_xs, solution):

        print("KOMMT HIER HIN WTF")

        print("CALC L STUFF BULLSHIT=====================")
        print("predxs",pred_xs.shape)
        print("solution",solution.shape)
        print("repeat stuff",np.repeat(solution,pred_xs.shape[0],axis=0).shape)

        if goal_xs is None:
            goal_xs = np.zeros((pred_xs.shape[0],pred_xs.shape[1]+solution.shape[0]))

        print("goalxs",goal_xs.shape)


        xugz = np.append(np.append(pred_xs,solution.reshape(),axis=1), goal_xs, axis=1)
        xgz = np.append(pred_xs,goal_xs, axis=1)
        jac_c = self.Jacobian_cost(xugz)
        jax_ct = self.Jacobian_terminal_cost(xgz)
        hess_c = self.Hessian_cost(xugz)
        hess_ct = self.Hessian_cost(xgz)

        return l_x, l_xx, l_u, l_uu, l_ux

    def backward_pass(self, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux):
        V_x_old = l_x[-1]
        V_xx_old = l_xx[-1]
        k = np.zeros((self.output_size, self.output_size))
        K = np.zeros((self.output_size, self.input_size, self.input_size))

        V_x = 0  # dummys
        V_xx = 0  # dummys
        for t in range(self.output_size - 1, -1, -1):
            # get Q val
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self.q(f_x[t], f_u[t], l_x[t],
                                                l_u[t], l_xx[t], l_ux[t],
                                                l_uu[t], V_x_old, V_xx_old)
            # calc gain
            k[t] = - np.linalg.solve(Q_uu, Q_u)  # 10c
            K[t] = - np.linalg.solve(Q_uu, Q_ux)  # 10d
            # update V_x val
            V_x = Q_x + K[t].T @ Q_uu @ k[t]  # 11b
            V_x += K[t].T @ Q_u + Q_ux.T @ k[t]  # 11b
            # update V_xx val
            V_xx = Q_xx + K[t].T @ Q_uu @ K[t]  # 11c
            V_xx += K[t].T @ Q_ux + Q_ux.T @ K[t]
            V_xx = 0.5 * (V_xx + V_xx.T)  # to maintain symmetry.

        return k, K

    def q(self, f_x, f_u, l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx):
        size = len(l_x)

        Q_x = l_x + f_x.T @ V_x  # 5a
        Q_u = l_u + f_u.T @ V_x  # 5b
        Q_xx = l_xx + f_x.T @ V_xx @ f_x  # 5c möglicherweise falsch

        reg = self.mu * np.eye(size)  # regularization term
        Q_uu = l_uu + f_u.T @ (V_xx + reg) @ f_u  # 10a möglicherweise falsch
        Q_ux = l_ux + f_u.T @ (V_xx + reg) @ f_x  # 10b möglicherweise falsch

        # TODO wenns nicht geht + V_x @ f_

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def calc_input_trajectory(self, k, K, xs, solution, alpha):
        # get size
        (pred_len, input_size, state_size) = K.shape
        # initializepred_len# init state is same
        new_solution = np.zeros((pred_len, input_size))

        for t in range(pred_len):
            new_solution[t] = solution[t] \
                              + alpha * k[t] \
                              + np.dot(K[t], (new_xs[t] - xs[t]))
            new_xs[t + 1] = self.model.predict(new_xs[t], new_solution[t])

        return new_xs, new_solution