import numpy as np
import numdifftools as nd

from MPCBenchmark.agents.agent import Agent
import multiprocessing as mp

class ILQR(Agent):
    def __init__(self, bounds_low, bounds_high, input_size, output_size, model, params, cores=12) -> None:
        super().__init__(bounds_low,bounds_high,input_size,output_size,model)
        self.pred_length = params["T"]
        self.max_iter = params["max_iter"]
        self.threshold = params["threshold"]

        def c(xu,g_z):
            xu = xu[np.newaxis,:]
            x = xu[:,:self.state_size]
            u = xu[:,self.state_size:(self.state_size+self.output_size)]
            z = self.model._transform(x, u)
            #g_z = xugz[:,(self.input_size+self.output_size):]
            return self.model._state_cost(z, g_z)[0]

        def ct(x,g_z):
            x = x[np.newaxis,:]
            #x = x[:,:self.input_size]
            #g_z = xgz[:,self.input_size:]
            z = self.model._transform(x, np.zeros((x.shape[0],self.output_size)))
            return self.model._terminal_cost(z, g_z)[0]
        
        def f(xu):
            x = xu[np.newaxis,:self.state_size]
            u = xu[np.newaxis,self.state_size:(self.state_size+self.output_size)]
            return self.model._dynamics(x,u)[0]

        self.c = c
        self.ct = ct
        self.f = f

        self.Jacobian_cost = nd.Jacobian(c)
        self.Jacobian_terminal_cost = nd.Jacobian(ct)
        self.Hessian_cost = nd.Hessian(c)
        self.Hessian_terminal_cost = nd.Hessian(ct)
        self.Jacobian_dynamics = nd.Jacobian(f)
        self.Hessian_dynamics = nd.Hessian(f)
    
        self.pool = mp.Pool(cores)
        self.prev_sol = np.zeros((self.pred_length,self.output_size))

        self.init_mu = 1
        self.mu_min = 1e-6
        self.delta_zero = 2
        self.init_delta = self.delta_zero
        self.alphas = 1.1**(-np.arange(10)**2)

    def calc_action(self, state, g_z=None, goal_state=None):
        self.mu = self.init_mu
        self.delta = self.init_delta
        sol = self.prev_sol
        goal_state = np.array([goal_state])
        converged_sol = False
        accepted_solution = False

        print("Goal State",goal_state)
        print("PredLength",self.pred_length)
        if g_z is None:
            if goal_state is None:
                raise AttributeError("goal_state can't be null if no target trajectory g_z is given!")
            g_z = np.repeat(goal_state, self.pred_length,axis=0)
        elif len(np.array(g_z).shape) <= 1:
            raise AttributeError("g_z can't be 1-Dimensional")
        g_z = np.array(g_z)

        #actual algorithm

        for iter in range(self.max_iter):
            print("Iteration",iter)
            xs, cost = self.simulate_trajectory(state,sol,g_z)
            print("Cost:",cost)
            print("Mu:",self.mu)
            l_x,l_u,l_xx,l_uu,l_ux,f_x,f_u = lfs = self.derivatives(xs[:-1],sol,g_z)
            k, K = self.backward_pass(l_x,l_u,l_xx,l_uu,l_ux,f_x,f_u)
            #check if backward pass failed
            if False:
                self.delta = max(self.delta_zero,self.delta*self.delta_zero)
                self.mu = max(self.mu_min,self.mu*self.delta)
                continue
            #end check

            # begin line search   
            for alpha in self.alphas:
                print("Alpha:",alpha)
                new_xs, new_us, new_cost = self.forward_pass(alpha,k,K,xs,sol)
                # check if forward pass has diverged
                if new_cost < cost:
                    # check of 13
                    if np.abs((cost - new_cost) / cost) < self.threshold:
                        converged_sol = True
                        print("Solution Converged")
                        break

                    cost = new_cost
                    xs = new_xs
                    sol = new_us

                     # decrease mu for next iteration
                    self.delta = min(1/self.delta_zero,self.delta/self.delta_zero)
                    self.mu = 0 if self.mu*self.delta < self.mu_min else self.mu*self.delta

                    accepted_solution = True
                    
            if not accepted_solution:
                self.delta = max(self.delta_zero,self.delta*self.delta_zero)
                self.mu = max(self.mu_min,self.mu*self.delta)
            
            if converged_sol:
                break

        self.prev_sol[:-1] = sol[1:]
        self.prev_sol[-1] = sol[-1]

        #print(ls)
        return sol[0]


    #return with terminal state (pred_len+1,state_size)
    def simulate_trajectory(self,x,us,g_z):
        xs = np.zeros((self.pred_length+1,self.state_size))
        xs[0,:] = x
        #Simulation
        cost = 0
        for i in range(1,self.pred_length+1):
            newstate = self.model.predict(xs[i-1,:], us[i-1, :], goal=g_z[i-1, :])
            cost -= self.model.get_reward()
            xs[i, :] = newstate
        #Simulateend
        return xs, cost


    ############################################################################
    #
    #                           Derivative Pass
    #
    ############################################################################

    def step_derive(self,xu_t,gz_t):
        #xu_t = xu[t]
        #x_t = xs[t]
        #gz_t = g_z[t]
        jac = self.Jacobian_cost(xu_t, gz_t) # doesnt work
        #jac_t = self.Jacobian_terminal_cost(x_t, gz_t)
        hess = self.Hessian_cost(xu_t, gz_t)
        #hess_t = self.Hessian_terminal_cost(x_t, gz_t)

        l_x = jac[:,:self.state_size]
        l_u = jac[:,self.state_size:]
        l_xx = np.array([hess[i,i] for i in range(self.state_size)]) # this code is horrible but i know it
        l_uu = np.array([hess[i,i] for i in range(self.state_size,self.state_size + self.output_size)])
        l_ux = np.array([hess[-1,i] for i in range(self.state_size)])
        # print("jac",jac)
        # #print("jac_t",jac_t)
        # print("hess",hess)
        # #print("hess_t",hess_t)

        # print("l_x",l_x)
        # print("l_u",l_u)
        # print("l_xx",l_xx)
        # print("l_uu",l_uu)
        # print("l_ux",l_ux)
        return l_x,l_u,l_xx,l_uu,l_ux

    def step_derive_dynamics(self,xu_t):
        jac = self.Jacobian_dynamics(xu_t)
        #hess = self.Hessian_dynamics(xu_t) # doesnt work for some reason
        f_x = jac[:,:self.state_size]
        f_u = jac[:,self.state_size:]
        return f_x, f_u

    def derivatives(self, xs, us, g_z):

        xu = np.append(xs,us, axis=1)
        #Calculate derivatives

        l_xs = np.zeros((self.pred_length, self.state_size))
        l_us = np.zeros((self.pred_length, self.output_size))
        l_xxs = np.zeros((self.pred_length, self.state_size))
        l_uus = np.zeros((self.pred_length, self.output_size))
        l_uxs = np.zeros((self.pred_length, self.state_size))
        f_xs = np.zeros((self.pred_length, self.state_size,self.state_size))
        f_us = np.zeros((self.pred_length, self.state_size,self.output_size))

        for t in range(self.pred_length):
            l_x, l_u, l_xx, l_uu, l_ux = self.step_derive(xu[t],g_z[t])
            f_x, f_u = self.step_derive_dynamics(xu[t])
            l_xs[t,:] = l_x
            l_us[t,:] = l_u
            l_xxs[t,:] = l_xx
            l_uus[t,:] = l_uu
            l_uxs[t,:] = l_ux
            f_xs[t,:] = f_x
            f_us[t,:] = f_u

        #calculate terminal cost derivatives
        lx_T = self.Jacobian_terminal_cost(xs[-1],g_z[-1])[[0]]
        lxx_T = self.Hessian_terminal_cost(xs[-1],g_z[-1])
        lxx_T = np.array([[lxx_T[i,i] for i in range(self.state_size)]])
        l_xs = np.append(l_xs, lx_T, axis=0)
        l_xxs = np.append(l_xxs, lxx_T, axis=0)
        return l_xs,l_us,l_xxs,l_uus,l_uxs,f_xs,f_us

    ############################################################################
    #
    #                           Backward Pass
    #
    ############################################################################

    def _Q(self,l_x,l_u,l_xx,l_uu,l_ux,f_x,f_u,V_x,V_xx):
        # print("l_x",l_x.shape)
        # print("l_u",l_u.shape)
        # print("l_xx",l_xx.shape)
        # print("l_uu",l_uu.shape)
        # print("l_ux",l_ux.shape)

        # print("f_x",f_x.shape)
        # print("f_u",f_u.shape)

        # print("V_x",V_x.shape)
        # print("V_xx",V_xx.shape)
        Q_x = l_x + f_x.T @ V_x
        # print("Q_x",Q_x.shape)
        Q_u = l_u + f_u.T @ V_x
        # print("Q_u",Q_u.shape)
        Q_xx = l_xx + f_x.T @ V_xx @ f_x #+ V_x@f_xx
        # print("Q_xx",Q_xx.shape)
        #import pdb; pdb.set_trace()
        Q_uu = l_uu + f_u.T @ (V_xx + self.mu * np.eye(self.state_size)) @ f_u #+ V_x @ f_uu # 10a
        Q_ux = l_ux + f_u.T @ (V_xx + self.mu * np.eye(self.state_size)) @ f_x #+ V_x @ f_ux # 10b
        
        return Q_x, Q_u, Q_xx, Q_uu, Q_ux


    def backward_pass(self,l_x,l_u,l_xx,l_uu,l_ux,f_x,f_u):
        
        V_x = l_x[-1]
        V_xx = l_xx[-1]

        ks = np.zeros((self.pred_length, self.output_size))
        Ks = np.zeros((self.pred_length, self.output_size, self.state_size))

        for t in range(self.pred_length-1, -1,-1):
            Q_x, Q_u, Q_xx, Q_uu, Q_ux = self._Q(l_x[t],l_u[t],l_xx[None, t],l_uu[None, t],l_ux[None,t],f_x[t],f_u[t],V_x,V_xx)
            ks[t] = k = -Q_uu**-1 @ Q_u # 10c
            Ks[t] = K = -Q_uu**-1 @ Q_ux # 10d

            #DeltaV = 1/2 * k.T @ Q_uu @ k @ Q_u # 11a
            V_x    = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k # 11b
            V_xx  = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K # 11c

            V_xx = 0.5 * (V_xx + V_xx.T) 
        
        return ks,Ks
        


    ############################################################################
    #
    #                           Forward Pass
    #
    ############################################################################
    def forward_pass(self,alpha,k,K,xs,us):
        x_hat, u_hat = np.zeros_like(xs), np.zeros_like(us)
        x_hat[0] = xs[0]
        c_hat = 0
        for i in range(self.pred_length):
            u_hat[i] = us[i] + alpha*k[i] + K[i]@(x_hat[i]-xs[i]) # 12
            x_hat[i+1] = self.model.predict(x_hat[i],u_hat[i])
            c_hat -= self.model.last_reward
        
        return x_hat, u_hat, c_hat
