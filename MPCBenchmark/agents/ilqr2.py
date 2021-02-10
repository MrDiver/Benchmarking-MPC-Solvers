import numpy as np
import numdifftools as nd

from MPCBenchmark.agents.agent import Agent
import multiprocessing as mp

class ILQR(Agent):
    def __init__(self, bounds_low, bounds_high, input_size, output_size, model, params, cores=12) -> None:
        super().__init__(bounds_low,bounds_high,input_size,output_size,model)
        self.pred_length = params["T"]

        def c(xu,g_z):
            xu = xu[np.newaxis,:]
            x = xu[:,:self.input_size]
            u = xu[:,self.input_size:(self.input_size+self.output_size)]
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
            x = xu[np.newaxis,:self.input_size]
            u = xu[np.newaxis,self.input_size:(self.input_size+self.output_size)]
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

        self.mu = 1

    def calc_action(self, state, g_z=None, goal_state=None):
        sol = self.prev_sol
        goal_state = np.array([goal_state])

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
        l_x,l_u,l_xx,l_uu,l_ux,f_x,f_u = lfs = self.derivatives(state,sol,g_z)
        self.backward_pass(l_x,l_u,l_xx,l_uu,l_ux,f_x,f_u)

        #print(ls)
        return np.array([0])


    def step_derive(self,xu_t,gz_t):
        #xu_t = xu[t]
        #x_t = xs[t]
        #gz_t = g_z[t]
        jac = self.Jacobian_cost(xu_t, gz_t) # doesnt work
        #jac_t = self.Jacobian_terminal_cost(x_t, gz_t)
        hess = self.Hessian_cost(xu_t, gz_t)
        #hess_t = self.Hessian_terminal_cost(x_t, gz_t)

        l_x = jac[:,:self.input_size]
        l_u = jac[:,self.input_size:]
        l_xx = np.array([hess[i,i] for i in range(self.input_size)]) # this code is horrible but i know it
        l_uu = np.array([hess[i,i] for i in range(self.input_size,self.input_size + self.output_size)])
        l_ux = np.array([hess[-1,i] for i in range(self.input_size)])
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
        f_x = jac[:,:self.input_size]
        f_u = jac[:,self.input_size:]
        return f_x, f_u

    def derivatives(self, x, us, g_z):
        xs = np.zeros((self.pred_length,self.input_size))
        xs[0,:] = x
        #Simulation
        cost = 0
        for i in range(1,self.pred_length):
            newstate = self.model.predict(xs[i-1,:], us[i-1, :], goal=g_z[i-1, :])
            cost -= self.model.get_reward()
            xs[i, :] = newstate
        #Simulateend

        xu = np.append(xs,us, axis=1)
        

        #for t in range(self.pred_length-1):
        #_inputs = [(xu[t],g_z[t]) for t in range(self.pred_length-1)]
        #print(_inputs)

        #Calculate derivatives

        l_xs = np.zeros((self.pred_length, self.input_size))
        l_us = np.zeros((self.pred_length, self.output_size))
        l_xxs = np.zeros((self.pred_length, self.input_size))
        l_uus = np.zeros((self.pred_length, self.output_size))
        l_uxs = np.zeros((self.pred_length, self.input_size))
        f_xs = np.zeros((self.pred_length, self.input_size,self.input_size))
        f_us = np.zeros((self.pred_length, self.input_size,self.output_size))

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
        lxx_T = np.array([[lxx_T[i,i] for i in range(self.input_size)]])
        l_xs = np.append(l_xs, lx_T, axis=0)
        l_xxs = np.append(l_xxs, lxx_T, axis=0)
        print(f_xs)
        return l_xs,l_us,l_xxs,l_uus,l_uxs,f_xs,f_us


    def _Q(self,l_x,l_u,l_xx,l_uu,l_ux,f_x,f_u,V_x,V_xx):

        print("l_x",l_x.shape)
        print("l_u",l_u.shape)
        print("l_xx",l_xx.shape)
        print("l_uu",l_uu.shape)
        print("l_ux",l_ux.shape)

        print("f_x",f_x.shape)
        print("f_u",f_u.shape)

        print("V_x",V_x.shape)
        print("V_xx",V_xx.shape)
        Q_x = l_x + f_x.T @ V_x
        print("Q_x",Q_x)
        Q_u = l_u + f_u.T @ V_x
        print("Q_u",Q_u)
        Q_xx = l_xx + f_x.T @ V_xx @ f_x #+ V_x@f_xx

        Q_uu = l_uu + f_u.T @ (V_xx + self.mu @ np.eye(self.input_size)) @ f_u #+ V_x @ f_uu # 10a
        Q_ux = l_ux + f_u.T @ (V_xx + self.mu @ np.eye(self.input_size)) @ f_x #+ V_x @ f_ux # 10b
        
        return Q_x, Q_u, Q_xx, Q_uu, Q_ux



    def backward_pass(self,l_x,l_u,l_xx,l_uu,l_ux,f_x,f_u):
        
        V_x = l_x[-1]
        V_xx = l_xx[-1]

        ks = np.zeros((self.pred_length, self.output_size))
        Ks = np.zeros((self.pred_length, self.output_size, self.input_size))

        for t in range(self.pred_length-1, -1,-1):
            Q_x, Q_u, Q_xx, Q_uu, Q_ux = self._Q(l_x[t],l_u[t],l_xx[t],l_uu[t],l_ux[t],f_x[t],f_u[t],V_x,V_xx)
            ks[t] = k = -Q_uu**-1 @ Q_u # 10c
            Ks[t] = K = -Q_uu**-1 @ Q_ux # 10d

            #DeltaV = 1/2 * k.T @ Q_uu @ k @ Q_u # 11a
            V_x    = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k # 11b
            V_xx  = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K # 11c

            V_xx = 0.5 * (V_xx + V_xx.T) 
        
        return ks,Ks
        

    


