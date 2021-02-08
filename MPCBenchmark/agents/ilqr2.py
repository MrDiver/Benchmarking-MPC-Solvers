import numpy as np
import numdifftools as nd

from MPCBenchmark.agents.agent import Agent


class ILQR(Agent):
    def __init__(self, bounds_low, bounds_high, input_size, output_size, model, params) -> None:
        super().__init__(bounds_low,bounds_high,input_size,output_size,model)
        self.pred_length = params["T"]

        def c(xugz):
            xugz = xugz[np.newaxis,:]
            x = xugz[:,:self.input_size]
            u = xugz[:,self.input_size:(self.input_size+self.output_size)]
            z = self.model._transform(x, u)
            g_z = xugz[:,(self.input_size+self.output_size):]
            return self.model._state_cost(z, g_z)[0]

        def ct(xgz):
            xgz = xgz[np.newaxis,:]
            x = xgz[:,:self.input_size]
            g_z = xgz[:,self.input_size:]
            z = self.model._transform(x, np.zeros((x.shape[0],self.output_size)))
            return self.model._terminal_cost(z, g_z)[0]

        self.c = c
        self.ct = ct
        self.Jacobian_cost = nd.Jacobian(c)
        self.Jacobian_terminal_cost = nd.Jacobian(ct)
        self.Hessian_cost = nd.Hessian(c)
        self.Hessian_terminal_cost = nd.Hessian(ct)
    
        self.prev_sol = np.zeros((self.pred_length,self.output_size))

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
        self.derivatives(state,sol,g_z)

        return np.array([0])


    def derivatives(self, x, u, g_z):
        xs = np.zeros((self.pred_length,self.input_size))

        #Simulation

        #Simulateend

        z = self.model._transform(xs,u)
        
        xugz = np.append(z,g_z,axis=1)
        xgz = np.append(xs,g_z,axis=1)
        print(xugz)
        print(xgz)

        print(self.c(xugz[0])) #works fine
        print(self.ct(xgz[0])) #works fine
        print(xugz[[0]]) #[[0. 0. 0. 1. 1. 0.]]
        jac = self.Jacobian_cost(xugz[0]) # doesnt work
        print("jac",jac)
        input()
        #jac_t = self.Jacobian_terminal_cost(xgz)
        #hess = self.Hessian_cost(xugz)
        #hess_t = self.Hessian_terminal_cost(xgz)

        return

    def backward_pass(self,l_x,l_u,l_xx,l_uu,l_ux,f_x,f_u,f_xx,f_uu,f_ux,V_x,V_xx,mu):


        Q_x = l_x + f_x.T @ V_x
        Q_u = l_u + f_u.T @ V_x
        Q_xx = l_xx + f_x.T @ V_xx @ f_x + V_x@f_xx

        Q_uu = l_uu + f_u.T @ (V_xx + mu @ np.eye(n)) @ f_u + V_x @ f_uu # 10a
        Q_ux = l_ux + f_u.T @ (V_xx + mu @ np.eye(n)) @ f_x + V_x @ f_ux # 10b

        k = -Q_uu**-1 @ Q_u # 10c
        K = -Q_uu**-1 @ Q_ux # 10d

        DeltaV = 1/2 * k.T @ Q_uu @ k @ Q_u # 11a
        V_x    = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k # 11b
        V_xx  = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K # 11c

