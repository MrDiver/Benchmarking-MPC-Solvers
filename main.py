import src.envs.Pendulum_CC as pend
import src.envs.Cartpole_CC as cartp
import old.controller.MPPI3 as mppi
import numpy as np

import gym

dec = 3

def F(x,u):
    th_cos = np.round(np.arccos(x[0]),dec)
    th_sin = np.round(np.arcsin(x[1]),dec)
    th_dot = x[2]
    th = (th_cos * (1 if th_sin > 0 else -1))
    
    max_speed = 8
    max_torque = 2.
    dt = .05
    g = 10
    m = 1.
    l = 1.
    
    def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)
    
    u = np.clip(u, -max_torque, max_torque)[0]
    costs = angle_normalize(th) ** 2 + .1 * th_dot ** 2 + .001 * (u ** 2)

    newthdot = th_dot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -max_speed, max_speed)

    return np.array([0,0,0,0])
    
tc = (lambda x: 0)

def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)
    
def ic (x,u):
    th_cos = np.round(np.arccos(x[0]),dec)
    th_sin = np.round(np.arcsin(x[1]),dec)
    th_dot = x[2]
    
    th = (th_cos * (1 if th_sin > 0 else -1))
    costs = angle_normalize(th) ** 2 + .1 * th_dot ** 2 + .001 * (u ** 2)
    return costs
#ic = (lambda x,u: (1-x[0])**2 + 0.1*(x[2])**2)

solver = mppi.MPPI3(K=100,T=20,output_size=1,input_size=3,F=pend.F,Sigma=np.diag([10]),tc=tc,ic=ic,l=1)

env = pend.PendulumEnv()#cartp.CartPoleEnv()
#ENV_NAME = "Pendulum-v0"
#env = gym.make(ENV_NAME)
env.seed(0)
obs = env.reset()
print("initial state",obs)
#input()
for t in range(1000):
    u = solver.step(t,obs)
    #ownobs = F(obs,u)
    env.seed(0)
    obs,reward,done,something = env.step(u)
    # print("GT",obs)
    # print("PR",ownobs)
    # print(np.round(obs)==np.round(ownobs))
    print(u,"with cost",-reward)
    env.render()
    if(done):
        env.reset()
    
#input()