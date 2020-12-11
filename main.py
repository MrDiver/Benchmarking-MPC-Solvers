#import scripts.envs.Pendulum_CC as pend
#import scripts.envs.Cartpole_CC as cartp
import old.controller.MPPI3 as mppi
import numpy as np

import gym
import gym_cartpole_swingup

dec = 3

# def F(x,u):
#     th_cos = np.round(np.arccos(x[0]),dec)
#     th_sin = np.round(np.arcsin(x[1]),dec)
#     th_dot = x[2]
#     th = (th_cos * (1 if th_sin > 0 else -1))
    
#     max_speed = 8
#     max_torque = 2.
#     dt = .05
#     g = 10
#     m = 1.
#     l = 1.
    
#     def angle_normalize(x):
#         return (((x+np.pi) % (2*np.pi)) - np.pi)
    
#     u = np.clip(u, -max_torque, max_torque)[0]
#     costs = angle_normalize(th) ** 2 + .1 * th_dot ** 2 + .001 * (u ** 2)

#     newthdot = th_dot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
#     newth = th + newthdot * dt
#     newthdot = np.clip(newthdot, -max_speed, max_speed)

#     return np.array([0,0,0,0])
    
tc = (lambda x: 0)
ic = (lambda x,u: 0)

def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)
    
# def ic (x,u):
#     th_cos = np.round(np.arccos(x[0]),dec)
#     th_sin = np.round(np.arcsin(x[1]),dec)
#     th_dot = x[2]
    
#     th = (th_cos * (1 if th_sin > 0 else -1))
#     costs = angle_normalize(th) ** 2 + .1 * th_dot ** 2 + .001 * (u ** 2)
#     return costs

#ic = (lambda x,u: 3*(np.pi-x[2])**2 + 0.1*x[1]**2 + 0.2*x[3]**2 + 3*(np.abs(x[0])+1)**2)
#tc = (lambda x: 5*(np.abs(x[0])+1)**2 + 1*x[1]**2 + 3*x[3]**2)


#env = pend.PendulumEnv()#cartp.CartPoleEnv()
ENVIRONMENT = "CartPoleSwingUp-v0"
ENVIRONMENT = "MountainCarContinuous-v0"
ENVIRONMENT = "Pendulum-v0"
ENVIRONMENT = "FetchPush-v1"
env = gym.make(ENVIRONMENT)
#ENV_NAME = "Pendulum-v0"
#env = gym.make(ENV_NAME)

obs = env.reset()
print("initial state", obs)

e = gym.make(ENVIRONMENT)
e.reset()


def F(x,u):
    e.env.state = x
    obs,r,done,_ = e.step(u)
    newx = e.env.state
    return newx, r, obs

def MJF(x,u):
    l = len(x)//2
    e.env.set_state(x[:l],x[:l])
    obs,r,done,_ = e.step(u)
    newx = np.append(env.sim.data.qpos,env.sim.data.qvel)
    return newx,r,obs

solver = mppi.MPPI3(K=100,T=100,output_size=1,input_size=6,F=F,Sigma=10,tc=tc,ic=ic,l=0.5)
#input()

is_mujoco = False

current_state = None
for t in range(1000):
    if is_mujoco:
        current_state = np.append(env.sim.data.qpos,env.sim.data.qvel)
    else:
        current_state = env.env.state
    u = solver.step(t,obs,current_state)
    #ownobs = F(obs,u)
    #env.seed(0)
    obs,reward,done,something = env.step(u)
    # print("GT",obs)
    # print("PR",ownobs)
    # print(np.round(obs)==np.round(ownobs))
    print(u,"with cost \t",reward)
    #print(env.env.state)
    #print(obs)
    env.render()
    if(done):
        env.reset()


#input()