from mujoco_py import load_model_from_path, MjSim, MjSimState, MjViewer
import numpy as np
import matplotlib.pyplot as plt
from src.MPPI2 import MPPI2
from src.MPPI import MPPI

model = load_model_from_path("mjxml/inverted_pendulum.xml")

tmpsim = sim = MjSim(model)
def TransitionModel(q,qd,u):
    tmpsim.set_state(MjSimState(1,q,qd,u,{}))
    tmpsim.data.ctrl[:] = u
    tmpsim.step()
    state = tmpsim.get_state()
    return state.qpos,state.qvel

def TransitionModel2(x,u):
    l = x.shape[0]//2
    tmpsim.set_state(MjSimState(0,x[:l],x[l:],u,{}))
    tmpsim.data.ctrl[:] = u
    tmpsim.step()
    state = tmpsim.get_state()
    return np.append(state.qpos,state.qvel)
    

ic2 = (lambda q,qd,u,: 10*q[1]**2 + qd[1]**2)
tc2 = (lambda q,qd: q[0]**2 )
#ic = (lambda q,qd,u,: q[0]**2 + u**4)
ic = (lambda x,u,: 10*x[1]**2 + x[3]**2)
tc = (lambda x: x[0]**2 )

solver = MPPI2(50,20,1,4,F=TransitionModel2,Sigma=np.eye(1)*0.01,tc=tc,ic=ic,l=0.25)
solver2 = MPPI(50,20,(1,1),F=TransitionModel,Sigma=np.eye(1)*0.01,tc=tc2,ic=ic2,l=0.25)
def main():
    
    sim = MjSim(model=model)
    viewer = MjViewer(sim=sim)

    sim_state = sim.get_state()
    bounds = model.actuator_ctrlrange.copy().astype(np.float32)
    print(bounds)
    print(sim_state)
    
    maxtime = 1000
    
    sim.set_state(sim_state)
    for iteration in range(maxtime):
        state = sim.get_state()
        t,q,qd = state.time,state.qpos,state.qvel
        u = solver.step(t,np.append(q,qd))
        #u2 = solver2.step(t,np.append(q,qd))
        #u = np.round(100*q[1])/10 + 1*(q[0]-q[1])
        sim.data.ctrl[:] = u
        # print(iteration)
        # print("u",u)
        # print("q",q)
        # print("qd",qd)
        # print("cost:",ic(q,qd,u))
        # print("tcost:",tc(q,qd))
        
        viewer.add_marker(pos=[u+q[0],0,0])
        #print(u,u2)
        #viewer.add_marker(pos=[qd[1],1,0])
        viewer.render()
        sim.step()

main()

#print(sim.data.body_xmat)
        #plt.scatter(i,sim.data.body_xmat[2,0])
        #print("\n geom\n",sim.data.geom_xpos)
        #print("\n anchor\n",sim.data.xanchor)
        #viewer.add_marker(pos=[sim.data.site_xpos[0]])
        #viewer.add_marker(pos=[sim.data.site_xpos[1]])