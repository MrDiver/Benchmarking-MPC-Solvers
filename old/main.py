from mujoco_py import load_model_from_path, MjSim, MjSimState, MjViewer
import numpy as np
import matplotlib.pyplot as plt
from controller.MPPI3 import MPPI3

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
    

ic = (lambda x,u,: x[1]**2+x[0]**2)
tc = (lambda x: 10*x[0]**2+x[1]**2+x[2]**2+x[3]**2)

solver = MPPI3(100,20,1,4,F=TransitionModel2,Sigma=np.eye(1)*3,tc=tc,ic=ic,l=10)
def main():
    
    sim = MjSim(model=model)
    viewer = MjViewer(sim=sim)

    sim_state = sim.get_state()
    bounds = model.actuator_ctrlrange.copy().astype(np.float32)
    #print(bounds)
    #print(sim_state)
    
    maxtime = 1000
    
    sim.set_state(sim_state)
    for iteration in range(maxtime):
        state = sim.get_state()
        t,q,qd = state.time,state.qpos,state.qvel
        u = solver.step(t,np.append(q,qd))
        print(u)
        sim.data.ctrl[:] = np.clip(u,bounds[:,0],bounds[:,1])
        viewer.add_marker(pos=[u/10,0,0])
        viewer.render()
        sim.step()
        #print(iteration)
        print(q,qd)

main()

#print(sim.data.body_xmat)
        #plt.scatter(i,sim.data.body_xmat[2,0])
        #print("\n geom\n",sim.data.geom_xpos)
        #print("\n anchor\n",sim.data.xanchor)
        #viewer.add_marker(pos=[sim.data.site_xpos[0]])
        #viewer.add_marker(pos=[sim.data.site_xpos[1]])