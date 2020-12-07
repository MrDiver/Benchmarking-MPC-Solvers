from mujoco_py import load_model_from_path, MjSim, MjSimState, MjViewer
import numpy as np
import matplotlib.pyplot as plt

from src.MPPI import MPPI

model = load_model_from_path("mjxml/inverted_pendulum.xml")

tmpsim = sim = MjSim(model)
def TransitionModel(q,qd,u):
    tmpsim.set_state(MjSimState(1,q,qd,u,{}))
    tmpsim.data.ctrl[:] = u
    tmpsim.step()
    state = tmpsim.get_state()
    return state.qpos,state.qvel
    

ic = (lambda q,qd,u,: q[1]**2 + 0.1*qd[1]**2 + 0.001*u**2)
tc = (lambda q,qd: 0.01*q[0]**2 + 0.01*qd[0]**2 + q[1]**2 + 0.1*qd[1]**2)
#ic = (lambda q,qd,u,: q[0]**2 + u**4)
#tc = (lambda q,qd: q[0]**2 )
solver = MPPI(50,20,(1,1),F=TransitionModel,Sigma=np.diag([0.9]),tc=tc,ic=ic)

def step(t,q,qd):
    u = solver.step(t,q,qd)
    #solver.print_state()
    return u

def main():
    
    sim = MjSim(model=model)
    viewer = MjViewer(sim=sim)

    sim_state = sim.get_state()
    bounds = model.actuator_ctrlrange.copy().astype(np.float32)
    print(bounds)
    print(sim_state)
    sim.set_state(sim_state)
    for iteration in range(1000):
        state = sim.get_state()
        t,q,qd = state.time,state.qpos,state.qvel
        u = step(t,q,qd)
        sim.data.ctrl[:] = u
        print(iteration)
        print("u",u)
        print("q",q)
        print("qd",qd)
        print("cost:",ic(q,qd,u))
        print("tcost:",tc(q,qd))
        viewer.add_marker(pos=[u/5+q[0],0,0])
        #viewer.add_marker(pos=[q[1],1,0])
        viewer.render()
        sim.step()
    plt.show()

x = compile(main(),optimize=3)
exec(x)

#print(sim.data.body_xmat)
        #plt.scatter(i,sim.data.body_xmat[2,0])
        #print("\n geom\n",sim.data.geom_xpos)
        #print("\n anchor\n",sim.data.xanchor)
        #viewer.add_marker(pos=[sim.data.site_xpos[0]])
        #viewer.add_marker(pos=[sim.data.site_xpos[1]])