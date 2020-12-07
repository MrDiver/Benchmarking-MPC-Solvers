from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np

from src.MPPI import MPPI

solver = MPPI(100,100,(1,1))
def step(t,q,qd):
    u = solver.step(t,q,qd)
    solver.print_state()
    return u

def main():
    model = load_model_from_path("mjxml/inverted_pendulum.xml")
    sim = MjSim(model=model)
    viewer = MjViewer(sim=sim)

    sim_state = sim.get_state()

    print(sim_state)
    sim.set_state(sim_state)
    for i in range(1000):
        state = sim.get_state()
        t,q,qd = state.time,state.qpos,state.qvel
        sim.data.ctrl[:] = step(t,q,qd)
        print(sim.data.ctrl)
        viewer.render()
        sim.step()

main()