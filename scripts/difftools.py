import numpy as np
import numdifftools as nd
from MPCBenchmark.models.gym_pendulum_model import PendulumModel

model = PendulumModel()

xs = np.array([[np.pi, 0], [0,1]])
us = np.array([[2], [2]])


z = model._transform(xs, us)
print(z)
cost = model._state_cost(z, np.zeros(z.shape))
print("costs1",cost)

print("d:",model._dynamics(xs,us))