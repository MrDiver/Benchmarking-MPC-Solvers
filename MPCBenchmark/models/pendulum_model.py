import gym
import numpy as np
from os import path
from MPCBenchmark.models.model import Model


class PendulumModel(Model):
    max_torque = 2.
    bounds_low = -np.array([max_torque])
    bounds_high = np.array([max_torque])
    action_size = 1
    state_size = 2

    def __init__(self, g=10.0):
        super().__init__("Pendulum Model (No Noise)")
        self.max_speed = 8
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.W = np.diag([1.0, .1, .001])
        self.W_t = np.diag([1.0, 2.0, 0.0])
        self.viewer = None
        self.last_u = None

        self.seed()

    def _dynamics(self, x, u):
        g, m, l, dt = self.g, self.m, self.l, self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        th = x[:, [0]]
        thdot = x[:, [1]]
        newthdot = thdot + (-3 * g / (2 * l) *
                            np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        return np.append(newth, newthdot, axis=1)

    def _transform(self, x, u):
        th = angle_normalize(x[:, [0]])
        thdot = x[:, [1]]
        z = -np.append(np.append(th, thdot, axis=1), u, axis=1)
        return z

    def _state_cost(self, z, g_z):
        _zd = z-g_z
        #costs = [(z @ self.W) @ z.T for z in _zd]
        costs = np.einsum("bi,ij,bj->b", _zd, self.W, _zd)
        return costs

    def _terminal_cost(self, x, g_x):
        _zd = x-g_x
        costs = np.einsum("bi,ij,bj->b", _zd, self.W_t, _zd)
        return costs


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
