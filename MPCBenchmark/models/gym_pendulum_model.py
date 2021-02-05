import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from MPCBenchmark.models.model import Model


class PendulumModel(Model):

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.W = np.diag([1.0,.1,.001])
        self.W_t = np.diag([1.0,2.0,0.0])
        self.viewer = None
        self.last_u = None

        self.bounds_low = -self.max_torque
        self.bounds_high = self.max_torque

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def predict(self, current_state: np.ndarray, action: np.ndarray, goal = None) -> np.ndarray:
        current_state = current_state.reshape(1,-1)
        action = action.reshape(1,-1)
        print("cur",current_state.shape)
        print("act",action.shape)
        z = self._transform(current_state, action)
        if goal is None:
            goal = np.zeros(z.shape)
        costs = self._state_cost(z, goal)
        newstate = self._dynamics(current_state,action)

        self.last_reward = -costs[0]
        self.last_observation = newstate[0]
        self.last_u = action[0]
        return newstate

    def batch_predict(self, current_state: np.ndarray, action: np.ndarray, goal = None) -> np.ndarray:
        z = self._transform(current_state, action)
        if goal is None:
            goal = np.zeros(z.shape)
        costs = self._state_cost(z, goal)
        newstate = self._dynamics(current_state,action)

        self.last_reward = -costs
        self.last_observation = newstate
        self.last_u = action
        return newstate

    def _dynamics(self, x, u):
        g, m, l, dt = self.g, self.m, self.l, self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        th = x[:,[0]]
        thdot = x[:,[1]]
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        return np.append(newth, newthdot, axis=1)


    def _transform(self, x, u):
        th = angle_normalize(x[:, [0]])
        thdot = x[:, [1]]
        z = np.append(np.append(th, thdot, axis=1), u, axis=1)
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


