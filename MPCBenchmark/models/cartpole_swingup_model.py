"""
Cart pole swing-up: Original version from:
https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py

Modified so that done=True when x is outside of -2.4 to 2.4
Reward is also reshaped to be similar to PyBullet/roboschool version

More difficult, since dt is 0.05 (not 0.01), and only 200 timesteps
"""
from MPCBenchmark.models.model import Model
import gym
import numpy as np


class CartPoleSwingUpModel(Model):
    def __init__(self):
        super().__init__()
        # TODO: maybe change rendering to use the model instead
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.6  # pole's length
        self.m_p_l = (self.m_p*self.l)
        self.force_mag = 10.0
        self.dt = 0.01  # seconds between state updates
        self.b = 0.1  # friction coefficient

        self.t = 0  # timestep
        self.t_limit = 1000

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.viewer = None
        self.state = None

        self.last_reward = 0
        self.last_observation = None
        self.bounds_low = np.array([-1])
        self.bounds_high = np.array([1])
        self.state_size = 4
        self.action_size = 1
        self.W = -np.diag([1, 0, 1, 0, 0])
        self.W_t = -np.diag([5, 0, 1, 0, 0])

    def reset(self):
        #self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        self.state = np.random.normal(loc=np.array(
            [0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))
        self.steps_beyond_done = None
        return self.state

    def get_reward(self):
        return self.last_reward

    def get_observation(self):
        return self.last_observation

    # used for linearization algorithms to have access to the dynamics

    def _dynamics(self, x, u):

        action = np.clip(u, -1.0, 1.0)[:, 0]
        action *= self.force_mag

        state = x
        x, x_dot, theta, theta_dot = state[:, [
            0]], state[:, [1]], state[:, [2]], state[:, [3]]

        s = np.sin(theta)
        c = np.cos(theta)

        xdot_update = (-2*self.m_p_l*(theta_dot**2)*s + 3*self.m_p*self.g *
                       s*c + 4*action - 4*self.b*x_dot)/(4*self.total_m - 3*self.m_p*c**2)
        thetadot_update = (-3*self.m_p_l*(theta_dot**2)*s*c + 6*self.total_m*self.g *
                           s + 6*(action - self.b*x_dot)*c)/(4*self.l*self.total_m - 3*self.m_p_l*c**2)
        x = x + x_dot*self.dt
        theta = theta + theta_dot*self.dt
        x_dot = x_dot + xdot_update*self.dt
        theta_dot = theta_dot + thetadot_update*self.dt
        newstate = np.append(
            np.append(np.append(x, x_dot, axis=1), theta, axis=1), theta_dot, axis=1)
        return newstate

    def _transform(self, x, u):
        xc, x_dot, theta, theta_dot = x[:, [
            0]], x[:, [1]], x[:, [2]], x[:, [3]]

        tmp = 1
        if (xc > self.x_threshold).any() or (xc < -self.x_threshold).any():
            tmp = -10000
        xc = np.cos((xc/self.x_threshold)*(np.pi/2.0))
        theta = np.cos(theta)+1.0
        xc = xc * theta * tmp
        theta = np.zeros_like(theta)
        return np.append(np.append(np.append(np.append(xc, x_dot, axis=1), theta, axis=1), theta_dot, axis=1), u, axis=1)

    def _state_cost(self, z, g_z):
        #reward_theta = (np.cos(theta)+1.0)/2.0
        #reward_x = np.cos((x/self.x_threshold)*(np.pi/2.0))

        #reward = reward_theta*reward_x

        _zd = z-g_z
        #costs = [(z @ self.W) @ z.T for z in _zd]
        costs = np.einsum("bi,ij,bj->b", _zd, self.W, _zd)
        return costs

    def _terminal_cost(self, x, g_x):
        _zd = x-g_x

        costs = np.einsum("bi,ij,bj->b", _zd, self.W_t, _zd)
        return costs
