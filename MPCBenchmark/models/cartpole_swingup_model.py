from MPCBenchmark.models.model import Model as Model

import gym
import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CartParams:
    """Parameters defining the Cart."""

    width: float = 1 / 3
    height: float = 1 / 6
    mass: float = 0.5


@dataclass(frozen=True)
class PoleParams:
    """Parameters defining the Pole."""

    width: float = 0.05
    length: float = 0.6
    mass: float = 0.5




State = namedtuple("State", "x_pos x_dot theta theta_dot")

@dataclass
class CartPoleSwingUpParams:  # pylint: disable=no-member,too-many-instance-attributes
    """Parameters for physics simulation."""

    gravity: float = 9.82
    forcemag: float = 10.0
    deltat: float = 0.01
    friction: float = 0.1
    x_threshold: float = 2.4
    cart: CartParams = field(default_factory=CartParams)
    pole: PoleParams = field(default_factory=PoleParams)
    masstotal: float = field(init=False)
    mpl: float = field(init=False)

    def __post_init__(self):
        self.masstotal = self.cart.mass + self.pole.mass
        self.mpl = self.pole.mass * self.pole.length


class CartpoleSwingupModel(Model):
    def __init__(self,) -> None:
        super().__init__()
        self.low = -1.0
        self.high = 1.0
        self.params = CartPoleSwingUpParams()
        self.state = None

    def _transition_fn(self, state, action):
        # pylint: disable=no-member
        action = action[0] * self.params.forcemag

        sin_theta = np.sin(state.theta)
        cos_theta = np.cos(state.theta)

        xdot_update = (
            -2 * self.params.mpl * (state.theta_dot ** 2) * sin_theta
            + 3 * self.params.pole.mass * self.params.gravity * sin_theta * cos_theta
            + 4 * action
            - 4 * self.params.friction * state.x_dot
        ) / (4 * self.params.masstotal - 3 * self.params.pole.mass * cos_theta ** 2)
        thetadot_update = (
            -3 * self.params.mpl * (state.theta_dot ** 2) * sin_theta * cos_theta
            + 6 * self.params.masstotal * self.params.gravity * sin_theta
            + 6 * (action - self.params.friction * state.x_dot) * cos_theta
        ) / (
            4 * self.params.pole.length * self.params.masstotal
            - 3 * self.params.mpl * cos_theta ** 2
        )

        delta_t = self.params.deltat
        return State(
            x_pos=state.x_pos + state.x_dot * delta_t,
            theta=state.theta + state.theta_dot * delta_t,
            x_dot=state.x_dot + xdot_update * delta_t,
            theta_dot=state.theta_dot + thetadot_update * delta_t,
        )


    @staticmethod
    def _get_obs(state):
        x_pos, x_dot, theta, theta_dot = state
        return np.array(
            [x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32
        )

    @staticmethod
    def _reward_fn(state, action, next_state):
        return (1 + np.cos(next_state.theta, dtype=np.float32)) / 2

    def _terminal(self, state):
        return bool(abs(state.x_pos) > self.params.x_threshold)

    def predict(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # if(np.clip(action, self.bounds_low, self.bounds_high) != action):
        #    print("RuntimeWarning: Actions out of action space for this model")
        # Valid action

        action = np.clip(action, self.low, self.high)
        self.state = next_state = self._transition_fn(current_state, action)
        self.last_observation = self._get_obs(next_state)
        self.last_reward = self._reward_fn(current_state, action, next_state)

        return next_state

    def batch_predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError
