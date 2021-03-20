from MPCBenchmark.envs.env import Environment
from MPCBenchmark.models.acrobot_model import AcrobotModel
import numpy as np


class AcrobotEnv(Environment):

    def __init__(self):
        super().__init__("AcrobotEnvironment")
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 15
        }

        self.viewer = None
        self.model = AcrobotModel()

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.model.LINK_LENGTH_1 + \
                self.model.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1 = [-self.model.LINK_LENGTH_1 *
              np.cos(s[0]), self.model.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.model.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.model.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi/2, s[0]+s[1]-np.pi/2]
        link_lengths = [self.model.LINK_LENGTH_1, self.model.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _done(self):
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)
