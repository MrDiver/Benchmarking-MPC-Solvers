import numpy as np

from MPCBenchmark.i2c.quadrature_inf import QuadratureInf


def dynamics(x, u):
    dt = 0.05
    m = 1.
    l = 1.
    d = 5e-1  # damping
    g = 9.80665
    u_mx = 2.
    u = np.clip(u, -u_mx, u_mx)
    th_dot_dot = -3.0 * g / (2 * l) * np.sin(x[:, 0] + np.pi) - d * x[:, 1]
    th_dot_dot += 3.0 / (m * l ** 2) * u.squeeze()
    x_dot = x[:, 1] + th_dot_dot * dt
    x_pos = x[:, 0] + x_dot * dt
    _x = np.vstack((x_pos, x_dot)).T
    return _x


def observe(x):
    return np.array([np.sin(x[:, 0]), np.cos(x[:, 0])]).T


class kalman_curb:

    def __init__(self, mu, sig, eta, zeta) -> None:
        super().__init__()

        self.mu = mu
        self.sig = sig

        self.measure_infer = QuadratureInf(1, 0, 0, 2)
        self.dynamics_infer = QuadratureInf(1, 0, 0, 2)

        self.mus_filt = []  # saving the filtered mu & covariance
        self.sigs_filt = []

        self.pred_mu = []
        self.pred_sig = []
        self.pred_sig_d = []

        self.mu_smo = self.mu
        self.sig_smo = self.sig

    def __call__(self, u, y):
        """
        :param u: torque
        :param y: true measurement
        :return: updated value for the state of current time step, its variance and observation
        """
        mu_x, sig_x, sig_d = self.predict_step(u)
        self.pred_mu.append(mu_x)
        self.pred_sig.append(sig_x)
        self.pred_sig_d.append(sig_d)

        self.update_step(mu_x, sig_x, y)
        self.mus_filt.append(self.mu)
        self.sigs_filt.append(self.sig)

        return self.mu[0, :], self.sig, observe(self.mu)[0, :]

    def predict_step(self, u):
        """
        :param u: torque
        :return: prediction of mu, sigma, cov_sigma, cross_cov sigma
        """
        dynam = lambda x: dynamics(x, u)
        mu_x, sig_x, sig_d = self.dynamics_infer(dynam, self.mu, self.sig)
        sig_x += self.sig_eta
        return mu_x, sig_x, sig_d

    def update_step(self, mu, sig, y):
        """
        Updates the value of mu and sigma by using measurement and prediction
        :param mu: predicted mu
        :param sig: predicted sigma
        :param y: true measurement
        """
        mu_y, sig_y, sig_xy = self.measure_infer(observe, mu, sig)
        sig_y += self.sig_zeta
        K = np.linalg.solve(sig_y.T, sig_xy.T).T
        self.mu = mu + (y.reshape((1, -1)) - mu_y) @ K.T
        self.sig = sig - K @ sig_y @ K.T

    def smooth(self, t, end):
        """
        Smoothing goes backwards in time and updates all values for mu, sigma for all states
        and their observation
        :param t: current time step
        :param end: of trajectory
        :return: smoothed value for current time step, its variance and observation
        """
        if end:
            self.mu_smo = self.mus_filt[t][0, :]
            self.sig_smo = self.sigs_filt[t]
        else:
            C = np.linalg.solve(self.sigs_filt[t].T, self.pred_sig_d[t+1]).T
            self.mu_smo = self.mus_filt[t][0, :] + C @ (self.mu_smo - self.pred_mu[t+1])
            self.sig_smo = self.sigs_filt[t] + C @ (self.sig_smo - self.pred_sig[t+1]) @ C.T
        return self.mu_smo, self.sig_smo, observe(self.mu_smo[:, None])[0, :]