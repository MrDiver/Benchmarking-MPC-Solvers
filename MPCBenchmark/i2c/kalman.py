import numpy as np


class kalman:

    def __init__(self, A, a, B, eta, zeta, C, y_t, mu_t, var_t) -> None:
        super().__init__()
        self.A = A
        self.a = a
        self.B = B
        self.eta = eta
        self.C = C
        self.y_t = y_t
        self.zeta = zeta

        self.mu_t = mu_t
        self.var_t = var_t
        self.mus = []
        self.sigs = []
        self.mus.append(self.mu_t)
        self.sigs.append(self.var_t)

    def dynamics(self, x, u):
        return self.A @ x + self.a + self.B @ u

    def calc_filter(self, u, y):
        self.predict_step(self.A, self.eta, u)
        self.update_step(self.C, self.zeta, y)
        # return the mean of state and the mean of observation
        # e.g. voltage = current*resistance -> mu is current and c*mu is voltage
        return self.mu_x[:, 0], (self.C @ self.mu_x)[:, 0]

    def predict_step(self, A, eta, u):
        """
        A = transition matrix
        eta = process noise covariance matrix
        """
        self.mu_x = self.dynamics(self.mu_x, u)
        self.sig_x = A @ self.sig_x @ A.T + eta

    def update_step(self, C, zeta, y_t):
        """
        L_t = Kalman gain matrix
        C = observation matrix
        zeta = observation noise covariance matrix
        y_t = true measurement
        """
        sig_y = C @ self.sig_x @ C.T + zeta
        L_t = np.linalg.solve(sig_y, self.sig_x @ C.T)
        self.mu_x = self.mu_x + L_t @ (y_t.reshape((-1, 1)) - C @ self.mu_x)
        self.sig_x = (np.eye(np.shape(L_t)[0]) - L_t @ C) @ self.sig_x

    def smooth(self, t):
        """
        t = current time step
        """
        J = np.linalg.solve(self.sigs[t-1], self.sigs[t-1] @ self.A.T)
        self.mus[t] = self.mus[t-1] + J @ (self.mus[t] - self.A @ self.mus[t-1])
        self.sigs[t] = self.sigs[t-1] + J @ (self.sigs[t] - self.sigs[t-1]) @ J.T
        return self.mus[t[:, 0]], (self.C @ self.mus[t])[:, 0]
