import numpy as np


class kalman:

    def __init__(self, A, a, B, eta, zeta, C, y_t, mu_x, sig_x) -> None:
        super().__init__()
        self.A = A
        self.a = a
        self.B = B
        self.sig_eta = eta
        self.C = C
        self.y_t = y_t
        self.sig_zeta = zeta

        self.mu_x = mu_x
        self.sig_x = sig_x

        self.mus_filt = []
        self.sigs_filt = []

        self.mu_smo = None
        self.sig_smo = None
        self.mus_filt.append(self.mu_x)
        self.sigs_filt.append(self.sig_x)
        self.mu_predict = []

    def dynamics(self, x, u):
        return self.A @ x + self.a + self.B @ u


    """
    Performs the kalman filtering
    :param u: torque
    :param y: measurement
    """
    def __call__(self, u, y):
        self.predict_step(self.A, self.sig_eta, u)
        self.update_step(self.C, self.sig_zeta, y)
        # return the mean of state and the mean of observation
        # e.g. voltage = current*resistance -> mu is current and c*mu is voltage
        self.mus_filt.append(self.mu_x)
        self.sigs_filt.append(self.sig_x)
        return self.mu_x[:, 0], (self.C @ self.mu_x)[:, 0], self.sig_x

    def predict_step(self, A, eta, u):
        """
        Predicts the values for mean and variance
        :param A: transition matrix
        :param eta: process noise covariance matrix
        """
        self.mu_x = self.dynamics(self.mu_x, u)
        self.mu_predict.append(self.mu_x)
        self.sig_x = A @ self.sig_x @ A.T + eta

    def update_step(self, C, zeta, y_t):
        """
        Updates the values of mean and variance
        :param L_t: Kalman gain matrix
        :param C: observation matrix
        :param zeta: observation noise covariance matrix
        :param y_t: true measurement
        """
        sig_y = C @ self.sig_x @ C.T + zeta
        L_t = self.sig_x @ C.T @ np.linalg.pinv(sig_y)
        self.mu_x = self.mu_x + L_t @ (y_t.reshape((-1, 1)) - C @ self.mu_x)
        self.sig_x = (np.eye(np.shape(L_t)[0]) - L_t @ C) @ self.sig_x

    def smooth(self, t, end):
        """
        Kalman smoothing for the current time step
        :param t: current time step
        :param end: if it is the end of the trajectory
        """
        if end:
            self.mu_smo = self.mus_filt[t]
            self.sig_smo = self.sigs_filt[t]
        else:
            J = self.sigs_filt[t] @ self.A.T @ np.linalg.inv(self.A @ self.sigs_filt[t]
                                                            @ self.A.T + self.sig_eta)
            self.mu_smo = self.mus_filt[t] + J @ (self.mu_smo - self.mu_predict[t])
            self.sig_smo = self.sigs_filt[t] + J @ (self.sig_smo - (self.A @ self.sigs_filt[t] @ self.A.T + self.sig_eta)) @ J.T
        return self.mu_smo[:, 0], (self.C @ self.mu_smo)[:, 0], self.sig_smo
