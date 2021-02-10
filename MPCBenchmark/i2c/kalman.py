import numpy as np


class kallman:

    def __init__(self, A, E, C, y_t, mu_t, var_t) -> None:
        super().__init__()
        self.A = A
        self.E = E
        self.C = C
        self.y_t = y_t

        self.mu_t = mu_t
        self.var_t = var_t

    def calc_filter(self):
        self.predict_step(self.A, self.E)
        self.update_step(self.C, self.E, self.y_t)

    def predict_step(self, A, E):
        """
        A = transition matrix
        E = process noise covariance matrix
        """
        self.mu_t = A @ self.mu_t
        self.var_t = A @ self.var_t @ A.T + E

    def update_step(self, C, E, y_t):
        """
        L_t =
        C = cost matrix
        E = process noise covariance matrix
        y_t = true measurement
        """
        L_t = self.var_t @ C.T @ np.linalg.pinv(C @ self.var_t @ C.T + E)
        self.mu_t = self.mu_t + L_t @ (y_t - C @ self.mu_t)
        self.var_t = (np.eye(np.shape(L_t)[0]) - L_t @ C) @ self.var_t
