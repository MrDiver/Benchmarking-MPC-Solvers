import numpy as np


class QuadratureInference(object):

    def __init__(self, alpha, beta, kappa, dim):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.dim = dim
        self.lam = self.alpha ** 2 * (self.dim + self.kappa) - self.dim
        self.sf = np.sqrt(self.dim + self.lam)

        self.n = 2 * self.dim + 1
        self.w0_sig_extra = 1. - self.alpha ** 2 + self.beta
        self.wghts_m = (1 / (2. * (self.dim + self.lam))) * np.ones((self.n,))
        self.wghts_m[0] = 2 * self.lam * self.wghts_m[0]
        self.wghts_sig = np.copy(self.wghts_m)
        self.wghts_sig[0] += self.w0_sig_extra
        self.wghts_m[0] = 0.
        self.wghts_sig[0] = 0.
        self.base_pts = np.vstack((np.zeros((1, self.dim)),
                                   np.eye(self.dim),
                                   -np.eye(self.dim)))
        self.prev_x_pts = None
        self.prev_m = None

    def propagate(self, m_x, sig_x):
        m_x = m_x.reshape((-1, self.dim))
        scale = self.sf * np.linalg.cholesky(sig_x)
        return m_x + self.base_pts.dot(scale.T)

    def __call__(self, f, m_x, sig_x):
        x_pts = self.propagate(m_x, sig_x)
        y_pts, m_y, sig_y, sig_xy = self.evaluate(f, m_x, x_pts)
        self.x_pts, self.y_pts = x_pts, y_pts
        return m_y.T, sig_y, sig_xy

    def evaluate(self, f, m_x, x_pts):
        y_pts = f(x_pts)
        m_y = (self.wghts_m @ y_pts).T
        # batched weighted outer products
        sig_y = np.einsum("b,bi,bj->ij", self.wghts_sig, y_pts, y_pts) - np.outer(m_y, m_y)
        sig_xy = np.einsum("b,bi,bj->ij", self.wghts_sig, x_pts, y_pts) - np.outer(m_x, m_y)
        self.prev_m = m_y
        self.prev_x_pts = x_pts
        return y_pts, m_y, sig_y, sig_xy