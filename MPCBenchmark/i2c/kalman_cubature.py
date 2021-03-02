class kalman_curb:

    def __init__(self, A, a, B, eta, zeta, C, y_t, mu_t, var_t) -> None:
        super().__init__()
