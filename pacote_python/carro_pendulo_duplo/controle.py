"""Controle LQR projetado sobre o modelo linearizado e aplicado ao não linear."""
import numpy as np
from scipy.linalg import solve_continuous_are


def lqr(A, B, Q, R):
    """Ganho K de u = -K dz para o sistema contínuo dz' = A dz + B du."""
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def controlabilidade(A, B):
    n = A.shape[0]
    Ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    return Ctrb, np.linalg.matrix_rank(Ctrb)


class ControladorLQR:
    """u = ubar - K (z - zbar), com saturação opcional em F e tau.

    Os ângulos do erro são "embrulhados" em (-pi, pi] para que o controlador
    escolha o caminho mais curto até o equilíbrio.
    """

    def __init__(self, K, zbar, ubar, F_max=None, tau_max=None):
        self.K, self.zbar, self.ubar = np.asarray(K), np.asarray(zbar), np.asarray(ubar)
        self.F_max, self.tau_max = F_max, tau_max

    def __call__(self, t, z):
        e = np.asarray(z) - self.zbar
        e[1:3] = (e[1:3] + np.pi) % (2 * np.pi) - np.pi
        u = self.ubar - self.K @ e
        if self.F_max is not None:
            u[0] = np.clip(u[0], -self.F_max, self.F_max)
        if self.tau_max is not None:
            u[1] = np.clip(u[1], -self.tau_max, self.tau_max)
        return u
