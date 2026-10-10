"""Integração numérica do modelo não linear e do modelo linearizado."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .modelo import dinamica, energia
from .parametros import Parametros


@dataclass
class Resultado:
    t: np.ndarray        # (N,)
    Z: np.ndarray        # (N, 6) estados
    U: np.ndarray        # (N, 2) entradas [F, tau]
    p: Parametros
    titulo: str = ""

    @property
    def energia(self):
        TV = np.array([energia(z, self.p) for z in self.Z])
        return TV[:, 0], TV[:, 1]


def _entrada(controlador):
    if controlador is None:
        return lambda t, z: np.zeros(2)
    if callable(controlador):
        return controlador
    u = np.asarray(controlador, dtype=float)   # entrada constante
    return lambda t, z: u


def simular(p: Parametros, z0, t_final=10.0, controlador=None, fps=60,
            metodo="RK45", rtol=1e-8, atol=1e-10, titulo=""):
    """Simula o modelo NÃO LINEAR.

    controlador: None (entradas nulas), vetor constante [F, tau],
                 ou função u = controlador(t, z).
    """
    ufun = _entrada(controlador)
    t_eval = np.linspace(0.0, t_final, int(t_final * fps) + 1)
    sol = solve_ivp(lambda t, z: dinamica(z, ufun(t, z), p), (0.0, t_final),
                    np.asarray(z0, dtype=float), method=metodo, t_eval=t_eval,
                    rtol=rtol, atol=atol, max_step=1.0 / fps)
    if not sol.success:
        print("Aviso: integração terminou com:", sol.message)
    Z = sol.y.T
    U = np.array([ufun(t, z) for t, z in zip(sol.t, Z)])
    return Resultado(sol.t, Z, U, p, titulo)


def simular_linear(A, B, zbar, ubar, z0, p: Parametros, t_final=10.0,
                   controlador=None, fps=60, titulo=""):
    """Simula o modelo LINEARIZADO dz' = A dz + B du (retorna z = zbar + dz)."""
    ufun = _entrada(controlador) if controlador is not None else (lambda t, z: ubar)
    t_eval = np.linspace(0.0, t_final, int(t_final * fps) + 1)

    def f(t, dz):
        z = zbar + dz
        return A @ dz + B @ (ufun(t, z) - ubar)

    sol = solve_ivp(f, (0.0, t_final), np.asarray(z0) - zbar, t_eval=t_eval,
                    rtol=1e-9, atol=1e-11, max_step=1.0 / fps)
    Z = sol.y.T + zbar
    U = np.array([ufun(t, z) for t, z in zip(sol.t, Z)])
    return Resultado(sol.t, Z, U, p, titulo)
