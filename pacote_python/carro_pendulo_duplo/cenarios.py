"""Cenários prontos de simulação.

Ângulos medidos da vertical para cima (theta = 0: barra em pé; theta = pi: pendente).

Cada função retorna um dicionário com:
    'res'      : Resultado do modelo não linear
    'fantasma' : Resultado do modelo linear (quando faz sentido) ou None
    'info'     : texto descritivo
"""
import numpy as np

from .parametros import Parametros
from .modelo import equilibrio, linearizar
from .simulacao import simular, simular_linear
from .controle import lqr, ControladorLQR, controlabilidade

D = np.pi / 180


def livre(p=None, t_final=12.0):
    """Malha aberta (F = tau = 0) a partir de uma condição inicial grande."""
    p = p or Parametros()
    z0 = [0.25, 110 * D, 220 * D, 0, 0, 0]
    res = simular(p, z0, t_final, titulo="Malha aberta: resposta livre amortecida (F = τ = 0)")
    return dict(res=res, fantasma=None,
                info="Carrinho deslocado 0,25 m, θ1 = 110°, θ2 = 220°, sem entradas. "
                     "Mostra a mola 'tombando', o acoplamento carrinho-pêndulo e o amortecimento.")


def caotico(p=None, t_final=15.0):
    """Sem dissipação e com ângulos grandes: dinâmica caótica do pêndulo duplo."""
    p = (p or Parametros()).sem_dissipacao()
    z0 = [0.0, 60 * D, 20 * D, 0, 0, 0]
    res = simular(p, z0, t_final, titulo="Sem dissipação, ângulos grandes (caótico)")
    return dict(res=res, fantasma=None,
                info="cx = c1 = c2 = 0. A energia T+V deve permanecer constante (verificação).")


def equilibrio_inclinado(p=None, t_final=10.0, theta1_graus=145.0):
    """Barra 1 inclinada sustentada pelo torque de regime, barra 2 pendente."""
    p = p or Parametros()
    zbar, ubar = equilibrio(p, theta1=theta1_graus * D, theta2=np.pi)
    A, B = linearizar(p, zbar)
    z0 = zbar + np.array([0.05, -8 * D, -15 * D, 0, 0, 0])
    res = simular(p, z0, t_final, controlador=ubar,
                  titulo=f"Equilíbrio com θ̄1 = {theta1_graus:.0f}°, τ̄ = {ubar[1]:.2f} N m (constante)")
    lin = simular_linear(A, B, zbar, ubar, z0, p, t_final, titulo="linear")
    return dict(res=res, fantasma=lin,
                info="Torque constante τ̄ = -g h1 sen θ̄1 mantém a barra 1 inclinada; "
                     "a barra 2 oscila em torno da vertical. Fantasma = modelo linear.")


def linear_vs_naolinear(p=None, t_final=8.0, amplitude_graus=50.0):
    """Compara modelo não linear e linearizado em torno do equilíbrio pendente."""
    p = p or Parametros()
    zbar, ubar = equilibrio(p, np.pi, np.pi)
    A, B = linearizar(p, zbar)
    z0 = [0.15, np.pi - amplitude_graus * D, np.pi + amplitude_graus * D, 0, 0, 0]
    res = simular(p, z0, t_final, titulo=f"Não linear × linear (perturbação de {amplitude_graus:.0f}°)")
    lin = simular_linear(A, B, zbar, ubar, z0, p, t_final, titulo="linear")
    return dict(res=res, fantasma=lin,
                info="Barras transparentes = modelo linear. Aumente a amplitude para ver a divergência.")


def mola_sem_precarga(p=None, t_final=12.0):
    """Ln = L0: rigidez linear nula em x = 0 (mola puramente cúbica)."""
    p = (p or Parametros()).com(Ln=Parametros().L0)
    z0 = [0.3, np.pi, np.pi, 0, 0, 0]
    res = simular(p, z0, t_final, titulo="Mola sem pré-carga (Ln = L0): força ≈ k x³ / (2 L0²)")
    zbar, ubar = equilibrio(p, np.pi, np.pi)
    A, B = linearizar(p, zbar)
    lin = simular_linear(A, B, zbar, ubar, z0, p, t_final, titulo="linear")
    return dict(res=res, fantasma=lin,
                info="O modelo linear (fantasma) não 'vê' a mola: o carrinho apenas desliza "
                     "e para por atrito, enquanto o não linear oscila.")


def _lqr(p, theta1, theta2, z0, t_final, titulo, F_max=60.0, tau_max=15.0):
    zbar, ubar = equilibrio(p, theta1, theta2)
    A, B = linearizar(p, zbar)
    _, posto = controlabilidade(A, B)
    Q = np.diag([10, 100, 100, 1, 1, 1])
    R = np.diag([0.1, 1.0])
    K = lqr(A, B, Q, R)
    ctrl = ControladorLQR(K, zbar, ubar, F_max=F_max, tau_max=tau_max)
    res = simular(p, z0, t_final, controlador=ctrl, titulo=titulo)
    return dict(res=res, fantasma=None, K=K, A=A, B=B,
                info=f"LQR projetado no modelo linear (posto de controlabilidade = {posto}) "
                     f"e aplicado ao modelo não linear com saturação |F| ≤ {F_max} N, |τ| ≤ {tau_max} N m.")


def lqr_invertido(p=None, t_final=8.0):
    """Estabilização das duas barras invertidas, em pé (θ̄1 = θ̄2 = 0)."""
    p = p or Parametros()
    z0 = [0.2, -12 * D, 15 * D, 0, 0, 0]
    return _lqr(p, 0.0, 0.0, z0, t_final, "LQR: pêndulo duplo invertido (θ̄1 = θ̄2 = 0°)")


def lqr_inclinado(p=None, t_final=8.0, theta1_graus=30.0):
    """Barra 1 inclinada e barra 2 invertida na vertical."""
    p = p or Parametros()
    z0 = [-0.15, (theta1_graus - 10) * D, -12 * D, 0, 0, 0]
    return _lqr(p, theta1_graus * D, 0.0, z0, t_final,
                f"LQR: barra 1 a {theta1_graus:.0f}°, barra 2 invertida")


CENARIOS = {
    "livre": livre,
    "caotico": caotico,
    "equilibrio": equilibrio_inclinado,
    "comparacao": linear_vs_naolinear,
    "mola": mola_sem_precarga,
    "lqr_invertido": lqr_invertido,
    "lqr_inclinado": lqr_inclinado,
}
