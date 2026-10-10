"""Equações de movimento (Lagrange) e linearização.

Estado:  z = [x, th1, th2, xd, th1d, th2d]
Entrada: u = [F, tau]

Ângulos absolutos medidos a partir da vertical PARA CIMA, positivos quando a barra
se inclina para +x (sentido horário no desenho): th = 0 é a barra em pé (invertida),
th = pi é a barra pendente. A mola fica presa abaixo do trilho.

Modelo:  M(q) qdd + n(q,qd) + C qd + g(q) = Bu u      (Seções 5 e 6 do documento)
"""
import numpy as np
from .parametros import Parametros

BU = np.array([[1.0, 0.0],
               [0.0, 1.0],
               [0.0, 0.0]])


# ------------------------------------------------------------------ mola
def comprimento_mola(x, p: Parametros):
    return np.sqrt(p.L0**2 + x**2)


def forca_mola(x, p: Parametros):
    """f_k(x) = k x (1 - Ln / sqrt(L0^2 + x^2))  (componente horizontal)."""
    return p.k * x * (1.0 - p.Ln / np.sqrt(p.L0**2 + x**2))


def rigidez_tangente(x, p: Parametros):
    """k_e(x) = d f_k / dx."""
    return p.k * (1.0 - p.Ln * p.L0**2 / (p.L0**2 + x**2) ** 1.5)


# ------------------------------------------------------------------ matrizes
def matriz_massa(q, p: Parametros):
    _, t1, t2 = q
    c1, c2, c12 = np.cos(t1), np.cos(t2), np.cos(t1 - t2)
    return np.array([[p.mt,        p.h1 * c1,    p.h2 * c2],
                     [p.h1 * c1,   p.I1,         p.h12 * c12],
                     [p.h2 * c2,   p.h12 * c12,  p.I2]])


def matriz_amortecimento(p: Parametros):
    return np.array([[p.cx, 0.0, 0.0],
                     [0.0, p.c1 + p.c2, -p.c2],
                     [0.0, -p.c2, p.c2]])


def termos_centripetos(q, qd, p: Parametros):
    _, t1, t2 = q
    _, w1, w2 = qd
    s1, s2, s12 = np.sin(t1), np.sin(t2), np.sin(t1 - t2)
    return np.array([-p.h1 * s1 * w1**2 - p.h2 * s2 * w2**2,
                     p.h12 * s12 * w2**2,
                     -p.h12 * s12 * w1**2])


def forcas_conservativas(q, p: Parametros):
    x, t1, t2 = q
    return np.array([forca_mola(x, p),
                     -p.g * p.h1 * np.sin(t1),
                     -p.g * p.h2 * np.sin(t2)])


def dinamica(z, u, p: Parametros):
    """zdot = f(z, u) -- modelo não linear completo."""
    z = np.asarray(z, dtype=float)
    q, qd = z[:3], z[3:]
    rhs = BU @ np.asarray(u, dtype=float) - termos_centripetos(q, qd, p) \
        - matriz_amortecimento(p) @ qd - forcas_conservativas(q, p)
    qdd = np.linalg.solve(matriz_massa(q, p), rhs)
    return np.concatenate([qd, qdd])


# ------------------------------------------------------------------ energia
def energia(z, p: Parametros):
    """Retorna (T, V) -- energia cinética e potencial."""
    q, qd = np.asarray(z[:3]), np.asarray(z[3:])
    T = 0.5 * qd @ matriz_massa(q, p) @ qd
    x, t1, t2 = q
    V = p.g * (p.h1 * np.cos(t1) + p.h2 * np.cos(t2)) \
        + 0.5 * p.k * (comprimento_mola(x, p) - p.Ln) ** 2
    return T, V


# ------------------------------------------------------------------ cinemática
def pontos(z, p: Parametros):
    """Coordenadas (x, y) de O (junta no carrinho), A (junta entre barras),
    ponta da barra 2 (B), G1 e G2. Eixo Y para cima; a mola fica presa em (0, -L0)."""
    x, t1, t2 = z[0], z[1], z[2]
    O = np.array([x, 0.0])
    A = O + p.l1 * np.array([np.sin(t1), np.cos(t1)])
    B = A + p.l2 * np.array([np.sin(t2), np.cos(t2)])
    G1 = O + p.a1 * np.array([np.sin(t1), np.cos(t1)])
    G2 = A + p.a2 * np.array([np.sin(t2), np.cos(t2)])
    return O, A, B, G1, G2


# ------------------------------------------------------------------ equilíbrio
def equilibrio(p: Parametros, theta1=0.0, theta2=0.0, x=0.0):
    """Ponto de equilíbrio (Seção 7): barra 2 deve ser vertical
    (theta2 = 0, em pé, ou pi, pendente).

    Retorna (zbar, ubar) com ubar = [F_bar, tau_bar] = [f_k(x), -g h1 sin(theta1)].
    """
    if abs(np.sin(theta2)) > 1e-9:
        raise ValueError("A barra 2 só está em equilíbrio na vertical: "
                         "theta2 deve ser 0 ou pi (junta A é passiva).")
    zbar = np.array([x, theta1, theta2, 0.0, 0.0, 0.0])
    ubar = np.array([forca_mola(x, p), -p.g * p.h1 * np.sin(theta1)])
    return zbar, ubar


# ------------------------------------------------------------------ linearização
def linearizar(p: Parametros, zbar):
    """Matrizes A (6x6) e B (6x2) analíticas (Seção 8 do documento)."""
    x, t1, t2 = zbar[0], zbar[1], zbar[2]
    if abs(np.sin(t2)) > 1e-9 or np.any(np.abs(zbar[3:]) > 1e-12):
        raise ValueError("zbar não é equilíbrio (theta2 deve ser 0/pi e velocidades nulas).")
    al, be, ga = np.sin(t1), np.cos(t1), np.cos(t2)
    mt, h1, h2, h12, I1, I2 = p.mt, p.h1, p.h2, p.h12, p.I1, p.I2

    Delta = mt * (I1 * I2 - h12**2 * be**2) - h1**2 * be**2 * I2 - h2**2 * I1 \
        + 2 * h1 * h2 * h12 * be**2
    n11 = I1 * I2 - h12**2 * be**2
    n12 = be * (h2 * h12 - h1 * I2)
    n13 = ga * (h1 * h12 * be**2 - h2 * I1)
    n22 = mt * I2 - h2**2
    n23 = be * ga * (h1 * h2 - mt * h12)
    n33 = mt * I1 - h1**2 * be**2
    N = np.array([[n11, n12, n13], [n12, n22, n23], [n13, n23, n33]])
    Minv = N / Delta

    K = np.diag([rigidez_tangente(x, p), -p.g * h1 * be, -p.g * h2 * ga])
    C = matriz_amortecimento(p)

    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = -Minv @ K
    A[3:, 3:] = -Minv @ C
    B = np.zeros((6, 2))
    B[3:, :] = Minv @ BU
    return A, B


def linearizar_numerico(p: Parametros, zbar, ubar, eps=1e-6):
    """Jacobiana por diferenças centrais (para conferência)."""
    A = np.zeros((6, 6))
    B = np.zeros((6, 2))
    for i in range(6):
        d = np.zeros(6); d[i] = eps
        A[:, i] = (dinamica(zbar + d, ubar, p) - dinamica(zbar - d, ubar, p)) / (2 * eps)
    for i in range(2):
        d = np.zeros(2); d[i] = eps
        B[:, i] = (dinamica(zbar, ubar + d, p) - dinamica(zbar, ubar - d, p)) / (2 * eps)
    return A, B
