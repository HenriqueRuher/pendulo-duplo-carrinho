"""Linearização em um equilíbrio à sua escolha + projeto LQR.

Equilíbrio (Seção 7 do documento): theta1 qualquer, theta2 = 0 (em pé) ou pi (pendente).
Ângulos medidos da vertical para cima.
"""
import numpy as np
from carro_pendulo_duplo import (Parametros, equilibrio, linearizar, linearizar_numerico,
                                 controlabilidade, lqr, ControladorLQR, simular, animar)

np.set_printoptions(precision=4, suppress=True, linewidth=120)

p = Parametros()
theta1_bar = np.radians(20)       # alpha = sin, beta = cos
theta2_bar = 0.0                  # barra 2 invertida, em pé (gamma = +1)

zbar, ubar = equilibrio(p, theta1_bar, theta2_bar, x=0.0)
A, B = linearizar(p, zbar)
An, Bn = linearizar_numerico(p, zbar, ubar)
print("F_bar, tau_bar =", ubar)
print("A =\n", A)
print("B =\n", B)
print("erro analítico x numérico:", np.abs(A - An).max(), np.abs(B - Bn).max())
print("autovalores de A:", np.linalg.eigvals(A))
print("posto de controlabilidade:", controlabilidade(A, B)[1])

Q = np.diag([10, 100, 100, 1, 1, 1])
R = np.diag([0.1, 1.0])
K = lqr(A, B, Q, R)
print("K =\n", K)
print("autovalores de malha fechada:", np.linalg.eigvals(A - B @ K))

ctrl = ControladorLQR(K, zbar, ubar, F_max=60, tau_max=15)
z0 = zbar + np.array([0.1, np.radians(-8), np.radians(10), 0, 0, 0])
res = simular(p, z0, t_final=8, controlador=ctrl, titulo="LQR no equilíbrio escolhido")
anim = animar(res)
