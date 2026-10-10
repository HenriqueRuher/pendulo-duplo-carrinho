"""Simulação em malha aberta com condição inicial escolhida por você."""
import numpy as np
from carro_pendulo_duplo import Parametros, simular, animar, plotar

p = Parametros()                       # edite aqui: Parametros(M=2.0, k=300, Ln=0.45, ...)
# ângulos medidos da vertical para cima (0 = em pé, 180° = pendente)
z0 = [0.20,                            # x [m]
      np.radians(120),                 # theta1
      np.radians(210),                 # theta2
      0.0, 0.0, 0.0]                   # velocidades

res = simular(p, z0, t_final=10, titulo="Meu teste em malha aberta")
plotar(res, mostrar=False)
anim = animar(res)                            # animar(res, salvar="meu_teste.gif") para salvar
