"""Entradas definidas no tempo: pulso de força no carrinho e torque senoidal na barra 1."""
import numpy as np
from carro_pendulo_duplo import Parametros, simular, animar


def entradas(t, z):
    F = 15.0 if 1.0 <= t < 1.3 else 0.0          # pulso de força [N]
    tau = 0.8 * np.sin(2 * np.pi * 0.7 * t)       # torque senoidal [N m]
    return np.array([F, tau])


p = Parametros()
z0 = [0.0, np.pi, np.pi, 0.0, 0.0, 0.0]      # barras pendentes, em repouso
res = simular(p, z0, t_final=12, controlador=entradas,
              titulo="Pulso de F e torque senoidal")
anim = animar(res)
