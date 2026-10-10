"""Quanto o modelo linear se afasta do não linear à medida que a amplitude cresce."""
import numpy as np
import matplotlib.pyplot as plt
from carro_pendulo_duplo import Parametros, equilibrio, linearizar, simular, simular_linear

p = Parametros()
zbar, ubar = equilibrio(p, np.pi, np.pi)     # ambas pendentes
A, B = linearizar(p, zbar)

fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for amp, cor in zip([5, 20, 45, 80], ["C0", "C1", "C2", "C3"]):
    z0 = [0.0, np.pi - np.radians(amp), np.pi, 0, 0, 0]
    nl = simular(p, z0, 6)
    li = simular_linear(A, B, zbar, ubar, z0, p, 6)
    for i, ax in enumerate(axs):
        f = 1 if i == 0 else 180 / np.pi
        ax.plot(nl.t, f * nl.Z[:, i], color=cor, label=f"{amp}° não linear")
        ax.plot(li.t, f * li.Z[:, i], "--", color=cor)
axs[1].axhline(180, color="0.6", lw=0.8); axs[2].axhline(180, color="0.6", lw=0.8)
axs[0].set_ylabel("x [m]"); axs[1].set_ylabel("θ1 [°]"); axs[2].set_ylabel("θ2 [°]")
axs[2].set_xlabel("t [s]"); axs[0].legend(fontsize=8, ncol=2)
for ax in axs: ax.grid(alpha=0.3)
fig.suptitle("Contínuo: não linear   |   tracejado: linear")
plt.show()
