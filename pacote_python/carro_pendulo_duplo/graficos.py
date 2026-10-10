"""Gráficos estáticos de resultados."""
import numpy as np
import matplotlib.pyplot as plt
from .animacao import COR_CARRO, COR_B1, COR_B2, COR_F, COR_TAU


def plotar(res, comparar=None, salvar=None, mostrar=True):
    """Estados, entradas e energia. `comparar` = outro Resultado (tracejado)."""
    t, Z, U = res.t, res.Z, res.U
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(res.titulo)
    nomes = [("$x$ [m]", 0, 1, COR_CARRO), (r"$\theta_1$ [°]", 1, 180 / np.pi, COR_B1),
             (r"$\theta_2$ [°]", 2, 180 / np.pi, COR_B2),
             (r"$\dot x$ [m/s]", 3, 1, COR_CARRO), (r"$\dot\theta_1$ [°/s]", 4, 180 / np.pi, COR_B1),
             (r"$\dot\theta_2$ [°/s]", 5, 180 / np.pi, COR_B2)]
    for (nome, i, f, cor), ax in zip(nomes, axs.T.ravel()):
        ax.plot(t, f * Z[:, i], color=cor, label=res.titulo or "não linear")
        if comparar is not None:
            ax.plot(comparar.t, f * comparar.Z[:, i], "--", color="k", alpha=0.7,
                    label=comparar.titulo or "comparação")
        ax.set_ylabel(nome); ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("t [s]")
    fig.tight_layout()

    fig2, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    a1.plot(t, U[:, 0], color=COR_F, label="$F$ [N]")
    a1.plot(t, U[:, 1], color=COR_TAU, label=r"$\tau$ [N m]")
    a1.legend(); a1.grid(alpha=0.3); a1.set_ylabel("entradas")
    T, V = res.energia
    a2.plot(t, T, label="T"); a2.plot(t, V - V[0], label="V - V(0)"); a2.plot(t, T + V - V[0], "k", label="T + V - V(0)")
    a2.legend(); a2.grid(alpha=0.3); a2.set_xlabel("t [s]"); a2.set_ylabel("energia [J]")
    fig2.tight_layout()
    if salvar:
        fig.savefig(salvar, dpi=120)
        base = str(salvar).rsplit(".", 1)
        fig2.savefig(base[0] + "_entradas_energia." + (base[1] if len(base) > 1 else "png"), dpi=120)
    if mostrar:
        plt.show()
    return fig, fig2


def plotar_mola(p, salvar=None, mostrar=True):
    """f_k(x) e k_e(x) para o conjunto de parâmetros."""
    from .modelo import forca_mola, rigidez_tangente
    x = np.linspace(-1.5 * p.L0, 1.5 * p.L0, 400)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(x, forca_mola(x, p), color="#8237aa"); a.set_title("$f_k(x)$"); a.set_xlabel("x [m]"); a.grid(alpha=0.3)
    b.plot(x, rigidez_tangente(x, p), color="#8237aa"); b.set_title("$k_e(x)$"); b.set_xlabel("x [m]"); b.grid(alpha=0.3)
    fig.tight_layout()
    if salvar:
        fig.savefig(salvar, dpi=120)
    if mostrar:
        plt.show()
    return fig
