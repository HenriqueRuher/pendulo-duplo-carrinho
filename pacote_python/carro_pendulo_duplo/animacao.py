"""Animação do sistema com gráficos sincronizados."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyArrowPatch

from .modelo import pontos, comprimento_mola
from .simulacao import Resultado

# mesmas cores do documento
COR_CARRO = "#285aa0"
COR_B1 = "#e1730f"
COR_B2 = "#1e914b"
COR_MOLA = "#8237aa"
COR_F = "#c81923"
COR_TAU = "#d2008c"


def _mola(p0, p1, n_espiras=12, largura=0.03, pontas=0.06):
    """Polilinha em zigue-zague entre p0 e p1."""
    p0, p1 = np.asarray(p0), np.asarray(p1)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        return np.array([p0, p1]).T
    e = d / L
    nrm = np.array([-e[1], e[0]])
    ponta = min(pontas, 0.2 * L)
    s = np.linspace(ponta, L - ponta, 2 * n_espiras + 1)
    amp = largura * np.array([0] + [(-1) ** i for i in range(1, len(s) - 1)] + [0])
    pts = [p0] + [p0 + si * e + ai * nrm for si, ai in zip(s, amp)] + [p1]
    return np.array(pts).T


def animar(res: Resultado, salvar=None, mostrar=True, fps=30, velocidade=1.0,
           fantasma: Resultado = None, rastro_s=1.5, dpi=100, lado_mola="abaixo"):
    """Anima o resultado de uma simulação.

    salvar     : caminho .gif ou .mp4 (opcional)
    velocidade : fator de tempo real (2.0 = duas vezes mais rápido)
    fantasma   : segundo Resultado desenhado em transparência (ex.: modelo linear)
    lado_mola  : 'abaixo' (padrão, como no documento), 'acima' ou 'auto'. Só muda o
                 DESENHO: a dinâmica depende apenas de L = sqrt(L0^2 + x^2), igual dos
                 dois lados do trilho. 'auto' desenha a mola do lado oposto ao pêndulo.
    """
    p = res.p
    t, Z, U = res.t, res.Z, res.U
    if lado_mola == "auto":
        lado_mola = "abaixo" if np.mean(np.cos(Z[:, 1])) > 0 else "acima"
    ya = p.L0 if lado_mola == "acima" else -p.L0
    # índices dos quadros
    passo = max(1, int(round((1.0 / fps) * velocidade / (t[1] - t[0]))))
    quadros = np.arange(0, len(t), passo)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    axx = fig.add_subplot(gs[0, 1])
    axt = fig.add_subplot(gs[1, 1], sharex=axx)
    axu = fig.add_subplot(gs[2, 1], sharex=axx)
    axe = fig.add_subplot(gs[3, 1], sharex=axx)
    fig.suptitle(res.titulo or "Carrinho -- mola inclinada -- pêndulo duplo", fontsize=13)

    # ---------------- cena
    lmax = p.l1 + p.l2
    xs = Z[:, 0]
    xmin = min(xs.min() - lmax, -0.3) - 0.15
    xmax = max(xs.max() + lmax, 0.3) + 0.15
    ymin, ymax = -max(p.L0, lmax) - 0.15, max(p.L0, lmax) + 0.15
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.plot([xmin, xmax], [0, 0], color="0.55", lw=6, solid_capstyle="butt", zorder=1)
    sgn = 1 if ya > 0 else -1
    ax.plot([-0.08, 0.08], [ya, ya], color="k", lw=2)
    for xh in np.linspace(-0.08, 0.06, 6):
        ax.plot([xh, xh + 0.025], [ya, ya + sgn * 0.03], color="k", lw=0.8)
    ax.axvline(0, color="0.7", ls=":", lw=1)

    wc, hc = 0.22, 0.12
    (mola,) = ax.plot([], [], color=COR_MOLA, lw=1.6, zorder=2)
    carro = Rectangle((0, 0), wc, hc, fc="#d6e1f0", ec=COR_CARRO, lw=2, zorder=3)
    ax.add_patch(carro)
    (rastro,) = ax.plot([], [], color=COR_B2, lw=1, alpha=0.5, zorder=2)
    if fantasma is not None:
        (fb1,) = ax.plot([], [], color=COR_B1, lw=7, alpha=0.25, solid_capstyle="round", zorder=3)
        (fb2,) = ax.plot([], [], color=COR_B2, lw=7, alpha=0.25, solid_capstyle="round", zorder=3)
    (b1,) = ax.plot([], [], color=COR_B1, lw=8, solid_capstyle="round", zorder=4)
    (b2,) = ax.plot([], [], color=COR_B2, lw=8, solid_capstyle="round", zorder=4)
    (juntas,) = ax.plot([], [], "o", mfc="white", mec="k", ms=7, zorder=5)
    seta_F = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=18,
                             color=COR_F, lw=2.5, zorder=6)
    ax.add_patch(seta_F)
    (arco_tau,) = ax.plot([], [], color=COR_TAU, lw=2.5, zorder=6)
    (ponta_tau,) = ax.plot([], [], marker=(3, 0, 0), color=COR_TAU, ms=9, zorder=6)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace",
                  fontsize=10, bbox=dict(fc="white", ec="0.8", alpha=0.85))

    Fmax = max(np.abs(U[:, 0]).max(), 1e-9)
    taumax = max(np.abs(U[:, 1]).max(), 1e-9)
    esc_F = 0.3 / Fmax

    # ---------------- gráficos
    axx.plot(t, Z[:, 0], color=COR_CARRO, label="$x$")
    axx.set_ylabel("$x$ [m]")
    axt.plot(t, np.degrees(Z[:, 1]), color=COR_B1, label=r"$\theta_1$")
    axt.plot(t, np.degrees(Z[:, 2]), color=COR_B2, label=r"$\theta_2$")
    axt.set_ylabel("ângulo [°]"); axt.legend(loc="upper right", fontsize=8)
    axu.plot(t, U[:, 0], color=COR_F, label="$F$ [N]")
    axu.plot(t, U[:, 1], color=COR_TAU, label=r"$\tau$ [N m]")
    axu.set_ylabel("entradas"); axu.legend(loc="upper right", fontsize=8)
    T, V = res.energia
    axe.plot(t, T + V, color="k", label="$T+V$")
    axe.set_ylabel("energia [J]"); axe.set_xlabel("t [s]")
    if fantasma is not None:
        axx.plot(fantasma.t, fantasma.Z[:, 0], "--", color=COR_CARRO, alpha=0.6)
        axt.plot(fantasma.t, np.degrees(fantasma.Z[:, 1]), "--", color=COR_B1, alpha=0.6)
        axt.plot(fantasma.t, np.degrees(fantasma.Z[:, 2]), "--", color=COR_B2, alpha=0.6)
        axx.plot([], [], "--", color="0.4", label="linear")
        axx.plot([], [], "-", color="0.4", label="não linear")
        axx.legend(loc="upper right", fontsize=8)
    cursores = [a.axvline(0, color="0.3", lw=1) for a in (axx, axt, axu, axe)]
    for a in (axx, axt, axu, axe):
        a.grid(alpha=0.3)

    n_rastro = int(rastro_s / (t[1] - t[0]))

    def desenhar(k):
        z = Z[k]
        O, A, B, _, _ = pontos(z, p)
        mola.set_data(*_mola((0.0, ya), (z[0], sgn * hc / 2)))
        carro.set_xy((z[0] - wc / 2, -hc / 2))
        b1.set_data([O[0], A[0]], [O[1], A[1]])
        b2.set_data([A[0], B[0]], [A[1], B[1]])
        juntas.set_data([O[0], A[0]], [O[1], A[1]])
        k0 = max(0, k - n_rastro)
        Bs = np.array([pontos(zz, p)[2] for zz in Z[k0:k + 1:max(1, passo // 2)]])
        rastro.set_data(Bs[:, 0], Bs[:, 1])
        if fantasma is not None:
            kf = min(k, len(fantasma.Z) - 1)
            Of, Af, Bf, _, _ = pontos(fantasma.Z[kf], p)
            fb1.set_data([Of[0], Af[0]], [Of[1], Af[1]])
            fb2.set_data([Af[0], Bf[0]], [Af[1], Bf[1]])
        # força F
        F, tau = U[k]
        xF = z[0] + np.sign(F) * wc / 2
        seta_F.set_positions((xF, 0.0), (xF + esc_F * F, 0.0))
        seta_F.set_visible(abs(F) > 1e-3 * Fmax)
        # torque tau: arco proporcional (tau > 0 no sentido de theta1: horário)
        frac = np.clip(tau / taumax, -1, 1) if taumax > 1e-6 else 0.0
        if abs(frac) > 1e-3:
            ang = np.linspace(-np.pi / 2, -np.pi / 2 - frac * 1.5 * np.pi, 40)
            r = 0.09
            arco_tau.set_data(O[0] + r * np.cos(ang), O[1] + r * np.sin(ang))
            ponta_tau.set_data([O[0] + r * np.cos(ang[-1])], [O[1] + r * np.sin(ang[-1])])
            ponta_tau.set_marker((3, 0, np.degrees(ang[-1]) + (180 if frac > 0 else 0)))
        else:
            arco_tau.set_data([], []); ponta_tau.set_data([], [])
        L = comprimento_mola(z[0], p)
        txt.set_text(f"t = {t[k]:6.2f} s\n"
                     f"x = {z[0]:+.3f} m   L = {L:.3f} m\n"
                     f"θ1 = {np.degrees(z[1]):+7.1f}°\n"
                     f"θ2 = {np.degrees(z[2]):+7.1f}°\n"
                     f"F = {F:+7.2f} N  τ = {tau:+6.2f} N m")
        for c in cursores:
            c.set_xdata([t[k], t[k]])
        return []

    anim = animation.FuncAnimation(fig, desenhar, frames=quadros,
                                   interval=1000.0 / fps, blit=False)
    if salvar:
        if str(salvar).lower().endswith(".mp4"):
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
                anim.save(salvar, writer=writer, dpi=dpi)
            except Exception as e:  # ffmpeg ausente
                alt = str(salvar)[:-4] + ".gif"
                print(f"ffmpeg indisponível ({e}); salvando GIF em {alt}")
                anim.save(alt, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        else:
            anim.save(salvar, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        print("Animação salva em", salvar)
    if mostrar:
        plt.show()
    return anim
