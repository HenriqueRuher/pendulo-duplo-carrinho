"""Linha de comando:  python -m carro_pendulo_duplo <cenario> [opções]"""
import argparse
import sys

import numpy as np


def main(argv=None):
    from . import cenarios
    from .animacao import animar
    from .graficos import plotar

    ap = argparse.ArgumentParser(
        prog="python -m carro_pendulo_duplo",
        description="Simula e anima o carrinho com mola inclinada e pêndulo duplo.")
    ap.add_argument("cenario", nargs="?", default="livre",
                    help="cenário: " + ", ".join(cenarios.CENARIOS) + " (padrão: livre)")
    ap.add_argument("--tf", type=float, default=None, help="tempo final [s]")
    ap.add_argument("--salvar", default=None, help="salva a animação (.gif ou .mp4)")
    ap.add_argument("--sem-janela", action="store_true", help="não abre janela (útil com --salvar)")
    ap.add_argument("--graficos", action="store_true", help="mostra também os gráficos estáticos")
    ap.add_argument("--velocidade", type=float, default=1.0, help="fator de tempo real da animação")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--listar", action="store_true", help="lista os cenários e sai")
    a = ap.parse_args(argv)

    if a.listar:
        for nome, f in cenarios.CENARIOS.items():
            print(f"  {nome:15s} {f.__doc__.strip().splitlines()[0]}")
        return 0
    if a.cenario not in cenarios.CENARIOS:
        ap.error(f"cenário desconhecido '{a.cenario}'. Use --listar.")

    if a.sem_janela:
        import matplotlib
        matplotlib.use("Agg")

    kw = {} if a.tf is None else {"t_final": a.tf}
    out = cenarios.CENARIOS[a.cenario](**kw)
    print(out["res"].titulo)
    print(out["info"])
    if "K" in out:
        np.set_printoptions(precision=3, suppress=True)
        print("Ganho LQR K =\n", out["K"])
    if a.graficos:
        plotar(out["res"], comparar=out["fantasma"], mostrar=False)
    animar(out["res"], salvar=a.salvar, mostrar=not a.sem_janela, fps=a.fps,
           velocidade=a.velocidade, fantasma=out["fantasma"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
