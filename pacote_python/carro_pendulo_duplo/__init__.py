"""Simulação do sistema carrinho -- mola inclinada -- pêndulo duplo.

Uso rápido:
    >>> from carro_pendulo_duplo import Parametros, simular, animar
    >>> p = Parametros()
    >>> res = simular(p, z0=[0.2, 0.5, -0.3, 0, 0, 0], t_final=10)
    >>> animar(res)
"""
from .parametros import Parametros
from .modelo import (dinamica, energia, equilibrio, linearizar, linearizar_numerico,
                     forca_mola, rigidez_tangente, matriz_massa, pontos)
from .simulacao import simular, simular_linear, Resultado
from .controle import lqr, ControladorLQR, controlabilidade
from .animacao import animar
from .graficos import plotar, plotar_mola
from . import cenarios

__version__ = "1.0.0"
