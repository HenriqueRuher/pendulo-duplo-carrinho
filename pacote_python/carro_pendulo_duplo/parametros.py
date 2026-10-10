"""Parâmetros físicos do sistema carrinho--mola inclinada--pêndulo duplo.

A notação segue o documento LaTeX do modelo (Seção 1):
    m_t  = M + m1 + m2
    h1   = m1*a1 + m2*l1
    h2   = m2*a2
    h12  = m2*l1*a2
    I1   = JG1 + m1*a1^2 + m2*l1^2
    I2   = JG2 + m2*a2^2
"""
from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class Parametros:
    # carrinho
    M: float = 1.5          # massa do carrinho [kg]
    # barra 1
    m1: float = 0.5         # massa [kg]
    l1: float = 0.5         # comprimento [m]
    a1: Optional[float] = None    # dist. junta O -> G1 [m] (padrão: l1/2)
    JG1: Optional[float] = None   # inércia em G1 [kg m^2] (padrão: m1 l1^2/12)
    # barra 2
    m2: float = 0.3
    l2: float = 0.4
    a2: Optional[float] = None    # dist. junta A -> G2 (padrão: l2/2)
    JG2: Optional[float] = None   # (padrão: m2 l2^2/12)
    # mola inclinada
    k: float = 200.0        # rigidez [N/m]
    L0: float = 0.5         # distância vertical do ponto de fixação ao trilho [m]
    Ln: float = 0.4         # comprimento livre da mola [m] (Ln < L0 => pré-tração)
    # dissipação viscosa
    cx: float = 2.0         # trilho [N s/m]
    c1: float = 0.02        # junta O [N m s/rad]
    c2: float = 0.01        # junta A [N m s/rad]
    # gravidade
    g: float = 9.81

    def __post_init__(self):
        if self.a1 is None:
            self.a1 = self.l1 / 2
        if self.JG1 is None:
            self.JG1 = self.m1 * self.l1**2 / 12
        if self.a2 is None:
            self.a2 = self.l2 / 2
        if self.JG2 is None:
            self.JG2 = self.m2 * self.l2**2 / 12

    # parâmetros agrupados
    @property
    def mt(self): return self.M + self.m1 + self.m2
    @property
    def h1(self): return self.m1 * self.a1 + self.m2 * self.l1
    @property
    def h2(self): return self.m2 * self.a2
    @property
    def h12(self): return self.m2 * self.l1 * self.a2
    @property
    def I1(self): return self.JG1 + self.m1 * self.a1**2 + self.m2 * self.l1**2
    @property
    def I2(self): return self.JG2 + self.m2 * self.a2**2

    def com(self, **mudancas) -> "Parametros":
        """Cópia com alguns parâmetros alterados. Ex.: p.com(cx=0, c1=0, c2=0)."""
        novo = replace(self, **mudancas)
        return novo

    def sem_dissipacao(self) -> "Parametros":
        return self.com(cx=0.0, c1=0.0, c2=0.0)
