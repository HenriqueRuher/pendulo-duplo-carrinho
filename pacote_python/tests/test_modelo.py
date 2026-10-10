import numpy as np
import pytest

from carro_pendulo_duplo import (Parametros, equilibrio, linearizar, linearizar_numerico,
                                 simular, dinamica, controlabilidade)


@pytest.mark.parametrize("th1,th2", [(0.0, 0.0), (0.6, 0.0), (0.6, np.pi), (np.pi, np.pi)])
@pytest.mark.parametrize("x", [0.0, 0.13])
def test_linearizacao_analitica_igual_numerica(th1, th2, x):
    p = Parametros()
    zbar, ubar = equilibrio(p, th1, th2, x)
    assert np.allclose(dinamica(zbar, ubar, p), 0, atol=1e-12)
    A, B = linearizar(p, zbar)
    An, Bn = linearizar_numerico(p, zbar, ubar)
    assert np.allclose(A, An, atol=1e-6)
    assert np.allclose(B, Bn, atol=1e-6)


def test_conservacao_energia_sem_dissipacao():
    p = Parametros().sem_dissipacao()
    res = simular(p, [0.2, 1.0, 2.0, 0, 0, 0], t_final=5.0, rtol=1e-10, atol=1e-12)
    T, V = res.energia
    E = T + V
    assert np.max(np.abs(E - E[0])) < 1e-5 * max(1.0, abs(E[0]))


def test_energia_decresce_com_dissipacao():
    p = Parametros()
    res = simular(p, [0.2, 1.0, 2.0, 0, 0, 0], t_final=5.0)
    T, V = res.energia
    E = T + V
    assert np.all(np.diff(E) <= 1e-7)


def test_controlabilidade():
    p = Parametros()
    zbar, _ = equilibrio(p, 0.0, 0.0)          # ambas em pé (invertidas)
    A, B = linearizar(p, zbar)
    assert controlabilidade(A, B)[1] == 6


def test_barra2_fora_da_vertical_nao_e_equilibrio():
    with pytest.raises(ValueError):
        equilibrio(Parametros(), 0.3, 0.3)
