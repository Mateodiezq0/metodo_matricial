import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from core.elementos import Nodo, CargaNodal, TipoCarga, CargaBarra, Elemento, Estructura

def test_nodo():
    n = Nodo(id=1, x=1.0, y=2.0, restricciones=[True, False, True], valores_prescritos=[0.0, 1.0, 0.0])
    assert np.allclose(n.get_coord(), [1.0, 2.0])
    assert n.restricciones == [True, False, True]
    assert n.valores_prescritos[1] == 1.0

def test_carga_nodal():
    c = CargaNodal(nodo_id=2, fx=10.0, fy=-5.0, mz=3.0)
    v = c.vector()
    assert np.allclose(v, [10.0, -5.0, 3.0])

def test_tipo_carga():
    t = TipoCarga(id=1, tipo=2, L1=0.2, L2=0.8, q1=10, q2=0, alpha=90)
    assert t.tipo == 2
    assert t.alpha == 90

def test_carga_barra():
    cb = CargaBarra(barra_id=3, carga_id=1)
    assert cb.barra_id == 3
    assert cb.carga_id == 1

def test_elemento():
    e = Elemento(id=1, nodo_i=1, nodo_f=2, E=200e9, b=0.3, h=0.5)
    e.calcular_longitud_y_angulo(np.array([0, 0]), np.array([3, 4]))
    assert np.isclose(e.L, 5.0)
    assert np.isclose(e.tita, 53.13010235415599)
    K = e.matriz_rigidez_portico()
    assert K.shape == (6, 6)
    assert np.allclose(K, K.T, atol=1e-6)  # Debe ser simétrica
    assert np.allclose(K, [[2.1984e+09, 2.8512e+09, -1.2000e+08, -2.1984e+09, -2.8512e+09, -1.2000e+08], [2.8512e+09, 3.8616e+09, 9.0000e+07, -2.8512e+09, -3.8616e+09, 9.0000e+07], [-1.2000e+08, 9.0000e+07, 5.0000e+08, 1.2000e+08, -9.0000e+07, 2.5000e+08], [-2.1984e+09, -2.8512e+09, 1.2000e+08, 2.1984e+09, 2.8512e+09, 1.2000e+08], [-2.8512e+09, -3.8616e+09, -9.0000e+07, 2.8512e+09, 3.8616e+09, -9.0000e+07], [-1.2000e+08, 9.0000e+07, 2.5000e+08, 1.2000e+08, -9.0000e+07, 5.0000e+08]], atol=1e-6)
    #print(K)
    #print(e)

def test_estructura():
    estructura = Estructura()
    estructura.agregar_nodo(Nodo(id=1, x=0, y=0))
    estructura.agregar_nodo(Nodo(id=2, x=3, y=4))
    estructura.agregar_elemento(Elemento(id=1, nodo_i=1, nodo_f=2, E=200e9, b=0.3, h=0.5))
    assert len(estructura.nodos) == 2
    assert len(estructura.elementos) == 1
