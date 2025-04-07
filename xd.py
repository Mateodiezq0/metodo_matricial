from core.elementos import Nodo, Elemento, TipoCarga
import numpy as np

# Parámetros de la carga trapezoidal parcial
carga = TipoCarga(
    id=1,
    tipo=1,
    L1=0.25,
    L2=0.5,
    q1=-0.5,
    q2=-0.2,
    alpha=90  # Vertical hacia abajo
)

# Casos a evaluar: (Nombre, restricciones nodo_i, restricciones nodo_f)
casos = [
    ("Empotrado - Libre", [True, True, True], [False, False, False]),
    ("Empotrado - Nudo", [True, True, True], None),
    ("Nudo - Empotrado", None, [True, True, True]),
    ("Libre - Empotrado", [False, False, False], [True, True, True]),
]

for nombre, rest_i, rest_f in casos:
    print(f"\n== Caso: {nombre} ==")

    nodo_i = Nodo(id=1, x=0, y=0, restricciones=rest_i)
    nodo_f = Nodo(id=2, x=1, y=0, restricciones=rest_f)

    barra = Elemento(
        id=1,
        nodo_i=1,
        nodo_f=2,
        E=210000,
        b=1,
        h=1,
        L=1.0,
        tita=0.0,
        nodo_i_obj=nodo_i,
        nodo_f_obj=nodo_f
    )

    f_eq_local = barra.calcular_cargas_equivalentes_locales(carga)
    print (nodo_i)
    print(nodo_f)
    print("Cargas equivalentes locales:", f_eq_local)
