import sympy as sp

def cargas_equivalentes_lineal_empotrado(q1: float, q2: float, L1: float, L2: float, L: float):
    # Definimos las variables simbólicas
    x = sp.Symbol('x')

    # Carga distribuida lineal entre L1 y L2
    m = (q2 - q1)/(L2 - L1)
    qx = q1 + m*(x - L1)

    # Funciones de forma para viga (modo vertical): N1 y N2
    N1 = 1 - x/L
    N2 = x/L

    # Fuerzas equivalentes verticales en nodo i y j
    Qi = sp.integrate(qx * N1, (x, L1, L2))
    Qj = sp.integrate(qx * N2, (x, L1, L2))

    # Momentos equivalentes
    Mi = sp.integrate(qx * N1 * x, (x, L1, L2))
    Mj = sp.integrate(qx * N2 * (x - L), (x, L1, L2))

    # Resultados con signos negativos por convención de equilibrio
    fe = [
        0,              # Ni (sin carga axial)
        float(Qi),      # Qi
        float(Mi),      # Mi
        0,              # Nj
        float(Qj),      # Qj
        float(Mj)       # Mj
    ]

    return fe

# Ejemplo de uso:
if __name__ == "__main__":
    q1 = -2.0    # Carga en L1 (t/m)
    q2 = -5.0    # Carga en L2 (t/m)
    L1 = 2.0     # Inicio de carga (m)
    L2 = 5.0     # Fin de carga (m)
    L = 10.0     # Longitud total de la viga (m)

    feq = cargas_equivalentes_lineal_empotrado(q1, q2, L1, L2, L)
    print("Fuerzas nodales equivalentes:")
    print(f"[Ni, Qi, Mi, Nj, Qj, Mj] = {feq}")
