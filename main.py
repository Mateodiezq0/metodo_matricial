from core.elementos import Nodo, Elemento, CargaNodal, TipoCarga, CargaBarra, Estructura
from core.ensamble import ensamblar_matriz_rigidez, ensamblar_vector_fuerzas
from core.solucion import resolver_sistema, calcular_solicitaciones, transformar_a_locales
import numpy as np

# --- Datos del problema ---
E = 210e4  # t/m^2
estructura = Estructura()

# Nodos
estructura.agregar_nodo(Nodo(1, 0, 0, [True, True, True], [0, 0, 0]))
estructura.agregar_nodo(Nodo(2, 1, 3))
estructura.agregar_nodo(Nodo(3, 3, 3))
estructura.agregar_nodo(Nodo(4, 5, 3, [True, True, True], [0, 0, 0]))

# Elementos
estructura.agregar_elemento(Elemento(1, 1, 2, E, 0.2, 0.2))  # 20x20
estructura.agregar_elemento(Elemento(2, 2, 4, E, 0.3, 0.2))  # 20x30
estructura.agregar_elemento(Elemento(3, 2, 3, E, 0.3, 0.2))

# Calcular geometría
coords = {n.id: n.get_coord() for n in estructura.nodos}
for elem in estructura.elementos:
    elem.calcular_longitud_y_angulo(coords[elem.nodo_i], coords[elem.nodo_f])

# Cargas distribuidas en elementos 2 (1 t/m vertical)
estructura.agregar_tipo_carga(TipoCarga(1, 1, 0, 0, 1, 1, 90))  # vertical
estructura.agregar_carga_barra(CargaBarra(2, 1))

# Carga puntual en barra 3, 20 t a 30 grados
estructura.agregar_tipo_carga(TipoCarga(2, 2, 0.5, 0, 20, 0, 30))
estructura.agregar_carga_barra(CargaBarra(3, 2))

# --- Ensamble y solución ---
K = ensamblar_matriz_rigidez(estructura)
F = ensamblar_vector_fuerzas(estructura)
D = resolver_sistema(K, F, estructura)
esf_globales = calcular_solicitaciones(estructura, D)
esf_locales = transformar_a_locales(esf_globales, estructura)

# --- Salida ---
np.set_printoptions(precision=3, suppress=True)
print("Matriz K:")
print(K)
print("\nVector de cargas F:")
print(F)
print("\nVector de desplazamientos D:")
print(D)
print("\nEsfuerzos globales (por barra):")
for i, ef in enumerate(esf_globales, 1):
    print(f"Barra {i}: {ef}")
print("\nEsfuerzos locales (N, Q, M):")
for i, ef in enumerate(esf_locales, 1):
    print(f"Barra {i}: {ef}")
