from core.elementos import Nodo, Elemento, CargaNodal, TipoCarga, CargaBarra, Estructura
from core.ensamble import ensamblar_matriz_rigidez, ensamblar_vector_fuerzas
from core.solucion import resolver_sistema, calcular_solicitaciones, transformar_a_locales
import numpy as np

# --- Datos del problema ---
E = 210  # t/cm^2
estructura = Estructura()

# Nodos
estructura.agregar_nodo(Nodo(1, 0, 0, [True, True, True], [0, 0, 0])) #Empotrado
estructura.agregar_nodo(Nodo(2, 100, 300, None, [0, 0, 0])) #Nudo
estructura.agregar_nodo(Nodo(3, 300, 300, [True, True, True], [0, 0, 0])) #Empotrado
estructura.agregar_nodo(Nodo(4, 0, 300, [False, False, False], [0, 0, 0])) #Libre

# Elementos
estructura.agregar_elemento(Elemento(1, 1, 2, E, 20, 20))  # 20x20
estructura.agregar_elemento(Elemento(2, 4, 2, E, 20, 30))  # 30x20
estructura.agregar_elemento(Elemento(3, 2, 3, E, 20, 30))  # 30x20

# Asociar objetos Nodo a los elementos
for elemento in estructura.elementos:
    elemento.nodo_i_obj = estructura.nodos[elemento.nodo_i-1]
    elemento.nodo_f_obj = estructura.nodos[elemento.nodo_f-1]

# Calcular geometría
coords = {n.id: n.get_coord() for n in estructura.nodos}
for elem in estructura.elementos:
    elem.calcular_longitud_y_angulo(coords[elem.nodo_i], coords[elem.nodo_f])

# Cargas distribuidas en elementos 2 (1 t/m vertical)
estructura.agregar_tipo_carga(TipoCarga(1, 1, 0, 100, -0.01, -0.01, 90))  # vertical
estructura.agregar_carga_barra(CargaBarra(2, 1))

# Cargas distribuidas en elementos 2 (1 t/m vertical)
estructura.agregar_tipo_carga(TipoCarga(3, 1, 0, 300, -0.01, -0.01, 90))  # vertical
estructura.agregar_carga_barra(CargaBarra(3, 3))

# Carga puntual en nodo 2, 20 t a 30 grados
estructura.agregar_tipo_carga(TipoCarga(2, 2, 100, 0, -20, 0, 30))
estructura.agregar_carga_nodal(CargaNodal(2, 2))


for elem in estructura.elementos:
    print(elem)
    print("\n")
    print("--------------------------------------------")

for nodo in estructura.nodos:
    print(nodo)
    print("\n")
    print("--------------------------------------------")


print("\nCargas equivalentes locales por barra:")
for elem in estructura.elementos:
    cargas = [cb for cb in estructura.cargas_barras if cb.barra_id == elem.id]
    if not cargas:
        print(f"Barra {elem.id}: sin carga → [0, 0, 0, 0, 0, 0]")
    else:
        for carga_barra in cargas:
            tipo = next((tc for tc in estructura.tipos_carga if tc.id == carga_barra.carga_id), None)
            if tipo is None or tipo.tipo == 0:
                fe_local = np.zeros(6)
            else:
                fe_local = elem.cargas_equivalentes_globales(tipo)
            print(f"Barra {elem.id} (TipoCarga {tipo.id}): {fe_local}")



 
#-----------------------------------
# Tamaño del sistema
ndofs_por_nodo = 3
n_nodos = len(estructura.nodos)
F = np.zeros(ndofs_por_nodo * n_nodos)

elementos_dict = {e.id: e for e in estructura.elementos}
tipos_carga_dict = {tc.id: tc for tc in estructura.tipos_carga}

# Ensamblaje de cargas nodales puntuales
for carga in estructura.cargas_nodales:
    idx = (carga.nodo_id - 1) * ndofs_por_nodo
    F[idx + 0] += carga.fx
    F[idx + 1] += carga.fy
    F[idx + 2] += carga.mz

# Ensamblaje de cargas equivalentes de barra
for carga_barra in estructura.cargas_barras:
    elem = elementos_dict.get(carga_barra.barra_id)
    tipo_carga = tipos_carga_dict.get(carga_barra.carga_id)

    if elem is None or tipo_carga is None:
        print(f"⚠️ Elemento o tipo de carga no encontrado para barra {carga_barra.barra_id}")
        continue

    fe_global = elem.cargas_equivalentes_globales(tipo_carga)
    ni = elem.nodo_i - 1
    nf = elem.nodo_f - 1

    dofs = [
        3 * ni, 3 * ni + 1, 3 * ni + 2,
        3 * nf, 3 * nf + 1, 3 * nf + 2,
    ]

    for i in range(6):
        F[dofs[i]] += fe_global[i]

print(F)




# --- Ensamble y solución ---
K = ensamblar_matriz_rigidez(estructura)

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
