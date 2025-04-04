import numpy as np
from core.elementos import Estructura

def ensamblar_matriz_rigidez(estructura: Estructura) -> np.ndarray:
    """
    Ensambla la matriz de rigidez global a partir de los elementos de la estructura.

    Parameters
    ----------
    estructura : Estructura
        Instancia de la estructura con nodos y elementos definidos.

    Returns
    -------
    np.ndarray
        Matriz de rigidez global del sistema.
    """
    ndofs_por_nodo = 3  # Pórtico plano: ux, uy, θ
    n_nodos = len(estructura.nodos)
    K = np.zeros((ndofs_por_nodo * n_nodos, ndofs_por_nodo * n_nodos))

    for elem in estructura.elementos:
        ke = elem.matriz_rigidez_portico()
        ni = elem.nodo_i - 1  # ID → índice
        nf = elem.nodo_f - 1

        dofs = [
            3 * ni, 3 * ni + 1, 3 * ni + 2,
            3 * nf, 3 * nf + 1, 3 * nf + 2,
        ]

        for i in range(6):
            for j in range(6):
                K[dofs[i], dofs[j]] += ke[i, j]

    return K

def ensamblar_vector_fuerzas(estructura: Estructura) -> np.ndarray:
    ndofs_por_nodo = 3
    n_nodos = len(estructura.nodos)
    F = np.zeros(ndofs_por_nodo * n_nodos)

    # Ensamblaje de cargas nodales
    for carga in estructura.cargas_nodales:
        idx = (carga.nodo_id - 1) * ndofs_por_nodo
        F[idx + 0] += carga.fx
        F[idx + 1] += carga.fy
        F[idx + 2] += carga.mz

    # Ensamblaje de cargas equivalentes de barra
    for carga_barra in estructura.cargas_barras:
        elem = estructura.elementos[carga_barra.barra_id - 1]
        tipo_carga = estructura.tipos_carga[carga_barra.carga_id - 1]

        fe_global = elem.cargas_equivalentes_globales(tipo_carga)
        ni = elem.nodo_i - 1
        nf = elem.nodo_f - 1
        dofs = [
            3 * ni, 3 * ni + 1, 3 * ni + 2,
            3 * nf, 3 * nf + 1, 3 * nf + 2,
        ]

        for i in range(6):
            F[dofs[i]] += fe_global[i]

    return F