import numpy as np
from core.elementos import Estructura
from typing import List

def resolver_sistema(K: np.ndarray, F: np.ndarray, estructura: Estructura) -> np.ndarray:
    """
    Resuelve el sistema de ecuaciones K D = F aplicando condiciones de borde.

    Parameters
    ----------
    K : np.ndarray
        Matriz de rigidez global.
    F : np.ndarray
        Vector de fuerzas global.
    estructura : Estructura
        Estructura con nodos que contienen las restricciones.

    Returns
    -------
    np.ndarray
        Vector global de desplazamientos.
    """
    ndofs = len(F)
    prescripciones = np.full(ndofs, np.nan)

    # Paso 1: Cargar desplazamientos prescritos (solo si hay restricciones reales)
    for nodo in estructura.nodos:
        base = (nodo.id - 1) * 3
        if nodo.restricciones is not None:
            for i in range(3):
                if nodo.restricciones[i]:
                    prescripciones[base + i] = nodo.valores_prescritos[i]

    # Armamos índices de fijos y libres
    idx_fijos = np.where(~np.isnan(prescripciones))[0]
    idx_libres = np.where(np.isnan(prescripciones))[0]

    # Vector de desplazamientos global
    D = np.zeros(ndofs)
    D[idx_fijos] = prescripciones[idx_fijos]

    # Sistema reducido
    Kll = K[np.ix_(idx_libres, idx_libres)]
    Klf = K[np.ix_(idx_libres, idx_fijos)]
    Fl = F[idx_libres]
    Df = D[idx_fijos]

    Dl = np.linalg.solve(Kll, Fl - Klf @ Df)
    D[idx_libres] = Dl

    return D






def calcular_solicitaciones(estructura: Estructura, D: np.ndarray) -> List[np.ndarray]:
    """
    Calcula los esfuerzos nodales de cada barra en coordenadas globales.

    Parameters
    ----------
    estructura : Estructura
        Instancia de la estructura.
    D : np.ndarray
        Vector global de desplazamientos resuelto.

    Returns
    -------
    List[np.ndarray]
        Lista con los vectores de esfuerzos nodales por barra (globales, 6 componentes).
    """
    esfuerzos = []

    for elem in estructura.elementos:
        ni = elem.nodo_i - 1
        nf = elem.nodo_f - 1

        dofs = [
            3 * ni, 3 * ni + 1, 3 * ni + 2,
            3 * nf, 3 * nf + 1, 3 * nf + 2
        ]

        D_elem = D[dofs]
        K_elem = elem.matriz_rigidez_portico()

        # Sumar todas las cargas equivalentes aplicadas a la barra
        A_elem = np.zeros(6)
        for carga_barra in estructura.cargas_barras:
            if carga_barra.barra_id == elem.id:
                tipo_carga = next((tc for tc in estructura.tipos_carga if tc.id == carga_barra.carga_id), None)
                if tipo_carga:
                    A_elem += elem.cargas_equivalentes_globales(tipo_carga)

        # Reacción interna: fuerza nodal generada por desplazamiento (K·D) más carga equivalente
        F_elem = K_elem @ D_elem - A_elem  # 👈 esto da la solicitación interna neta
        esfuerzos.append(F_elem)

    return esfuerzos




def transformar_a_locales(esfuerzos_globales: List[np.ndarray], estructura: Estructura) -> List[np.ndarray]:
    """
    Transforma los esfuerzos nodales de cada barra desde coordenadas globales a locales.

    Parameters
    ----------
    esfuerzos_globales : List[np.ndarray]
        Lista de esfuerzos nodales por barra (globales).
    estructura : Estructura
        Lista de elementos estructurales con matriz de rotación.

    Returns
    -------
    List[np.ndarray]
        Lista de esfuerzos nodales transformados a coordenadas locales.
    """
    esfuerzos_locales = []

    for elem, Fg in zip(estructura.elementos, esfuerzos_globales):
        R = elem.matriz_rotacion()
        Fl = R @ Fg
        esfuerzos_locales.append(Fl)

    return esfuerzos_locales
