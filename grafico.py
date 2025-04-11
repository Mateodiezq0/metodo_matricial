# Archivo: graf_esfuerzos.py
import numpy as np
import matplotlib.pyplot as plt
from core.elementos import Estructura


def calcular_distribucion_caso_constante(Ni, Qi, Mi, q, L, n_points=50):
    x = np.linspace(0, L, n_points)
    N = np.full_like(x, Ni)  # axial constante
    Q = Qi - q * x           # cortante lineal
    M = Mi + Qi * x - 0.5 * q * x**2  # momento cuadrático
    return x, N, Q, M


def graficar_esfuerzos_estructura(estructura: Estructura, esfuerzos_locales):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].set_title("Distribución de esfuerzos locales en barras")

    for idx, elem in enumerate(estructura.elementos):
        L = elem.L
        Q1 = None
        cargas = [cb for cb in estructura.cargas_barras if cb.barra_id == elem.id]
        tipo = None
        for cb in cargas:
            tipo = next((tc for tc in estructura.tipos_carga if tc.id == cb.carga_id), None)
            if tipo and tipo.tipo == 1 and tipo.q1 == tipo.q2:
                Q1 = tipo.q1
                break

        if Q1 is not None:
            x, N, Q, M = calcular_distribucion_caso_constante(
                esfuerzos_locales[idx][0], esfuerzos_locales[idx][1], esfuerzos_locales[idx][2], Q1, L)

            axs[0].plot(x + elem.nodo_i_obj.x, N, label=f"Barra {elem.id}")
            axs[1].plot(x + elem.nodo_i_obj.x, Q, label=f"Barra {elem.id}")
            axs[2].plot(x + elem.nodo_i_obj.x, M, label=f"Barra {elem.id}")

    axs[0].set_ylabel("Axial N [t]")
    axs[1].set_ylabel("Cortante Q [t]")
    axs[2].set_ylabel("Momento M [t*cm]")
    axs[2].set_xlabel("Posición x [cm]")
    for ax in axs:
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()


# Llamado desde main.py
if __name__ == "__main__":
    from main import estructura, esf_locales
    graficar_esfuerzos_estructura(estructura, esf_locales)
