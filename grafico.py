import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon, Circle

def plot_estructura(estructura, escala=1.0):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.grid(True)

    # --- Dibujar barras ---
    for elem in estructura.elementos:
        ni = elem.nodo_i_obj.get_coord()
        nf = elem.nodo_f_obj.get_coord()
        ax.plot([ni[0], nf[0]], [ni[1], nf[1]], 'k-', linewidth=3)

    # --- Dibujar nodos y numeración ---
    for nodo in estructura.nodos:
        x, y = nodo.get_coord()
        ax.plot(x, y, 'ro', markersize=8, zorder=3)  # zorder para que estén encima
        ax.text(x + 0.5, y + 0.5, f'N{nodo.id}', color='red', fontsize=12, weight='bold')

    # --- Apoyos (PRIMERO LOS DIBUJAMOS CON ZORDER BAJO) ---
    for nodo in estructura.nodos:
        if hasattr(nodo, 'restricciones') and nodo.restricciones is not None:
            x, y = nodo.get_coord()
            restr = nodo.restricciones
            
            # Tamaño base para los apoyos (ajústalo según tu estructura)
            support_size = max(2, 0.015 * max(abs(x) for x in plt.xlim()))  
            
            if restr == [True, True, True]:  # Empotramiento
                rect = plt.Rectangle((x - support_size, y - 2*support_size), 
                                   2*support_size, 2*support_size, 
                                   color='lime', alpha=0.7, zorder=1)
                ax.add_patch(rect)
                ax.text(x, y - 2.8*support_size, 'EMPOTRE', 
                       ha='center', color='darkgreen', weight='bold')
                
            elif sum(restr[:2]) == 2:  # Apoyo doble (X e Y fijos)
                tri = plt.Polygon([[x, y], 
                                  [x - support_size, y - 2*support_size],
                                  [x + support_size, y - 2*support_size]], 
                                 color='lime', alpha=0.7, zorder=1)
                ax.add_patch(tri)
                ax.text(x, y - 2.8*support_size, 'APOYO DOBLE', 
                       ha='center', color='darkgreen', weight='bold')
                
            elif any(restr[:2]):  # Apoyo simple (X o Y fijo)
                circ = plt.Circle((x, y-support_size), radius=support_size, 
                                color='lime', alpha=0.7, zorder=1)
                ax.add_patch(circ)
                ax.text(x, y - 2.8*support_size, 'APOYO SIMPLE', 
                       ha='center', color='darkgreen', weight='bold')

    # --- Cargas nodales ---
    for carga in estructura.cargas_nodales:
        nodo = estructura.nodos[carga.nodo_id - 1]
        x, y = nodo.get_coord()
        if carga.fx != 0 or carga.fy != 0:
            ax.arrow(x, y, carga.fx * escala, carga.fy * escala,
                    head_width=0.5, head_length=1, fc='red', ec='red', zorder=4)
            ax.text(x + carga.fx * escala, y + carga.fy * escala, 
                   f'P={carga.fx, carga.fy}', color='red', weight='bold')

    # --- Cargas distribuidas ---
    for carga_barra in estructura.cargas_barras:
        elem = next(e for e in estructura.elementos if e.id == carga_barra.barra_id)
        tipo = next(tc for tc in estructura.tipos_carga if tc.id == carga_barra.carga_id)

        if tipo.tipo == 1:
            ni = elem.nodo_i_obj.get_coord()
            nf = elem.nodo_f_obj.get_coord()
            x_mid = (ni[0] + nf[0]) / 2
            y_mid = (ni[1] + nf[1]) / 2

            alpha = np.radians(tipo.alpha)
            qx = escala * tipo.q1 * np.cos(alpha)
            qy = escala * tipo.q1 * np.sin(alpha)

            ax.arrow(x_mid, y_mid, qx, qy,
                    head_width=0.5, head_length=1, fc='blue', ec='blue', zorder=4)
            ax.text(x_mid + qx, y_mid + qy, f'q={tipo.q1}', 
                   color='blue', weight='bold')

    ax.set_title('ESTRUCTURA INDEFORMADA', weight='bold', size=14)
    plt.xlabel('X [m]', weight='bold')
    plt.ylabel('Y [m]', weight='bold')
    plt.tight_layout()
    plt.show()