from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional
from math import cos, sin, radians

@dataclass
class Nodo:
    id: int
    x: float
    y: float
    restricciones: List[bool] = field(default_factory=lambda: [False, False, False])
    valores_prescritos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def get_coord(self):
        return np.array([self.x, self.y])


@dataclass
class CargaNodal:
    nodo_id: int
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0

    def vector(self):
        return np.array([self.fx, self.fy, self.mz])


@dataclass
class TipoCarga:
    id: int
    tipo: int  # 1: distribuida, 2: puntual
    L1: float  # Posición de inicio de la carga (sólo en cargas distribuidas o puntuales)
    L2: float  # Para cargas distribuidas, marca el fin de la carga
    q1: float  # Magnitud de la carga (si tipo == 2: carga puntual, si tipo == 1: intensidad)
    q2: float  # (Solo si tipo == 1) Varía la carga en la barra (por ejemplo, lineal)
    alpha: float  # Ángulo respecto al eje X


@dataclass
class CargaBarra:
    barra_id: int
    carga_id: int



@dataclass
class Elemento:
    id: int
    nodo_i: int
    nodo_f: int
    E: float
    b: float
    h: float
    artic_i: bool = False
    artic_f: bool = False
    tipo: int = 2
    L: Optional[float] = None
    tita: Optional[float] = None

    # Nuevos atributos para guardar los objetos Nodo
    nodo_i_obj: Optional["Nodo"] = None
    nodo_f_obj: Optional["Nodo"] = None

    def area(self):
        return self.b * self.h

    def inercia(self):
        return (self.b * self.h ** 3) / 12

    def calcular_longitud_y_angulo(self, coord_i: np.ndarray, coord_f: np.ndarray):
        dx = coord_f[0] - coord_i[0]
        dy = coord_f[1] - coord_i[1]
        self.L = np.hypot(dx, dy)
        self.tita = np.degrees(np.arctan2(dy, dx))

    def matriz_rigidez_portico(self):
        assert self.L is not None and self.tita is not None, "Longitud y ángulo deben estar definidos"
        A = self.area()
        I = self.inercia()
        L = self.L
        E = self.E
        c = np.cos(np.radians(self.tita))
        s = np.sin(np.radians(self.tita))

        Kloc = np.array([
            [E*A/L,       0,           0,      -E*A/L,       0,           0],
            [0,     12*E*I/L**3,  6*E*I/L**2,     0, -12*E*I/L**3,  6*E*I/L**2],
            [0,     6*E*I/L**2,   4*E*I/L,        0, -6*E*I/L**2,   2*E*I/L],
            [-E*A/L,      0,           0,       E*A/L,       0,           0],
            [0,    -12*E*I/L**3, -6*E*I/L**2,     0,  12*E*I/L**3, -6*E*I/L**2],
            [0,     6*E*I/L**2,   2*E*I/L,        0, -6*E*I/L**2,   4*E*I/L],
        ])

        R = self.matriz_rotacion()
        return R.T @ Kloc @ R

    def calcular_cargas_equivalentes_locales(self, tipo_carga) -> np.ndarray:
        def _nodo_soporta_carga(nodo, comp: list[int]) -> bool:
            return nodo is not None and nodo.restricciones is not None and any(nodo.restricciones[i] for i in comp)

        if tipo_carga is None or tipo_carga.tipo == 0:
            return np.zeros(6)

        L = self.L
        L1 = tipo_carga.L1
        L2 = tipo_carga.L2
        tita = radians(self.tita)

        nodo_i = self.nodo_i_obj
        nodo_j = self.nodo_f_obj

        nodo_i_fijo =  _nodo_soporta_carga(nodo_i, [0, 1, 2])
        nodo_j_fijo =  _nodo_soporta_carga(nodo_j, [0, 1, 2])

        if tipo_carga.tipo == 1:
            q = tipo_carga.q1
            alpha = radians(tipo_carga.alpha)
            cos_a = cos(alpha - tita)
            sen_a = sin(alpha - tita)

            if nodo_i_fijo and not nodo_j_fijo:
                Fx = q * (L2-L1) * cos_a 
                Fy = q * (L2-L1) * sen_a 
                MA = -q * (L2-L1) * (L - (L2-L1)/2) * sen_a
                return np.array([Fx, Fy, MA, 0, 0, 0])

            elif nodo_j_fijo and not nodo_i_fijo:
                Fx = q * (L2-L1) * cos_a 
                Fy = q * (L2-L1) * sen_a 
                MB = -q * (L2-L1) * (L - (L2-L1)/2) * sen_a
                return np.array([0, 0, 0, Fx, Fy, MB])

            elif nodo_i_fijo and nodo_j_fijo: #NO ES GENERICO HAY QUE CAMBIAR ESTO XD
                N = cos_a * q * L / 2
                Q = sen_a * q * L / 2
                M = sen_a * q * L**2 / 12
                return np.array([N, Q, M, N, Q, -M])

            else:
                raise ValueError("Ambos nodos son libres, no puede aplicarse carga.")

        elif tipo_carga.tipo == 2:
            p = tipo_carga.q1
            li = self.L * tipo_carga.L1
            lj = self.L - li
            alpha = radians(tipo_carga.alpha)
            cos_a = cos(alpha - tita)
            sen_a = sin(alpha - tita)

            if nodo_i_fijo and not nodo_j_fijo:
                Fx = -p * cos_a
                Fy = -p * sen_a
                MA = -p * sen_a * (L - li)
                return np.array([Fx, Fy, MA, 0, 0, 0])

            elif nodo_j_fijo and not nodo_i_fijo:
                Fx = -p * cos_a
                Fy = -p * sen_a
                MB = p * sen_a * li
                return np.array([0, 0, 0, Fx, Fy, -MB])

            elif nodo_i_fijo and nodo_j_fijo:
                Ni = -cos_a * p * lj / L
                Nj = -cos_a * p * li / L
                Mi = -sen_a * p * li * (lj / L)**2
                Mj = -sen_a * p * lj * (li / L)**2
                Qi = -sen_a * p * ((lj / L)**2) * (3 - 2 * lj / L)
                Qj = -sen_a * p * ((li / L)**2) * (3 - 2 * li / L)
                return np.array([Ni, Qi, Mi, Nj, Qj, -Mj])

            else:
                raise ValueError("Ambos nodos son libres, no puede aplicarse carga.")
        else:
            raise NotImplementedError("Tipo de carga no soportado")

    def matriz_rotacion(self) -> np.ndarray:
        c = np.cos(np.radians(self.tita))
        s = np.sin(np.radians(self.tita))
        return np.array([
            [ c,  s, 0,  0, 0, 0],
            [-s,  c, 0,  0, 0, 0],
            [ 0,  0, 1,  0, 0, 0],
            [ 0,  0, 0,  c, s, 0],
            [ 0,  0, 0, -s, c, 0],
            [ 0,  0, 0,  0, 0, 1],
        ])

    def cargas_equivalentes_globales(self, tipo_carga) -> np.ndarray:
        f_local = self.calcular_cargas_equivalentes_locales(tipo_carga)
        R = self.matriz_rotacion()
        return R.T @ f_local



@dataclass
class Estructura:
    nodos: List[Nodo] = field(default_factory=list)
    elementos: List[Elemento] = field(default_factory=list)
    cargas_nodales: List[CargaNodal] = field(default_factory=list)
    cargas_barras: List[CargaBarra] = field(default_factory=list)
    tipos_carga: List[TipoCarga] = field(default_factory=list)

    def agregar_nodo(self, nodo: Nodo):
        self.nodos.append(nodo)

    def agregar_elemento(self, elem: Elemento):
        self.elementos.append(elem)

    def agregar_carga_nodal(self, carga_nodal: CargaNodal):
        tipo = next((tc for tc in self.tipos_carga if tc.id == carga_nodal.nodo_id), None)
        if tipo is None:
            raise ValueError(f"Tipo de carga {carga_nodal.nodo_id} no encontrado.")

        fx = tipo.q1 * np.cos(np.radians(tipo.alpha))
        fy = tipo.q1 * np.sin(np.radians(tipo.alpha))
        self.cargas_nodales.append(CargaNodal(carga_nodal.nodo_id, fx, fy, 0))

    def agregar_tipo_carga(self, tipo: TipoCarga):
        self.tipos_carga.append(tipo)

    def agregar_carga_barra(self, carga_barra: CargaBarra):
        self.cargas_barras.append(carga_barra)
