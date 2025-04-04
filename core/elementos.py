from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional

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
    tipo: int  # 1: uniforme, 2: puntual
    L1: float
    L2: float
    q1: float
    q2: float
    alpha: float


@dataclass
class CargaBarra:
    barra_id: int
    carga_id: int


@dataclass
class Elemento:
    id: int               # Identificador único del elemento
    nodo_i: int           # Nodo inicial (ID)
    nodo_f: int           # Nodo final (ID)
    E: float              # Módulo de elasticidad
    b: float              # Base de la sección
    h: float              # Altura de la sección
    artic_i: bool = False # ¿Está articulado en el extremo inicial?
    artic_f: bool = False # ¿En el extremo final?
    tipo: int = 2         # Tipo de elemento (por defecto pórtico plano)
    L: Optional[float] = None   # Longitud (se calcula luego)
    tita: Optional[float] = None # Ángulo respecto del eje X (en grados)


    def area(self):
        return self.b * self.h

    def inercia(self):
        self.inercia = (self.b * self.h ** 3) / 12
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

        R = np.array([
            [ c,  s, 0,  0, 0, 0],
            [-s,  c, 0,  0, 0, 0],
            [ 0,  0, 1,  0, 0, 0],
            [ 0,  0, 0,  c, s, 0],
            [ 0,  0, 0, -s, c, 0],
            [ 0,  0, 0,  0, 0, 1],
        ])

        return R.T @ Kloc @ R


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

    def agregar_carga_nodal(self, carga: CargaNodal):
        self.cargas_nodales.append(carga)

    def agregar_tipo_carga(self, tipo: TipoCarga):
        self.tipos_carga.append(tipo)

    def agregar_carga_barra(self, carga_barra: CargaBarra):
        self.cargas_barras.append(carga_barra)
