#Módulo MAC-A: Espacio de Estados, Discretización ZOH y LQR (DARE)

import numpy as np
from scipy.signal import tf2ss, cont2discrete
from scipy.linalg import solve_discrete_are, LinAlgError
import control as ctrl  # <--- Usamos esta librería para el Padé

class StateSpaceEngine:
    @staticmethod
    def fopdt_to_ss(K, tau, theta):
        # UPGRADE: Aproximación de Padé de Orden 4 para precisión absoluta en el State Space
        if theta > 1e-4:
            num_p, den_p = ctrl.pade(theta, n=4)    
        else:
            num_p = [1.0]
            den_p = [1.0]

        num = np.polymul([K], num_p)
        den = np.polymul([tau, 1.0], den_p)
        
        A, B, C, D = tf2ss(num, den)
        return A, B, C, D

    @staticmethod
    def discretize_zoh(A, B, C, D, ts):
        sys_d = cont2discrete((A, B, C, D), ts, method='zoh')
        return sys_d[0], sys_d[1], sys_d[2], sys_d[3]

    @staticmethod
    def compute_lqr(Ad, Bd, Q=None, R=None):
        n_states = Ad.shape[0]
        # FIX 1: Detección dinámica de entradas para preparar compatibilidad MIMO
        n_inputs = Bd.shape[1] if Bd.ndim > 1 else 1

        # FIX 2: Conversión inteligente de escalares a matrices de penalización
        if Q is None:
            Q = np.eye(n_states)
        else:
            Q = np.atleast_2d(Q)
            # Expandir escalar a matriz diagonal del tamaño del estado
            if Q.shape == (1, 1) and n_states > 1:
                Q = Q[0, 0] * np.eye(n_states)

        if R is None:
            R = np.eye(n_inputs)
        else:
            R = np.atleast_2d(R)
            # Expandir escalar a matriz diagonal del número de entradas
            if R.shape == (1, 1) and n_inputs > 1:
                R = R[0, 0] * np.eye(n_inputs)

        try:
            # Solución de la Ecuación Algebraica de Riccati Discreta (DARE)
            P = solve_discrete_are(Ad, Bd, Q, R)
            
            # Cálculo de la Ganancia Óptima
            K_lqr = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
            return K_lqr
            
        except LinAlgError:
            # FIX 3: Captura de fallo si el algoritmo de búsqueda prueba un sistema no estabilizable
            return None

class SubspaceN4SID:
    """Reserva de Arquitectura para Identificación Multivariable MIMO"""
    def __init__(self):
        pass

    def fit_mimo(self, u_data, y_data, ts):
        return "N4SID_MIMO_MODEL_READY"