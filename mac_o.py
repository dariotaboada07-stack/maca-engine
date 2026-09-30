#Módulo MAC-O: Optimización Multiobjetivo (ITAE + Energía + Saturación)

import numpy as np
from scipy.optimize import minimize
from mac_c import ClassicalSimulator  # Asegúrate de que este módulo esté accesible

class MultiObjectiveOptimizer:
    def __init__(self, model_params, hardware_config=None, alpha=1.0, beta=0.01):
        self.model = model_params
        self.hw = hardware_config
        self.alpha = alpha  # Peso para ITAE
        self.beta = beta    # Peso para Energía del Actuador u(t)^2

    def _cost_function(self, gains, t_sim):
        Kp, Ki, Kd = gains
        
        # Penalización severa si el optimizador prueba ganancias negativas
        if Kp <= 0 or Ki < 0 or Kd < 0:
            return 1e6

        K = self.model.get('K', 1.0)
        tau = self.model.get('tau', 1.0)
        theta = self.model.get('theta', 0.0)

        try:
            t_out, y_out, u_out = ClassicalSimulator.simulate_fopdt_loop(
                K, tau, theta, Kp, Ki, Kd, t_sim, self.hw
            )

            # 1. ITAE (Integral Time-weighted Absolute Error)
            error = np.abs(1.0 - y_out)
            itae = np.trapezoid(t_out * error, t_out) if hasattr(np, 'trapezoid') else np.trapz(t_out * error, t_out)

            # 2. Control Effort (Energía del Actuador)
            # FIX: Ignoramos el índice 0 para suprimir el impulso infinito del "Derivative Kick" inicial
            if len(u_out) > 1:
                energy = np.trapezoid(u_out[1:]**2, t_out[1:]) if hasattr(np, 'trapezoid') else np.trapz(u_out[1:]**2, t_out[1:])
            else:
                energy = 0.0

            # 3. Penalización de Overshoot (>5%)
            peak = np.max(y_out)
            overshoot_pen = 0.0
            if peak > 1.05:
                # Castigo cuadrático escalado para picos altos
                overshoot_pen = 1e3 * ((peak - 1.05) ** 2)
                
            # 4. FIX: Penalización de Error en Estado Estacionario (Terminal Error)
            # Obliga al optimizador a que la señal termine en la referencia (tolerancia del 2%)
            ss_error = np.abs(1.0 - y_out[-1])
            ss_pen = 1e4 * (ss_error ** 2) if ss_error > 0.02 else 0.0

            total_cost = self.alpha * itae + self.beta * energy + overshoot_pen + ss_pen
            return total_cost if np.isfinite(total_cost) else 1e6

        except Exception:
            return 1e6

    def optimize(self, initial_gains, t_sim):
        # FIX: Asegurar que las suposiciones iniciales no rompan los límites estrictos
        x0 = [
            max(0.01, initial_gains['Kp']), 
            max(0.001, initial_gains['Ki']), 
            max(0.0, initial_gains.get('Kd', 0.0))
        ]
        
        # Limites (Kp, Ki, Kd) - Evitamos que Kp y Ki sean 0 para asegurar control
        bounds = [(0.01, 100.0), (0.001, 100.0), (0.0, 50.0)]
        
        res = minimize(
            self._cost_function, 
            x0, 
            args=(t_sim,), 
            method='L-BFGS-B', 
            bounds=bounds
        )
        
        return {
            "Kp": res.x[0],
            "Ki": res.x[1],
            "Kd": res.x[2],
            "cost": res.fun,
            "success": res.success
        }