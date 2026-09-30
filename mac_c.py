#Módulo MAC-C: Control Clásico, SysID (FO, FOPDT, SO) y SIMC


import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lti, step

class ClassicalSysID:
    @staticmethod
    def _fo_model(t, K, tau):
        return K * (1.0 - np.exp(-t / np.maximum(tau, 1e-4)))

    @staticmethod
    def _fopdt_model(t, K, tau, theta):
        t_shift = np.maximum(0.0, t - theta)
        return K * (1.0 - np.exp(-t_shift / np.maximum(tau, 1e-4)))

    @staticmethod
    def _so_model(t, K, wn, zeta):
        wn = max(wn, 1e-4)
        zeta = max(zeta, 1e-4)
        
        # FIX: Evitar división por cero si zeta == 1 (Críticamente amortiguado)
        if abs(zeta - 1.0) < 1e-4:
            return K * (1.0 - (1.0 + wn * t) * np.exp(-wn * t))
            
        elif zeta < 1.0: # Subamortiguado
            wd = wn * np.sqrt(1.0 - zeta**2)
            phi = np.arctan(np.sqrt(1.0 - zeta**2) / zeta)
            return K * (1.0 - (1.0 / np.sqrt(1.0 - zeta**2)) * np.exp(-zeta * wn * t) * np.sin(wd * t + phi))
        else: # Sobreamortiguado
            p1 = -wn * (zeta - np.sqrt(zeta**2 - 1.0))
            p2 = -wn * (zeta + np.sqrt(zeta**2 - 1.0))
            return K * (1.0 + (p2 * np.exp(p1 * t) - p1 * np.exp(p2 * t)) / (p1 - p2))

    def fit_all(self, t, y):
        # FIX: Normalizar a t=0 y y=0 para que las exponenciales funcionen correctamente
        t_arr = np.asarray(t)
        y_arr = np.asarray(y)
        t_norm = t_arr - t_arr[0]
        y0 = y_arr[0]
        y_norm = y_arr - y0 
        
        y_final = y_norm[-1]
        t_final = t_norm[-1]
        
        # FIX: Prevenir colisión de bounds si la muestra es muy rápida
        theta_max = max(t_final * 0.5, 1e-3)
        p0_theta = min(0.1, theta_max * 0.5)

        try:
            # 1. Ajuste FO
            popt_fo, _ = curve_fit(self._fo_model, t_norm, y_norm, 
                                   p0=[y_final, t_final/3.0], bounds=(0, [np.inf, np.inf]))
            mse_fo = np.mean((y_norm - self._fo_model(t_norm, *popt_fo))**2)
        except RuntimeError:
            mse_fo = np.inf

        try:
            # 2. Ajuste FOPDT
            popt_fopdt, _ = curve_fit(self._fopdt_model, t_norm, y_norm, 
                                      p0=[y_final, t_final/3.0, p0_theta], bounds=(0, [np.inf, np.inf, theta_max]))
            mse_fopdt = np.mean((y_norm - self._fopdt_model(t_norm, *popt_fopdt))**2)
            penalized_mse_fopdt = mse_fopdt * 1.05
        except RuntimeError:
            penalized_mse_fopdt = np.inf

        try:
            # 3. Ajuste SO
            popt_so, _ = curve_fit(self._so_model, t_norm, y_norm, 
                                   p0=[y_final, 2.0, 0.7], bounds=(0, [np.inf, np.inf, 5.0]))
            mse_so = np.mean((y_norm - self._so_model(t_norm, *popt_so))**2)
        except RuntimeError:
            mse_so = np.inf

        # Selección de Modelo
        if mse_fo <= penalized_mse_fopdt and mse_fo <= mse_so:
            return {"type": "FO", "K": popt_fo[0], "tau": popt_fo[1], "theta": 0.0, "mse": mse_fo, "y0": y0}
        elif penalized_mse_fopdt <= mse_so:
            return {"type": "FOPDT", "K": popt_fopdt[0], "tau": popt_fopdt[1], "theta": popt_fopdt[2], "mse": mse_fopdt, "y0": y0}
        else:
            return {"type": "SO", "K": popt_so[0], "wn": popt_so[1], "zeta": popt_so[2], "mse": mse_so, "y0": y0}

class AnalyticalTuning:
    @staticmethod
    def skogestad_simc(K, tau, theta, tau_c=None):
        if tau_c is None:
            tau_c = max(theta, 0.1 * tau)
        
        theta_eff = max(theta, 1e-3)
        Kp = (1.0 / K) * (tau / (tau_c + theta_eff))
        Ti = min(tau, 4.0 * (tau_c + theta_eff))
        Ki = Kp / Ti
        Kd = 0.0
        return {"Kp": Kp, "Ki": Ki, "Kd": Kd, "Ti": Ti, "tau_c": tau_c}

class ClassicalSimulator:
    @staticmethod
    def simulate_fopdt_loop(K, tau, theta, Kp, Ki, Kd, t_sim, hardware_config=None):
        if theta > 1e-4:
            num_pade = [theta**2, -6*theta, 12]
            den_pade = [theta**2, 6*theta, 12]
        else:
            num_pade = [1.0]
            den_pade = [1.0]
            
        num_p = np.polymul([K], num_pade)
        den_p = np.polymul([tau, 1.0], den_pade)

        tf_filter = max(0.05 * tau, 0.005)
        num_c = [(Kp * tf_filter) + Kd, Kp + (Ki * tf_filter), Ki]
        den_c = [tf_filter, 1.0, 0.0]

        num_open = np.polymul(num_c, num_p)
        den_open = np.polymul(den_c, den_p)
        den_cl = np.polyadd(den_open, num_open)
        
        system_cl = lti(num_open, den_cl)
        t_eval = np.linspace(0, t_sim, 200)
        t_out, y_out = step(system_cl, T=t_eval)

        # Simulación de señal de control u(t)
        e = 1.0 - y_out
        dt = t_out[1] - t_out[0] if len(t_out) > 1 else 0.01
        
        # FIX: Añadida la acción derivativa en la reconstrucción discreta de la señal
        de = np.gradient(e, dt) 
        u_unfiltered = Kp * e + Ki * np.cumsum(e) * dt + Kd * de
        
        if hardware_config is not None:
            # Nota: Esto recorta u(t) visualmente, pero al usar scipy.signal.lti la planta real
            # siguió recibiendo la u(t) infinita. No observarás windup integral aquí.
            u_out = np.array([hardware_config.quantize_and_saturate(u) for u in u_unfiltered])
        else:
            u_out = u_unfiltered
            
        return t_out, y_out, u_out