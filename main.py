import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importamos las clases desde sus respectivos módulos modulares 
from mac_c import ClassicalSysID, AnalyticalTuning
from mac_a import StateSpaceEngine
from mac_o import MultiObjectiveOptimizer
from mac_lm import IndustrialTranspiler

def generar_datos_prueba():
    """Genera datos simulados para no depender de un CSV al probar"""
    t = np.linspace(0, 20, 200)
    K_real, tau_real, theta_real = 1.8, 2.5, 0.8
    y = np.zeros_like(t)
    indices = t >= theta_real
    y[indices] = K_real * 50.0 * (1 - np.exp(-(t[indices] - theta_real) / tau_real))
    y += np.random.normal(0, 0.5, len(t)) + 25.0 # Ruido y offset
    return t, y, 50.0 # tiempo, salida, pwm (amplitud del escalón)

if __name__ == "__main__":
    print("=== INICIANDO MACA ENGINE MK3 ===")
    
    # 1. Ingesta de datos
    t_data, y_data, u_step = generar_datos_prueba()
    
    # 2. IDENTIFICACIÓN DE PLANTA (El foco principal)
    print("\n[1/3] Ejecutando Identificación Clásica (SysID)...")
    sysid = ClassicalSysID()
    
    # Dividimos y_data entre u_step temporalmente para que la ganancia K sea unitaria respecto a la entrada
    y_norm = (y_data - y_data[0]) / u_step 
    
    modelo = sysid.fit_all(t_data, y_norm)
    print(f"-> Mejor modelo encontrado: {modelo['type']}")
    print(f"-> Parámetros: K={modelo['K']:.3f}, Tau={modelo['tau']:.3f}s, Theta={modelo.get('theta', 0):.3f}s")
    
    # 3. TRANSFORMACIÓN A ESPACIO DE ESTADOS (Para el proyecto de Rodrigo)
    print("\n[2/3] Generando Matrices de Espacio de Estados (State Space)...")
    if modelo['type'] in ['FO', 'FOPDT']:
        A, B, C, D = StateSpaceEngine.fopdt_to_ss(modelo['K'], modelo['tau'], modelo.get('theta', 0.0))
        
        # Discretizar para control digital (Ej. Ts = 100ms)
        Ts = 0.1 
        Ad, Bd, Cd, Dd = StateSpaceEngine.discretize_zoh(A, B, C, D, Ts)
        
        print("-> Matrices de Tiempo Discreto (Ad, Bd) generadas exitosamente.")
        print("Matriz Ad (Dinámica del sistema):")
        print(np.round(Ad, 4))
        
    # 4. SINTONIZACIÓN Y GENERACIÓN DE CÓDIGO (Demostración de conexión)
    print("\n[3/3] Sintonización y Transpilación Industrial...")
    # Sintonización base con Skogestad
    gains_base = AnalyticalTuning.skogestad_simc(modelo['K'], modelo['tau'], modelo.get('theta', 0.0))
    
    # Generar código para PLC
    plc_code = IndustrialTranspiler.generate_structured_text(
        gains_base['Kp'], gains_base['Ki'], gains_base['Kd'], ts=0.1
    )
    
    print("-> Código IEC 61131-3 generado. Listo para inyectar en hardware.")
    print("==================================================")
