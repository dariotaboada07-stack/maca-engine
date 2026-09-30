#Configuración de Hardware, Presets y Clasificación AUTO


import numpy as np

class HardwareConfig:
    def __init__(self, name, u_min=0.0, u_max=5.0, resolution_bits=8, ts=0.01, deadband=0.0, max_current=2.0):
        self.name = name
        self.u_min = u_min
        self.u_max = u_max
        self.resolution_bits = resolution_bits
        self.ts = ts
        self.deadband = deadband
        self.max_current = max_current
        self.levels = (2 ** resolution_bits) - 1

    def quantize_and_saturate(self, u_calculated):
        # 1. Aplicar Zona Muerta (Deadband)
        u_val = u_calculated
        if abs(u_val) < self.deadband:
            u_val = 0.0
            
        # 2. Aplicar Saturación Física
        u_sat = np.clip(u_val, self.u_min, self.u_max)
        
        # 3. Aplicar Cuantización Digital (DAC / PWM)
        if self.resolution_bits > 0 and self.u_max > self.u_min:
            span = self.u_max - self.u_min
            norm = (u_sat - self.u_min) / span
            step = round(norm * self.levels)
            u_quant = self.u_min + (step / self.levels) * span
        else:
            u_quant = u_sat
            
        # Retornar como float nativo para evitar arrastrar tipos np.float64
        return float(u_quant) 

PRESETS = {
    "ARDUINO_AVR_ARM": HardwareConfig("Arduino AVR/ARM (0-5V / PWM 8-bit)", u_min=0.0, u_max=5.0, resolution_bits=8, ts=0.001, deadband=0.0),
    "H_BRIDGE_DRIVER": HardwareConfig("Puente H Driver (-12V a +12V)", u_min=-12.0, u_max=12.0, resolution_bits=10, ts=0.001, deadband=0.5),
    "RAMPS_3D_BOARD": HardwareConfig("RAMPS 3D Heater Board (Soft-PWM 0-100%)", u_min=0.0, u_max=100.0, resolution_bits=8, ts=0.1, deadband=0.0),
    "PLC_ARDUINO_OPTA": HardwareConfig("PLC Industrial Arduino Opta (0-10V / 12-bit)", u_min=0.0, u_max=10.0, resolution_bits=12, ts=0.005, deadband=0.0)
}

class ActuatorClassifier:
    @staticmethod
    def classify(t, y):
        # FIX 1: Forzar coerción a NumPy Arrays para habilitar vectorización matemática
        t = np.asarray(t)
        y = np.asarray(y)

        # Manejo de arreglos inválidos o demasiado cortos
        if len(t) < 2 or len(y) < 2:
            return "UNKNOWN", "Datos insuficientes para clasificar"

        dt = np.mean(np.diff(t))
        t_total = t[-1] - t[0]
        y_span = np.max(y) - np.min(y)
        
        # Evitar evaluación si la señal es plana (no hubo step/respuesta)
        if y_span == 0:
            return "STATIC", "Señal constante, sin dinámica observable"

        # Estimar constante de tiempo tau (63.2%)
        y_63 = np.min(y) + 0.632 * y_span
        idx_63 = np.where(y >= y_63)[0]
        
        # FIX 2: Calcular la constante tau relativa al inicio del vector (o step)
        if len(idx_63) > 0:
            t_63 = t[idx_63[0]]
            tau = t_63 - t[0] 
        else:
            tau = t_total * 0.5 # Estimación de respaldo si nunca llega al 63%

        # Evaluación basada en la constante de tiempo relativa (tau)
        if tau > 1.0:
            return "THERMAL_PROCESS", f"Proceso Térmico / Lento (FOPDT Dominante, Inercia Alta, tau={tau:.2f}s)"
        elif tau < 0.1:
            return "ELECTROMECHANICAL", f"Electromecánico / Rápido (FO Dominante, Retardo ~0, tau={tau:.2f}s)"
        else:
            return "FLUIDIC_NEUMATIC", f"Fluídico / Neumático (SO Dominante, Posible Turbulencia, tau={tau:.2f}s)"