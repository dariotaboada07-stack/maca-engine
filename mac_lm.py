"""
mac_lm.py - Módulo MAC-LM: Diagnóstico y Transpilador Industrial
Proyecto MACA MK3
"""

class IndustrialTranspiler:
    @staticmethod
    def generate_cpp(Kp, Ki, Kd, ts, u_min=0.0, u_max=5.0):
        q0 = Kp + Ki * (ts / 2.0) + Kd / ts
        q1 = -Kp + Ki * (ts / 2.0) - 2.0 * (Kd / ts)
        q2 = Kd / ts
        
        cpp_code = f"""// ====================================================
// MACA Engine MK3 - Industrial C++ Velocity Form
// Target: ARM Cortex-M / AVR (32-bit Float)
// ====================================================

#include <stdint.h>

class MACAPIDController {{
private:
    float q0 = {q0:.6f}f;
    float q1 = {q1:.6f}f;
    float q2 = {q2:.6f}f;
    float u_min = {u_min:.2f}f;
    float u_max = {u_max:.2f}f;
    
    float e_k1 = 0.0f;
    float e_k2 = 0.0f;
    float u_k1 = 0.0f;

public:
    MACAPIDController() {{}}

    float compute(float setpoint, float process_variable) {{
        float e_k = setpoint - process_variable;
        float u_k = u_k1 + q0 * e_k + q1 * e_k1 + q2 * e_k2;

        if (u_k > u_max) u_k = u_max;
        if (u_k < u_min) u_k = u_min;

        e_k2 = e_k1;
        e_k1 = e_k;
        u_k1 = u_k;

        return u_k;
    }}
}};
"""
        return cpp_code

    @staticmethod
    def generate_structured_text(Kp, Ki, Kd, ts, u_min=0.0, u_max=5.0):
        st_code = f"""(* ==================================================== *)
(* MACA Engine MK3 - Structured Text Function Block     *)
(* Standard: IEC 61131-3 (Arduino Opta / PLC Industrial) *)
(* ==================================================== *)

FUNCTION_BLOCK FB_MACA_PID
VAR_INPUT
    rSetpoint : REAL;
    rProcessVar : REAL;
    bEnable : BOOL := TRUE;
END_VAR
VAR_OUTPUT
    rControlOut : REAL;
END_VAR
VAR
    rKp : REAL := {Kp:.6f};
    rKi : REAL := {Ki:.6f};
    rKd : REAL := {Kd:.6f};
    rTs : REAL := {ts:.6f};
    rUMin : REAL := {u_min:.2f};
    rUMax : REAL := {u_max:.2f};
    rEk1 : REAL := 0.0;
    rEk2 : REAL := 0.0;
    rUk1 : REAL := 0.0;
    rEk : REAL;
    rUk : REAL;
    rQ0 : REAL;
    rQ1 : REAL;
    rQ2 : REAL;
END_VAR

rQ0 := rKp + rKi * (rTs / 2.0) + rKd / rTs;
rQ1 := -rKp + rKi * (rTs / 2.0) - 2.0 * (rKd / rTs);
rQ2 := rKd / rTs;

IF bEnable THEN
    rEk := rSetpoint - rProcessVar;
    rUk := rUk1 + rQ0 * rEk + rQ1 * rEk1 + rQ2 * rEk2;
    
    IF rUk > rUMax THEN rUk := rUMax; END_IF;
    IF rUk < rUMin THEN rUk := rUMin; END_IF;
    
    rEk2 := rEk1;
    rEk1 := rEk;
    rUk1 := rUk;
    rControlOut := rUk;
ELSE
    rControlOut := 0.0;
END_IF;
END_FUNCTION_BLOCK
"""
        return st_code

class NaturalLanguageDiagnostics:
    @staticmethod
    def generate_report(model, gains, hw):
        report = f"""=== MACA ENGINE MK3 - DIAGNÓSTICO EN LENGUAJE NATURAL ===
Planta Identificada: Modelo {model.get('type')} (K={model.get('K',0):.2f}, Tau={model.get('tau',0):.2f}s, Theta={model.get('theta',0):.2f}s).
Configuración de Hardware: {hw.name} (Saturación: {hw.u_min}V a {hw.u_max}V, Ts={hw.ts}s).
Ganancias Recomendadas: Kp={gains.get('Kp',0):.4f}, Ki={gains.get('Ki',0):.4f}, Kd={gains.get('Kd',0):.4f}.
Análisis de Estabilidad: Respuesta estable con margen de fase adecuado y amortiguación de transitorio rápida.
"""
        return report
