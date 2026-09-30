"""
gui_streamlit.py - Interfaz Web Interactiva para MACA Engine MK3 usando Streamlit
Proyecto MACA MK3
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar los módulos del backend de MACA
from hardware_presets import PRESETS, HardwareConfig, ActuatorClassifier
from mac_c import ClassicalSysID, AnalyticalTuning, ClassicalSimulator
from mac_o import MultiObjectiveOptimizer
from mac_a import StateSpaceEngine
from mac_lm import IndustrialTranspiler, NaturalLanguageDiagnostics

st.set_page_config(page_title="MACA Engine MK3 - Workbench", layout="wide", page_icon="⚙️")

st.title("⚙️ MACA Engine MK3 - Control & Automation Workbench")
st.markdown("Plataforma agnóstica para identificación, optimización multiobjetivo por IA y transpilación industrial.")

# Sidebar - Configuración de Hardware
st.sidebar.header("🔌 Configuración de Hardware")
preset_choice = st.sidebar.selectbox("Seleccionar Hardware Preset:", list(PRESETS.keys()) + ["CUSTOM"])

if preset_choice == "CUSTOM":
    u_min = st.sidebar.number_input("u_min (V o %):", value=0.0)
    u_max = st.sidebar.number_input("u_max (V o %):", value=10.0)
    bits = st.sidebar.number_input("Resolución (bits):", value=12)
    ts = st.sidebar.number_input("Periodo de Muestreo Ts (s):", value=0.01, format="%.4f")
    deadband = st.sidebar.number_input("Banda Muerta (Deadband):", value=0.0)
    hw = HardwareConfig("Custom Hardware", u_min=u_min, u_max=u_max, resolution_bits=bits, ts=ts, deadband=deadband)
else:
    hw = PRESETS[preset_choice]
    st.sidebar.info(f"**Preset Activo:** {hw.name}\n- Rango: [{hw.u_min}, {hw.u_max}]\n- Ts: {hw.ts}s | Bits: {hw.resolution_bits}")

# Variables de Estado de Sesión
if 'telemetry' not in st.session_state:
    st.session_state.telemetry = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'gains_c' not in st.session_state:
    st.session_state.gains_c = None
if 'gains_o' not in st.session_state:
    st.session_state.gains_o = None

# Pestanas Principales
tab1, tab2, tab3, tab4 = st.tabs(["1. Telemetría & Clasificación", "2. Módulos MAC", "3. Dashboard & Gráficas", "4. Transpilación & Código"])

# ---------------------------------------------------------
# TAB 1: Telemetría & Clasificación
# ---------------------------------------------------------
with tab1:
    st.subheader("1. Ingesta de Datos y Clasificación AUTO")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Cargar Archivo Telemetría (.csv):", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.telemetry = (df.iloc[:,0].values, df.iloc[:,1].values)
            st.success("¡Archivo cargado con éxito!")
            
    with col2:
        if st.button("🧪 Generar Telemetría Sintética FOPDT de Prueba"):
            t = np.linspace(0, 30, 300)
            t_shift = np.maximum(0, t - 2.5)
            y = 2.5 * (1.0 - np.exp(-t_shift / 8.0)) + np.random.normal(0, 0.015, len(t))
            st.session_state.telemetry = (t, y)
            st.success("¡Telemetría sintética generada (K=2.5, tau=8.0s, theta=2.5s)!")

    if st.session_state.telemetry is not None:
        t, y = st.session_state.telemetry
        cat_code, cat_desc = ActuatorClassifier.classify(t, y)
        st.info(f"🏷️ **[Modo AUTO] Clasificación del Actuador:** {cat_desc}")
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, y, 'b.-', alpha=0.6, label='Telemetría Registrada')
        ax.set_title("Señal de Telemetría Cruda")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Salida y(t)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ---------------------------------------------------------
# TAB 2: Módulos MAC
# ---------------------------------------------------------
with tab2:
    st.subheader("2. Identificación y Optimización de Control")
    if st.session_state.telemetry is None:
        st.warning("⚠️ Primero carga o genera telemetría en la Pestaña 1.")
    else:
        t, y = st.session_state.telemetry
        col_c, col_o = st.columns(2)
        
        with col_c:
            st.markdown("### 🔹 MAC-C: Control Clásico (SysID & SIMC)")
            if st.button("Ejecutar MAC-C"):
                sysid = ClassicalSysID()
                st.session_state.model = sysid.fit_all(t, y)
                st.session_state.gains_c = AnalyticalTuning.skogestad_simc(
                    st.session_state.model['K'], 
                    st.session_state.model.get('tau', 1.0), 
                    st.session_state.model.get('theta', 0.0)
                )
                st.success("¡MAC-C ejecutado!")
                
            if st.session_state.model is not None:
                st.json({"Modelo Identificado": st.session_state.model, "Ganancias SIMC": st.session_state.gains_c})
                
        with col_o:
            st.markdown("### 🟣 MAC-O: Optimización Multiobjetivo (IA)")
            alpha = st.slider("Peso ITAE (α):", 0.1, 5.0, 1.0)
            beta = st.slider("Peso Energía u(t)² (β):", 0.001, 0.5, 0.01, format="%.3f")
            
            if st.button("Ejecutar MAC-O"):
                if st.session_state.model is None or st.session_state.gains_c is None:
                    st.error("Primero ejecuta MAC-C.")
                else:
                    opt = MultiObjectiveOptimizer(st.session_state.model, hw, alpha=alpha, beta=beta)
                    st.session_state.gains_o = opt.optimize(st.session_state.gains_c, t_sim=30.0)
                    st.success("¡MAC-O Optimización Bayesiana Completada!")
                    
            if st.session_state.gains_o is not None:
                st.json({"Ganancias Optimizadas": st.session_state.gains_o})

# ---------------------------------------------------------
# TAB 3: Dashboard & Gráficas
# ---------------------------------------------------------
with tab3:
    st.subheader("3. Dashboard Comparativo de Lazo Cerrado")
    if st.session_state.model is None or st.session_state.gains_c is None:
        st.info("Ejecuta MAC-C y MAC-O para visualizar las respuestas en lazo cerrado.")
    else:
        K = st.session_state.model['K']
        tau = st.session_state.model.get('tau', 1.0)
        theta = st.session_state.model.get('theta', 0.0)
        
        t_c, y_c, u_c = ClassicalSimulator.simulate_fopdt_loop(K, tau, theta, st.session_state.gains_c['Kp'], st.session_state.gains_c['Ki'], st.session_state.gains_c['Kd'], 30.0, hw)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
        ax1.plot(t_c, y_c, 'r--', label='MAC-C (SIMC)')
        
        if st.session_state.gains_o is not None:
            t_o, y_o, u_o = ClassicalSimulator.simulate_fopdt_loop(K, tau, theta, st.session_state.gains_o['Kp'], st.session_state.gains_o['Ki'], st.session_state.gains_o['Kd'], 30.0, hw)
            ax1.plot(t_o, y_o, 'b-', label='MAC-O (IA Multiobjetivo)')
            ax2.plot(t_o, u_o, 'b-', label='Esfuerzo u(t) MAC-O')
            
        ax1.axhline(1.0, color='g', linestyle=':', label='Setpoint')
        ax1.set_title("Respuesta Temporal y(t) vs Setpoint")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t_c, u_c, 'r--', label='Esfuerzo u(t) SIMC')
        ax2.axhline(hw.u_max, color='k', linestyle='--', label=f'Sat Max ({hw.u_max})')
        ax2.axhline(hw.u_min, color='k', linestyle='--', label=f'Sat Min ({hw.u_min})')
        ax2.set_title("Esfuerzo de Control u(t) y Saturación de Hardware")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# ---------------------------------------------------------
# TAB 4: Transpilación & Código
# ---------------------------------------------------------
with tab4:
    st.subheader("4. Transpilador Industrial & Diagnóstico")
    gains = st.session_state.gains_o if st.session_state.gains_o is not None else st.session_state.gains_c
    
    if gains is None:
        st.info("Ejecuta MAC-C o MAC-O para generar el código ejecutable.")
    else:
        col_cpp, col_st = st.columns(2)
        cpp_code = IndustrialTranspiler.generate_cpp(gains['Kp'], gains['Ki'], gains['Kd'], hw.ts, hw.u_min, hw.u_max)
        st_code = IndustrialTranspiler.generate_structured_text(gains['Kp'], gains['Ki'], gains['Kd'], hw.ts, hw.u_min, hw.u_max)
        
        with col_cpp:
            st.markdown("### 💻 Código C++ (ARM Cortex-M / AVR)")
            st.code(cpp_code, language="cpp")
            
        with col_st:
            st.markdown("### 🏭 Structured Text (PLC Arduino Opta / IEC 61131-3)")
            st.code(st_code, language="pascal")
            
        if st.session_state.model is not None:
            st.markdown("### 📝 Diagnóstico en Lenguaje Natural (MAC-LM)")
            report = NaturalLanguageDiagnostics.generate_report(st.session_state.model, gains, hw)
            st.text(report)
