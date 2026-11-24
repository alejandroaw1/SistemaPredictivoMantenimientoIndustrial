from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
from groq import Groq

# --- 1. Configuración Inicial de la Página ---
st.set_page_config(
    page_title="Mantenimiento Predictivo Industrial",
    page_icon="🏭",
    layout="wide"
)

# --- 2. Configuración Segura de la API Key de Groq ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = "gsk_TU_API_KEY_LOCAL_AQUI"

client = Groq(api_key=api_key or "clave_no_configurada")

# --- 3. Cargar el Modelo Entrenado (Desacoplado del CSV) ---
@st.cache_resource
def cargar_modelo_ml():
    return joblib.load('modelo_mantenimiento.pkl')

modelo = cargar_modelo_ml()

# --- 4. Interfaz Gráfica en Streamlit ---
st.title("🏭 Sistema Predictivo de Mantenimiento Industrial")
st.write("Diagnóstico mediante **Machine Learning (Random Forest)** y generación automatizada de **Órdenes de Trabajo con Groq API**.")

# Controles laterales para los 4 sensores
st.sidebar.header("🎛️ Telemetría de Sensores")
temperatura = st.sidebar.slider("Temperatura (°C)", min_value=40.0, max_value=120.0, value=60.0, step=0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", min_value=0.0, max_value=15.0, value=1.5, step=0.1)
horas = st.sidebar.slider("Horas de Operación (hrs)", min_value=0, max_value=6000, value=300, step=10)
presion = st.sidebar.slider("Presión (PSI)", min_value=70.0, max_value=180.0, value=92.0, step=0.5)

# Tarjetas visuales de métricas en 4 columnas
st.subheader("📡 Lecturas Actuales en Línea")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

st.divider()

# --- 5. Procesamiento y Predicción ---
if st.button("🔍 Ejecutar Diagnóstico y Generar Orden", type="primary"):
    datos_entrada = pd.DataFrame([{
        'temperatura_c': temperatura,
        'vibracion_mms': vibracion,
        'horas_operacion': horas,
        'presion_psi': presion
    }])

    prediccion = int(modelo.predict(datos_entrada)[0])
    probabilidades = modelo.predict_proba(datos_entrada)[0]
    indice_clase = list(modelo.classes_).index(prediccion)
    confianza = probabilidades[indice_clase] * 100

    st.subheader("📊 Diagnóstico del Modelo Predictivo")
    if prediccion == 0:
        st.success(f"✅ **Estado 0: Operación Normal** (Probabilidad: {confianza:.1f}%)")
        estado_texto = "0 - OPERACIÓN NORMAL (Parámetros estables)"
    elif prediccion == 1:
        st.warning(f"⚠️ **Estado 1: Mantenimiento Preventivo** (Probabilidad: {confianza:.1f}%)")
        estado_texto = "1 - MANTENIMIENTO PREVENTIVO (Desgaste moderado; revisión programada)"
    else:
        st.error(f"🚨 **Estado 2: Fallo Crítico** (Probabilidad: {confianza:.1f}%)")
        estado_texto = "2 - FALLO CRÍTICO (Riesgo inminente de avería; paro inmediato)"

    # --- 6. Generación Automatizada con Groq API ---
    st.subheader("🤖 Reporte Técnico Automatizado (Groq API)")
    folio = f"OT-{datetime.now().strftime('%Y%m%d-%H%M')}"

    prompt_sistema = (
        "Eres un Jefe de Mantenimiento Industrial experto en confiabilidad de maquinaria, "
        "análisis de vibraciones, termografía y seguridad industrial. Responde siempre en español con precisión técnica."
    )

    if prediccion == 0:
        prompt_usuario = f"""Analiza las siguientes lecturas de telemetría industrial:
- Temperatura: {temperatura} °C
- Vibración: {vibracion} mm/s
- Horas de operación: {horas} hrs
- Presión: {presion} PSI
- Diagnóstico ML: {estado_texto} (Confianza: {confianza:.1f}%)

Emite un dictamen breve en Markdown de monitoreo preventivo confirmando que la maquinaria opera en rango seguro y señalando qué parámetro vigilar en la siguiente ronda."""
    else:
        urgencia = "URGENTE CON PROTOCOLO DE PARO INMEDIATO" if prediccion == 2 else "PROGRAMADA PREVENTIVA"
        prompt_usuario = f"""Se detectó una anomalía en línea de producción con las siguientes lecturas:
- Temperatura: {temperatura} °C
- Vibración: {vibracion} mm/s
- Horas de operación: {horas} hrs
- Presión: {presion} PSI
- Diagnóstico ML: {estado_texto} (Confianza: {confianza:.1f}%)
- Prioridad: {urgencia}

Genera una **Orden de Trabajo Técnico** estructurada en Markdown con exactamente estas 4 secciones:
1. **Folio:** `{folio}` (incluye nivel de prioridad)
2. **Diagnóstico:** Análisis técnico de la causa raíz probable según temperatura, vibración, horas y presión.
3. **3 Acciones Técnicas:** Lista numerada con 3 intervenciones concretas a ejecutar.
4. **Equipo de Seguridad (EPP):** Lista del EPP obligatorio para intervenir el equipo de forma segura."""

    with st.spinner("Consultando modelo de IA en Groq..."):
        try:
            modelos_activos = [
                m.id for m in client.models.list().data
                if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "playai", "embed"])
            ]

            if not modelos_activos:
                st.warning("⚠️ No se encontraron modelos de texto activos en Groq. Se muestra únicamente el diagnóstico ML.")
            else:
                modelo_groq = next(
                    (m for m in modelos_activos if "llama" in m.lower()),
                    modelos_activos[0]
                )

                completion = client.chat.completions.create(
                    model=modelo_groq,
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_usuario}
                    ],
                    temperature=0.5,
                    max_tokens=1024
                )

                st.caption(f"⚡ Generado con modelo activo: `{modelo_groq}`")
                st.markdown(completion.choices[0].message.content)

        except Exception as e:
            st.warning(f"⚠️ No fue posible generar el reporte con Groq API (verifica tu API Key o límite de peticiones). El diagnóstico de Machine Learning sigue operativo.\n\nDetalle: `{e}`")