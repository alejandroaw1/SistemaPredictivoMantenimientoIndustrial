import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("📂 1. Cargando conjunto de datos desde 'sensores_planta.csv'...")

try:
    df = pd.read_csv('sensores_planta.csv', sep=',')

    # Validar si toda la fila quedó agrupada en una sola columna
    if len(df.columns) == 1 and ',' in df.columns[0]:
        col_unica = df.columns[0]
        nuevas_cols = [c.strip().lower() for c in col_unica.split(',')]
        df = df[col_unica].str.split(',', expand=True)
        df.columns = nuevas_cols
        df = df.apply(pd.to_numeric, errors='coerce')
    else:
        df.columns = df.columns.str.strip().str.lower()

    print(f"✅ Archivo cargado correctamente. Registros totales: {len(df)}")
    print(f"📋 Columnas detectadas: {df.columns.tolist()}")
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo 'sensores_planta.csv'. Revisa la ruta.")
    exit()

# 2. Limpieza básica: eliminar filas incompletas
df = df.dropna()

# 3. Verificar columnas requeridas
columnas_requeridas = ['temperatura_c', 'vibracion_mms', 'horas_operacion', 'presion_psi', 'estado']
if not all(col in df.columns for col in columnas_requeridas):
    print(f"❌ Error: Tu CSV debe tener las columnas: {columnas_requeridas}")
    exit()

# 4. Definición de Variables de Entrada (X) y Objetivo (y)
X = df[['temperatura_c', 'vibracion_mms', 'horas_operacion', 'presion_psi']]
y = df['estado']

# 5. Dividir datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Entrenar el Modelo de Clasificación (Random Forest)
print("🧠 2. Entrenando el algoritmo RandomForestClassifier...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 7. Evaluar precisión del modelo
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"🎯 Precisión del modelo (Accuracy): {precision * 100:.2f}%")

if precision < 0.80:
    print("⚠️ Advertencia: La precisión es menor al 80%. Revisa la calidad de los datos.")

# 8. Exportar el modelo entrenado a un archivo .pkl
joblib.dump(modelo, 'modelo_mantenimiento.pkl')
print("✅ 3. ¡Éxito! Modelo exportado como 'modelo_mantenimiento.pkl'.")