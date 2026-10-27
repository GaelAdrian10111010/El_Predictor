import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from modelo import entrenar_modelo_knn, predecir_alumno

st.set_page_config(page_title="Predictor KNN", page_icon="🎓", layout="centered")

st.title("🎓 Predictor de Rendimiento Escolar (Modelo KNN)")
st.caption("Aprendizaje Supervisado por K-Vecinos Más Cercanos")

# Control interactivo de vecinos K en la barra lateral
st.sidebar.header("⚙️ Parámetros del Modelo")
k_vecinos = st.sidebar.slider("Número de Vecinos (K):", min_value=1, max_value=15, value=5, step=2)

# Re-entrenar modelo según el número de vecinos K seleccionado
exactitud, matriz, n_train, n_test = entrenar_modelo_knn(n_neighbors=k_vecinos)

st.sidebar.metric("Exactitud (Accuracy)", f"{exactitud * 100:.1f}%")
st.sidebar.write(f"**Datos de entrenamiento:** {n_train}")
st.sidebar.write(f"**Datos de prueba:** {n_test}")

# Visualización de la Matriz de Confusión
st.sidebar.write("### Matriz de Confusión")
fig, ax = plt.subplots(figsize=(3, 2.5))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Reprueba', 'Aprueba'],
            yticklabels=['Reprueba', 'Aprueba'], ax=ax)
plt.ylabel('Real')
plt.xlabel('Predicho')
st.sidebar.pyplot(fig)

# Formulario para datos del alumno
st.header("📋 Ingrese los datos del alumno")

col1, col2, col3 = st.columns(3)
with col1:
    horas = st.slider("Horas de estudio (diarias):", 0.0, 10.0, 5.0, 0.5)
with col2:
    asistencia = st.slider("Asistencia (%):", 0.0, 100.0, 80.0, 1.0)
with col3:
    tareas = st.slider("Tareas entregadas (%):", 0.0, 100.0, 70.0, 1.0)

if st.button("🔮 Predecir con KNN", use_container_width=True):
    resultado, confianza = predecir_alumno(horas, asistencia, tareas)

    st.write("---")
    if resultado == 1:
        st.success(f"### Result: 🎉 APRUEBA")
        st.write(f"Nivel de certeza por voto de vecinos: **{confianza:.1f}%**")
    else:
        st.error(f"### Result: ⚠️ REPRUEBA")
        st.write(f"Nivel de certeza por voto de vecinos: **{confianza:.1f}%**")