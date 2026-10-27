# 🎓 Predictor de Rendimiento Académico — Machine Learning (KNN)

Este proyecto es un sistema interactivo de **Aprendizaje Automático Supervisado** enfocado en predecir si un alumno **APRUEBA** o **REPRUEBA** una asignatura basándose en sus hábitos académicos.

El modelo utiliza el algoritmo de **K-Vecinos Más Cercanos (K-Nearest Neighbors - KNN)** y cuenta con una interfaz web desarrollada en **Streamlit**.

---

## 👥 Equipo de Trabajo (Equipo 1)
- **Integrante 1: Gael Adrian Hernandez Avila
- **Integrante 2: Uriel Alexander Cano Garcia
- **Integrante 3: Kevin Gerardo Resendez Cadena 

---

## 🚀 Características del Sistema

- **Backend (`modelo.py`):**
  - Generación de dataset sintético con margen de variación realista (ruido estadístico).
  - Normalización/Escalamiento de características con `StandardScaler`.
  - División de datos en conjunto de Entrenamiento (80%) y Prueba (20%) mediante `train_test_split`.
  - Entrenamiento del clasificador `KNeighborsClassifier`.
  - Evaluación del modelo calculando el **Accuracy** y la **Matriz de Confusión**.

- **Frontend (`app.py`):**
  - Interfaz web interactiva construida con `Streamlit`.
  - Ajuste dinámico en tiempo real del hiperparámetro $K$ (número de vecinos).
  - Visualización gráfica de la **Matriz de Confusión** mediante `seaborn` y `matplotlib`.
  - Sliders interactivos para ingresar datos del alumno (*Horas de estudio*, *Porcentaje de asistencia* y *Tareas entregadas*).
  - Diagnóstico visual inmediato (**APRUEBA** / **REPRUEBA**) con el nivel de certeza o porcentaje de votación de los vecinos.

---

## 🛠️ Requisitos e Instalación

### 1. Requisitos Previos
Tener instalado Python 3.8 o superior.

### 2. Instalación de Librerías
Abre la terminal en la raíz del proyecto e instala los paquetes necesarios ejecutando:

### 3. Ecender Localhost
Al terminar la instalacion de las librerias ejecuta el comando que esta debajo del primer comando "pip":

```bash
pip install numpy pandas scikit-learn streamlit matplotlib seaborn joblib

streamlit run app.py
