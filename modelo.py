import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def generar_datos_simulados(n_samples=500):
    """
    Genera un dataset de alumnos con un porcentaje de 'ruido'
    para simular un entorno real impredecible (sin reglas fijas absolutas).
    """
    rng = np.random.default_rng(42)

    horas = rng.uniform(0, 10, n_samples)
    asistencia = rng.uniform(0, 100, n_samples)
    tareas = rng.uniform(0, 100, n_samples)

    # Probabilidad base según las variables
    prob = (horas / 10) * 0.4 + (asistencia / 100) * 0.3 + (tareas / 100) * 0.3

    # Agregar un factor aleatorio (ruido) para que no sea una regla matemática perfecta
    ruido = rng.normal(0, 0.1, n_samples)
    prob_real = np.clip(prob + ruido, 0, 1)

    # 1 = APRUEBA, 0 = REPRUEBA
    y = (prob_real >= 0.55).astype(int)
    X = np.column_stack((horas, asistencia, tareas))

    return X, y


def entrenar_modelo_knn(n_neighbors=5):
    """
    Entrena un modelo KNN (K-Nearest Neighbors).
    """
    X, y = generar_datos_simulados()

    # 1. Separar datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Escalar características (CRUCIAL para KNN para medir distancias correctamente)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Crear y entrenar el modelo KNN
    modelo_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    modelo_knn.fit(X_train_scaled, y_train)

    # 4. Evaluación
    y_pred = modelo_knn.predict(X_test_scaled)
    exactitud = accuracy_score(y_test, y_pred)
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Guardar modelo y escalador
    joblib.dump(modelo_knn, 'modelo_knn.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return exactitud, matriz, len(X_train), len(X_test)


def predecir_alumno(horas, asistencia, tareas):
    """
    Carga el modelo y el escalador para predecir sobre un alumno nuevo.
    """
    modelo = joblib.load('modelo_knn.pkl')
    scaler = joblib.load('scaler.pkl')

    nuevo_alumno = np.array([[horas, asistencia, tareas]])
    nuevo_alumno_scaled = scaler.transform(nuevo_alumno)

    prediccion = int(modelo.predict(nuevo_alumno_scaled)[0])
    probabilidades = modelo.predict_proba(nuevo_alumno_scaled)[0]
    confianza = float(probabilidades[prediccion]) * 100

    return prediccion, confianza


if __name__ == '__main__':
    acc, cm, n_train, n_test = entrenar_modelo_knn()
    print("Modelo KNN entrenado exitosamente.")
    print(f"Exactitud: {acc:.2%}")
    print("Matriz de confusión:\n", cm)