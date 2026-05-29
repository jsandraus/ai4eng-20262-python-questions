import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def predecir_prestamos_libros(X: np.ndarray, y: np.ndarray):
    """
    Entrena un modelo de Regresión Lineal para predecir préstamos de libros
    y devuelve el MSE y las predicciones sobre el conjunto de prueba.

    Parámetros:
        X: Matriz de features de forma (n_muestras, n_features).
        y: Vector de etiquetas de forma (n_muestras,).

    Retorna:
        tuple: (mse: float, preds: np.ndarray)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return (mse, preds)
