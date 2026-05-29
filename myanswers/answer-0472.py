import numpy as np
from sklearn.feature_selection import VarianceThreshold


def seleccionar_sensores(X: np.ndarray, umbral: float) -> np.ndarray:
    """
    Filtra las columnas (sensores) de la matriz X cuya varianza sea
    mayor al umbral indicado.

    Parámetros:
        X:      Matriz de datos de forma (n_muestras, n_sensores).
        umbral: Valor mínimo de varianza para conservar una columna.

    Retorna:
        np.ndarray: Submatriz con solo las columnas que superan el umbral.
    """
    selector = VarianceThreshold(threshold=umbral)
    return selector.fit_transform(X)
