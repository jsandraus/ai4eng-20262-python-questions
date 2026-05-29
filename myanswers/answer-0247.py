import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def predecir_conversion_clientes(df: pd.DataFrame) -> float:
    """
    Entrena un modelo de Regresión Logística para predecir la conversión
    de clientes y devuelve el accuracy sobre el conjunto de prueba.

    Parámetros:
        df: DataFrame con columnas edad, ingresos, visitas_web,
            tiempo_en_pagina y conversion.

    Retorna:
        float: accuracy del modelo sobre el conjunto de prueba.
    """
    X = df.drop(columns=["conversion"])
    y = df["conversion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LogisticRegression(max_iter=500)
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)
    return float(accuracy_score(y_test, pred))
