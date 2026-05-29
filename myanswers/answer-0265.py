import pandas as pd


def seleccionar_productos(df: pd.DataFrame) -> list:
    """
    Selecciona los índices de productos que cumplen al menos una
    de las siguientes condiciones:
      - price > 100
      - margin > 12 (solo si la columna 'margin' existe en el DataFrame)

    Parámetros:
        df: DataFrame con columna 'price', columna 'category' y,
            opcionalmente, columna 'margin'. El índice empieza en 1000.

    Retorna:
        list: Lista de índices (int) de los productos seleccionados,
              en el mismo orden en que aparecen en el DataFrame.
    """
    resultado = []

    for idx, row in df.iterrows():
        cumple = False

        if row["price"] > 100:
            cumple = True

        if "margin" in df.columns:
            if row["margin"] > 12:
                cumple = True

        if cumple:
            resultado.append(idx)

    return resultado
