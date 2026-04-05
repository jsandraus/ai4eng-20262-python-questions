import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random

def generar_caso_de_uso_regresion_con_interaccion():
    """
    Genera un caso de prueba aleatorio para la función
    regresion_con_interaccion(df, target_col, col_ordinal, categorias_orden).

    La función evaluada debe:
    1. Codificar col_ordinal con OrdinalEncoder(categories=[categorias_orden]).
    2. Reemplazar la columna original con los valores codificados.
    3. Crear columna 'interaccion' = col_ordinal_codificada * primera_col_numerica.
    4. Separar X e y con target_col.
    5. Dividir 75/25 con random_state=7 y entrenar LinearRegression.
    6. Devolver dict con 'r2' (4 dec.) y 'coef_interaccion' (float).
    """

    # ── 1. Configuración aleatoria ──────────────────────────────────
    n_samples  = random.randint(40, 100)
    n_num_cols = random.randint(2, 4)   # columnas numéricas adicionales
    random_seed = random.randint(0, 999)
    np.random.seed(random_seed)

    categorias_orden = ["deficiente", "regular", "buena", "excelente"]
    col_ordinal = "calidad_construccion"
    target_col  = "precio"

    # ── 2. Generar datos ────────────────────────────────────────────
    num_col_names = [f"num_col_{i}" for i in range(n_num_cols)]
    df = pd.DataFrame(
        np.random.randn(n_samples, n_num_cols),
        columns=num_col_names
    )
    df[col_ordinal] = np.random.choice(categorias_orden, size=n_samples)

    # Target con algo de señal
    ord_map = {"deficiente": 0, "regular": 1, "buena": 2, "excelente": 3}
    ord_vals = df[col_ordinal].map(ord_map).values
    df[target_col] = (
        ord_vals * 50_000
        + df[num_col_names[0]].values * 20_000
        + np.random.randn(n_samples) * 5_000
        + 100_000
    )

    # ── 3. Input ────────────────────────────────────────────────────
    input_data = {
        'df': df.copy(),
        'target_col': target_col,
        'col_ordinal': col_ordinal,
        'categorias_orden': categorias_orden
    }

    # ── 4. Ground truth ─────────────────────────────────────────────
    df_work = df.copy()

    encoder = OrdinalEncoder(categories=[categorias_orden])
    df_work[col_ordinal] = encoder.fit_transform(
        df_work[[col_ordinal]]
    ).astype(float)

    # Primera columna numérica distinta de col_ordinal y target_col
    first_num = [c for c in df_work.columns
                 if c != col_ordinal and c != target_col][0]
    df_work['interaccion'] = df_work[col_ordinal] * df_work[first_num]

    X = df_work.drop(columns=[target_col])
    y = df_work[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2 = round(model.score(X_test, y_test), 4)
    interaccion_idx = list(X.columns).index('interaccion')
    coef_interaccion = float(model.coef_[interaccion_idx])

    output_data = {'r2': r2, 'coef_interaccion': coef_interaccion}

    return input_data, output_data


# ── Ejemplo de uso ──────────────────────────────────────────────────
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_regresion_con_interaccion()

    print("=== INPUT ===")
    print(f"Columna ordinal: {entrada['col_ordinal']}")
    print(f"Categorías orden: {entrada['categorias_orden']}")
    print(f"Target col: {entrada['target_col']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)

