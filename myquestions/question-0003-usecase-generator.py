import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import random

def generar_caso_de_uso_agrupar_municipios():
    """
    Genera un caso de prueba aleatorio para la función
    agrupar_municipios(df, cols_numericas, k_valores).

    La función evaluada debe:
    1. Seleccionar cols_numericas del DataFrame con pandas.
    2. Imputar NaN con SimpleImputer(strategy='median').
    3. Escalar con StandardScaler.
    4. Para cada k en k_valores, ajustar KMeans y calcular
       calinski_harabasz_score.
    5. Elegir el k con mayor score (menor k en empate).
    6. Devolver dict con 'best_k', 'best_score' y 'labels'.
    """

    # ── 1. Configuración aleatoria ──────────────────────────────────
    n_samples    = random.randint(60, 150)
    n_num_cols   = random.randint(3, 6)
    n_extra_cols = random.randint(1, 3)   # columnas binarias extra
    k_max        = random.randint(4, 6)
    k_valores    = list(range(2, k_max + 1))
    random_seed  = random.randint(0, 999)
    np.random.seed(random_seed)

    # ── 2. Generar datos ────────────────────────────────────────────
    cols_num = [f"indicador_{i}" for i in range(n_num_cols)]
    cols_bin = [f"binaria_{j}" for j in range(n_extra_cols)]

    data_num = np.random.randn(n_samples, n_num_cols)
    data_bin = np.random.randint(0, 2, size=(n_samples, n_extra_cols)).astype(float)

    df = pd.DataFrame(
        np.hstack([data_num, data_bin]),
        columns=cols_num + cols_bin
    )

    # Introducir NaNs solo en columnas numéricas (~8%)
    for col in cols_num:
        mask = np.random.choice([True, False], size=n_samples, p=[0.08, 0.92])
        df.loc[mask, col] = np.nan

    # ── 3. Input ────────────────────────────────────────────────────
    input_data = {
        'df': df.copy(),
        'cols_numericas': cols_num,
        'k_valores': k_valores
    }

    # ── 4. Ground truth ─────────────────────────────────────────────
    X_raw = df[cols_num].copy()

    imputer = SimpleImputer(strategy='median')
    X_imp   = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_imp)

    scores = {}
    for k in k_valores:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = km.fit_predict(X_sc)
        scores[k] = calinski_harabasz_score(X_sc, labels_k)

    best_k = min(
        (k for k in k_valores if scores[k] == max(scores.values())),
    )
    best_score = round(scores[best_k], 2)

    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    best_labels = km_best.fit_predict(X_sc)

    output_data = {
        'best_k': best_k,
        'best_score': best_score,
        'labels': best_labels
    }

    return input_data, output_data


# ── Ejemplo de uso ──────────────────────────────────────────────────
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_agrupar_municipios()

    print("=== INPUT ===")
    print(f"Columnas numéricas: {entrada['cols_numericas']}")
    print(f"k_valores a evaluar: {entrada['k_valores']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())

    print("\n=== OUTPUT ESPERADO ===")
    print(f"best_k     : {salida_esperada['best_k']}")
    print(f"best_score : {salida_esperada['best_score']}")
    print(f"labels     : {salida_esperada['labels'][:10]} ...")

