
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import random

def generar_caso_de_uso_analizar_error_por_cuartil():
    """
    Genera un caso de prueba aleatorio para la función
    analizar_error_por_cuartil(X, y, n_cuartiles=4).

    La función entrenada debe:
    1. Dividir en train/test 80/20 con random_state=42.
    2. Escalar X con StandardScaler (fit solo en train).
    3. Entrenar Ridge(alpha=1.0) sobre train.
    4. Calcular el error absoluto por muestra en test.
    5. Segmentar y_test en n_cuartiles grupos por valor real.
    6. Calcular el MAE promedio de cada cuartil.
    7. Devolver un DataFrame con columnas [cuartil, mae_promedio].
    """

    # ── 1. Configuración aleatoria ──────────────────────────────────
    n_samples   = random.randint(60, 120)   # suficientes para train/test
    n_features  = random.randint(3, 7)
    n_cuartiles = random.choice([3, 4, 5])
    random_seed = random.randint(0, 999)
    np.random.seed(random_seed)

    # ── 2. Generar datos ────────────────────────────────────────────
    X = np.random.randn(n_samples, n_features)
    # y con algo de señal (combinación lineal + ruido)
    coef = np.random.randn(n_features)
    y = X @ coef + np.random.randn(n_samples) * 0.5

    # ── 3. Input ────────────────────────────────────────────────────
    input_data = {
        'X': X.copy(),
        'y': y.copy(),
        'n_cuartiles': n_cuartiles
    }

    # ── 4. Ground truth ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    abs_errors = np.abs(y_test - y_pred)

    # Límites de cuartiles sobre y_test
    percentiles = np.linspace(0, 100, n_cuartiles + 1)
    limits = np.percentile(y_test, percentiles)

    rows = []
    for i in range(n_cuartiles):
        lo, hi = limits[i], limits[i + 1]
        if i == 0:
            mask = (y_test >= lo) & (y_test <= hi)
        else:
            mask = (y_test > lo) & (y_test <= hi)
        mae_q = float(np.mean(abs_errors[mask])) if mask.sum() > 0 else float('nan')
        rows.append({'cuartil': i + 1, 'mae_promedio': mae_q})

    output_data = pd.DataFrame(rows).sort_values('cuartil').reset_index(drop=True)

    return input_data, output_data


# ── Ejemplo de uso ──────────────────────────────────────────────────
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_analizar_error_por_cuartil()

    print("=== INPUT ===")
    print(f"Shape de X: {entrada['X'].shape}")
    print(f"Shape de y: {entrada['y'].shape}")
    print(f"n_cuartiles: {entrada['n_cuartiles']}")

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
