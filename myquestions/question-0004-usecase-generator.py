import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import random

def generar_caso_de_uso_comparar_clasificadores_estratificado():
    """
    Genera un caso de prueba aleatorio para la función
    comparar_clasificadores_estratificado(X, y, n_folds=5).

    La función evaluada debe:
    1. Instanciar LogisticRegression(max_iter=500, random_state=42)
       y DecisionTreeClassifier(max_depth=5, random_state=42).
    2. Crear StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).
    3. Implementar manualmente el bucle de CV (sin cross_val_score).
    4. Calcular f1_score(average='weighted', zero_division=0) en cada fold.
    5. Usar numpy para media y std de cada modelo.
    6. Devolver dict con métricas y 'modelo_mas_estable' (menor std).
    """

    # ── 1. Configuración aleatoria ──────────────────────────────────
    n_samples   = random.randint(80, 200)
    n_features  = random.randint(4, 10)
    n_classes   = random.randint(2, 4)
    n_folds     = random.choice([3, 5])
    random_seed = random.randint(0, 999)
    np.random.seed(random_seed)

    # ── 2. Generar datos ────────────────────────────────────────────
    X = np.random.randn(n_samples, n_features)
    # Clases balanceadas aproximadamente
    y = np.array([i % n_classes for i in range(n_samples)])
    np.random.shuffle(y)

    # ── 3. Input ────────────────────────────────────────────────────
    input_data = {
        'X': X.copy(),
        'y': y.copy(),
        'n_folds': n_folds
    }

    # ── 4. Ground truth ─────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    models = {
        'logistic': LogisticRegression(max_iter=500, random_state=42),
        'tree':     DecisionTreeClassifier(max_depth=5, random_state=42)
    }

    f1_scores = {'logistic': [], 'tree': []}

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
            f1_scores[name].append(f1)

    log_arr  = np.array(f1_scores['logistic'])
    tree_arr = np.array(f1_scores['tree'])

    log_std  = round(float(np.std(log_arr)),  4)
    tree_std = round(float(np.std(tree_arr)), 4)

    if log_std <= tree_std:
        mas_estable = "LogisticRegression"
    else:
        mas_estable = "DecisionTreeClassifier"

    output_data = {
        'logistic_f1_mean': round(float(np.mean(log_arr)),  4),
        'logistic_f1_std':  log_std,
        'tree_f1_mean':     round(float(np.mean(tree_arr)), 4),
        'tree_f1_std':      tree_std,
        'modelo_mas_estable': mas_estable
    }

    return input_data, output_data


# ── Ejemplo de uso ──────────────────────────────────────────────────
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_comparar_clasificadores_estratificado()

    print("=== INPUT ===")
    print(f"Shape de X : {entrada['X'].shape}")
    print(f"Clases únicas en y: {np.unique(entrada['y'])}")
    print(f"n_folds : {entrada['n_folds']}")

    print("\n=== OUTPUT ESPERADO ===")
    for k, v in salida_esperada.items():
        print(f"  {k}: {v}")

