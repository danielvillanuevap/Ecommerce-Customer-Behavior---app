"""Entrena el pipeline final de churn (fiel al notebook Proyecto_FINAL.ipynb) y lo
congela en disco para que la app Streamlit lo cargue sin reentrenar.

Uso:
    python train_model.py
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import churn_core as cc

IMPORTANCE_COLUMNS = cc.BASE_NUMERIC_COLUMNS + ["Genero", "Pais", "Trimestre_Registro", "Diversidad_Metodo_Pago"]
IMPORTANCE_SAMPLE_SIZE = 4000
IMPORTANCE_N_REPEATS = 8


def load_training_data():
    if cc.LOCAL_DATA_FILE.exists():
        raw = pd.read_csv(cc.LOCAL_DATA_FILE)
        print(f"Datos cargados desde {cc.LOCAL_DATA_FILE.name}: {raw.shape}")
    else:
        raw = cc.build_demo_dataset()
        print(f"No se encontro {cc.LOCAL_DATA_FILE.name}; datos demo sinteticos generados: {raw.shape}")
    return cc.ensure_base_columns(cc.normalize_columns(raw))


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    }
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return metrics, matrix, report, y_pred, y_prob


def compute_feature_importance(model, X_test, y_test):
    rng = np.random.RandomState(cc.SEED)
    sample_size = min(IMPORTANCE_SAMPLE_SIZE, len(X_test))
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)

    X_sample = X_test.iloc[sample_idx][IMPORTANCE_COLUMNS]
    y_sample = y_test.iloc[sample_idx]

    print(f"Calculando importancia por permutacion sobre {sample_size} filas, {IMPORTANCE_N_REPEATS} repeticiones...")
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="roc_auc",
        n_repeats=IMPORTANCE_N_REPEATS,
        random_state=cc.SEED,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "Variable": IMPORTANCE_COLUMNS,
            "Importancia_media": result.importances_mean,
            "Importancia_std": result.importances_std,
        }
    ).sort_values("Importancia_media", ascending=False).reset_index(drop=True)
    return importance_df


def main():
    start = time.time()
    cc.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = load_training_data()
    X = cc.model_input_frame(data)
    y = data[cc.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=cc.SEED
    )
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    model = cc.make_final_pipeline()
    print("Entrenando HistGradientBoostingClassifier con el pipeline fiel al notebook...")
    model.fit(X_train, y_train)

    metrics, matrix, report, _, y_prob = evaluate(model, X_test, y_test)
    print("\nMetricas en el set de prueba (held-out, 20%):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    selected = cc.selected_feature_table(model)
    print(f"\nVariables seleccionadas por SelectKBest ANOVA k={cc.SELECTED_FEATURES_K}: {len(selected)}")

    importance_df = compute_feature_importance(model, X_test, y_test)
    print("\nTop 10 variables por importancia de permutacion (caida de ROC AUC):")
    print(importance_df.head(10).to_string(index=False))

    joblib.dump(model, cc.MODEL_PATH)
    print(f"\nModelo guardado en: {cc.MODEL_PATH}")

    model_metadata = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "dataset_shape": list(data.shape),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": cc.TARGET,
        "hyperparameters": {
            "max_leaf_nodes": 63,
            "max_iter": 150,
            "learning_rate": 0.05,
            "l2_regularization": 0.1,
            "class_weight": "balanced",
            "random_state": cc.SEED,
            "selector": f"SelectKBest(f_classif, k={cc.SELECTED_FEATURES_K})",
        },
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "feature_importance": importance_df.round(6).to_dict(orient="records"),
        "y_test": [int(v) for v in y_test.tolist()],
        "y_prob_test": [round(float(v), 4) for v in y_prob.tolist()],
    }

    with open(cc.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata guardada en: {cc.MODEL_METADATA_PATH}")
    print(f"\nListo en {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
