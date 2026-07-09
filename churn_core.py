import json
import platform
from pathlib import Path
from importlib import metadata

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_FILE = PROJECT_ROOT / "ecommerce_customer_churn_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "churn_hist_gradient_boosting.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "model" / "model_metadata.json"
TARGET = "Fuga_Cliente"

RENAME_COLUMNS = {
    "Age": "Edad",
    "Gender": "Genero",
    "Country": "Pais",
    "City": "Ciudad",
    "Membership_Years": "Anos_Membresia",
    "Login_Frequency": "Frecuencia_Login",
    "Session_Duration_Avg": "Duracion_Sesion_Promedio",
    "Pages_Per_Session": "Paginas_Por_Sesion",
    "Cart_Abandonment_Rate": "Tasa_Abandono_Carrito",
    "Wishlist_Items": "Articulos_Lista_Deseos",
    "Total_Purchases": "Total_Compras",
    "Average_Order_Value": "Valor_Promedio_Pedido",
    "Days_Since_Last_Purchase": "Dias_Desde_Ultima_Compra",
    "Discount_Usage_Rate": "Tasa_Uso_Descuento",
    "Returns_Rate": "Tasa_Devoluciones",
    "Email_Open_Rate": "Tasa_Apertura_Email",
    "Customer_Service_Calls": "Llamadas_Servicio_Cliente",
    "Product_Reviews_Written": "Resenas_Productos_Escritas",
    "Social_Media_Engagement_Score": "Puntuacion_Engagement_Redes_Sociales",
    "Mobile_App_Usage": "Uso_App_Movil",
    "Payment_Method_Diversity": "Diversidad_Metodo_Pago",
    "Lifetime_Value": "Valor_Vida_Util",
    "Credit_Balance": "Balance_Credito",
    "Churned": "Fuga_Cliente",
    "Signup_Quarter": "Trimestre_Registro",
}

BASE_NUMERIC_COLUMNS = [
    "Edad",
    "Anos_Membresia",
    "Frecuencia_Login",
    "Duracion_Sesion_Promedio",
    "Paginas_Por_Sesion",
    "Tasa_Abandono_Carrito",
    "Articulos_Lista_Deseos",
    "Total_Compras",
    "Valor_Promedio_Pedido",
    "Dias_Desde_Ultima_Compra",
    "Tasa_Uso_Descuento",
    "Tasa_Devoluciones",
    "Tasa_Apertura_Email",
    "Llamadas_Servicio_Cliente",
    "Resenas_Productos_Escritas",
    "Puntuacion_Engagement_Redes_Sociales",
    "Uso_App_Movil",
    "Valor_Vida_Util",
    "Balance_Credito",
]

NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + ["Uso_App_Movil_Binario", "Tiene_Balance_Credito"]
CATEGORICAL_COLUMNS = [
    "Genero",
    "Pais",
    "Trimestre_Registro",
    "Diversidad_Metodo_Pago",
    "GrupoEdad",
    "Segmento_Compras",
    "Segmento_Recencia",
    "Segmento_Valor_Vida_Util",
]
PREDICTION_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
SELECTED_FEATURES_K = 35

PLOT_TEMPLATE = "plotly_white"
RISK_COLORS = {"Bajo": "#15803d", "Medio": "#b45309", "Alto": "#b91c1c"}
CLUSTER_COLORS = ["#2563eb", "#f97316", "#0f766e", "#b91c1c"]

NOTEBOOK_LOG_COLUMNS = [
    "Valor_Vida_Util",
    "Total_Compras",
    "Dias_Desde_Ultima_Compra",
    "Valor_Promedio_Pedido",
    "Balance_Credito",
]

NOTEBOOK_CAPPING_COLUMNS = [
    "Edad",
    "Anos_Membresia",
    "Frecuencia_Login",
    "Duracion_Sesion_Promedio",
    "Paginas_Por_Sesion",
    "Tasa_Abandono_Carrito",
    "Articulos_Lista_Deseos",
    "Tasa_Uso_Descuento",
    "Tasa_Devoluciones",
    "Tasa_Apertura_Email",
    "Llamadas_Servicio_Cliente",
    "Resenas_Productos_Escritas",
    "Puntuacion_Engagement_Redes_Sociales",
    "Uso_App_Movil",
]

NOTEBOOK_BINARY_COLUMNS = ["Uso_App_Movil_Binario", "Tiene_Balance_Credito"]
NOTEBOOK_NOMINAL_COLUMNS = ["Genero", "Pais", "Trimestre_Registro", "GrupoEdad", "Diversidad_Metodo_Pago"]
NOTEBOOK_ORDINAL_COLUMNS = ["Segmento_Recencia", "Segmento_Valor_Vida_Util", "Segmento_Compras"]

VARIABLE_LABELS = {
    "Edad": "Edad",
    "Genero": "Genero",
    "Pais": "Pais",
    "Ciudad": "Ciudad",
    "Anos_Membresia": "Anos de Membresia",
    "Frecuencia_Login": "Frecuencia de Login",
    "Duracion_Sesion_Promedio": "Duracion de Sesion",
    "Paginas_Por_Sesion": "Paginas por Sesion",
    "Tasa_Abandono_Carrito": "Tasa de Abandono de Carrito",
    "Articulos_Lista_Deseos": "Articulos en Lista de Deseos",
    "Total_Compras": "Total de Compras",
    "Valor_Promedio_Pedido": "Valor Promedio del Pedido",
    "Dias_Desde_Ultima_Compra": "Dias desde Ultima Compra",
    "Tasa_Uso_Descuento": "Tasa de Uso de Descuento",
    "Tasa_Devoluciones": "Tasa de Devoluciones",
    "Tasa_Apertura_Email": "Tasa de Apertura de Email",
    "Llamadas_Servicio_Cliente": "Llamadas a Servicio al Cliente",
    "Resenas_Productos_Escritas": "Resenas Escritas",
    "Puntuacion_Engagement_Redes_Sociales": "Engagement en Redes Sociales",
    "Uso_App_Movil": "Uso de App Movil",
    "Diversidad_Metodo_Pago": "Diversidad de Metodo de Pago",
    "Valor_Vida_Util": "Valor de Vida del Cliente",
    "Balance_Credito": "Balance de Credito",
    "Trimestre_Registro": "Trimestre de Registro",
    "Fuga_Cliente": "Fuga de Cliente",
    "GrupoEdad": "Grupo de Edad",
    "Segmento_Compras": "Segmento de Compras",
    "Segmento_Recencia": "Segmento de Recencia",
    "Segmento_Valor_Vida_Util": "Segmento de Valor de Vida",
    "Uso_App_Movil_Binario": "Usa App Movil",
    "Tiene_Balance_Credito": "Tiene Balance de Credito",
}


def pretty_label(name):
    if name is None:
        return ""
    clean = str(name).split("__", 1)[-1]
    if clean in VARIABLE_LABELS:
        return VARIABLE_LABELS[clean]
    for base, label in VARIABLE_LABELS.items():
        if clean.startswith(f"{base}_"):
            suffix = clean[len(base) + 1 :].replace("_", " ")
            return f"{label}: {suffix}"
    return clean.replace("_", " ")


def pretty_series(series):
    return series.map(pretty_label)


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 17px;
        }

        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
            font-size: 1.03rem;
            line-height: 1.58;
        }

        [data-testid="stMarkdownContainer"] h4 {
            font-size: 1.28rem;
        }

        [data-testid="stMarkdownContainer"] h5 {
            font-size: 1.12rem;
        }

        [data-testid="stCaptionContainer"] {
            font-size: 0.95rem !important;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 28rem),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.08), transparent 26rem),
                #f6f8fb;
        }

        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2.4rem;
            max-width: 1380px;
        }

        section[data-testid="stSidebar"] {
            background: #101827;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #e5edf7;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #cbd5e1;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #172033;
            letter-spacing: 0;
        }

        .app-hero {
            border: 1px solid #e3e8ef;
            background: linear-gradient(135deg, #ffffff 0%, #f6f8fb 56%, #eef7f5 100%);
            border-radius: 10px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
            animation: fadeUp 420ms ease-out;
        }

        .eyebrow {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            color: #172033;
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }

        .hero-copy {
            color: #5f6b7a;
            max-width: 900px;
            font-size: 1.1rem;
            line-height: 1.55;
        }

        .panel {
            border: 1px solid #e3e8ef;
            background: rgba(255, 255, 255, 0.92);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.045);
        }

        .risk-card {
            border: 1px solid #e3e8ef;
            border-left: 6px solid #2563eb;
            background: #ffffff;
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin: 0.7rem 0 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
        }

        .risk-card h3 {
            margin: 0 0 0.25rem;
            font-size: 1.18rem;
        }

        .risk-card p {
            color: #5f6b7a;
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.5;
        }

        .signal-list {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            margin: 0.6rem 0 1rem;
        }

        .signal-row {
            display: flex;
            gap: 0.7rem;
            align-items: flex-start;
            background: #ffffff;
            border: 1px solid #e3e8ef;
            border-radius: 9px;
            padding: 0.75rem 0.95rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }

        .signal-dot {
            width: 9px;
            height: 9px;
            min-width: 9px;
            border-radius: 50%;
            background: #b45309;
            margin-top: 0.42rem;
        }

        .signal-row strong {
            display: block;
            color: #172033;
            font-size: 1.08rem;
        }

        .signal-meta {
            display: block;
            color: #5f6b7a;
            font-size: 0.9rem;
            margin: 0.15rem 0 0.32rem;
        }

        .signal-row p {
            margin: 0;
            color: #5f6b7a;
            font-size: 1rem;
            line-height: 1.45;
        }

        .whatif-card {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            background: #ffffff;
            border: 1px solid #e3e8ef;
            border-radius: 10px;
            padding: 0.9rem 1.05rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        }

        .whatif-card strong {
            display: block;
            color: #172033;
            font-size: 1.08rem;
        }

        .whatif-bar-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .whatif-bar-label {
            width: 74px;
            flex-shrink: 0;
            color: #5f6b7a;
            font-size: 0.88rem;
        }

        .whatif-bar-track {
            flex: 1;
            height: 10px;
            border-radius: 999px;
            background: #eef1f5;
            overflow: hidden;
        }

        .whatif-bar-fill {
            height: 100%;
            border-radius: 999px;
        }

        .whatif-bar-fill.actual {
            background: #94a3b8;
        }

        .whatif-bar-fill.simulated {
            background: #2563eb;
        }

        .whatif-bar-value {
            width: 48px;
            flex-shrink: 0;
            text-align: right;
            color: #172033;
            font-size: 0.9rem;
            font-weight: 700;
        }

        .whatif-badge {
            align-self: flex-start;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            white-space: nowrap;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }

        .feature-card {
            border: 1px solid #e3e8ef;
            background: #ffffff;
            border-radius: 10px;
            padding: 1rem;
            min-height: 138px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        }

        .feature-card strong {
            display: block;
            color: #172033;
            font-size: 1.08rem;
            margin-bottom: 0.35rem;
        }

        .feature-card span {
            color: #5f6b7a;
            line-height: 1.5;
            font-size: 1rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e3e8ef;
            background: #ffffff;
            border-radius: 10px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricValue"] {
            color: #172033;
            font-size: 1.7rem;
            font-weight: 800;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 1rem;
        }

        .cluster-card {
            border: 1px solid #e3e8ef;
            background: #ffffff;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
        }

        .cluster-card h3 {
            margin: 0 0 0.15rem;
            font-size: 1.25rem;
        }

        .cluster-card .cluster-badge {
            display: inline-block;
            font-size: 0.82rem;
            font-weight: 700;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            margin-bottom: 0.7rem;
        }

        .cluster-metric-row {
            display: flex;
            justify-content: space-between;
            padding: 0.4rem 0;
            border-bottom: 1px solid #eef1f5;
            font-size: 1rem;
        }

        .cluster-metric-row:last-child {
            border-bottom: none;
        }

        .cluster-metric-row span:first-child {
            color: #5f6b7a;
        }

        .cluster-metric-row span:last-child {
            color: #172033;
            font-weight: 700;
        }

        .stDownloadButton button, .stButton button {
            border-radius: 7px;
            border: 1px solid #1f2937;
            background: #1f2937;
            color: #ffffff;
            transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
        }

        .stDownloadButton button:hover, .stButton button:hover {
            border-color: #111827;
            background: #111827;
            color: #ffffff;
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 900px) {
            .feature-grid {
                grid-template-columns: 1fr;
            }
            .hero-title {
                font-size: 1.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle="", eyebrow="Churn Intelligence"):
    copy_html = f'<div class="hero-copy">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_cards(items):
    cards = []
    for title, body in items:
        cards.append(f"<div class='feature-card'><strong>{title}</strong><span>{body}</span></div>")
    st.markdown(f"<div class='feature-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def sidebar_source():
    with st.sidebar:
        st.title("Churn Intelligence")
        st.caption("Sistema analitico para priorizar retencion de clientes.")
        st.divider()
        show_environment_validation()


def load_context(require_model=True):
    sidebar_source()
    data = load_data()

    if TARGET not in data.columns:
        st.error("El dataset de entrenamiento debe incluir la columna Fuga_Cliente o Churned.")
        st.stop()

    data = ensure_base_columns(data)

    if not require_model:
        return data, None, None, None, None

    model, metrics, matrix, report = load_trained_model()
    return data, model, metrics, matrix, report


@st.cache_data(show_spinner="Cargando dataset...")
def load_data():
    if LOCAL_DATA_FILE.exists():
        data = pd.read_csv(LOCAL_DATA_FILE)
    else:
        data = build_demo_dataset()
    return normalize_columns(data)


def build_demo_dataset(n_rows=1200):
    rng = np.random.default_rng(SEED)
    countries = np.array(["Peru", "Chile", "Colombia", "Mexico", "Argentina", "Brasil"])
    genders = np.array(["Female", "Male", "Other"])
    quarters = np.array(["Q1", "Q2", "Q3", "Q4"])
    payment = np.array(["Low", "Medium", "High"])

    data = pd.DataFrame(
        {
            "Edad": rng.integers(18, 69, n_rows),
            "Genero": rng.choice(genders, n_rows, p=[0.48, 0.48, 0.04]),
            "Pais": rng.choice(countries, n_rows),
            "Ciudad": rng.choice(["Lima", "Santiago", "Bogota", "CDMX", "Buenos Aires"], n_rows),
            "Anos_Membresia": rng.integers(0, 10, n_rows),
            "Frecuencia_Login": rng.poisson(9, n_rows).clip(0, 35),
            "Duracion_Sesion_Promedio": rng.normal(18, 7, n_rows).clip(1, 55),
            "Paginas_Por_Sesion": rng.normal(5.5, 2.0, n_rows).clip(1, 14),
            "Tasa_Abandono_Carrito": rng.beta(2.2, 2.4, n_rows) * 100,
            "Articulos_Lista_Deseos": rng.poisson(5, n_rows).clip(0, 30),
            "Total_Compras": rng.poisson(14, n_rows).clip(0, 80),
            "Valor_Promedio_Pedido": rng.gamma(5, 18, n_rows).clip(10, 420),
            "Dias_Desde_Ultima_Compra": rng.gamma(2.5, 18, n_rows).clip(0, 180),
            "Tasa_Uso_Descuento": rng.beta(2, 5, n_rows) * 100,
            "Tasa_Devoluciones": rng.beta(1.4, 8, n_rows) * 100,
            "Tasa_Apertura_Email": rng.beta(3, 3, n_rows) * 100,
            "Llamadas_Servicio_Cliente": rng.poisson(1.8, n_rows).clip(0, 15),
            "Resenas_Productos_Escritas": rng.poisson(2.2, n_rows).clip(0, 20),
            "Puntuacion_Engagement_Redes_Sociales": rng.beta(2.5, 2.5, n_rows) * 100,
            "Uso_App_Movil": rng.poisson(10, n_rows).clip(0, 45),
            "Diversidad_Metodo_Pago": rng.choice(payment, n_rows, p=[0.28, 0.47, 0.25]),
            "Valor_Vida_Util": rng.gamma(4, 620, n_rows).clip(120, 12000),
            "Balance_Credito": rng.gamma(2, 85, n_rows).clip(0, 1500),
            "Trimestre_Registro": rng.choice(quarters, n_rows),
        }
    )

    score = (
        0.030 * data["Tasa_Abandono_Carrito"]
        + 0.018 * data["Dias_Desde_Ultima_Compra"]
        + 0.110 * data["Llamadas_Servicio_Cliente"]
        + 0.018 * data["Tasa_Devoluciones"]
        - 0.026 * data["Tasa_Apertura_Email"]
        - 0.055 * data["Frecuencia_Login"]
        - 0.016 * data["Uso_App_Movil"]
        - 0.0020 * data["Total_Compras"]
        - 0.00016 * data["Valor_Vida_Util"]
        + rng.normal(0, 0.85, n_rows)
    )
    probability = 1 / (1 + np.exp(-(score - 2.3)))
    data[TARGET] = rng.binomial(1, probability)
    return data


def normalize_columns(data):
    data = data.copy().rename(columns=RENAME_COLUMNS)

    if TARGET in data.columns:
        data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce").fillna(0).astype(int)

    for col in BASE_NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in ["Genero", "Pais", "Ciudad", "Trimestre_Registro", "Diversidad_Metodo_Pago"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    return data


class BusinessRuleCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rate_cols = [
            "Tasa_Abandono_Carrito",
            "Tasa_Uso_Descuento",
            "Tasa_Devoluciones",
            "Tasa_Apertura_Email",
            "Puntuacion_Engagement_Redes_Sociales",
        ]
        self.non_negative_cols = [
            "Valor_Vida_Util",
            "Total_Compras",
            "Dias_Desde_Ultima_Compra",
            "Valor_Promedio_Pedido",
            "Balance_Credito",
            "Anos_Membresia",
            "Frecuencia_Login",
            "Duracion_Sesion_Promedio",
            "Paginas_Por_Sesion",
            "Articulos_Lista_Deseos",
            "Llamadas_Servicio_Cliente",
            "Resenas_Productos_Escritas",
            "Uso_App_Movil",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "Edad" in X.columns:
            X.loc[(X["Edad"] < 18) | (X["Edad"] > 100), "Edad"] = np.nan

        for col in self.rate_cols:
            if col in X.columns:
                X.loc[(X[col] < 0) | (X[col] > 100), col] = np.nan

        for col in self.non_negative_cols:
            if col in X.columns:
                X.loc[X[col] < 0, col] = np.nan

        return X


class SkewnessBasedImputer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = X.columns
        self.skewness_ = X.skew(numeric_only=True)
        self.fill_values_ = {}

        for col in self.columns_:
            skew = self.skewness_[col]
            fill_value = X[col].median() if abs(skew) > self.threshold else X[col].mean()
            self.fill_values_[col] = 0 if pd.isna(fill_value) else fill_value

        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        for col in self.columns_:
            X[col] = X[col].fillna(self.fill_values_[col])
        return X.values


class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1).values


class Log1pTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = np.clip(X, a_min=0, a_max=None)
        return np.log1p(X)


class FixedFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_business_features(X)


def ensure_base_columns(data):
    data = data.copy()
    for col in BASE_NUMERIC_COLUMNS:
        if col not in data.columns:
            data[col] = np.nan

    defaults = {
        "Genero": "Desconocido",
        "Pais": "Desconocido",
        "Trimestre_Registro": "Q1",
        "Diversidad_Metodo_Pago": "Desconocido",
    }
    for col, default in defaults.items():
        if col not in data.columns:
            data[col] = default

    return data


def add_business_features(data):
    data = ensure_base_columns(data)

    data["Uso_App_Movil_Binario"] = (pd.to_numeric(data["Uso_App_Movil"], errors="coerce") > 0).astype(int)
    data["Tiene_Balance_Credito"] = (pd.to_numeric(data["Balance_Credito"], errors="coerce") > 0).astype(int)

    data["GrupoEdad"] = pd.cut(
        pd.to_numeric(data["Edad"], errors="coerce"),
        bins=[17, 25, 40, 60, 100],
        labels=["Joven", "Adulto_Joven", "Adulto", "Adulto_Mayor"],
    ).astype("object")

    data["Segmento_Compras"] = pd.cut(
        pd.to_numeric(data["Total_Compras"], errors="coerce"),
        bins=[-np.inf, 5, 15, 30, np.inf],
        labels=["Comprador_Ocasional", "Comprador_Regular", "Comprador_Frecuente", "Comprador_VIP"],
    ).astype("object")

    data["Segmento_Recencia"] = pd.cut(
        pd.to_numeric(data["Dias_Desde_Ultima_Compra"], errors="coerce"),
        bins=[-np.inf, 30, 60, 90, np.inf],
        labels=["Reciente", "Medio", "Riesgo", "Inactivo"],
    ).astype("object")

    data["Segmento_Valor_Vida_Util"] = pd.cut(
        pd.to_numeric(data["Valor_Vida_Util"], errors="coerce"),
        bins=[-np.inf, 1000, 2500, 5000, np.inf],
        labels=["Bajo", "Medio", "Alto", "Muy_Alto"],
    ).astype("object")

    return data


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def selected_classifier():
    return HistGradientBoostingClassifier(
        max_leaf_nodes=63,
        max_iter=150,
        learning_rate=0.05,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=SEED,
    )


def make_notebook_column_transformer():
    numeric_log_pipeline = Pipeline(
        steps=[
            ("imputer", SkewnessBasedImputer(threshold=0.5)),
            ("log1p", Log1pTransformer()),
            ("scaler", RobustScaler()),
        ]
    )

    numeric_capping_pipeline = Pipeline(
        steps=[
            ("imputer", SkewnessBasedImputer(threshold=0.5)),
            ("capper", IQRCapper(factor=1.5)),
            ("scaler", RobustScaler()),
        ]
    )

    binary_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[
                        ["Reciente", "Medio", "Riesgo", "Inactivo"],
                        ["Bajo", "Medio", "Alto", "Muy_Alto"],
                        [
                            "Comprador_Ocasional",
                            "Comprador_Regular",
                            "Comprador_Frecuente",
                            "Comprador_VIP",
                        ],
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num_log", numeric_log_pipeline, NOTEBOOK_LOG_COLUMNS),
            ("num_cap", numeric_capping_pipeline, NOTEBOOK_CAPPING_COLUMNS),
            ("binarias", binary_pipeline, NOTEBOOK_BINARY_COLUMNS),
            ("cat_nom", nominal_pipeline, NOTEBOOK_NOMINAL_COLUMNS),
            ("cat_ord", ordinal_pipeline, NOTEBOOK_ORDINAL_COLUMNS),
        ],
        remainder="drop",
    )


def make_notebook_cluster_preprocessor():
    return Pipeline(
        steps=[
            ("rules", BusinessRuleCleaner()),
            ("features", FixedFeatureEngineer()),
            ("preprocessor", make_notebook_column_transformer()),
        ]
    )


def notebook_model_pipeline(classifier):
    """Pipeline fiel al notebook: limpieza -> feature engineering -> preprocesamiento
    log/capping/ordinal -> seleccion ANOVA k=35 -> clasificador. Toma datos base (ver
    model_input_frame) y aplica su propia limpieza y feature engineering internamente."""
    return Pipeline(
        steps=[
            ("rules", BusinessRuleCleaner()),
            ("features", FixedFeatureEngineer()),
            ("preprocessor", make_notebook_column_transformer()),
            ("selector", SelectKBest(score_func=f_classif, k=SELECTED_FEATURES_K)),
            ("model", classifier),
        ]
    )


def make_final_pipeline():
    """Pipeline del modelo final: HistGradientBoosting optimizado sobre el
    preprocesamiento fiel al notebook. Es el pipeline que entrena train_model.py
    y que la app sirve congelada (ver load_trained_model)."""
    return notebook_model_pipeline(selected_classifier())


def model_input_frame(data):
    """Datos base (sin ingenieria de variables) listos para entrar al pipeline final,
    que aplica su propia limpieza y feature engineering internamente."""
    return ensure_base_columns(normalize_columns(data)).drop(columns=[TARGET], errors="ignore")


def notebook_feature_names(fitted_column_transformer):
    names = []
    for name, transformer, columns in fitted_column_transformer.transformers_:
        if name == "remainder":
            continue
        columns = list(columns)
        if name == "cat_nom":
            ohe = transformer.named_steps["onehot"]
            ohe_names = ohe.get_feature_names_out(columns)
            names.extend([f"{name}__{col}" for col in ohe_names])
        else:
            names.extend([f"{name}__{col}" for col in columns])
    return np.array(names)


@st.cache_resource(show_spinner="Cargando modelo entrenado...")
def load_trained_model():
    """Carga el pipeline final ya entrenado (train_model.py) en vez de reentrenar
    en cada sesion. La app siempre sirve este mismo modelo congelado."""
    if not MODEL_PATH.exists():
        st.error(
            f"No se encontro el modelo entrenado en `{MODEL_PATH.relative_to(PROJECT_ROOT)}`. "
            "Corre `python train_model.py` para generarlo antes de usar la app."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)

    model_metadata = {}
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)

    metrics = model_metadata.get("metrics", {})
    matrix = np.array(model_metadata.get("confusion_matrix", [[0, 0], [0, 0]]))
    report = model_metadata.get("classification_report", {})
    return model, metrics, matrix, report


def load_model_metadata():
    if not MODEL_METADATA_PATH.exists():
        return {}
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def feature_origin(feature_name):
    clean_name = feature_name.split("__", 1)[-1]
    if clean_name in PREDICTION_COLUMNS:
        return clean_name
    for col in CATEGORICAL_COLUMNS:
        if clean_name.startswith(f"{col}_"):
            return col
    return clean_name


def selected_feature_table(model):
    if "selector" not in model.named_steps:
        return pd.DataFrame()

    feature_names = notebook_feature_names(model.named_steps["preprocessor"])
    selector = model.named_steps["selector"]
    scores = selector.scores_
    selected = selector.get_support()
    table = pd.DataFrame(
        {
            "Feature transformada": feature_names[selected],
            "Variable base": [feature_origin(name) for name in feature_names[selected]],
            "Score ANOVA": scores[selected],
        }
    )
    return table.sort_values("Score ANOVA", ascending=False).reset_index(drop=True)


def metric_bundle(y_test, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    }


def comparison_dataset(data, max_rows=20000):
    if len(data) <= max_rows:
        return data
    sample, _ = train_test_split(
        data,
        train_size=max_rows,
        stratify=data[TARGET],
        random_state=SEED,
    )
    return sample.reset_index(drop=True)


@st.cache_data(show_spinner="Comparando modelos candidatos...")
def train_model_comparison(data):
    benchmark = comparison_dataset(data)
    X = model_input_frame(benchmark)
    y = benchmark[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    candidates = [
        (
            "Baseline mayoritario",
            "Baseline",
            Pipeline(steps=[("model", DummyClassifier(strategy="most_frequent"))]),
            "Referencia minima: predice la clase mas frecuente.",
        ),
        (
            "Regresion Logistica",
            "Machine Learning",
            notebook_model_pipeline(LogisticRegression(max_iter=600, class_weight="balanced", random_state=SEED)),
            "Modelo lineal interpretable y rapido.",
        ),
        (
            "Random Forest",
            "Machine Learning",
            notebook_model_pipeline(
                RandomForestClassifier(
                    n_estimators=160,
                    max_depth=12,
                    min_samples_leaf=12,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                )
            ),
            "Ensamble robusto, pero mas pesado para explicar.",
        ),
        (
            "Gradient Boosting",
            "Machine Learning",
            notebook_model_pipeline(GradientBoostingClassifier(n_estimators=130, learning_rate=0.06, max_depth=3, random_state=SEED)),
            "Boosting clasico cercano al enfoque ganador.",
        ),
        (
            "MLP",
            "Deep Learning",
            notebook_model_pipeline(
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=90,
                    early_stopping=True,
                    random_state=SEED,
                )
            ),
            "Red neuronal multicapa; flexible, pero menos transparente y mas sensible a tuning.",
        ),
        (
            "HistGradientBoosting",
            "Machine Learning seleccionado",
            notebook_model_pipeline(selected_classifier()),
            "Mejor balance entre desempeno, estabilidad, costo y explicabilidad operativa.",
        ),
    ]

    rows = []
    for name, model_type, candidate, comment in candidates:
        candidate.fit(X_train, y_train)
        y_pred = candidate.predict(X_test)
        if hasattr(candidate, "predict_proba"):
            y_prob = candidate.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
        metrics = metric_bundle(y_test, y_pred, y_prob)
        rows.append(
            {
                "Modelo": name,
                "Tipo": model_type,
                "Muestra usada": len(benchmark),
                "Seleccionado": "Si" if name == "HistGradientBoosting" else "No",
                "Comentario": comment,
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)


@st.cache_resource(show_spinner="Calculando segmentacion de clientes...")
def train_clustering(data, n_clusters=2):
    X_cluster_raw = ensure_base_columns(data.drop(columns=[TARGET], errors="ignore"))
    X_prep = make_notebook_cluster_preprocessor().fit_transform(X_cluster_raw)
    labels = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=SEED).fit_predict(X_prep)
    pca = PCA(n_components=3, random_state=SEED)
    coords = pca.fit_transform(X_prep)

    segmented = data.copy()
    segmented["Cluster_KMeans"] = labels.astype(str)
    segmented["PC1"] = coords[:, 0]
    segmented["PC2"] = coords[:, 1]
    segmented["PC3"] = coords[:, 2]
    return segmented, pca.explained_variance_ratio_


def predict_probabilities(model, data):
    return model.predict_proba(model_input_frame(data))[:, 1]


def risk_level(probability):
    if probability >= 0.70:
        return "Alto"
    if probability >= 0.40:
        return "Medio"
    return "Bajo"


def risk_action(level):
    actions = {
        "Alto": "Priorizar contacto, incentivo personalizado y revision de friccion en soporte o carrito.",
        "Medio": "Activar campana preventiva, monitorear recencia y reforzar beneficios de retencion.",
        "Bajo": "Mantener fidelizacion regular y observar cambios bruscos de comportamiento.",
    }
    return actions[level]


def prediction_output(model, data, threshold=0.50):
    output = normalize_columns(data)
    output["Probabilidad_Fuga"] = predict_probabilities(model, output)
    output["Riesgo"] = output["Probabilidad_Fuga"].apply(risk_level)
    output["Prediccion"] = np.where(output["Probabilidad_Fuga"] >= threshold, "Fuga", "Activo")
    output["Accion_Recomendada"] = output["Riesgo"].apply(risk_action)
    return output


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "no instalado"


def show_environment_validation():
    st.caption("Entorno")
    packages = ["streamlit", "pandas", "numpy", "scikit-learn", "plotly"]
    checks = pd.DataFrame(
        {
            "Elemento": ["Python"] + packages,
            "Version": [platform.python_version()] + [package_version(pkg) for pkg in packages],
        }
    )
    st.dataframe(checks, hide_index=True, use_container_width=True)


def safe_mode(series):
    return series.dropna().mode().iloc[0] if not series.dropna().mode().empty else ""


def sorted_options(series):
    return sorted(series.dropna().unique().tolist(), key=lambda value: str(value))


def numeric_bounds(data, col):
    values = pd.to_numeric(data[col], errors="coerce").dropna()
    if values.empty:
        return 0.0, 1.0, 0.0

    low = float(values.quantile(0.01))
    high = float(values.quantile(0.99))
    default = float(values.median())
    if low == high:
        high = low + 1.0
    return low, high, default


def quantile_value(data, col, q):
    values = pd.to_numeric(data[col], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.quantile(q))


def option_index(options, value):
    if value in options:
        return options.index(value)
    return 0


def default_customer_values(data):
    customer = {}
    for col in BASE_NUMERIC_COLUMNS:
        _, _, default = numeric_bounds(data, col)
        customer[col] = default

    for col in ["Genero", "Pais", "Trimestre_Registro", "Diversidad_Metodo_Pago"]:
        customer[col] = safe_mode(data[col]) if col in data.columns else ""

    customer["Ciudad"] = safe_mode(data["Ciudad"]) if "Ciudad" in data.columns else "No_usado"
    return customer


def demo_customer_preset(data, scenario):
    customer = default_customer_values(data)
    if scenario == "Cliente estable":
        customer.update(
            {
                "Frecuencia_Login": quantile_value(data, "Frecuencia_Login", 0.80),
                "Dias_Desde_Ultima_Compra": quantile_value(data, "Dias_Desde_Ultima_Compra", 0.20),
                "Tasa_Abandono_Carrito": quantile_value(data, "Tasa_Abandono_Carrito", 0.20),
                "Tasa_Apertura_Email": quantile_value(data, "Tasa_Apertura_Email", 0.80),
                "Llamadas_Servicio_Cliente": quantile_value(data, "Llamadas_Servicio_Cliente", 0.20),
                "Uso_App_Movil": quantile_value(data, "Uso_App_Movil", 0.80),
                "Total_Compras": quantile_value(data, "Total_Compras", 0.75),
                "Valor_Vida_Util": quantile_value(data, "Valor_Vida_Util", 0.75),
            }
        )
    elif scenario == "Cliente en riesgo medio":
        customer.update(
            {
                "Frecuencia_Login": quantile_value(data, "Frecuencia_Login", 0.45),
                "Dias_Desde_Ultima_Compra": quantile_value(data, "Dias_Desde_Ultima_Compra", 0.65),
                "Tasa_Abandono_Carrito": quantile_value(data, "Tasa_Abandono_Carrito", 0.65),
                "Tasa_Apertura_Email": quantile_value(data, "Tasa_Apertura_Email", 0.40),
                "Llamadas_Servicio_Cliente": quantile_value(data, "Llamadas_Servicio_Cliente", 0.65),
                "Uso_App_Movil": quantile_value(data, "Uso_App_Movil", 0.40),
            }
        )
    elif scenario == "Cliente en riesgo alto":
        customer.update(
            {
                "Frecuencia_Login": quantile_value(data, "Frecuencia_Login", 0.10),
                "Dias_Desde_Ultima_Compra": quantile_value(data, "Dias_Desde_Ultima_Compra", 0.90),
                "Tasa_Abandono_Carrito": quantile_value(data, "Tasa_Abandono_Carrito", 0.90),
                "Tasa_Apertura_Email": quantile_value(data, "Tasa_Apertura_Email", 0.10),
                "Llamadas_Servicio_Cliente": quantile_value(data, "Llamadas_Servicio_Cliente", 0.90),
                "Uso_App_Movil": quantile_value(data, "Uso_App_Movil", 0.10),
                "Total_Compras": quantile_value(data, "Total_Compras", 0.25),
                "Valor_Vida_Util": quantile_value(data, "Valor_Vida_Util", 0.25),
                "Tasa_Devoluciones": quantile_value(data, "Tasa_Devoluciones", 0.85),
            }
        )
    return customer


def build_customer_form(data, preset=None, widget_namespace="manual"):
    customer = default_customer_values(data)
    preset = preset or default_customer_values(data)
    customer.update(preset)

    st.markdown("#### Datos generales")
    left, right = st.columns(2)
    with left:
        country_options = sorted_options(data["Pais"])
        quarter_options = sorted_options(data["Trimestre_Registro"])
        payment_options = sorted_options(data["Diversidad_Metodo_Pago"])
        customer["Pais"] = st.selectbox(
            pretty_label("Pais"),
            country_options,
            index=option_index(country_options, preset.get("Pais")),
            key=f"{widget_namespace}_pais",
        )
        customer["Trimestre_Registro"] = st.selectbox(
            pretty_label("Trimestre_Registro"),
            quarter_options,
            index=option_index(quarter_options, preset.get("Trimestre_Registro")),
            key=f"{widget_namespace}_trimestre",
        )
        customer["Diversidad_Metodo_Pago"] = st.selectbox(
            pretty_label("Diversidad_Metodo_Pago"),
            payment_options,
            index=option_index(payment_options, preset.get("Diversidad_Metodo_Pago")),
            key=f"{widget_namespace}_pago",
        )

    with right:
        for col in ["Edad", "Anos_Membresia", "Valor_Vida_Util", "Balance_Credito"]:
            low, high, default = numeric_bounds(data, col)
            value = min(max(float(preset.get(col, default)), low), high)
            customer[col] = st.number_input(pretty_label(col), min_value=low, max_value=high, value=value, key=f"{widget_namespace}_{col}")

    st.markdown("#### Variables clave para estimar churn")
    cols = st.columns(3)
    input_columns = [
        "Frecuencia_Login",
        "Dias_Desde_Ultima_Compra",
        "Tasa_Abandono_Carrito",
        "Tasa_Apertura_Email",
        "Llamadas_Servicio_Cliente",
        "Uso_App_Movil",
        "Total_Compras",
        "Valor_Promedio_Pedido",
        "Tasa_Devoluciones",
    ]

    for idx, col in enumerate(input_columns):
        low, high, default = numeric_bounds(data, col)
        value = min(max(float(preset.get(col, default)), low), high)
        with cols[idx % 3]:
            customer[col] = st.number_input(pretty_label(col), min_value=low, max_value=high, value=value, key=f"{widget_namespace}_{col}")

    customer["Ciudad"] = safe_mode(data["Ciudad"]) if "Ciudad" in data.columns else "No_usado"
    return pd.DataFrame([customer])


def risk_signal_table(customer, reference):
    row = customer.iloc[0]
    signals = [
        (
            "Abandono de carrito alto",
            "Tasa_Abandono_Carrito",
            "> p75",
            row.get("Tasa_Abandono_Carrito", 0) >= quantile_value(reference, "Tasa_Abandono_Carrito", 0.75),
            "Friccion directa antes de la compra.",
        ),
        (
            "Recencia elevada",
            "Dias_Desde_Ultima_Compra",
            "> p75",
            row.get("Dias_Desde_Ultima_Compra", 0) >= quantile_value(reference, "Dias_Desde_Ultima_Compra", 0.75),
            "El cliente lleva demasiado tiempo sin comprar.",
        ),
        (
            "Baja apertura de email",
            "Tasa_Apertura_Email",
            "< p25",
            row.get("Tasa_Apertura_Email", 0) <= quantile_value(reference, "Tasa_Apertura_Email", 0.25),
            "Menor respuesta a campanas de retencion.",
        ),
        (
            "Muchas llamadas a soporte",
            "Llamadas_Servicio_Cliente",
            "> p75",
            row.get("Llamadas_Servicio_Cliente", 0) >= quantile_value(reference, "Llamadas_Servicio_Cliente", 0.75),
            "Puede indicar friccion, reclamos o mala experiencia.",
        ),
        (
            "Bajo uso de app",
            "Uso_App_Movil",
            "< p25",
            row.get("Uso_App_Movil", 0) <= quantile_value(reference, "Uso_App_Movil", 0.25),
            "Menor habito digital y menor recurrencia.",
        ),
        (
            "Valor de vida bajo",
            "Valor_Vida_Util",
            "< p25",
            row.get("Valor_Vida_Util", 0) <= quantile_value(reference, "Valor_Vida_Util", 0.25),
            "Menor historial economico acumulado.",
        ),
    ]
    rows = []
    for signal, variable, criterio, active, impact in signals:
        if active:
            rows.append(
                {
                    "Senal detectada": signal,
                    "Variable": variable,
                    "Valor cliente": row.get(variable, np.nan),
                    "Criterio": criterio,
                    "Lectura": impact,
                }
            )
    if not rows:
        rows.append(
            {
                "Senal detectada": "Sin senales criticas fuertes",
                "Variable": "-",
                "Valor cliente": "-",
                "Criterio": "Dentro de rangos centrales",
                "Lectura": "El riesgo depende de combinaciones mas suaves del comportamiento.",
            }
        )
    return pd.DataFrame(rows)


def what_if_scenarios(model, customer, reference, threshold):
    base = customer.copy()
    scenarios = [
        (
            "Reducir friccion en carrito",
            {"Tasa_Abandono_Carrito": quantile_value(reference, "Tasa_Abandono_Carrito", 0.35)},
        ),
        (
            "Reactivar compra reciente",
            {"Dias_Desde_Ultima_Compra": quantile_value(reference, "Dias_Desde_Ultima_Compra", 0.35)},
        ),
        (
            "Mejorar engagement digital",
            {
                "Tasa_Apertura_Email": quantile_value(reference, "Tasa_Apertura_Email", 0.70),
                "Uso_App_Movil": quantile_value(reference, "Uso_App_Movil", 0.70),
            },
        ),
        (
            "Resolver friccion de soporte",
            {"Llamadas_Servicio_Cliente": quantile_value(reference, "Llamadas_Servicio_Cliente", 0.25)},
        ),
    ]

    current_probability = float(prediction_output(model, base, threshold)["Probabilidad_Fuga"].iloc[0])
    rows = []
    for name, changes in scenarios:
        simulated = base.copy()
        for col, value in changes.items():
            simulated[col] = value
        probability = float(prediction_output(model, simulated, threshold)["Probabilidad_Fuga"].iloc[0])
        rows.append(
            {
                "Escenario": name,
                "Probabilidad actual": current_probability,
                "Probabilidad simulada": probability,
                "Reduccion estimada": current_probability - probability,
                "Lectura": "Priorizar" if probability < current_probability else "Monitorear",
            }
        )
    return pd.DataFrame(rows).sort_values("Reduccion estimada", ascending=False)


def render_signal_cards(signals_df):
    if len(signals_df) == 1 and signals_df.iloc[0]["Senal detectada"] == "Sin senales criticas fuertes":
        st.markdown(
            """
            <div class="risk-card" style="border-left-color:#15803d">
                <h3>Sin senales criticas fuertes</h3>
                <p>El riesgo de este cliente depende de combinaciones mas suaves del comportamiento, no de una senal aislada.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows_html = []
    for _, row in signals_df.iterrows():
        value = row["Valor cliente"]
        value_text = f"{value:,.1f}" if isinstance(value, (int, float, np.floating)) else str(value)
        rows_html.append(
            '<div class="signal-row"><div class="signal-dot"></div><div>'
            f"<strong>{row['Senal detectada']}</strong>"
            f'<span class="signal-meta">{pretty_label(row["Variable"])} = {value_text} ({row["Criterio"]})</span>'
            f"<p>{row['Lectura']}</p>"
            "</div></div>"
        )
    st.markdown(f'<div class="signal-list">{"".join(rows_html)}</div>', unsafe_allow_html=True)


def render_whatif_cards(simulation_df):
    cards_html = []
    for _, row in simulation_df.iterrows():
        is_priority = row["Lectura"] == "Priorizar"
        badge_color = "#15803d" if is_priority else "#5f6b7a"
        badge_bg = "#dcfce7" if is_priority else "#eef1f5"
        current_pct = row["Probabilidad actual"] * 100
        sim_pct = row["Probabilidad simulada"] * 100
        delta = row["Reduccion estimada"] * 100
        delta_text = f"Baja {delta:.1f} pp" if delta >= 0 else f"Sube {abs(delta):.1f} pp"
        cards_html.append(
            '<div class="whatif-card">'
            f"<strong>{row['Escenario']}</strong>"
            '<div class="whatif-bar-row">'
            '<span class="whatif-bar-label">Actual</span>'
            f'<div class="whatif-bar-track"><div class="whatif-bar-fill actual" style="width:{min(current_pct, 100):.1f}%"></div></div>'
            f'<span class="whatif-bar-value">{current_pct:.0f}%</span>'
            "</div>"
            '<div class="whatif-bar-row">'
            '<span class="whatif-bar-label">Simulado</span>'
            f'<div class="whatif-bar-track"><div class="whatif-bar-fill simulated" style="width:{min(sim_pct, 100):.1f}%"></div></div>'
            f'<span class="whatif-bar-value">{sim_pct:.0f}%</span>'
            "</div>"
            f'<span class="whatif-badge" style="color:{badge_color};background:{badge_bg}">{delta_text} de probabilidad de fuga &middot; {row["Lectura"]}</span>'
            "</div>"
        )
    st.markdown(f'<div class="signal-list">{"".join(cards_html)}</div>', unsafe_allow_html=True)


def variable_groups_table():
    return pd.DataFrame(
        [
            {
                "Categoria": "Demografia",
                "Variables": "Edad, Genero, Pais, Ciudad, Anos de Membresia",
                "Lectura": "Perfil general del cliente.",
            },
            {
                "Categoria": "Engagement digital",
                "Variables": "Frecuencia de Login, Duracion de Sesion, Paginas, Email, App, Redes",
                "Lectura": "Mide habito, interes y cercania con la plataforma.",
            },
            {
                "Categoria": "Compra",
                "Variables": "Total de Compras, Ticket Promedio, Recencia, Descuentos, Devoluciones",
                "Lectura": "Describe valor transaccional y senales de inactividad.",
            },
            {
                "Categoria": "Servicio",
                "Variables": "Llamadas a Servicio al Cliente, Resenas Escritas",
                "Lectura": "Captura friccion, reclamos o compromiso con la marca.",
            },
            {
                "Categoria": "Financiero / target",
                "Variables": "Valor de Vida del Cliente, Balance de Credito, Fuga de Cliente",
                "Lectura": "Permite priorizar retencion segun valor y riesgo.",
            },
        ]
    )


def official_model_results():
    return pd.DataFrame(
        [
            ["Hist Gradient Boosting", 0.9179, 0.9134, 0.7907, 0.8957, 0.9252, 0.8801],
            ["Gradient Boosting", 0.9180, 0.9289, 0.7756, 0.8948, 0.9245, 0.8757],
            ["Random Forest", 0.9147, 0.9009, 0.7923, 0.8923, 0.9207, 0.8784],
            ["XGBoost", 0.9092, 0.8376, 0.8511, 0.8901, 0.9246, 0.8920],
            ["LightGBM", 0.9094, 0.8421, 0.8451, 0.8899, 0.9253, 0.8903],
            ["Decision Tree", 0.8713, 0.7639, 0.8031, 0.8458, 0.8931, 0.8511],
            ["SVC", 0.8586, 0.7233, 0.8270, 0.8346, 0.9020, 0.8492],
            ["Logistic Regression", 0.7181, 0.5086, 0.7244, 0.6903, 0.7899, 0.7200],
            ["Dummy Baseline", 0.7110, 0.0000, 0.0000, 0.4155, 0.5000, 0.5000],
        ],
        columns=["Modelo", "Accuracy", "Precision", "Recall", "F1_macro", "ROC_AUC", "Balanced Acc."],
    )


def render_context_and_data(data):
    hero(
        "Contexto del negocio y entendimiento de datos",
        "",
        "1. Descripcion del problema",
    )

    st.markdown("#### 1.1 Contexto del proyecto")
    st.markdown(
        """
        El proyecto aborda churn en e-commerce: cuando un cliente abandona, la empresa pierde ingresos presentes
        y tambien el potencial futuro representado por su Lifetime Value. La pregunta central es si podemos detectar
        anticipadamente clientes en riesgo usando comportamiento de compra, engagement digital, soporte y perfil.
        """
    )

    st.markdown("#### 1.2 Que predice el modelo?")
    st.markdown(
        """
        El modelo predice la **probabilidad de que un cliente abandone la plataforma** (`Fuga_Cliente`),
        considerando:

        - **Comportamiento digital:** frecuencia de login, duracion de sesion, uso de app movil, apertura de email.
        - **Comportamiento de compra:** total de compras, valor promedio del pedido, dias desde la ultima compra.
        - **Senales de friccion:** abandono de carrito, devoluciones, llamadas a servicio al cliente.
        - **Perfil y valor:** antiguedad como miembro, valor de vida del cliente (LTV), pais, metodo de pago.

        La prediccion permite estimar el **nivel de riesgo** de cada cliente y la **accion de retencion** mas adecuada.
        """
    )

    st.markdown("#### 1.3 A quien ayuda?")
    feature_cards(
        [
            ("Marketing y CRM", "Priorizar campanas de retencion y personalizar ofertas segun el riesgo de fuga."),
            ("Atencion al cliente", "Detectar friccion (soporte, devoluciones) antes de que el cliente se fugue."),
            ("Finanzas y growth", "Proteger el Lifetime Value acumulado y anticipar el impacto en ingresos futuros."),
            ("Gerencia comercial", "Asignar presupuesto de retencion donde el riesgo y el valor del cliente son mas altos."),
        ]
    )

    left, right = st.columns([0.92, 1.08])
    with left:
        target_counts = data[TARGET].value_counts().rename(index={0: "Activo", 1: "Fuga"}).reset_index()
        target_counts.columns = ["Estado", "Clientes"]
        fig = px.bar(
            target_counts,
            x="Estado",
            y="Clientes",
            color="Estado",
            color_discrete_map={"Activo": "#0f766e", "Fuga": "#b91c1c"},
            title="Distribucion de la variable objetivo",
            template=PLOT_TEMPLATE,
            text="Clientes",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Categorias de variables")
        st.dataframe(variable_groups_table(), hide_index=True, use_container_width=True)

    st.markdown("#### Exploracion interactiva")
    numeric_options = [col for col in BASE_NUMERIC_COLUMNS if col in data.columns]
    selected_col = st.selectbox(
        "Variable numerica",
        numeric_options,
        index=numeric_options.index("Dias_Desde_Ultima_Compra"),
        format_func=pretty_label,
    )
    eda_data = data.copy()
    eda_data["Estado"] = eda_data[TARGET].map({0: "Activo", 1: "Fuga"})
    fig_hist = px.histogram(
        eda_data,
        x=selected_col,
        color="Estado",
        marginal="box",
        nbins=45,
        title=f"Distribucion de {pretty_label(selected_col)} por estado de fuga",
        labels={selected_col: pretty_label(selected_col)},
        template=PLOT_TEMPLATE,
        color_discrete_map={"Activo": "#0f766e", "Fuga": "#b91c1c"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)


def render_visualization_communication(data, model, metrics, matrix, report):
    hero(
        "Visualizacion y comunicacion",
        "Cierre del proyecto: resultados finales y conclusiones, con base en el informe del proyecto.",
        "Resultados y conclusiones",
    )

    st.markdown("#### Resultados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.3f}")
    c2.metric("Recall", f"{metrics.get('Recall', 0):.3f}")
    c3.metric("F1 (fuga)", f"{metrics.get('F1', 0):.3f}")
    c4.metric("ROC AUC", f"{metrics.get('ROC AUC', 0):.3f}")

    if matrix is not None and matrix.size == 4:
        tn, fp, fn, tp = matrix.ravel()
        detection_rate = tp / (tp + fn) * 100 if (tp + fn) else 0
        st.markdown(
            f"El modelo detecta **{int(tp):,} clientes** que realmente estan en riesgo de fuga y deja pasar "
            f"**{int(fn):,}** (falsos negativos). El equipo de retencion puede priorizar cerca del "
            f"**{detection_rate:.0f}%** de los clientes que realmente se fugarian."
        )

    st.markdown("##### Hallazgos principales")
    feature_cards(
        [
            ("Dataset y calidad", "50,000 clientes y 25 variables. Se detectaron nulos en 14 variables y variables con fuerte asimetria, tratadas antes del modelado."),
            ("Desbalance moderado", "71.1% clientes activos vs 28.9% en fuga (ratio 2.46:1). No requirio tecnicas de remuestreo costosas."),
            ("Senales mas fuertes", "Llamadas a Servicio, Abandono de Carrito y Dias desde Ultima Compra son las variables con mayor poder discriminante."),
            ("Comportamiento digital", "Baja apertura de email y bajo uso de app movil se repiten en los clientes que terminan fugandose."),
        ]
    )

    st.markdown("#### Conclusiones")
    st.markdown("##### Aprendizaje supervisado")
    st.markdown(
        """
        - **HistGradientBoosting** es el modelo optimo: mejor equilibrio entre detectar clientes en riesgo y evitar falsas alarmas.
        - El desbalance moderado no exige remuestreo: `class_weight='balanced'` es suficiente y preserva mejor la precision.
        - Las variables de comportamiento digital pesan mas que las demograficas: el churn es sobre todo un fenomeno de desenganche conductual.
        - La seleccion de variables por consenso redujo el espacio de 48 a 35 sin perder desempeno.
        - El MLP es una alternativa solida, pero no supera al modelo de boosting seleccionado.
        """
    )

    st.markdown("##### Aprendizaje no supervisado")
    cluster_cols = st.columns(2)
    with cluster_cols[0]:
        st.markdown(
            """
            <div class="risk-card" style="border-left-color:#15803d">
                <h3>Segmento mayoritario (~94%)</h3>
                <p>Tasa de fuga ~27.7%, alto engagement digital. Estrategia: fidelizacion y monitoreo temprano de senales de deterioro.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cluster_cols[1]:
        st.markdown(
            """
            <div class="risk-card" style="border-left-color:#b91c1c">
                <h3>Segmento critico (~6%)</h3>
                <p>Tasa de fuga ~47.7%, bajo engagement y alto abandono de carrito. Estrategia: retencion urgente.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Tres metodologias (K-Means, Clustering Jerarquico Ward y metricas internas de validacion) "
        "convergieron en 2 segmentos como la segmentacion optima."
    )

    st.markdown("##### Recomendaciones estrategicas")
    st.markdown(
        """
        - Implementar el modelo en produccion para generar scores de riesgo de fuga de forma periodica.
        - Priorizar campanas de retencion automatizadas para el segmento critico: incentivos personalizados y mejoras en checkout.
        - Para el segmento mayoritario, programas de fidelizacion y alertas tempranas ante cambios de comportamiento.
        - Monitorear mensualmente las variables mas predictivas y reentrenar el modelo con datos recientes.
        - Combinar el score de churn con el valor de vida del cliente para priorizar los esfuerzos de retencion.
        """
    )


DECISION_THRESHOLD = 0.50


def render_individual_prediction(data, model):
    hero(
        "Prediccion individual",
        "Completa los datos del cliente y presiona Predecir para obtener su riesgo de fuga y una accion recomendada.",
        "2. Modelo de prediccion",
    )

    st.markdown("#### Complete los datos del cliente")
    scenario = st.selectbox(
        "Caso de demostracion",
        ["Cliente manual", "Cliente estable", "Cliente en riesgo medio", "Cliente en riesgo alto"],
        help="Usa un caso prearmado para acelerar la demostracion o ajusta los campos manualmente.",
    )
    preset = demo_customer_preset(data, scenario) if scenario != "Cliente manual" else default_customer_values(data)

    with st.form(key="customer_prediction_form"):
        customer = build_customer_form(data, preset=preset, widget_namespace=scenario.replace(" ", "_").lower())
        submitted = st.form_submit_button("Predecir", use_container_width=True)

    if submitted:
        st.session_state["customer_snapshot"] = customer

    if "customer_snapshot" not in st.session_state:
        st.info("Completa los datos del cliente y presiona **Predecir** para ver el resultado.")
        return

    customer = st.session_state["customer_snapshot"]
    result = prediction_output(model, customer, threshold=DECISION_THRESHOLD)

    probability = float(result["Probabilidad_Fuga"].iloc[0])
    level = result["Riesgo"].iloc[0]
    prediction = result["Prediccion"].iloc[0]
    color = RISK_COLORS[level]

    st.markdown("#### Resultado de la prediccion")
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de fuga", f"{probability * 100:.1f}%")
    c2.metric("Riesgo", level)
    c3.metric("Decision", prediction)
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"Umbral de decision: {DECISION_THRESHOLD:.2f} (el mismo usado en la evaluacion final del notebook).")

    st.markdown(
        f"""
        <div class="risk-card" style="border-left-color:{color}">
            <h3>Recomendaciones</h3>
            <p>{risk_action(level)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Principales senales detectadas")
    signals = risk_signal_table(customer, data)
    render_signal_cards(signals)

    st.markdown("#### Simulador que pasaria si")
    simulation = what_if_scenarios(model, customer, data, DECISION_THRESHOLD)
    render_whatif_cards(simulation)


def render_segments(data):
    hero(
        "Segmentacion de clientes",
        "Explora segmentos con K-Means y una proyeccion PCA 3D interactiva para detectar grupos de comportamiento.",
    )

    segmented, explained = train_clustering(data, n_clusters=2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", segmented["Cluster_KMeans"].nunique())
    c2.metric("Varianza PCA 2D", f"{explained[:2].sum() * 100:.1f}%")
    c3.metric("Varianza PCA 3D", f"{explained.sum() * 100:.1f}%")

    profile = (
        segmented.groupby("Cluster_KMeans")
        .agg(
            Clientes=(TARGET, "size"),
            Tasa_Fuga=(TARGET, "mean"),
            Compras_Promedio=("Total_Compras", "mean"),
            Valor_Vida_Util_Promedio=("Valor_Vida_Util", "mean"),
            Abandono_Carrito_Promedio=("Tasa_Abandono_Carrito", "mean"),
            Apertura_Email_Promedio=("Tasa_Apertura_Email", "mean"),
            Llamadas_Servicio_Promedio=("Llamadas_Servicio_Cliente", "mean"),
            Uso_App_Movil_Promedio=("Uso_App_Movil", "mean"),
        )
        .reset_index()
    )
    profile["Pct_Clientes"] = profile["Clientes"] / len(segmented) * 100
    profile = profile.sort_values("Tasa_Fuga", ascending=False).reset_index(drop=True)

    dist_fig = px.bar(
        profile.sort_values("Cluster_KMeans"),
        x="Cluster_KMeans",
        y="Pct_Clientes",
        color="Cluster_KMeans",
        text=profile.sort_values("Cluster_KMeans")["Pct_Clientes"].map(lambda value: f"{value:.3f}%"),
        title="Participacion de clientes por cluster",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=CLUSTER_COLORS,
    )
    dist_fig.update_traces(textposition="outside")
    dist_fig.update_layout(showlegend=False, yaxis_title="% de clientes", xaxis_title="Cluster")
    st.plotly_chart(dist_fig, use_container_width=True)

    use_all_points = st.checkbox(
        "Mostrar todos los clientes en el PCA",
        value=True,
        help="Con 50,000 clientes el grafico puede tardar un poco mas, pero representa todo el dataset.",
    )

    if use_all_points:
        sample = segmented
    else:
        max_points = st.slider(
            "Cantidad de puntos a visualizar",
            min_value=1000,
            max_value=len(segmented),
            value=min(12000, len(segmented)),
            step=1000,
        )
        sample = segmented.sample(min(max_points, len(segmented)), random_state=SEED)

    fig3d = px.scatter_3d(
        sample,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster_KMeans",
        hover_data=["Fuga_Cliente", "Total_Compras", "Valor_Vida_Util", "Tasa_Abandono_Carrito"],
        title="Clusters de clientes proyectados con PCA 3D",
        template=PLOT_TEMPLATE,
        opacity=0.68,
        color_discrete_sequence=CLUSTER_COLORS,
    )
    marker_size = 2 if len(sample) > 20000 else 3
    fig3d.update_traces(marker=dict(size=marker_size))
    fig3d.update_layout(
        height=720,
        scene=dict(
            xaxis_title=f"PC1 ({explained[0] * 100:.1f}% var.)",
            yaxis_title=f"PC2 ({explained[1] * 100:.1f}% var.)",
            zaxis_title=f"PC3 ({explained[2] * 100:.1f}% var.)",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    with st.expander("Ver tambien PCA 2D"):
        fig2d = px.scatter(
            sample,
            x="PC1",
            y="PC2",
            color="Cluster_KMeans",
            hover_data=["Fuga_Cliente", "Total_Compras", "Valor_Vida_Util"],
            title="Vista PCA 2D",
            template=PLOT_TEMPLATE,
            opacity=0.62,
            color_discrete_sequence=CLUSTER_COLORS,
        )
        st.plotly_chart(fig2d, use_container_width=True)

    st.markdown("#### Perfil de clusters")
    render_cluster_cards(profile)


def render_cluster_cards(profile):
    metric_defs = [
        ("Clientes", "Clientes", lambda v: f"{int(v):,}"),
        ("Tasa de fuga", "Tasa_Fuga", lambda v: f"{v * 100:.1f}%"),
        ("Compras promedio", "Compras_Promedio", lambda v: f"{v:.1f}"),
        ("Valor de vida promedio", "Valor_Vida_Util_Promedio", lambda v: f"${v:,.0f}"),
        ("Abandono de carrito promedio", "Abandono_Carrito_Promedio", lambda v: f"{v:.1f}%"),
        ("Apertura de email promedio", "Apertura_Email_Promedio", lambda v: f"{v:.1f}%"),
        ("Llamadas a servicio promedio", "Llamadas_Servicio_Promedio", lambda v: f"{v:.1f}"),
        ("Uso de app movil promedio", "Uso_App_Movil_Promedio", lambda v: f"{v:.1f}"),
    ]
    avg_churn = profile["Tasa_Fuga"].mean()
    labels = ["Segmento de mayor riesgo", "Segmento de menor riesgo"]

    cols = st.columns(len(profile))
    for idx, (col_widget, (_, row)) in enumerate(zip(cols, profile.iterrows())):
        with col_widget:
            is_high_risk = row["Tasa_Fuga"] >= avg_churn
            badge_color = "#b91c1c" if is_high_risk else "#15803d"
            badge_bg = "#fee2e2" if is_high_risk else "#dcfce7"
            label = labels[idx] if idx < len(labels) else f"Cluster {row['Cluster_KMeans']}"
            metrics_html = "".join(
                f'<div class="cluster-metric-row"><span>{name}</span><span>{fmt(row[col])}</span></div>'
                for name, col, fmt in metric_defs
            )
            st.markdown(
                f"""
                <div class="cluster-card">
                    <span class="cluster-badge" style="color:{badge_color};background:{badge_bg}">{label}</span>
                    <h3>{row['Pct_Clientes']:.1f}% de la cartera</h3>
                    {metrics_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def feature_importance_table():
    rows = load_model_metadata().get("feature_importance", [])
    return pd.DataFrame(rows)


def feature_importance_insight_text(importance_table):
    if importance_table.empty:
        return ""
    lider = importance_table.iloc[0]
    return (
        f"**{pretty_label(lider['Variable'])}** es la senal que mas mueve la prediccion: perderla "
        "es lo que mas empeora la capacidad del modelo para distinguir clientes en riesgo."
    )


def render_feature_importance_section():
    st.markdown("#### Importancia de variables")
    importance_table = feature_importance_table()
    if importance_table.empty:
        st.info("Ejecuta `train_model.py` para calcular la importancia de variables del modelo final.")
        return

    top_importance = importance_table.head(8).copy()
    top_importance["Variable"] = top_importance["Variable"].map(pretty_label)
    top_importance = top_importance.sort_values("Importancia_media", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=top_importance["Importancia_media"],
            y=top_importance["Variable"],
            orientation="h",
            marker_color="#2563eb",
        )
    )
    fig.update_layout(
        title="Cuanto se apoya el modelo en cada variable",
        template=PLOT_TEMPLATE,
        xaxis_title="Caida de ROC AUC al mezclar la variable",
        yaxis_title="",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(feature_importance_insight_text(importance_table))


def top_variable_importance_table(model, top_n=10):
    selected_features = selected_feature_table(model)
    base_summary = (
        selected_features.groupby("Variable base")
        .agg(Score=("Score ANOVA", "max"))
        .reset_index()
        .sort_values("Score", ascending=False)
        .head(top_n)
    )
    base_summary["Variable"] = base_summary["Variable base"].map(pretty_label)
    max_score = base_summary["Score"].max() or 1
    base_summary["Relevancia"] = (base_summary["Score"] / max_score * 100).round(1)
    return base_summary[["Variable", "Relevancia"]].reset_index(drop=True)


def render_radar_chart(df, model_col, metric_cols, axis_labels, title, top_n=5, sort_col=None):
    sort_col = sort_col or metric_cols[0]
    top = df.sort_values(sort_col, ascending=False).head(top_n)

    colors = ["#2563eb", "#0f766e", "#b45309", "#7c3aed", "#b91c1c"]
    theta = axis_labels + [axis_labels[0]]

    fig = go.Figure()
    for i, (_, row) in enumerate(top.iterrows()):
        values = [float(row[col]) for col in metric_cols]
        values.append(values[0])
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself",
                name=str(row[model_col]),
                line_color=colors[i % len(colors)],
                opacity=0.7,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        template=PLOT_TEMPLATE,
        height=460,
        margin=dict(l=40, r=40, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_methodology(data, model, metrics, matrix, report):
    hero(
        "Modelo y metodologia",
        "Resumen del modelo, sus variables mas importantes y como se compara contra otros candidatos.",
        "3. Analisis de modelos",
    )

    st.markdown("#### Modelo seleccionado")
    st.markdown(
        f"**HistGradientBoostingClassifier**, el mejor de 9 clasificadores evaluados. "
        f"En el set de prueba: ROC AUC = {metrics.get('ROC AUC', 0):.3f}, "
        f"Recall = {metrics.get('Recall', 0):.3f}, F1 (clase fuga) = {metrics.get('F1', 0):.3f}."
    )
    feature_cards(
        [
            ("Seleccion de variables", "ANOVA k=35: redujo 48 variables transformadas a 35 sin perder desempeno."),
            ("Hiperparametros", "max_leaf_nodes=63, learning_rate=0.05, l2_regularization=0.1."),
            ("Balanceo de clases", "class_weight='balanced': preserva mejor la precision que SMOTE o ADASYN."),
            ("Despliegue", "Entrenado una vez con train_model.py y servido congelado, sin reentrenar por sesion."),
        ]
    )

    st.markdown("#### Variables principales del modelo")
    st.caption(
        "Relevancia relativa (0-100) segun el score ANOVA de cada variable dentro del selector que usa el modelo final."
    )
    top_vars = top_variable_importance_table(model, top_n=10)
    st.dataframe(
        top_vars,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Relevancia": st.column_config.ProgressColumn("Relevancia", min_value=0, max_value=100, format="%.0f"),
        },
    )

    render_feature_importance_section()

    st.markdown("#### Comparacion de modelos")
    st.markdown("##### Resultados oficiales del informe")
    official = official_model_results()
    render_radar_chart(
        official,
        model_col="Modelo",
        metric_cols=["Accuracy", "Precision", "Recall", "F1_macro", "ROC_AUC", "Balanced Acc."],
        axis_labels=["Accuracy", "Precision", "Recall", "F1", "ROC AUC", "Balanced Acc."],
        title="Top 5 modelos - metricas oficiales del informe",
        top_n=5,
        sort_col="F1_macro",
    )

    st.markdown("##### Comparacion recalculada en la app, incluyendo MLP")
    comparison = train_model_comparison(data)
    render_radar_chart(
        comparison,
        model_col="Modelo",
        metric_cols=["Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy", "ROC AUC"],
        axis_labels=["Accuracy", "Precision", "Recall", "F1", "Balanced Acc.", "ROC AUC"],
        title="Top 5 modelos recalculados en la app (incluye MLP)",
        top_n=5,
        sort_col="F1",
    )

    best_row = comparison.sort_values("F1", ascending=False).iloc[0]
    mlp_row = comparison[comparison["Modelo"] == "MLP"]
    mlp_note = (
        f" El MLP (F1={float(mlp_row.iloc[0]['F1']):.3f}) es competitivo pero no lo supera."
        if not mlp_row.empty
        else ""
    )
    st.markdown(f"**{best_row['Modelo']}** sigue liderando esta comparacion (F1 = {best_row['F1']:.3f}).{mlp_note}")
