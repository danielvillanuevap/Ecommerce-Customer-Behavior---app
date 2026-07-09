import platform
from importlib import metadata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


SEED = 42
DATA_URL = "https://drive.google.com/uc?id=1KY4QYHM20j3ZWZqY8AeEgHpfPU9O4gNx"
TARGET = "Fuga_Cliente"

PAGE_OPTIONS = [
    "Dashboard",
    "Prediccion individual",
    "Prediccion masiva",
    "Segmentacion",
    "Modelo y metodologia",
]

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

PLOT_TEMPLATE = "plotly_white"
RISK_COLORS = {"Bajo": "#15803d", "Medio": "#b45309", "Alto": "#b91c1c"}


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "no instalado"


def apply_theme():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2.4rem;
            max-width: 1360px;
        }

        h1, h2, h3 {
            color: #172033;
            letter-spacing: 0;
        }

        .app-hero {
            border: 1px solid #e3e8ef;
            background: linear-gradient(135deg, #ffffff 0%, #f6f8fb 62%, #eef7f5 100%);
            border-radius: 8px;
            padding: 1.2rem 1.35rem;
            margin-bottom: 1rem;
        }

        .eyebrow {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            color: #172033;
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }

        .hero-copy {
            color: #5f6b7a;
            max-width: 860px;
            font-size: 1rem;
            line-height: 1.5;
        }

        .risk-card {
            border: 1px solid #e3e8ef;
            border-left: 6px solid #2563eb;
            background: #ffffff;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.7rem 0 1rem;
        }

        .risk-card h3 {
            margin: 0 0 0.25rem;
            font-size: 1.05rem;
        }

        .risk-card p {
            color: #5f6b7a;
            margin: 0;
            line-height: 1.45;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e3e8ef;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
        }

        div[data-testid="stMetricValue"] {
            color: #172033;
            font-size: 1.55rem;
            font-weight: 800;
        }

        .stDownloadButton button {
            border-radius: 6px;
            border: 1px solid #1f2937;
            background: #1f2937;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle, eyebrow="Prototipo analitico"):
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-copy">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Cargando dataset...")
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv(DATA_URL)
    return normalize_columns(data)


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
        bins=[0, 25, 35, 50, 65, np.inf],
        labels=["Joven", "Adulto_Joven", "Adulto", "Adulto_Mayor", "Senior"],
        include_lowest=True,
    ).astype("object")

    data["Segmento_Compras"] = pd.cut(
        pd.to_numeric(data["Total_Compras"], errors="coerce"),
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=["Bajo", "Medio", "Alto", "Muy_Alto"],
    ).astype("object")

    data["Segmento_Recencia"] = pd.cut(
        pd.to_numeric(data["Dias_Desde_Ultima_Compra"], errors="coerce"),
        bins=[-np.inf, 7, 30, 60, np.inf],
        labels=["Reciente", "Medio", "Lejano", "Muy_Lejano"],
    ).astype("object")

    data["Segmento_Valor_Vida_Util"] = pd.cut(
        pd.to_numeric(data["Valor_Vida_Util"], errors="coerce"),
        bins=[-np.inf, 500, 1000, 2000, np.inf],
        labels=["Bajo", "Medio", "Alto", "Premium"],
    ).astype("object")

    return data


def feature_frame(data):
    return add_business_features(data)[PREDICTION_COLUMNS]


def make_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


@st.cache_resource(show_spinner="Entrenando modelo supervisado...")
def train_supervised_model(data):
    X = feature_frame(data)
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    model = Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_leaf_nodes=63,
                    max_iter=150,
                    learning_rate=0.05,
                    l2_regularization=0.1,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

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

    return model, metrics, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)


@st.cache_resource(show_spinner="Calculando segmentacion de clientes...")
def train_clustering(data, n_clusters=2):
    X_prep = make_preprocessor().fit_transform(feature_frame(data))
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
    data = normalize_columns(data)
    return model.predict_proba(feature_frame(data))[:, 1]


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


def build_customer_form(data):
    customer = {}

    st.markdown("#### Datos generales")
    left, right = st.columns(2)
    with left:
        customer["Genero"] = st.selectbox("Genero", sorted_options(data["Genero"]))
        customer["Pais"] = st.selectbox("Pais", sorted_options(data["Pais"]))
        customer["Trimestre_Registro"] = st.selectbox("Trimestre de registro", sorted_options(data["Trimestre_Registro"]))
        customer["Diversidad_Metodo_Pago"] = st.selectbox(
            "Diversidad metodo pago",
            sorted_options(data["Diversidad_Metodo_Pago"]),
        )

    with right:
        for col in ["Edad", "Anos_Membresia", "Frecuencia_Login", "Dias_Desde_Ultima_Compra"]:
            low, high, default = numeric_bounds(data, col)
            customer[col] = st.number_input(col, min_value=low, max_value=high, value=default)

    st.markdown("#### Comportamiento de compra y engagement")
    cols = st.columns(3)
    input_columns = [
        "Duracion_Sesion_Promedio",
        "Paginas_Por_Sesion",
        "Tasa_Abandono_Carrito",
        "Articulos_Lista_Deseos",
        "Total_Compras",
        "Valor_Promedio_Pedido",
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

    for idx, col in enumerate(input_columns):
        low, high, default = numeric_bounds(data, col)
        with cols[idx % 3]:
            customer[col] = st.number_input(col, min_value=low, max_value=high, value=default)

    customer["Ciudad"] = safe_mode(data["Ciudad"]) if "Ciudad" in data.columns else "No_usado"
    return pd.DataFrame([customer])


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


def page_dashboard(data, model):
    hero(
        "Dashboard de fuga de clientes",
        "Vista ejecutiva para priorizar retencion: volumen de clientes, riesgo estimado y senales de friccion.",
    )

    scored = prediction_output(model, data)
    high_risk = int((scored["Riesgo"] == "Alto").sum())
    medium_risk = int((scored["Riesgo"] == "Medio").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes", f"{len(data):,}")
    c2.metric("Tasa real de fuga", f"{data[TARGET].mean() * 100:.1f}%")
    c3.metric("Riesgo alto estimado", f"{high_risk:,}")
    c4.metric("Probabilidad media", f"{scored['Probabilidad_Fuga'].mean() * 100:.1f}%")

    left, right = st.columns([1.08, 0.92])
    with left:
        risk_counts = scored["Riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto"]).fillna(0).reset_index()
        risk_counts.columns = ["Riesgo", "Clientes"]
        fig = px.bar(
            risk_counts,
            x="Riesgo",
            y="Clientes",
            color="Riesgo",
            color_discrete_map=RISK_COLORS,
            title="Clientes por nivel de riesgo",
            template=PLOT_TEMPLATE,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        churn_counts = data[TARGET].value_counts().rename(index={0: "Activo", 1: "Fuga"}).reset_index()
        churn_counts.columns = ["Estado", "Clientes"]
        fig = px.pie(
            churn_counts,
            names="Estado",
            values="Clientes",
            hole=0.55,
            color="Estado",
            color_discrete_map={"Activo": "#0f766e", "Fuga": "#b91c1c"},
            title="Distribucion real de fuga",
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="risk-card" style="border-left-color:#b91c1c">
            <h3>{high_risk:,} clientes en riesgo alto y {medium_risk:,} en riesgo medio</h3>
            <p>La prioridad operativa debe empezar por riesgo alto, especialmente clientes con abandono de carrito alto,
            poca apertura de email, baja actividad en app o muchas llamadas a soporte.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        country = (
            scored.groupby("Pais", observed=False)
            .agg(Probabilidad_Fuga=("Probabilidad_Fuga", "mean"), Clientes=("Probabilidad_Fuga", "size"))
            .reset_index()
            .sort_values("Probabilidad_Fuga", ascending=False)
            .head(12)
        )
        fig = px.bar(
            country,
            x="Probabilidad_Fuga",
            y="Pais",
            orientation="h",
            title="Riesgo promedio por pais",
            template=PLOT_TEMPLATE,
            hover_data=["Clientes"],
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with chart_cols[1]:
        sample = scored.sample(min(6000, len(scored)), random_state=SEED)
        fig = px.scatter(
            sample,
            x="Tasa_Abandono_Carrito",
            y="Tasa_Apertura_Email",
            color="Riesgo",
            color_discrete_map=RISK_COLORS,
            hover_data=["Total_Compras", "Valor_Vida_Util", "Llamadas_Servicio_Cliente"],
            title="Friccion digital vs engagement",
            template=PLOT_TEMPLATE,
            opacity=0.62,
        )
        st.plotly_chart(fig, use_container_width=True)


def page_individual_prediction(data, model):
    hero(
        "Prediccion individual",
        "Simula un cliente, ajusta el umbral de decision y obtiene una accion recomendada para retencion.",
    )

    threshold = st.slider("Umbral de decision", 0.10, 0.90, 0.50, 0.01)
    customer = build_customer_form(data)
    result = prediction_output(model, customer, threshold=threshold)

    probability = float(result["Probabilidad_Fuga"].iloc[0])
    level = result["Riesgo"].iloc[0]
    prediction = result["Prediccion"].iloc[0]
    color = RISK_COLORS[level]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probabilidad de fuga", f"{probability * 100:.1f}%")
    c2.metric("Riesgo", level)
    c3.metric("Decision", prediction)
    c4.metric("Umbral", f"{threshold:.2f}")
    st.progress(min(max(probability, 0.0), 1.0))

    st.markdown(
        f"""
        <div class="risk-card" style="border-left-color:{color}">
            <h3>Accion recomendada</h3>
            <p>{risk_action(level)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(result, hide_index=True, use_container_width=True)


def page_batch_prediction(data, model):
    hero(
        "Prediccion masiva",
        "Carga un CSV de clientes, calcula probabilidad de fuga para cada registro y descarga una cartera priorizada.",
    )

    threshold = st.slider("Umbral de decision para lote", 0.10, 0.90, 0.50, 0.01)
    batch_file = st.file_uploader("CSV para scoring masivo", type=["csv"], key="batch_scoring")

    if batch_file is None:
        st.info("Sube un CSV con las mismas columnas base del dataset. Tambien se aceptan nombres originales en ingles.")
        preview = data.drop(columns=[TARGET], errors="ignore").head(5)
        st.caption("Ejemplo de estructura esperada")
        st.dataframe(preview, hide_index=True, use_container_width=True)
        return

    batch_data = pd.read_csv(batch_file)
    scored = prediction_output(model, batch_data, threshold=threshold).sort_values("Probabilidad_Fuga", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros procesados", f"{len(scored):,}")
    c2.metric("Riesgo alto", f"{int((scored['Riesgo'] == 'Alto').sum()):,}")
    c3.metric("Probabilidad media", f"{scored['Probabilidad_Fuga'].mean() * 100:.1f}%")

    risk_counts = scored["Riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto"]).fillna(0).reset_index()
    risk_counts.columns = ["Riesgo", "Clientes"]
    fig = px.bar(
        risk_counts,
        x="Riesgo",
        y="Clientes",
        color="Riesgo",
        color_discrete_map=RISK_COLORS,
        title="Resultado del lote por nivel de riesgo",
        template=PLOT_TEMPLATE,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(scored.head(100), hide_index=True, use_container_width=True)
    st.download_button(
        "Descargar predicciones CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_fuga_clientes.csv",
        mime="text/csv",
    )


def page_segments(data):
    hero(
        "Segmentacion de clientes",
        "Explora segmentos con K-Means y una proyeccion PCA 3D interactiva para detectar grupos de comportamiento.",
    )

    segmented, explained = train_clustering(data, n_clusters=2)
    sample = segmented.sample(min(12000, len(segmented)), random_state=SEED)

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", segmented["Cluster_KMeans"].nunique())
    c2.metric("Varianza PCA 2D", f"{explained[:2].sum() * 100:.1f}%")
    c3.metric("Varianza PCA 3D", f"{explained.sum() * 100:.1f}%")

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
        color_discrete_sequence=["#2563eb", "#f97316", "#0f766e", "#b91c1c"],
    )
    fig3d.update_traces(marker=dict(size=3))
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
            color_discrete_sequence=["#2563eb", "#f97316", "#0f766e", "#b91c1c"],
        )
        st.plotly_chart(fig2d, use_container_width=True)

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
    st.markdown("#### Perfil de clusters")
    st.dataframe(profile.round(3), hide_index=True, use_container_width=True)


def page_model_methodology(metrics, matrix, report):
    hero(
        "Modelo y metodologia",
        "Resumen tecnico del flujo aplicado: datos, preparacion, modelo optimizado, validacion y prototipo.",
    )

    st.markdown("#### Modelo seleccionado")
    st.markdown(
        """
        Se utiliza `HistGradientBoostingClassifier` optimizado, porque en el notebook fue el mejor candidato global
        entre los modelos principales. El prototipo conserva los hiperparametros ganadores:
        `max_leaf_nodes=63`, `max_iter=150`, `learning_rate=0.05`, `l2_regularization=0.1`
        y `class_weight='balanced'`.
        """
    )

    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        metrics_df = pd.DataFrame({"Metrica": list(metrics.keys()), "Valor": list(metrics.values())})
        metrics_df["Valor"] = metrics_df["Valor"].round(4)
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with c2:
        cm_df = pd.DataFrame(matrix, index=["Real Activo", "Real Fuga"], columns=["Pred Activo", "Pred Fuga"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title="Matriz de confusion")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Classification report")
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Clase"})
    st.dataframe(report_df.round(4), hide_index=True, use_container_width=True)

    st.markdown("#### Flujo del prototipo")
    flow = pd.DataFrame(
        [
            {"Etapa": "Datos", "Aplicacion": "Carga CSV desde Google Drive o archivo local."},
            {"Etapa": "Preparacion", "Aplicacion": "Imputacion, escalado robusto, one-hot encoding y variables de negocio."},
            {"Etapa": "Modelo", "Aplicacion": "HistGradientBoosting optimizado para clasificar fuga."},
            {"Etapa": "Scoring", "Aplicacion": "Probabilidad, nivel de riesgo y accion recomendada."},
            {"Etapa": "Segmentacion", "Aplicacion": "K-Means con visualizacion PCA 3D para lectura de grupos."},
        ]
    )
    st.dataframe(flow, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="Churn Intelligence", layout="wide", initial_sidebar_state="expanded")
    apply_theme()

    with st.sidebar:
        st.title("Churn Intelligence")
        page = st.radio("Navegacion", PAGE_OPTIONS, label_visibility="collapsed")
        st.divider()
        st.caption("Fuente de entrenamiento")
        uploaded_file = st.file_uploader("CSV alternativo", type=["csv"], key="training_data")
        st.divider()
        show_environment_validation()

    data = load_data(uploaded_file)

    if TARGET not in data.columns:
        st.error("El dataset de entrenamiento debe incluir la columna Fuga_Cliente o Churned.")
        return

    model, metrics, matrix, report = train_supervised_model(data)

    if page == "Dashboard":
        page_dashboard(data, model)
    elif page == "Prediccion individual":
        page_individual_prediction(data, model)
    elif page == "Prediccion masiva":
        page_batch_prediction(data, model)
    elif page == "Segmentacion":
        page_segments(data)
    elif page == "Modelo y metodologia":
        page_model_methodology(metrics, matrix, report)


if __name__ == "__main__":
    main()
