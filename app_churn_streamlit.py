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

NUMERIC_COLUMNS = [
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
    "Uso_App_Movil_Binario",
    "Tiene_Balance_Credito",
]

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

EXCLUDED_COLUMNS = ["Ciudad"]


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


@st.cache_data(show_spinner="Cargando dataset...")
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv(DATA_URL)

    data = data.rename(columns=RENAME_COLUMNS)

    if TARGET in data.columns:
        data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce").fillna(0).astype(int)

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in ["Genero", "Pais", "Ciudad", "Trimestre_Registro", "Diversidad_Metodo_Pago"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    return data


def add_business_features(data):
    data = data.copy()

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


def split_features(data):
    model_data = add_business_features(data)
    feature_columns = [col for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if col in model_data.columns]
    X = model_data[feature_columns]
    y = model_data[TARGET]
    return X, y


@st.cache_resource(show_spinner="Entrenando modelo supervisado...")
def train_supervised_model(data):
    X, y = split_features(data)
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
    X, _ = split_features(data)
    prep = make_preprocessor()
    X_prep = prep.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=SEED)
    labels = kmeans.fit_predict(X_prep)

    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X_prep)

    segmented = data.copy()
    segmented["Cluster_KMeans"] = labels
    segmented["PC1"] = coords[:, 0]
    segmented["PC2"] = coords[:, 1]

    return segmented, pca.explained_variance_ratio_


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

    left, right = st.columns(2)
    with left:
        customer["Genero"] = st.selectbox("Genero", sorted_options(data["Genero"]))
        customer["Pais"] = st.selectbox("Pais", sorted_options(data["Pais"]))
        customer["Trimestre_Registro"] = st.selectbox(
            "Trimestre de registro",
            sorted_options(data["Trimestre_Registro"]),
        )
        customer["Diversidad_Metodo_Pago"] = st.selectbox(
            "Diversidad metodo pago",
            sorted_options(data["Diversidad_Metodo_Pago"]),
        )

    with right:
        for col in ["Edad", "Anos_Membresia", "Frecuencia_Login", "Dias_Desde_Ultima_Compra"]:
            low, high, default = numeric_bounds(data, col)
            customer[col] = st.number_input(col, min_value=low, max_value=high, value=default)

    st.divider()

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
    st.subheader("Validacion de entorno")
    packages = ["streamlit", "pandas", "numpy", "scikit-learn", "plotly"]
    checks = pd.DataFrame(
        {
            "Elemento": ["Python"] + packages,
            "Version": [platform.python_version()] + [package_version(pkg) for pkg in packages],
        }
    )
    st.dataframe(checks, hide_index=True, use_container_width=True)


def show_eda(data):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes", f"{len(data):,}")
    c2.metric("Variables", data.shape[1])
    c3.metric("Tasa fuga", f"{data[TARGET].mean() * 100:.1f}%")
    c4.metric("Nulos", f"{int(data.isna().sum().sum()):,}")

    left, right = st.columns(2)
    with left:
        churn_counts = data[TARGET].value_counts().rename(index={0: "Activo", 1: "Fuga"}).reset_index()
        churn_counts.columns = ["Estado", "Clientes"]
        st.plotly_chart(px.bar(churn_counts, x="Estado", y="Clientes", color="Estado"), use_container_width=True)

    with right:
        missing = data.isna().mean().mul(100).sort_values(ascending=False).head(12).reset_index()
        missing.columns = ["Variable", "Porcentaje nulos"]
        st.plotly_chart(px.bar(missing, x="Porcentaje nulos", y="Variable", orientation="h"), use_container_width=True)

    x_axis = st.selectbox("Variable X", ["Total_Compras", "Valor_Vida_Util", "Tasa_Abandono_Carrito", "Tasa_Apertura_Email"])
    y_axis = st.selectbox("Variable Y", ["Valor_Vida_Util", "Dias_Desde_Ultima_Compra", "Llamadas_Servicio_Cliente", "Uso_App_Movil"])
    sample = data.sample(min(5000, len(data)), random_state=SEED)
    st.plotly_chart(
        px.scatter(
            sample,
            x=x_axis,
            y=y_axis,
            color=TARGET,
            opacity=0.55,
            title="Relacion entre variables de comportamiento y fuga",
        ),
        use_container_width=True,
    )


def show_prediction(data, model):
    threshold = st.slider("Umbral de decision", 0.10, 0.90, 0.50, 0.01)
    customer = build_customer_form(data)

    probability = float(model.predict_proba(add_business_features(customer)[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS])[:, 1][0])
    prediction = int(probability >= threshold)

    st.subheader("Resultado del cliente simulado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de fuga", f"{probability * 100:.1f}%")
    c2.metric("Decision", "Fuga" if prediction else "Activo")
    c3.metric("Umbral", f"{threshold:.2f}")
    st.progress(min(max(probability, 0.0), 1.0))

    if probability >= 0.70:
        st.error("Riesgo alto: priorizar contacto, beneficio personalizado y revision de friccion en soporte/carrito.")
    elif probability >= 0.40:
        st.warning("Riesgo medio: monitorear comportamiento y activar campana preventiva.")
    else:
        st.success("Riesgo bajo: mantener acciones de fidelizacion y seguimiento regular.")

    st.dataframe(customer, hide_index=True, use_container_width=True)


def show_model_validation(metrics, matrix, report):
    metrics_df = pd.DataFrame({"Metrica": list(metrics.keys()), "Valor": list(metrics.values())})
    metrics_df["Valor"] = metrics_df["Valor"].round(4)
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    cm_df = pd.DataFrame(matrix, index=["Real Activo", "Real Fuga"], columns=["Pred Activo", "Pred Fuga"])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title="Matriz de confusion")
    st.plotly_chart(fig, use_container_width=True)

    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Clase"})
    st.dataframe(report_df.round(4), hide_index=True, use_container_width=True)


def show_segments(data):
    segmented, explained = train_clustering(data, n_clusters=2)

    c1, c2 = st.columns(2)
    c1.metric("Clusters", segmented["Cluster_KMeans"].nunique())
    c2.metric("Varianza PCA 2D", f"{explained.sum() * 100:.1f}%")

    fig = px.scatter(
        segmented.sample(min(10000, len(segmented)), random_state=SEED),
        x="PC1",
        y="PC2",
        color="Cluster_KMeans",
        hover_data=["Fuga_Cliente", "Total_Compras", "Valor_Vida_Util", "Tasa_Abandono_Carrito"],
        title="Segmentacion de clientes con K-Means y PCA",
        opacity=0.6,
    )
    st.plotly_chart(fig, use_container_width=True)

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
        )
        .reset_index()
    )
    profile["Pct_Clientes"] = profile["Clientes"] / len(segmented) * 100
    st.dataframe(profile.round(3), hide_index=True, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Prototipo Churn E-commerce",
        layout="wide",
    )

    st.title("Prototipo analitico de fuga de clientes")
    st.caption("Aplicacion Streamlit basada en el notebook Proyecto_oficial_version1.ipynb")

    with st.sidebar:
        st.header("Datos")
        uploaded_file = st.file_uploader("CSV alternativo", type=["csv"])
        show_environment_validation()

    data = load_data(uploaded_file)
    model, metrics, matrix, report = train_supervised_model(data)

    tab_eda, tab_predict, tab_model, tab_segments = st.tabs(
        ["EDA", "Prediccion", "Validacion", "Segmentacion"]
    )

    with tab_eda:
        show_eda(data)

    with tab_predict:
        show_prediction(data, model)

    with tab_model:
        show_model_validation(metrics, matrix, report)

    with tab_segments:
        show_segments(data)


if __name__ == "__main__":
    main()
