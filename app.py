import streamlit as st

from churn_core import apply_theme

st.set_page_config(page_title="Churn Intelligence", layout="wide")
apply_theme()

pages = [
    st.Page("pages/0_Contexto_y_datos.py", title="Contexto y datos", default=True),
    st.Page("pages/2_Prediccion_individual.py", title="Prediccion individual"),
    st.Page("pages/4_Segmentacion.py", title="Segmentacion"),
    st.Page("pages/5_Modelo_y_metodologia.py", title="Modelo y metodologia"),
    st.Page("pages/6_Visualizacion_y_comunicacion.py", title="Visualizacion y comunicacion"),
]

st.navigation(pages).run()
