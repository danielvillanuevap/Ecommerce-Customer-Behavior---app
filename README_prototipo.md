# Churn Intelligence — App de prediccion de fuga de clientes

App en Streamlit para predecir la probabilidad de fuga (churn) de clientes de e-commerce, explorar los
factores detras del modelo y priorizar acciones de retencion. Usa un modelo `HistGradientBoostingClassifier`
entrenado y congelado (no reentrena en cada sesion).

## Paginas

**1. Contexto y datos** — pagina de entrada. Explica el problema de negocio, que predice el modelo, a quien
ayuda, la distribucion real de la fuga y permite explorar cualquier variable numerica de forma interactiva.

**2. Prediccion individual** — dos formas de predecir:
- Formulario de un cliente (con casos demo prearmados: estable, riesgo medio, riesgo alto) que solo calcula
  al presionar **Predecir**. Muestra probabilidad, riesgo, decision, recomendacion, señales de riesgo
  detectadas y un simulador "que pasaria si" con barras antes/despues por escenario.
- **Prediccion por CSV**, al final de la pagina: sube un archivo con varios clientes (columnas en español o
  ingles) y obtiene probabilidad de fuga para todos a la vez, con tarjetas de riesgo (Alto/Medio/Bajo),
  balance de la cartera predicha (Activo/Fuga), tabla de resultados, descarga en CSV y recomendaciones
  cortas segun el lote.

**3. Segmentacion** — clustering no supervisado (K-Means, k=2) con proyeccion PCA 3D interactiva y perfil de
cada segmento (mayor riesgo vs. menor riesgo) en tarjetas con metricas clave.

**4. Modelo y metodologia** — que modelo se uso y por que, sus variables mas importantes, importancia por
permutacion, y dos graficas de araña comparando los 5 mejores modelos (segun el informe oficial, y
recalculada en la app incluyendo el MLP).

**5. Visualizacion y comunicacion** — cierre del proyecto: metricas finales, hallazgos principales del
dataset, y conclusiones (aprendizaje supervisado, no supervisado y recomendaciones estrategicas).

## Modelo

- **Algoritmo:** `HistGradientBoostingClassifier` (scikit-learn), elegido entre 9 clasificadores comparados
  en el notebook del proyecto.
- **Pipeline:** limpieza por reglas de negocio, feature engineering (segmentos de recencia, valor de vida y
  compras), transformacion log1p / capping IQR segun asimetria, encoding categorico, seleccion de variables
  ANOVA (`SelectKBest`, k=35 de 48 transformadas).
- **Hiperparametros:** `max_leaf_nodes=63`, `max_iter=150`, `learning_rate=0.05`, `l2_regularization=0.1`,
  `class_weight='balanced'`.
- **Umbral de decision:** 0.50 (fijo, el mismo de la evaluacion final del notebook).
- **Metricas en test:** Accuracy 0.909 · Precision 0.836 · Recall 0.853 · F1 (fuga) 0.845 · ROC AUC 0.929.
- El modelo se entrena **una sola vez** con `train_model.py` (no en cada sesion de la app) y se guarda en
  `model/churn_hist_gradient_boosting.joblib` + `model/model_metadata.json` (metricas, matriz de confusion,
  importancia de variables). La app carga ese archivo congelado.

## Datos

La app lee `ecommerce_customer_churn_dataset.csv` directamente de la raiz del proyecto (50,000 clientes,
25 variables). Si el archivo no esta presente, usa un dataset demo sintetico generado localmente para que
la app nunca se caiga por falta de datos.

## Ejecutar localmente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

`train_model.py` solo hace falta correrlo una vez (o cuando cambie el dataset). Genera el `.joblib` y el
`.json` que la app carga despues sin reentrenar.

## Estructura

```text
app.py
churn_core.py
train_model.py
ecommerce_customer_churn_dataset.csv
requirements.txt
runtime.txt
model/
  churn_hist_gradient_boosting.joblib
  model_metadata.json
pages/
  0_Contexto_y_datos.py
  2_Prediccion_individual.py
  4_Segmentacion.py
  5_Modelo_y_metodologia.py
  6_Visualizacion_y_comunicacion.py
.streamlit/
  config.toml
```

## Despliegue

Pensada para Streamlit Community Cloud: archivo principal `app.py`, dependencias con version exacta en
`requirements.txt` y `runtime.txt` fijando Python 3.12 (evita incompatibilidades al deserializar el modelo
si Streamlit Cloud asigna una version de Python mas nueva que la usada para entrenar).

## Diseño

- Paleta: fondo claro `#f6f8fb`, texto `#172033`, azul analitico `#2563eb`, verde estabilidad `#15803d`,
  ambar alerta media `#b45309`, rojo riesgo alto `#b91c1c`. Sidebar oscuro.
- Tipografia: Inter, tamaños de letra ampliados para mejor lectura.
- Nombres de variables presentables en toda la app (sin guiones bajos).
- Resultados y recomendaciones en tarjetas visuales en vez de tablas crudas; las tablas se reservan para
  contenido genuinamente tabular (comparaciones, listados de datos).
