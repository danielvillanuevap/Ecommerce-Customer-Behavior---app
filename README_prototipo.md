# Prototipo Streamlit para churn de clientes

Este prototipo aplica la logica del PDF de prototipado al notebook `Proyecto_oficial_version1.ipynb`.

Incluye:

- Validacion basica del entorno y versiones.
- Carga del dataset desde `ecommerce_customer_churn_dataset.csv` local, con fallback a datos demo sinteticos si el archivo no esta presente.
- Navegacion directa (sin pantalla de inicio generica): contexto y datos, dashboard, prediccion individual, segmentacion y metodologia.
- Contexto de negocio y entendimiento de datos con EDA interactivo, incluyendo que predice el modelo y a quien ayuda.
- Dashboard ejecutivo con KPIs de fuga y niveles de riesgo.
- Modelo `HistGradientBoostingClassifier` entrenado una sola vez (`train_model.py`) y servido congelado desde `model/`, sin reentrenar en cada sesion.
- Simulador de cliente con widgets de entrada y acciones recomendadas presentadas como tarjetas de texto, no tablas.
- Casos demo prearmados para cliente estable, riesgo medio y riesgo alto.
- Senales explicables por cliente y simulador "que pasaria si" para acciones de retencion.
- Importancia de variables por permutacion con insights interpretativos.
- Validacion del modelo con metricas reales, matriz de confusion y comparacion contra baseline y otros modelos (incluye MLP como referencia de deep learning).
- Segmentacion no supervisada con K-Means y PCA 3D interactivo.
- Pagina final de visualizacion y comunicacion alineada a la seccion 5 del informe.

## Ejecutar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

`train_model.py` solo se corre una vez (o cuando cambie el dataset/notebook) y genera `model/churn_hist_gradient_boosting.joblib`
y `model/model_metadata.json`. La app carga ese modelo congelado en cada sesion en vez de reentrenar.

La app y `train_model.py` leen `ecommerce_customer_churn_dataset.csv` directamente desde la raiz del proyecto
(mismo dataset que antes se descargaba desde Google Drive). Si el archivo no esta presente, el prototipo usa
un dataset demo deterministico generado localmente para evitar bloqueos durante la exposicion.

## Estructura

```text
app.py
churn_core.py
train_model.py
ecommerce_customer_churn_dataset.csv
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

## Guia de diseno aplicada

- Paleta: fondo claro `#f6f8fb`, texto principal `#172033`, azul analitico `#2563eb`, verde estabilidad `#15803d`, ambar alerta media `#b45309` y rojo riesgo alto `#b91c1c`.
- Tipografia: Inter, con fallback a fuentes del sistema.
- UX: cada pantalla inicia con contexto, sigue con KPIs y termina con detalle accionable.
- UI: tarjetas con radios suaves, sombras ligeras, sidebar oscuro, graficos Plotly limpios y colores consistentes por riesgo.
- Dinamismo: microanimacion de entrada en encabezados y hover sutil en botones.
- Funcionalidad: scoring individual con modelo congelado, importancia de variables y PCA 3D para segmentacion.

## Guion sugerido de presentacion

1. Contexto y datos: presentar problema de churn, que predice el modelo, a quien ayuda, dataset, desbalance y EDA interactivo.
2. Dashboard: revisar volumen de clientes, tasa real de fuga, riesgo alto estimado y plan de accion comercial.
3. Prediccion individual: cargar el caso "Cliente en riesgo alto", leer las senales detectadas y mostrar el simulador "que pasaria si".
4. Segmentacion: mostrar los clusters, el PCA 3D y el perfil de grupos para lectura de comportamiento.
5. Modelo y metodologia: defender el pipeline, matriz de confusion, importancia de variables, comparacion oficial y MLP.
6. Visualizacion y comunicacion: cerrar con hallazgos, comparativas, justificacion del modelo y conclusiones.

## Decision de modelo

El modelo final es `HistGradientBoostingClassifier` por su balance entre desempeno, estabilidad, velocidad e interpretabilidad operativa.
La app compara contra baseline, regresion logistica, random forest, gradient boosting clasico y `MLPClassifier`.
El MLP se incluye como aproximacion de deep learning; aunque puede acercarse al mejor modelo, no se selecciona porque requiere mas tuning,
es menos transparente para explicar acciones de retencion y su complejidad no compensa una mejora marginal en una demo ejecutiva.

## Seccion 5 del informe cubierta en la app

- 5.1 Hallazgos: desbalance, senales tempranas, drivers de churn y metricas principales.
- 5.2 Comparativas: tabla oficial del informe y comparacion recalculada con modelos candidatos.
- 5.3 Justificacion: seleccion de Hist Gradient Boosting y explicacion frente a MLP/deep learning.
- 5.4 Conclusiones: prediccion accionable, priorizacion de retencion y valor para negocio.
