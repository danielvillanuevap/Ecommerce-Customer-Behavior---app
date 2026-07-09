from churn_core import load_context, render_model_methodology


data, model, metrics, matrix, report = load_context()
render_model_methodology(data, model, metrics, matrix, report)
