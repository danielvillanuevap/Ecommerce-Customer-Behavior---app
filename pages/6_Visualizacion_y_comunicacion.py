from churn_core import load_context, render_visualization_communication


data, model, metrics, matrix, report = load_context()
render_visualization_communication(data, model, metrics, matrix, report)
