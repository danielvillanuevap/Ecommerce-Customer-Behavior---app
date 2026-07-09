from churn_core import load_context, render_individual_prediction


data, model, _, _, _ = load_context()
render_individual_prediction(data, model)
