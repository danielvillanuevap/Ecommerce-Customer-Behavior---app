from churn_core import load_context, render_context_and_data


data, _, _, _, _ = load_context(require_model=False)
render_context_and_data(data)
