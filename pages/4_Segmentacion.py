from churn_core import load_context, render_segments


data, _, _, _, _ = load_context(require_model=False)
render_segments(data)
