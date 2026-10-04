"""
Router predictor — thin wrapper around the trained DeBERTa ordinal router
(trained on UltraFeedback-derived routing labels).

The real implementation lives in api_gateway.routes.RouterPredictor so that
the API and any offline tooling share the exact same loading/inference logic.
"""


class RouterPredictor:
    def __init__(self, model_path: str = None):
        from api_gateway.routes import RouterPredictor as _RP
        if model_path:
            _RP.LOCAL_MODEL_DIR = model_path
        self._rp = _RP

    def predict_difficulty(self, prompt: str) -> float:
        """Predicts difficulty score between 0 and 1 (expected ordinal class / 2)."""
        return self._rp.predict(prompt)["difficulty_score"]

    def predict_route(self, prompt: str) -> int:
        """Returns routing class: 0=weak, 1=medium, 2=strong."""
        return self._rp.predict(prompt)["label"]
