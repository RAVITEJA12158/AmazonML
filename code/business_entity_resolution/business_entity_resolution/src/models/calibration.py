import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def fit(self, raw_scores: np.ndarray, labels: np.ndarray):
        self.model = LogisticRegression()
        self.model.fit(raw_scores.reshape(-1, 1), labels)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, raw_scores: np.ndarray, labels: np.ndarray):
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(raw_scores, labels)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.model.predict(raw_scores)


def get_calibrator(method: str):
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    if method == "none":
        return None
    raise ValueError(f"Unknown calibration method: {method}")
