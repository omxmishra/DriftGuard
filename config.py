import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FULL_LOG_PATH = os.path.join(DATA_DIR, "access_logs_full.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "access_logs_features.csv")
BASELINE_SCORES_PATH = os.path.join(DATA_DIR, "baseline_scores.csv")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "access_logs_ground_truth.csv")
SEQUENCE_SCORES_PATH = os.path.join(DATA_DIR, "sequence_scores.csv")
CLASSIFIED_ALERTS_PATH = os.path.join(DATA_DIR, "classified_alerts.csv")
EXPLAINED_ALERTS_PATH = os.path.join(DATA_DIR, "explained_alerts.csv")
CLASSIFIER_METRICS_PATH = os.path.join(DATA_DIR, "classifier_metrics.csv")

DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, "detector_lstm.pt")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "classifier_xgb.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.joblib")

WINDOW_SIZE = 10
ALERT_BUDGET_PCT = 0.02
RANDOM_SEED = 42