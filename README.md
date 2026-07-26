# DriftGuard

**AI-Powered Behavioral Anomaly Detection for Enterprise SOCs**

Built for Honeywell Hackathon Q4: Behavioral Anomaly Detection for Cybersecurity.

DriftGuard models "normal" access and connection behaviour for users, service accounts, and edge devices, detects deviations in near real-time, classifies the type of anomaly, and surfaces an explainable, ranked alert queue for SOC analysts.

## Features

- Synthetic access-log generator with 7 labeled attack behaviours + 1 ambiguous edge case
- Per-entity statistical baseline profiling (fast, interpretable first-pass detector)
- LSTM autoencoder sequence-aware detector, ensembled with the baseline via percentile-rank
- XGBoost multi-class attack-type classifier, including a "normal" rejection class to catch detector false positives
- SHAP-based explainability, translated into analyst-readable reasons per alert
- Dark-themed Streamlit dashboard: ranked alert queue, risk scores, entity history, cold-start monitoring, live model performance metrics
- Explicit cold-start and concept-drift handling
- Auto-generated PDF technical report with cover page, architecture, results, references, and embedded screenshots

## Architecture

```text
Access Logs
      ↓
Feature Engineering
      ↓
Baseline Profile + LSTM Sequence Model
      ↓
Ensemble Scoring
      ↓
Attack Classifier (XGBoost)
      ↓
SHAP Explainability
      ↓
SOC Analyst Dashboard
```

<img width="786" height="980" alt="image" src="https://github.com/user-attachments/assets/af47550d-47ea-400d-a21f-cfd117da4c22" />

## Folder Structure

```text
DriftGuard/
│
├── config.py
├── generate_data.py
├── baseline_profile.py
├── detector.py
├── classifier.py
├── explainability.py
├── dashboard.py
├── build_report.py
├── data/
├── models/
├── reports/
├── screenshots/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run in order -- each script depends on the previous one's output:

```bash
python generate_data.py        # generates data/access_logs_full.csv + ground truth
python baseline_profile.py      # generates engineered features + baseline scores
python detector.py             # trains LSTM, generates sequence_scores.csv
python classifier.py           # trains XGBoost, generates classified_alerts.csv + metrics
python explainability.py        # generates explained_alerts.csv + feature importance chart
python build_report.py          # generates the technical report PDF
```

## Dashboard

```bash
streamlit run dashboard.py
```

## Technical Report

```bash
python build_report.py
```

Generates `reports/DriftGuard_Technical_Report.pdf`, covering problem summary, innovation, assumptions, architecture, dataset, detection approach, results, explainability, cold-start/concept-drift handling, scalability, known limitations, and references.

To include screenshots in the report, drop PNG/JPG files into `screenshots/` before running -- each image is auto-embedded with a caption derived from its filename.

Before finalizing, edit the placeholder fields at the top of `build_report.py` (`[Your Name]`, `[Your Candidate ID]`, `[Your Email]`) with your actual details.

## Results (current dataset)

- 250 entities, 30 simulated days, 16,294 total sessions, 2.081% anomaly rate
- Ensemble detection recall (top 2% alert budget): 100% brute_force, 100% device_spoofing, 98% credential_stuffing, 83% low_and_slow_exfil, 34% lateral_movement, 22% impossible_travel
- Attack-type classification agreement on real flagged alerts: 97.8%
- Full metrics, architecture rationale, and known limitations: see `reports/DriftGuard_Technical_Report.pdf`

<img width="1917" height="955" alt="image" src="https://github.com/user-attachments/assets/a490d949-d271-443e-ade5-1f043ac7093c" />

<img width="1905" height="857" alt="image" src="https://github.com/user-attachments/assets/d4afe5c1-dae5-47ea-90ce-a415252ef077" />

<img width="1911" height="847" alt="image" src="https://github.com/user-attachments/assets/ecc21c8f-5a9e-41e3-bf79-38523340fc4d" />

<img width="1896" height="705" alt="image" src="https://github.com/user-attachments/assets/ef1fcd0b-f185-4806-9efe-b9a041c40be3" />

<img width="1885" height="832" alt="image" src="https://github.com/user-attachments/assets/bee2c34f-5460-4f03-9c8d-632bf26f1664" />


## Known Limitations

- Synthetic data is more cleanly separable than real logs; held-out metrics are an upper bound, not a real-world guarantee
- lateral_movement and impossible_travel recall are comparatively weak due to shared alert-budget competition
- No live streaming deployment or continuous retraining loop is implemented; both are documented as design extensions

See the technical report for the full list.
