# Separating Exoplanets from False Positives in Kepler Data

**CMPS 261 Machine Learning, Section L2, American University of Beirut**
Instructor: Aya Kayal
Group: Hasan Jomaa, Mohammad Ammar, Ali Marji

We classify Kepler Objects of Interest (KOIs) as **confirmed planets** or **false positives** using only
measurements available before human vetting, then rank the 1,873 unresolved candidates for follow-up.

## Main results (held-out test set, 1,467 objects)

| Model | PR-AUC | ROC-AUC | F1 | Accuracy |
|---|---|---|---|---|
| Logistic regression (Level 1) | 0.950 | 0.971 | 0.885 | 90.9% |
| **Gradient boosting (Level 1)** | **0.975** | **0.987** | **0.939** | **95.4%** |
| Neural network, MLP (Level 2) | 0.963 | 0.981 | 0.913 | 93.4% |

- Gradient boosting wins on all 5 alternative grouped splits (PR-AUC 0.971 ± 0.006) and halves false alarms vs. logistic regression.
- It ranks as well as Kepler's own `koi_score` (0.975) without using any vetting output.
- Pipeline diagnostics raise PR-AUC to 0.997 but leak vetting decisions, so they are excluded from the main model.
- Main failure modes: weak signals (SNR < 10), small and isolated planets, blended light from nearby stars.

## Repository layout

```
kepler_exoplanet_classification.ipynb   full pipeline, executed with outputs
src/clean_kepler.py                     standalone cleaning script (same steps as the notebook)
data/cumulative.csv                     raw NASA Exoplanet Archive KOI table (downloaded 19 Nov 2026)
data/kepler_clean.csv                   7,328 labeled objects after cleaning
data/kepler_candidates.csv              1,873 unresolved candidates
models/final_model.joblib               final trained model (download this)
results/results.json                    every number used in the report and poster
results/candidate_ranking.csv           calibrated planet probability for every candidate
figures/                                all figures
report/                                 final report (PDF and Word)
poster/                                 final poster (PDF)
```

## Reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute kepler_exoplanet_classification.ipynb
```

Runs in about 10 minutes on a 2-core CPU. All randomness is fixed (`RANDOM_STATE = 42`).
Tested with Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, shap 0.54.1.
In Google Colab: upload `data/cumulative.csv` into a `data/` folder and run all cells.

## Use the trained model

```python
import joblib, pandas as pd
bundle = joblib.load("models/final_model.joblib")
X = pd.read_csv("data/kepler_candidates.csv")
raw = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
planet_probability = bundle["calibrator"].predict(raw)   # calibrated probability
```

The model expects the 31 core features produced by `src/clean_kepler.py` (listed in `bundle["features"]`).

## Data source

NASA Exoplanet Archive, Kepler Objects of Interest cumulative table:
https://exoplanetarchive.ipac.caltech.edu
