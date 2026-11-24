"""Rebuilds the cleaned Kepler KOI dataset described in the progress report.

Usage: python src/clean_kepler.py data/cumulative.csv data/
Writes kepler_clean.csv (labeled objects) and kepler_candidates.csv.
"""
import sys
import numpy as np
import pandas as pd

TRANSIT = ["koi_period", "koi_duration", "koi_depth", "koi_impact", "koi_model_snr",
           "koi_prad", "koi_ror", "koi_dor", "koi_incl", "koi_sma", "koi_teq",
           "koi_insol", "koi_srho", "koi_count"]
STAR = ["koi_steff", "koi_slogg", "koi_srad", "koi_smass", "koi_smet", "koi_kepmag"]
LOGGED = ["koi_period", "koi_depth", "koi_model_snr", "koi_prad", "koi_insol",
          "koi_ror", "koi_dor", "koi_srad"]
ENGINEERED = ["n_transits_est", "duration_ratio", "log_density_ratio"] + ["log_" + c for c in LOGGED]
CORE = TRANSIT + STAR + ENGINEERED                     # 31 core features
DIAGNOSTICS = ["koi_num_transits", "koi_max_sngle_ev", "koi_max_mult_ev",
               "koi_bin_oedp_sig", "koi_fwm_stat_sig", "koi_dicco_msky", "koi_dikco_msky"]
LEAKY = ["koi_pdisposition", "koi_score", "koi_fpflag_nt", "koi_fpflag_ss",
         "koi_fpflag_co", "koi_fpflag_ec", "kepler_name", "koi_comment"]
IDS = ["kepid", "kepoi_name"]
KEPLER_BASELINE_DAYS = 1470            # approximate length of the Kepler primary mission
SOLAR_DENSITY = 1.41                   # g/cm^3


def load_raw(path):
    return pd.read_csv(path, comment="#")


def clean(raw):
    df = raw.copy()
    # 1. Drop objects with no transit model fit (missingness reveals the label)
    no_fit = df["koi_prad"].isna() & df["koi_model_snr"].isna()
    df = df[~no_fit].copy()
    # 2. Physically impossible zeros -> missing
    for c in ["koi_depth", "koi_model_snr", "koi_insol", "koi_smass"]:
        df.loc[df[c] == 0, c] = np.nan
    # 3. Engineered features
    df["n_transits_est"] = KEPLER_BASELINE_DAYS / df["koi_period"]
    predicted_duration_h = df["koi_period"] * 24 / np.pi / df["koi_dor"]
    df["duration_ratio"] = df["koi_duration"] / predicted_duration_h
    catalog_density = SOLAR_DENSITY * df["koi_smass"] / df["koi_srad"] ** 3
    df["log_density_ratio"] = np.log10(df["koi_srho"] / catalog_density)
    for c in LOGGED:
        df["log_" + c] = np.log10(df[c])
    df = df.replace([np.inf, -np.inf], np.nan)   # guard: log10(0) would give -inf
    # 4. Labels
    labeled = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    labeled["label"] = (labeled["koi_disposition"] == "CONFIRMED").astype(int)
    candidates = df[df["koi_disposition"] == "CANDIDATE"].copy()
    cols = IDS + CORE + DIAGNOSTICS
    return (labeled[cols + ["label"]].reset_index(drop=True),
            candidates[cols].reset_index(drop=True),
            int(no_fit.sum()))


if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/cumulative.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/"
    labeled, candidates, n_drop = clean(load_raw(raw_path))
    labeled.to_csv(out + "kepler_clean.csv", index=False)
    candidates.to_csv(out + "kepler_candidates.csv", index=False)
    print(f"dropped {n_drop} no-fit rows; labeled {len(labeled)}; candidates {len(candidates)}")
