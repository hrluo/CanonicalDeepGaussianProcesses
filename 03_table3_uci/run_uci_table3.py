#!/usr/bin/env python3
"""
Full UCI benchmark:
    augmented identity-biased residual RFF-DGP
    regular residual RFF-DGP
    exact RBF Gaussian process regression
    ordinary linear regression

This script is self-contained except for standard packages.

Install:
    pip install numpy pandas torch scikit-learn openpyxl ucimlrepo

Paper Table 3 run:
    python run_uci_table3.py \
      --datasets auto_mpg,parkinsons_motor,air_quality_nox \
      --seeds 0,1,2,3,4,5,6,7,8,9 \
      --max-train 800 \
      --max-test 2000 \
      --epochs 1000 \
      --eta 1.0,0.6,0.3 \
      --n-features-hidden 96 \
      --n-features-output 192 \
      --gp-n-restarts 0 \
      --outdir table3_uci

Outputs:
    uci_runs.csv
    uci_aggregate.csv
    uci_leaderboard.csv
    uci_config.json
    uci_skipped.csv

Interpretation:
    Lower RMSE/MAE is better.
    Positive aug_vs_regular_dgp_rmse_gain means augmented DGP beats regular DGP.
    Positive aug_vs_gp_rmse_gain means augmented DGP beats exact RBF GP.
    Positive aug_vs_linear_rmse_gain means augmented DGP beats linear regression.
    Positive identity_gain means augmented DGP has more identity-like hidden layers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zipfile
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise RuntimeError("Install PyTorch first: pip install torch") from exc

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
except Exception as exc:
    raise RuntimeError("Install scikit-learn first: pip install scikit-learn") from exc


# ---------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------

DATASETS: Dict[str, Dict[str, Any]] = {
    # Prior pilot / focused candidates.
    "auto_mpg": {
        "description": "Auto MPG fuel-consumption regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
        "loader": "auto_mpg",
        "columns": [
            "mpg", "cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model_year", "origin", "car_name",
        ],
        "target_col": "mpg",
    },
    "forest_fires_log_area": {
        "description": "Forest Fires log1p burned-area regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv",
        "loader": "csv",
        "target_col": "area",
        "target_transform": "log1p",
        "categorical_cols": ["month", "day"],
    },
    "wine_red": {
        "description": "Wine Quality red wine regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "loader": "semicolon_csv",
        "target_col": "quality",
    },
    "wine_white": {
        "description": "Wine Quality white wine regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        "loader": "semicolon_csv",
        "target_col": "quality",
    },

    # Next-try candidates.
    "real_estate": {
        "description": "Real estate valuation regression.",
        "source": "static_zip",
        "url": "https://archive.ics.uci.edu/static/public/477/real+estate+valuation+data+set.zip",
        "loader": "zip_excel_auto",
        "target_col": "Y house price of unit area",
        "drop_cols": ["No"],
    },
    "qsar_fish": {
        "description": "QSAR fish toxicity regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv",
        "loader": "semicolon_headerless",
        "columns": ["CIC0", "SM1_Dz(Z)", "GATS1i", "NdsCH", "NdssC", "MLOGP", "LC50"],
        "target_col": "LC50",
    },
    "student_math": {
        "description": "Student Performance math final-grade regression.",
        "source": "legacy_zip",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
        "loader": "zip_member_semicolon",
        "member": "student-mat.csv",
        "target_col": "G3",
        "categorical": "auto",
    },
    "student_portuguese": {
        "description": "Student Performance Portuguese final-grade regression.",
        "source": "legacy_zip",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
        "loader": "zip_member_semicolon",
        "member": "student-por.csv",
        "target_col": "G3",
        "categorical": "auto",
    },
    "parkinsons_motor": {
        "description": "Parkinsons telemonitoring motor UPDRS regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
        "loader": "csv",
        "target_col": "motor_UPDRS",
        "drop_cols": ["subject#", "total_UPDRS"],
    },
    "parkinsons_total": {
        "description": "Parkinsons telemonitoring total UPDRS regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
        "loader": "csv",
        "target_col": "total_UPDRS",
        "drop_cols": ["subject#", "motor_UPDRS"],
    },
    "appliances_energy": {
        "description": "Appliances energy prediction regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv",
        "loader": "csv",
        "target_col": "Appliances",
        "drop_cols": ["date"],
    },
    "abalone": {
        "description": "Abalone rings regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        "loader": "csv_headerless",
        "columns": [
            "Sex", "Length", "Diameter", "Height", "Whole_weight",
            "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings",
        ],
        "target_col": "Rings",
        "categorical_cols": ["Sex"],
    },
    "bike_day": {
        "description": "Bike Sharing daily count regression.",
        "source": "legacy_zip",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
        "loader": "zip_member_csv",
        "member": "day.csv",
        "target_col": "cnt",
        "drop_cols": ["instant", "dteday", "casual", "registered"],
    },

    # New candidates motivated by the Parkinsons wins:
    # medium/large sensor, condition-monitoring, biomedical, or dynamic-count regression.
    "naval_compressor": {
        "description": "Naval propulsion condition monitoring: GT compressor decay coefficient.",
        "source": "ucimlrepo",
        "uci_id": 316,
        "loader": "ucimlrepo",
        "target_col": "__target__",
        "target_index": 0,
    },
    "naval_turbine": {
        "description": "Naval propulsion condition monitoring: GT turbine decay coefficient.",
        "source": "ucimlrepo",
        "uci_id": 316,
        "loader": "ucimlrepo",
        "target_col": "__target__",
        "target_index": 1,
    },
    "ct_slice_location": {
        "description": "Relative location of CT slices on axial axis.",
        "source": "ucimlrepo",
        "uci_id": 206,
        "loader": "ucimlrepo",
        "target_col": "__target__",
        "target_index": 0,
        "drop_cols": ["patientId", "PatientId", "patient_id"],
    },
    "sgemm_log_runtime": {
        "description": "SGEMM GPU kernel performance, log1p average runtime.",
        "source": "ucimlrepo",
        "uci_id": 440,
        "loader": "ucimlrepo",
        "target_col": "__target__",
        "target_strategy": "mean_targets",
        "target_transform": "log1p",
    },
    "air_quality_nox": {
        "description": "Air Quality sensor calibration: NOx concentration.",
        "source": "ucimlrepo",
        "uci_id": 360,
        "loader": "ucimlrepo",
        "target_col": "NOx(GT)",
        "missing_sentinel": -200,
        "drop_cols": ["Date", "Time", "CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"],
    },
    "air_quality_no2": {
        "description": "Air Quality sensor calibration: NO2 concentration.",
        "source": "ucimlrepo",
        "uci_id": 360,
        "loader": "ucimlrepo",
        "target_col": "NO2(GT)",
        "missing_sentinel": -200,
        "drop_cols": ["Date", "Time", "CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"],
    },
    "air_quality_benzene": {
        "description": "Air Quality sensor calibration: benzene concentration.",
        "source": "ucimlrepo",
        "uci_id": 360,
        "loader": "ucimlrepo",
        "target_col": "C6H6(GT)",
        "missing_sentinel": -200,
        "drop_cols": ["Date", "Time", "CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"],
    },
    "blog_feedback_log_comments": {
        "description": "BlogFeedback log1p next-24h comment count.",
        "source": "ucimlrepo",
        "uci_id": 304,
        "loader": "ucimlrepo",
        "target_col": "__target__",
        "target_index": 0,
        "target_transform": "log1p",
    },
    "facebook_comments_log": {
        "description": "Facebook Comment Volume log1p future comment count.",
        "source": "ucimlrepo",
        "uci_id": 363,
        "loader": "ucimlrepo",
        "target_col": "__target__",
        "target_index": 0,
        "target_transform": "log1p",
    },
    "communities_crime": {
        "description": "Communities and Crime violent crimes per population.",
        "source": "ucimlrepo",
        "uci_id": 183,
        "loader": "ucimlrepo",
        "target_col": "__target__",
        "target_index": 0,
        "drop_cols": ["communityname"],
    },

    # Larger optional robustness candidates. These are not in the default command
    # because exact GP can be slow; use smaller max-train if you include them.
    "online_news_logshares": {
        "description": "Online News Popularity log1p shares regression.",
        "source": "legacy_zip",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip",
        "loader": "zip_member_csv",
        "member": "OnlineNewsPopularity/OnlineNewsPopularity.csv",
        "target_col": "shares",
        "target_transform": "log1p",
        "drop_cols": ["url"],
        "strip_columns": True,
    },
    "protein_tertiary": {
        "description": "Protein tertiary structure RMSD regression.",
        "source": "legacy_url",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        "loader": "csv",
        "target_col": "RMSD",
    },
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def _read_zip_member_csv(url: str, member: str, sep: str = ",") -> pd.DataFrame:
    data = _download_bytes(url)
    with zipfile.ZipFile(BytesIO(data)) as zf:
        names = zf.namelist()
        if member not in names:
            # Try suffix match to be robust to directory prefixes.
            matches = [n for n in names if n.endswith(member)]
            if not matches:
                raise FileNotFoundError(f"Could not find {member} in zip. Members: {names[:10]}")
            member = matches[0]
        with zf.open(member) as f:
            return pd.read_csv(f, sep=sep)


def _read_zip_excel_auto(url: str) -> pd.DataFrame:
    data = _download_bytes(url)
    with zipfile.ZipFile(BytesIO(data)) as zf:
        members = zf.namelist()
        excel_members = [
            m for m in members
            if m.lower().endswith((".xlsx", ".xls"))
            and not Path(m).name.startswith("~$")
        ]
        if not excel_members:
            raise FileNotFoundError(f"No Excel file found in zip. Members: {members}")
        with zf.open(excel_members[0]) as f:
            return pd.read_excel(f)


def _load_dataset_frame(name: str, spec: Dict[str, Any]) -> pd.DataFrame:
    loader = spec["loader"]

    if loader == "ucimlrepo":
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError("Install ucimlrepo first: pip install ucimlrepo") from exc

        ds = fetch_ucirepo(id=int(spec["uci_id"]))

        Xdf = ds.data.features.copy() if ds.data.features is not None else pd.DataFrame()
        ydf = ds.data.targets.copy() if ds.data.targets is not None else pd.DataFrame()

        # Normalize column names to strings while preserving exact UCI labels.
        Xdf.columns = [str(c).strip() for c in Xdf.columns]
        ydf.columns = [str(c).strip() for c in ydf.columns]

        if spec.get("target_strategy") == "mean_targets":
            if ydf.empty:
                raise ValueError(f"{name}: target_strategy=mean_targets but ucimlrepo returned no targets.")
            y = ydf.apply(pd.to_numeric, errors="coerce").mean(axis=1)
            df = Xdf.copy()
            df["__target__"] = y
            return df

        if "target_index" in spec:
            idx = int(spec["target_index"])
            if not ydf.empty:
                y = ydf.iloc[:, idx]
                df = Xdf.copy()
                df["__target__"] = y
                return df
            # Fall back: some UCI imports put everything in features.
            if Xdf.shape[1] <= idx:
                raise ValueError(f"{name}: cannot find target index {idx}.")
            y = Xdf.iloc[:, -(1 + idx)]
            df = Xdf.iloc[:, :-(1 + idx)].copy()
            df["__target__"] = y
            return df

        target_col = str(spec.get("target_col", "")).strip()
        if target_col:
            if target_col in ydf.columns:
                df = Xdf.copy()
                df[target_col] = ydf[target_col]
                return df
            if target_col in Xdf.columns:
                return Xdf.copy()

        if not ydf.empty:
            df = Xdf.copy()
            df["__target__"] = ydf.iloc[:, 0]
            return df

        raise ValueError(f"{name}: ucimlrepo returned no usable target column.")

    if loader == "auto_mpg":
        return pd.read_csv(
            spec["url"],
            sep=r"\s+",
            header=None,
            names=spec["columns"],
            na_values=["?"],
            quotechar='"',
            engine="c",
        )

    if loader == "csv":
        return pd.read_csv(spec["url"])

    if loader == "semicolon_csv":
        return pd.read_csv(spec["url"], sep=";")

    if loader == "semicolon_headerless":
        return pd.read_csv(spec["url"], sep=";", header=None, names=spec["columns"])

    if loader == "csv_headerless":
        return pd.read_csv(spec["url"], header=None, names=spec["columns"])

    if loader == "zip_member_csv":
        return _read_zip_member_csv(spec["url"], spec["member"], sep=",")

    if loader == "zip_member_semicolon":
        return _read_zip_member_csv(spec["url"], spec["member"], sep=";")

    if loader == "zip_excel_auto":
        return _read_zip_excel_auto(spec["url"])

    raise ValueError(f"Unknown loader {loader} for {name}.")


def fetch_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}. Choices: {sorted(DATASETS)}")

    spec = DATASETS[name]
    df = _load_dataset_frame(name, spec)

    if spec.get("strip_columns", False):
        df.columns = [str(c).strip() for c in df.columns]

    if "missing_sentinel" in spec:
        df = df.replace(spec["missing_sentinel"], np.nan)

    target_col = spec["target_col"]
    if target_col not in df.columns:
        # Be forgiving about spaces in public CSVs.
        lookup = {str(c).strip(): c for c in df.columns}
        if target_col in lookup:
            target_col = lookup[target_col]
        else:
            raise KeyError(f"Target {spec['target_col']} not in columns: {list(df.columns)}")

    drop_cols = [c for c in spec.get("drop_cols", []) if c in df.columns]

    y = df[target_col].copy()
    X = df.drop(columns=[target_col] + drop_cols).copy()

    # Optional target transform.
    if spec.get("target_transform") == "log1p":
        y = np.log1p(pd.to_numeric(y, errors="coerce"))

    # Categorical handling.
    if spec.get("categorical") == "auto":
        cat_cols = [
            c for c in X.columns
            if X[c].dtype == "object" or str(X[c].dtype).startswith("category")
        ]
    else:
        cat_cols = [c for c in spec.get("categorical_cols", []) if c in X.columns]

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    ok = y.notna()
    X = X.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)

    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))

    # Drop zero-variance columns.
    keep = X.std(numeric_only=True) > 1e-12
    X = X.loc[:, keep.values]

    meta = {
        "dataset": name,
        "description": spec["description"],
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target": str(spec["target_col"]),
        "target_transform": spec.get("target_transform", "none"),
        "source": spec.get("source", ""),
    }

    return X, y, meta


def train_test_split_indices(n: int, train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(np.floor(float(train_frac) * n))
    n_train = max(10, min(n_train, n - 5))
    return idx[:n_train], idx[n_train:]


def standardize_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    max_train: int | None,
    max_test: int | None,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed + 999)

    if max_train is not None and max_train > 0 and train_idx.size > max_train:
        train_idx = rng.choice(train_idx, size=int(max_train), replace=False)
    if max_test is not None and max_test > 0 and test_idx.size > max_test:
        test_idx = rng.choice(test_idx, size=int(max_test), replace=False)

    X_train = X.iloc[train_idx].to_numpy(dtype=np.float32)
    X_test = X.iloc[test_idx].to_numpy(dtype=np.float32)
    y_train = y.iloc[train_idx].to_numpy(dtype=np.float32)
    y_test = y.iloc[test_idx].to_numpy(dtype=np.float32)

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_sd = X_train.std(axis=0, keepdims=True)
    x_sd[x_sd < 1e-8] = 1.0

    y_mean = float(y_train.mean())
    y_sd = float(y_train.std())
    if y_sd < 1e-8:
        y_sd = 1.0

    X_train_s = (X_train - x_mean) / x_sd
    X_test_s = (X_test - x_mean) / x_sd
    y_train_s = (y_train - y_mean) / y_sd
    y_test_s = (y_test - y_mean) / y_sd

    return (
        X_train_s.astype(np.float32),
        y_train_s.astype(np.float32),
        X_test_s.astype(np.float32),
        y_test_s.astype(np.float32),
        y_sd,
    )


# ---------------------------------------------------------------------
# Residual RFF-DGP
# ---------------------------------------------------------------------

class RFFMap(nn.Module):
    def __init__(self, in_dim: int, n_features: int, lengthscale: float):
        super().__init__()
        W = torch.randn(in_dim, n_features) / max(float(lengthscale), 1e-6)
        b = 2.0 * math.pi * torch.rand(n_features)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / float(n_features))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(X @ self.W + self.b)


class ResidualRFFDGP(nn.Module):
    def __init__(
        self,
        d: int,
        n_hidden_layers: int,
        n_features_hidden: int,
        n_features_output: int,
        lengthscale_hidden: float,
        lengthscale_output: float,
        residual_scale: float,
        init_sd: float = 0.02,
    ):
        super().__init__()
        self.d = int(d)
        self.n_hidden_layers = int(n_hidden_layers)
        self.residual_scale = float(residual_scale)

        self.hidden_features = nn.ModuleList(
            [RFFMap(self.d, n_features_hidden, lengthscale_hidden) for _ in range(self.n_hidden_layers)]
        )
        self.hidden_coef = nn.ParameterList(
            [nn.Parameter(init_sd * torch.randn(n_features_hidden, self.d)) for _ in range(self.n_hidden_layers)]
        )

        self.output_features = RFFMap(self.d, n_features_output, lengthscale_output)
        self.output_coef = nn.Parameter(init_sd * torch.randn(n_features_output))
        self.output_bias = nn.Parameter(torch.zeros(()))

    def forward(self, X: torch.Tensor, return_hidden: bool = False):
        H = X
        residuals = []
        hidden_states = [H]

        for feat, coef in zip(self.hidden_features, self.hidden_coef):
            R = self.residual_scale * (feat(H) @ coef)
            H = H + R
            residuals.append(R)
            hidden_states.append(H)

        yhat = self.output_features(H) @ self.output_coef + self.output_bias

        if return_hidden:
            return yhat, hidden_states, residuals
        return yhat

    def coefficient_norm2(self) -> torch.Tensor:
        out = torch.sum(self.output_coef ** 2) + self.output_bias ** 2
        for coef in self.hidden_coef:
            out = out + torch.sum(coef ** 2)
        return out


@dataclass
class DGPFitResult:
    rmse: float
    mae: float
    identity_error: float
    layer_identity_errors: List[float]
    effective_depth: float
    final_loss: float


def fit_dgp(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_test_np: np.ndarray,
    y_test_np: np.ndarray,
    seed: int,
    augmented: bool,
    args: argparse.Namespace,
) -> DGPFitResult:
    set_seed(seed)

    d = int(X_train_np.shape[1])
    lengthscale_hidden = float(np.sqrt(d)) if args.lengthscale_hidden <= 0 else float(args.lengthscale_hidden)
    lengthscale_output = float(np.sqrt(d)) if args.lengthscale_output <= 0 else float(args.lengthscale_output)

    eta = [float(v.strip()) for v in args.eta.split(",") if v.strip()]
    if len(eta) < args.n_hidden_layers:
        eta = eta + [0.0] * (args.n_hidden_layers - len(eta))
    eta = eta[:args.n_hidden_layers]

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    model = ResidualRFFDGP(
        d=d,
        n_hidden_layers=args.n_hidden_layers,
        n_features_hidden=args.n_features_hidden,
        n_features_output=args.n_features_output,
        lengthscale_hidden=lengthscale_hidden,
        lengthscale_output=lengthscale_output,
        residual_scale=args.residual_scale,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = float(X_train.shape[0])
    sigma2 = max(float(args.noise_sd) ** 2, 1e-4)
    eta_tensor = torch.tensor(eta, dtype=torch.float32)

    last_loss = float("nan")
    for epoch in range(int(args.epochs)):
        opt.zero_grad()

        yhat, _, residuals = model(X_train, return_hidden=True)
        mse_term = 0.5 * torch.mean((yhat - y_train) ** 2) / sigma2
        prior_term = 0.5 * float(args.prior_weight) * model.coefficient_norm2() / n

        identity_term = torch.zeros(())
        if augmented:
            per_layer = [torch.sqrt(torch.mean(R ** 2) + 1e-12) for R in residuals]
            identity_term = torch.sum(eta_tensor * torch.stack(per_layer))

        loss = mse_term + prior_term + identity_term
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

        if args.verbose and (epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1):
            kind = "aug" if augmented else "reg"
            print(f"      {kind} epoch={epoch:5d} loss={last_loss:.6f}", flush=True)

    with torch.no_grad():
        yhat, _, residuals = model(X_test, return_hidden=True)
        rmse = float(torch.sqrt(torch.mean((yhat - y_test) ** 2)).cpu())
        mae = float(torch.mean(torch.abs(yhat - y_test)).cpu())
        layer_errs = [float(torch.sqrt(torch.mean(R ** 2)).cpu()) for R in residuals]
        identity_error = float(np.mean(layer_errs)) if layer_errs else float("nan")
        effective_depth = float(np.sum(np.asarray(layer_errs) > float(args.depth_threshold)))

    return DGPFitResult(
        rmse=rmse,
        mae=mae,
        identity_error=identity_error,
        layer_identity_errors=layer_errs,
        effective_depth=effective_depth,
        final_loss=last_loss,
    )


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

def rmse_mae(yhat: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    yhat = np.asarray(yhat, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae = float(np.mean(np.abs(yhat - y)))
    return rmse, mae


def fit_linear_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse, mae = rmse_mae(pred, y_test)
    return {"rmse": rmse, "mae": mae}


def fit_gp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    d = int(X_train.shape[1])

    # Exact GP can be cubic in n. Optionally subsample only for GP.
    if args.gp_max_train is not None and args.gp_max_train > 0 and X_train.shape[0] > args.gp_max_train:
        rng = np.random.default_rng(seed + 12345)
        idx = rng.choice(np.arange(X_train.shape[0]), size=int(args.gp_max_train), replace=False)
        X_fit = X_train[idx]
        y_fit = y_train[idx]
    else:
        X_fit = X_train
        y_fit = y_train

    length = float(np.sqrt(d)) if args.gp_lengthscale <= 0 else float(args.gp_lengthscale)

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=length, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=args.gp_noise_level, noise_level_bounds=(1e-5, 1e1))
    )

    optimizer = None if args.gp_no_optimize else "fmin_l_bfgs_b"

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=args.gp_alpha,
        optimizer=optimizer,
        n_restarts_optimizer=int(args.gp_n_restarts),
        normalize_y=False,
        random_state=seed,
    )

    gp.fit(X_fit, y_fit)
    pred = gp.predict(X_test)
    rmse, mae = rmse_mae(pred, y_test)

    return {
        "rmse": rmse,
        "mae": mae,
        "gp_train_n_used": int(X_fit.shape[0]),
        "gp_kernel": str(gp.kernel_),
    }


# ---------------------------------------------------------------------
# Rows, aggregation, leaderboard
# ---------------------------------------------------------------------

def run_one_dataset_seed(
    dataset_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    meta: Dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    train_idx, test_idx = train_test_split_indices(len(y), args.train_frac, seed)
    X_train, y_train, X_test, y_test, y_sd = standardize_train_test(
        X, y, train_idx, test_idx,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=seed,
    )

    # Same seed for regular and augmented DGP, so random features and
    # initialization are matched.
    dgp_seed = 50_000 + 1009 * int(seed) + 17 * int(X_train.shape[1]) + int(X_train.shape[0])

    regular = fit_dgp(
        X_train, y_train, X_test, y_test,
        seed=dgp_seed,
        augmented=False,
        args=args,
    )
    augmented = fit_dgp(
        X_train, y_train, X_test, y_test,
        seed=dgp_seed,
        augmented=True,
        args=args,
    )

    linear = fit_linear_baseline(X_train, y_train, X_test, y_test)

    if args.no_gp:
        gp = {
            "rmse": float("nan"),
            "mae": float("nan"),
            "gp_train_n_used": 0,
            "gp_kernel": "",
        }
    else:
        gp = fit_gp_baseline(X_train, y_train, X_test, y_test, seed=seed, args=args)

    row: Dict[str, Any] = {
        "dataset": dataset_name,
        "description": meta["description"],
        "seed": int(seed),
        "n_rows": meta["n_rows"],
        "n_features": meta["n_features"],
        "target": meta["target"],
        "target_transform": meta["target_transform"],
        "train_n_used": int(X_train.shape[0]),
        "test_n_used": int(X_test.shape[0]),
        "y_sd": float(y_sd),
    }

    # Standardized scale.
    row.update({
        "regular_dgp_rmse_std": regular.rmse,
        "augmented_dgp_rmse_std": augmented.rmse,
        "gp_rmse_std": gp["rmse"],
        "linear_rmse_std": linear["rmse"],
        "regular_dgp_mae_std": regular.mae,
        "augmented_dgp_mae_std": augmented.mae,
        "gp_mae_std": gp["mae"],
        "linear_mae_std": linear["mae"],
    })

    # Original target scale.
    for key in ["rmse", "mae"]:
        row[f"regular_dgp_{key}_original"] = row[f"regular_dgp_{key}_std"] * y_sd
        row[f"augmented_dgp_{key}_original"] = row[f"augmented_dgp_{key}_std"] * y_sd
        row[f"gp_{key}_original"] = row[f"gp_{key}_std"] * y_sd
        row[f"linear_{key}_original"] = row[f"linear_{key}_std"] * y_sd

    # Gains: positive favors augmented DGP.
    for scale in ["std", "original"]:
        for metric in ["rmse", "mae"]:
            aug = row[f"augmented_dgp_{metric}_{scale}"]
            row[f"aug_vs_regular_dgp_{metric}_gain_{scale}"] = row[f"regular_dgp_{metric}_{scale}"] - aug
            row[f"aug_vs_gp_{metric}_gain_{scale}"] = row[f"gp_{metric}_{scale}"] - aug
            row[f"aug_vs_linear_{metric}_gain_{scale}"] = row[f"linear_{metric}_{scale}"] - aug

    # DGP layer diagnostics.
    row.update({
        "regular_dgp_identity_error": regular.identity_error,
        "augmented_dgp_identity_error": augmented.identity_error,
        "identity_gain": regular.identity_error - augmented.identity_error,
        "regular_dgp_effective_depth": regular.effective_depth,
        "augmented_dgp_effective_depth": augmented.effective_depth,
        "depth_reduction": regular.effective_depth - augmented.effective_depth,
        "regular_dgp_final_loss": regular.final_loss,
        "augmented_dgp_final_loss": augmented.final_loss,
        "gp_train_n_used": gp.get("gp_train_n_used", 0),
        "gp_kernel": gp.get("gp_kernel", ""),
    })

    return row


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    metadata_keys = {
        "dataset", "description", "seed", "target", "target_transform", "gp_kernel",
    }

    numeric_keys = [
        k for k, v in rows[0].items()
        if k not in metadata_keys and isinstance(v, (int, float, np.floating))
    ]

    out = []
    for dataset in sorted(set(r["dataset"] for r in rows)):
        sub = [r for r in rows if r["dataset"] == dataset]
        agg: Dict[str, Any] = {
            "dataset": dataset,
            "description": sub[0]["description"],
            "n_rows": sub[0]["n_rows"],
            "n_features": sub[0]["n_features"],
            "target": sub[0]["target"],
            "target_transform": sub[0]["target_transform"],
            "n_runs": len(sub),
        }
        for key in numeric_keys:
            vals = np.asarray([r[key] for r in sub], float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                agg[f"{key}_mean"] = float("nan")
                agg[f"{key}_se"] = float("nan")
            else:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_se"] = (
                    float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                    if vals.size > 1 else 0.0
                )

        out.append(agg)

    return out


def make_leaderboard(aggregate: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for a in aggregate:
        methods = ["regular_dgp", "augmented_dgp", "gp", "linear"]

        rmse_vals = {
            m: float(a.get(f"{m}_rmse_original_mean", np.nan))
            for m in methods
        }
        mae_vals = {
            m: float(a.get(f"{m}_mae_original_mean", np.nan))
            for m in methods
        }

        finite_rmse = {m: v for m, v in rmse_vals.items() if np.isfinite(v)}
        finite_mae = {m: v for m, v in mae_vals.items() if np.isfinite(v)}

        best_rmse_method = min(finite_rmse, key=finite_rmse.get) if finite_rmse else ""
        best_mae_method = min(finite_mae, key=finite_mae.get) if finite_mae else ""

        best_non_aug_rmse = min(
            [v for m, v in finite_rmse.items() if m != "augmented_dgp"],
            default=float("nan"),
        )
        best_non_aug_mae = min(
            [v for m, v in finite_mae.items() if m != "augmented_dgp"],
            default=float("nan"),
        )

        rmse_rank = 1 + sum(
            v < rmse_vals.get("augmented_dgp", float("inf"))
            for m, v in finite_rmse.items() if m != "augmented_dgp"
        )
        mae_rank = 1 + sum(
            v < mae_vals.get("augmented_dgp", float("inf"))
            for m, v in finite_mae.items() if m != "augmented_dgp"
        )

        rows.append({
            "dataset": a["dataset"],
            "n_rows": a["n_rows"],
            "n_features": a["n_features"],
            "n_runs": a["n_runs"],
            "augmented_dgp_rmse": rmse_vals.get("augmented_dgp", float("nan")),
            "regular_dgp_rmse": rmse_vals.get("regular_dgp", float("nan")),
            "gp_rmse": rmse_vals.get("gp", float("nan")),
            "linear_rmse": rmse_vals.get("linear", float("nan")),
            "augmented_dgp_mae": mae_vals.get("augmented_dgp", float("nan")),
            "regular_dgp_mae": mae_vals.get("regular_dgp", float("nan")),
            "gp_mae": mae_vals.get("gp", float("nan")),
            "linear_mae": mae_vals.get("linear", float("nan")),
            "best_rmse_method": best_rmse_method,
            "best_mae_method": best_mae_method,
            "augmented_rmse_rank": rmse_rank,
            "augmented_mae_rank": mae_rank,
            "aug_vs_best_non_aug_rmse_gain": best_non_aug_rmse - rmse_vals.get("augmented_dgp", float("nan")),
            "aug_vs_best_non_aug_mae_gain": best_non_aug_mae - mae_vals.get("augmented_dgp", float("nan")),
            "aug_vs_regular_dgp_rmse_gain": a.get("aug_vs_regular_dgp_rmse_gain_original_mean", float("nan")),
            "aug_vs_gp_rmse_gain": a.get("aug_vs_gp_rmse_gain_original_mean", float("nan")),
            "aug_vs_linear_rmse_gain": a.get("aug_vs_linear_rmse_gain_original_mean", float("nan")),
            "identity_gain": a.get("identity_gain_mean", float("nan")),
            "depth_reduction": a.get("depth_reduction_mean", float("nan")),
        })

    rows = sorted(
        rows,
        key=lambda r: (
            r["augmented_rmse_rank"],
            -float(r["aug_vs_best_non_aug_rmse_gain"]) if np.isfinite(r["aug_vs_best_non_aug_rmse_gain"]) else float("inf"),
        ),
    )
    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        default="auto_mpg,parkinsons_motor,air_quality_nox",
        help=f"Comma-separated dataset names. Choices: {sorted(DATASETS)}",
    )
    parser.add_argument("--outdir", default="table3_uci")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--max-test", type=int, default=2000)

    # DGP options.
    parser.add_argument("--noise-sd", type=float, default=0.15)
    parser.add_argument("--n-hidden-layers", type=int, default=3)
    parser.add_argument("--n-features-hidden", type=int, default=96)
    parser.add_argument("--n-features-output", type=int, default=192)
    parser.add_argument("--lengthscale-hidden", type=float, default=-1.0)
    parser.add_argument("--lengthscale-output", type=float, default=-1.0)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--eta", default="1.0,0.6,0.3")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--depth-threshold", type=float, default=0.025)

    # GP options.
    parser.add_argument("--no-gp", action="store_true", help="Skip exact GP baseline.")
    parser.add_argument("--gp-max-train", type=int, default=0, help="Optional smaller train size for exact GP only; 0 means same train set.")
    parser.add_argument("--gp-lengthscale", type=float, default=-1.0)
    parser.add_argument("--gp-noise-level", type=float, default=0.05)
    parser.add_argument("--gp-alpha", type=float, default=1e-6)
    parser.add_argument("--gp-n-restarts", type=int, default=0)
    parser.add_argument("--gp-no-optimize", action="store_true", help="Use fixed GP kernel hyperparameters.")

    parser.add_argument("--skip-unavailable", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Normalize gp-max-train.
    if args.gp_max_train <= 0:
        args.gp_max_train = None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    all_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    for dataset_name in dataset_names:
        print(f"\n=== Dataset: {dataset_name} ===", flush=True)
        try:
            X, y, meta = fetch_dataset(dataset_name)
            print(
                f"Loaded {dataset_name}: n={meta['n_rows']}, d={meta['n_features']}, "
                f"target={meta['target']}, transform={meta['target_transform']}",
                flush=True,
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if args.skip_unavailable:
                print(f"Skipping {dataset_name}: {msg}", flush=True)
                skipped.append({"dataset": dataset_name, "error": msg})
                continue
            raise

        for seed in seeds:
            print(f"  seed={seed}", flush=True)
            try:
                row = run_one_dataset_seed(dataset_name, X, y, meta, seed, args)
                all_rows.append(row)
                print(
                    "    RMSE orig: "
                    f"aug={row['augmented_dgp_rmse_original']:.4g}, "
                    f"regDGP={row['regular_dgp_rmse_original']:.4g}, "
                    f"GP={row['gp_rmse_original']:.4g}, "
                    f"linear={row['linear_rmse_original']:.4g}; "
                    f"id_gain={row['identity_gain']:.4g}",
                    flush=True,
                )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                print(f"    failed seed={seed}: {msg}", flush=True)
                skipped.append({"dataset": dataset_name, "seed": str(seed), "error": msg})
                continue

        # Save incrementally.
        write_csv(all_rows, outdir / "uci_runs.csv")
        aggregate = aggregate_rows(all_rows)
        write_csv(aggregate, outdir / "uci_aggregate.csv")
        write_csv(make_leaderboard(aggregate), outdir / "uci_leaderboard.csv")
        write_csv(skipped, outdir / "uci_skipped.csv")

    runs_path = outdir / "uci_runs.csv"
    agg_path = outdir / "uci_aggregate.csv"
    leaderboard_path = outdir / "uci_leaderboard.csv"
    skipped_path = outdir / "uci_skipped.csv"

    aggregate = aggregate_rows(all_rows)
    leaderboard = make_leaderboard(aggregate)

    write_csv(all_rows, runs_path)
    write_csv(aggregate, agg_path)
    write_csv(leaderboard, leaderboard_path)
    write_csv(skipped, skipped_path)

    with (outdir / "uci_config.json").open("w") as f:
        json.dump(
            {
                "datasets": dataset_names,
                "seeds": seeds,
                "train_frac": args.train_frac,
                "max_train": args.max_train,
                "max_test": args.max_test,
                "skipped": skipped,
                "dgp_args": {
                    "noise_sd": args.noise_sd,
                    "n_hidden_layers": args.n_hidden_layers,
                    "n_features_hidden": args.n_features_hidden,
                    "n_features_output": args.n_features_output,
                    "lengthscale_hidden": args.lengthscale_hidden,
                    "lengthscale_output": args.lengthscale_output,
                    "residual_scale": args.residual_scale,
                    "prior_weight": args.prior_weight,
                    "eta": args.eta,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "depth_threshold": args.depth_threshold,
                },
                "gp_args": {
                    "no_gp": args.no_gp,
                    "gp_max_train": args.gp_max_train,
                    "gp_lengthscale": args.gp_lengthscale,
                    "gp_noise_level": args.gp_noise_level,
                    "gp_alpha": args.gp_alpha,
                    "gp_n_restarts": args.gp_n_restarts,
                    "gp_no_optimize": args.gp_no_optimize,
                },
            },
            f,
            indent=2,
            default=str,
        )

    print("\nDone.")
    print(f"Per-run CSV:     {runs_path.resolve()}")
    print(f"Aggregate CSV:   {agg_path.resolve()}")
    print(f"Leaderboard CSV: {leaderboard_path.resolve()}")
    if skipped:
        print(f"Skipped CSV:     {skipped_path.resolve()}")

    if leaderboard:
        print("\nTop rows by augmented RMSE rank:")
        for r in leaderboard[:10]:
            print(
                f"  {r['dataset']:24s} rank={r['augmented_rmse_rank']} "
                f"aug_vs_best_non_aug_rmse_gain={r['aug_vs_best_non_aug_rmse_gain']:.4g} "
                f"id_gain={r['identity_gain']:.4g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
