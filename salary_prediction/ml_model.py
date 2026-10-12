"""
Machine-learning core (Regression).

  * Loads employee data from MySQL
  * Trains and compares 5 regression models
  * Picks the best by cross-validated R²
  * Saves the model + evaluation results with joblib
  * Predicts salary with an 80% range and runs "what-if" analysis

Shared by both the CLI (cli_app.py) and the web UI (web_app.py).
"""
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

import config

FEATURE_LABELS = {
    "experience_years": "Experience",
    "certifications": "Certifications",
    "performance_rating": "Performance rating",
    "education": "Education",
    "job_role": "Job role",
    "location": "Location",
    "company_size": "Company size",
    **{config.skill_col(s): f"Skill: {s}" for s in config.SKILLS},
}


# ------------------------------------------------------------- input check ---
def validate_input(raw: dict) -> dict:
    """Clean and validate user input (from CLI or web form). Raises ValueError."""
    errors = []

    def num(key, lo, hi, cast=float):
        try:
            v = cast(raw.get(key))
        except (TypeError, ValueError):
            errors.append(f"{key.replace('_', ' ').title()} must be a number")
            return None
        if not lo <= v <= hi:
            errors.append(f"{key.replace('_', ' ').title()} must be between {lo} and {hi}")
        return v

    def choice(key, options):
        v = raw.get(key)
        if v not in options:
            errors.append(f"{key.replace('_', ' ').title()} must be one of: {', '.join(options)}")
        return v

    skills = raw.get("skills") or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    bad = [s for s in skills if s not in config.SKILLS]
    if bad:
        errors.append(f"Unknown skill(s): {', '.join(bad)}")

    cleaned = {
        "experience_years": num("experience_years", 0, 40),
        "education": choice("education", config.EDUCATION_LEVELS),
        "job_role": choice("job_role", config.JOB_ROLES),
        "location": choice("location", config.LOCATIONS),
        "company_size": choice("company_size", config.COMPANY_SIZES),
        "certifications": num("certifications", 0, 20, int),
        "performance_rating": num("performance_rating", 1, 5, int),
        "skills": [s for s in config.SKILLS if s in skills],   # canonical order, no dupes
    }
    if errors:
        raise ValueError("; ".join(errors))
    return cleaned


def to_frame(inputs: list) -> pd.DataFrame:
    """List of validated input dicts -> DataFrame with model feature columns."""
    rows = []
    for inp in inputs:
        row = {k: inp[k] for k in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES}
        for s in config.SKILLS:
            row[config.skill_col(s)] = int(s in inp["skills"])
        rows.append(row)
    return pd.DataFrame(rows, columns=config.ALL_FEATURES)


# ------------------------------------------------------------------ models ---
def _preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ("skills", "passthrough", config.SKILL_FEATURES),
    ])


def _poly_preprocessor():
    """Experience as a cubic curve + pairwise interactions (e.g. role x experience)."""
    return Pipeline([
        ("cols", ColumnTransformer([
            ("exp", Pipeline([("poly", PolynomialFeatures(3, include_bias=False)), ("scale", StandardScaler())]),
             ["experience_years"]),
            ("num", StandardScaler(), ["certifications", "performance_rating"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
            ("skills", "passthrough", config.SKILL_FEATURES),
        ])),
        ("interactions", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
    ])


def _candidates():
    """name -> (estimator, preprocessor factory)"""
    rs = config.RANDOM_STATE
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Polynomial Regression": (Ridge(alpha=10.0), _poly_preprocessor),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                                       subsample=0.9, random_state=rs),
    }


def _make_pipeline(spec):
    estimator, prep = spec if isinstance(spec, tuple) else (spec, _preprocessor)
    # Salaries grow multiplicatively, so the model learns log(salary)
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep()), ("model", estimator)]),
        func=np.log, inverse_func=np.exp)


class SalaryModel:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]

    # ------------------------------------------------------------- train ---
    @classmethod
    def train(cls, df: pd.DataFrame, progress=print) -> "SalaryModel":
        if len(df) < 50:
            raise ValueError("Need at least 50 employee records to train the model.")
        X, y = df[config.ALL_FEATURES], df[config.TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE)
        kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

        results, fitted = [], {}
        for name, est in _candidates().items():
            progress(f"  training {name} ...")
            pipe = _make_pipeline(est)
            cv = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            results.append({
                "model_name": name,
                "r2": float(r2_score(y_test, pred)),
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "mape": float(mean_absolute_percentage_error(y_test, pred) * 100),
                "cv_r2": float(cv.mean()),
                "cv_std": float(cv.std()),
            })
            fitted[name] = (pipe, pred)

        best = max(results, key=lambda r: r["cv_r2"])
        best_name = best["model_name"]
        pipe, pred = fitted[best_name]
        progress(f"  best model: {best_name} (CV R² = {best['cv_r2']:.4f})")

        # 80% prediction range from the spread of actual/predicted on unseen data
        ratio = y_test.values / pred
        lo_q, hi_q = np.quantile(ratio, [0.10, 0.90])

        progress("  computing feature importance ...")
        perm = permutation_importance(pipe, X_test, y_test, n_repeats=8,
                                      random_state=config.RANDOM_STATE, scoring="r2")
        importance = sorted(
            [{"feature": f, "label": FEATURE_LABELS[f], "importance": float(m), "std": float(s)}
             for f, m, s in zip(config.ALL_FEATURES, perm.importances_mean, perm.importances_std)],
            key=lambda d: d["importance"], reverse=True)

        test_df = X_test.copy()
        test_df["actual"] = y_test.values
        test_df["predicted"] = pred
        bundle = {
            "pipeline": pipe,
            "model_name": best_name,
            "metrics": best,
            "all_results": sorted(results, key=lambda r: r["cv_r2"], reverse=True),
            "ratio_quantiles": (float(lo_q), float(hi_q)),
            "importance": importance,
            "test_points": test_df[["job_role", "experience_years", "actual", "predicted"]].to_dict("records"),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "total_rows": len(df),
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return cls(bundle)

    # --------------------------------------------------------- save/load ---
    def save(self, path=config.MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.bundle, path)

    @classmethod
    def load(cls, path=config.MODEL_PATH):
        if not os.path.exists(path):
            return None
        return cls(joblib.load(path))

    # ----------------------------------------------------------- predict ---
    @property
    def name(self):
        return self.bundle["model_name"]

    @property
    def metrics(self):
        return self.bundle["metrics"]

    def predict_many(self, inputs: list) -> np.ndarray:
        return self.pipeline.predict(to_frame(inputs))

    def predict(self, inp: dict) -> dict:
        value = float(self.predict_many([inp])[0])
        lo_q, hi_q = self.bundle["ratio_quantiles"]
        return {
            "predicted": value,
            "lower": value * lo_q,
            "upper": value * hi_q,
            "monthly": value / 12,
            "model_name": self.name,
        }

    # ----------------------------------------------------------- what-if ---
    def experience_curve(self, inp: dict, max_years: int = 25) -> list:
        years = list(range(0, max_years + 1))
        preds = self.predict_many([{**inp, "experience_years": y} for y in years])
        return [{"years": y, "salary": float(p)} for y, p in zip(years, preds)]

    def skill_impact(self, inp: dict) -> list:
        """Change in predicted salary from adding a missing skill / removing an owned one."""
        base = float(self.predict_many([inp])[0])
        variants, labels = [], []
        for s in config.SKILLS:
            if s in inp["skills"]:
                variants.append({**inp, "skills": [x for x in inp["skills"] if x != s]})
                labels.append((s, True))
            else:
                variants.append({**inp, "skills": inp["skills"] + [s]})
                labels.append((s, False))
        preds = self.predict_many(variants)
        out = []
        for (s, has), p in zip(labels, preds):
            # for owned skills: how much it currently adds (base - without)
            delta = base - float(p) if has else float(p) - base
            out.append({"skill": s, "has": has, "delta": delta, "pct": delta / base * 100})
        return sorted(out, key=lambda d: d["delta"], reverse=True)

    def category_impact(self, inp: dict, field: str, options: list) -> list:
        preds = self.predict_many([{**inp, field: o} for o in options])
        return [{"option": o, "salary": float(p), "current": o == inp[field]} for o, p in zip(options, preds)]


def train_and_save(db, progress=print) -> SalaryModel:
    df = db.load_training_frame()
    progress(f"  loaded {len(df)} employee records from {db.backend.upper()}")
    model = SalaryModel.train(df, progress=progress)
    model.save()
    db.log_model_run({**model.metrics, "train_rows": model.bundle["train_rows"]})
    return model
