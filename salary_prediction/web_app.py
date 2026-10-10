"""
Employee Salary Prediction System - Web Version (Flask)
=======================================================

    python web_app.py          ->  open http://127.0.0.1:5000

Pages:
    /            Dashboard: dataset insights with interactive charts
    /predict     Prediction form + in-depth result visualisation
    /model       Model comparison, accuracy plots, feature importance, retrain
    /employees   Browse / search / add / delete MySQL employee records, CSV export
    /history     All predictions saved in MySQL
"""
import csv
import io
import math

import numpy as np
import pandas as pd
from flask import Flask, Response, flash, redirect, render_template, request, url_for

import config
from config import fmt_inr, fmt_lpa
from data_generator import generate_employees
from db import get_db
from ml_model import SalaryModel, train_and_save, validate_input

app = Flask(__name__)
app.secret_key = "salary-prediction-mini-project"
app.jinja_env.filters["inr"] = fmt_inr
app.jinja_env.filters["lpa"] = fmt_lpa

_model_cache = {"model": None}


def get_model():
    if _model_cache["model"] is None:
        model = SalaryModel.load()
        if model is None:
            model = train_and_save(get_db(), progress=lambda *_: None)
        _model_cache["model"] = model
    return _model_cache["model"]


def ensure_data():
    db = get_db()
    if db.employee_count() == 0:
        db.bulk_insert_employees(generate_employees())
    return db


@app.context_processor
def inject_globals():
    db = get_db()
    return {"db_backend": db.backend.upper(), "db_desc": db.description, "cfg": config}


# ----------------------------------------------------------------- helpers ---
def dashboard_data(df: pd.DataFrame) -> dict:
    s = df.salary
    counts, edges = np.histogram(s / 1e5, bins=14)
    hist = {"labels": [f"{lo:.0f}–{hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:])],
            "counts": counts.tolist()}

    by_role = df.groupby("job_role").salary.agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
    by_edu = df.groupby("education").salary.agg(["mean", "count"]).reindex(config.EDUCATION_LEVELS).dropna()

    bands = [0, 2, 5, 8, 12, 16, 50]
    band_labels = ["0–2", "2–5", "5–8", "8–12", "12–16", "16+"]
    df = df.assign(band=pd.cut(df.experience_years, bins=bands, labels=band_labels, right=False))
    by_band = df.groupby("band", observed=False).salary.agg(["mean", "median"]).fillna(0)

    skill_rows = []
    for sk in config.SKILLS:
        col = config.skill_col(sk)
        w, wo = df[df[col] == 1].salary, df[df[col] == 0].salary
        skill_rows.append({"skill": sk, "with": w.mean(), "without": wo.mean(),
                           "premium": (w.mean() / wo.mean() - 1) * 100, "count": int(len(w))})
    skill_rows.sort(key=lambda r: r["premium"], reverse=True)

    pivot = df.pivot_table(index="location", columns="company_size", values="salary", aggfunc="mean") \
              .reindex(index=config.LOCATIONS, columns=config.COMPANY_SIZES)

    sample = df.sample(min(len(df), 900), random_state=1)
    scatter = [{"x": float(r.experience_years), "y": float(r.salary), "role": r.job_role,
                "name": r.full_name} for r in sample.itertuples()]

    return {
        "hist": hist,
        "role": {"labels": by_role.index.tolist(), "mean": by_role["mean"].round(0).tolist(),
                 "median": by_role["median"].round(0).tolist(), "count": by_role["count"].tolist()},
        "edu": {"labels": by_edu.index.tolist(), "mean": by_edu["mean"].round(0).tolist(),
                "count": by_edu["count"].astype(int).tolist()},
        "band": {"labels": band_labels, "mean": by_band["mean"].round(0).tolist(),
                 "median": by_band["median"].round(0).tolist()},
        "skills": skill_rows,
        "loc_size": {"labels": config.LOCATIONS,
                     "series": [{"name": c, "values": [None if math.isnan(v) else round(v) for v in pivot[c]]}
                                for c in config.COMPANY_SIZES]},
        "scatter": scatter,
        "roles": config.JOB_ROLES,
    }


# ------------------------------------------------------------------- pages ---
@app.route("/")
def dashboard():
    db = ensure_data()
    df = db.load_training_frame()
    model = get_model()
    kpi = {
        "employees": len(df),
        "avg": df.salary.mean(),
        "median": df.salary.median(),
        "avg_exp": df.experience_years.mean(),
        "r2": model.metrics["r2"],
        "mape": model.metrics["mape"],
        "predictions": int(db.scalar("SELECT COUNT(*) AS n FROM predictions") or 0),
        "top_role": df.groupby("job_role").salary.mean().idxmax(),
    }
    return render_template("dashboard.html", kpi=kpi, data=dashboard_data(df), model=model, active="dashboard")


DEFAULT_FORM = {
    "employee_name": "", "experience_years": 3, "education": "Bachelor's", "job_role": "Software Developer",
    "location": "Tier-1 City", "company_size": "Mid-size", "certifications": 1, "performance_rating": 3,
    "skills": ["Python", "SQL"],
}


@app.route("/predict", methods=["GET", "POST"])
def predict():
    db = ensure_data()
    model = get_model()
    form = dict(DEFAULT_FORM)
    result = viz = None

    if request.method == "POST":
        form = {k: request.form.get(k, "") for k in DEFAULT_FORM if k != "skills"}
        form["skills"] = request.form.getlist("skills")
        try:
            inp = validate_input(form)
        except ValueError as e:
            flash(str(e), "error")
            return render_template("predict.html", form=form, result=None, viz=None, model=model, active="predict")

        result = model.predict(inp)
        df = db.load_training_frame()
        peers = df[df.job_role == inp["job_role"]].salary
        similar = df[(df.job_role == inp["job_role"]) & (abs(df.experience_years - inp["experience_years"]) <= 2)]

        counts, edges = np.histogram(peers / 1e5, bins=12)
        you_bin = int(np.clip(np.searchsorted(edges, result["predicted"] / 1e5, side="right") - 1, 0, len(counts) - 1))
        result.update({
            "percentile": float((peers < result["predicted"]).mean() * 100) if len(peers) else None,
            "peer_count": int(len(peers)),
            "peer_median": float(peers.median()) if len(peers) else None,
            "similar_count": int(len(similar)),
            "similar_avg": float(similar.salary.mean()) if len(similar) else None,
            "role_avg": float(peers.mean()) if len(peers) else None,
            "peer_min": float(min(peers.min(), result["lower"])) if len(peers) else result["lower"],
            "peer_max": float(max(peers.max(), result["upper"])) if len(peers) else result["upper"],
        })
        skill_imp = model.skill_impact(inp)
        viz = {
            "curve": model.experience_curve(inp, 25),
            "current_exp": inp["experience_years"],
            "current_salary": result["predicted"],
            "skills": skill_imp,
            "education": model.category_impact(inp, "education", config.EDUCATION_LEVELS),
            "location": model.category_impact(inp, "location", config.LOCATIONS),
            "company": model.category_impact(inp, "company_size", config.COMPANY_SIZES),
            "rating": [{"option": f"{r}★", "salary": float(p), "current": r == inp["performance_rating"]}
                       for r, p in zip(range(1, 6), model.predict_many(
                           [{**inp, "performance_rating": r} for r in range(1, 6)]))],
            "peer_hist": {"labels": [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(edges[:-1], edges[1:])],
                          "counts": counts.tolist(), "you": you_bin},
            "similar": [{"x": float(r.experience_years), "y": float(r.salary)} for r in
                        df[df.job_role == inp["job_role"]].itertuples()],
        }
        top_new = [d for d in skill_imp if not d["has"] and d["delta"] > 0][:3]
        result["tips"] = top_new
        result["inp"] = inp
        db.save_prediction(inp, result, "web", form.get("employee_name", "").strip())

    return render_template("predict.html", form=form, result=result, viz=viz, model=model, active="predict")


@app.route("/model")
def model_page():
    db = ensure_data()
    model = get_model()
    b = model.bundle
    pts = b["test_points"]
    resid_pct = [(p["predicted"] / p["actual"] - 1) * 100 for p in pts]
    counts, edges = np.histogram(resid_pct, bins=16)
    within10 = float(np.mean(np.abs(resid_pct) <= 10) * 100)
    data = {
        "results": b["all_results"],
        "selected": model.name,
        "scatter": [{"x": p["actual"], "y": p["predicted"], "role": p["job_role"]} for p in pts],
        "resid": {"labels": [f"{lo:+.0f}%" for lo in edges[:-1]], "counts": counts.tolist(),
                  "edges": edges.tolist()},
        "importance": b["importance"],
    }
    return render_template("model.html", model=model, b=b, data=data, within10=within10,
                           runs=db.fetch_model_runs(10), active="model")


@app.route("/train", methods=["POST"])
def train():
    model = train_and_save(get_db(), progress=lambda *_: None)
    _model_cache["model"] = model
    flash(f"Model retrained on {model.bundle['total_rows']} records. Best: {model.name} "
          f"(R² {model.metrics['r2']:.4f})", "success")
    return redirect(url_for("model_page"))


@app.route("/employees")
def employees():
    db = ensure_data()
    search = request.args.get("q", "").strip()
    role = request.args.get("role", "")
    page = max(1, int(request.args.get("page", 1) or 1))
    per_page = 20
    total = db.count_employees(search, role)
    rows = db.fetch_employees(search, role, limit=per_page, offset=(page - 1) * per_page)
    pages = max(1, math.ceil(total / per_page))
    return render_template("employees.html", rows=rows, total=total, page=page, pages=pages,
                           q=search, role=role, active="employees", form=DEFAULT_FORM)


@app.route("/employees/add", methods=["POST"])
def add_employee():
    form = {k: request.form.get(k, "") for k in DEFAULT_FORM if k != "skills"}
    form["skills"] = request.form.getlist("skills")
    name = request.form.get("full_name", "").strip()
    try:
        inp = validate_input(form)
        salary = float(request.form.get("salary", ""))
        if not name:
            raise ValueError("Full name is required")
        if not 50000 <= salary <= 50000000:
            raise ValueError("Salary must be between ₹50,000 and ₹5,00,00,000")
    except ValueError as e:
        flash(str(e) if "could not convert" not in str(e) else "Salary must be a number", "error")
        return redirect(url_for("employees"))
    emp_id = get_db().insert_employee({**inp, "full_name": name, "salary": salary})
    flash(f"Employee #{emp_id} ({name}) added. Retrain the model to include this record.", "success")
    return redirect(url_for("employees"))


@app.route("/employees/<int:emp_id>/delete", methods=["POST"])
def delete_employee(emp_id):
    get_db().delete_employee(emp_id)
    flash(f"Employee #{emp_id} deleted.", "success")
    return redirect(request.referrer or url_for("employees"))


@app.route("/employees.csv")
def export_csv():
    rows = get_db().fetch_employees()
    buf = io.StringIO()
    cols = ["employee_id", "full_name", "experience_years", "education", "job_role", "location",
            "company_size", "certifications", "performance_rating", "skills", "salary"]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=employees.csv"})


@app.route("/history")
def history():
    rows = get_db().fetch_predictions(200)
    return render_template("history.html", rows=rows, active="history")


@app.route("/history/clear", methods=["POST"])
def clear_history():
    get_db().clear_predictions()
    flash("Prediction history cleared.", "success")
    return redirect(url_for("history"))


if __name__ == "__main__":
    ensure_data()
    get_model()
    print(f" * Database: {get_db().description}")
    app.run(debug=False, host="127.0.0.1", port=5000)
