"""
Employee Salary Prediction System - Command Line Version
=========================================================

Interactive menu:
    python cli_app.py

One-shot prediction (no menu):
    python cli_app.py predict --experience 4 --education "Master's" --role "Data Scientist" \
        --location "Tier-1 City" --company MNC --certifications 2 --rating 4 \
        --skills "Python,SQL,Machine Learning"

Other commands:
    python cli_app.py stats | model | train | history
"""
import argparse
import os
import sys

import numpy as np

import config
from config import fmt_inr, fmt_lpa
from db import get_db
from ml_model import SalaryModel, train_and_save, validate_input

# ------------------------------------------------------------- terminal UI ---
config.fix_console()
USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else str(text)


def bold(t): return c(t, "1")
def green(t): return c(t, "32")
def red(t): return c(t, "31")
def cyan(t): return c(t, "36")
def dim(t): return c(t, "2")
def yellow(t): return c(t, "33")


WIDTH = 72


def header(title):
    print()
    print(cyan("═" * WIDTH))
    print(bold(title.center(WIDTH)))
    print(cyan("═" * WIDTH))


def section(title):
    print()
    print(bold(title))
    print(dim("─" * WIDTH))


def bar_chart(items, value_fmt=lambda v: f"{v:,.0f}", width=34, label_w=22, highlight=None, mark_text=" ◄ you"):
    """Horizontal ASCII bar chart. items = [(label, value), ...]."""
    if not items:
        return
    vmax = max(abs(v) for _, v in items) or 1
    for label, v in items:
        n = int(round(abs(v) / vmax * width))
        bar = "█" * n if v >= 0 else "░" * n
        line_bar = green(bar) if label == highlight else cyan(bar) if v >= 0 else red(bar)
        mark = yellow(mark_text) if label == highlight else ""
        print(f"  {str(label)[:label_w]:<{label_w}} {line_bar}{' ' * (width - n)} {value_fmt(v)}{mark}")


def histogram(values, bins=10, width=34):
    counts, edges = np.histogram(values, bins=bins)
    cmax = counts.max() or 1
    for cnt, lo, hi in zip(counts, edges[:-1], edges[1:]):
        n = int(round(cnt / cmax * width))
        print(f"  {lo/1e5:6.1f} - {hi/1e5:5.1f} LPA {cyan('█' * n)}{' ' * (width - n)} {cnt}")


def table(rows, cols, widths):
    print("  " + " ".join(bold(f"{h:<{w}}") for h, w in zip(cols, widths)))
    print("  " + " ".join("─" * w for w in widths))
    for r in rows:
        print("  " + " ".join(f"{str(v)[:w]:<{w}}" for v, w in zip(r, widths)))


# ------------------------------------------------------------------ inputs ---
def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    try:
        val = input(f"  {prompt}{suffix}: ").strip()
    except EOFError:
        print()
        sys.exit(0)
    return val if val else (str(default) if default is not None else "")


def ask_number(prompt, lo, hi, cast=float, default=None):
    while True:
        raw = ask(f"{prompt} ({lo}-{hi})", default)
        try:
            v = cast(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(red(f"    ✗ Enter a number between {lo} and {hi}"))


def ask_choice(prompt, options, default=1):
    print(f"  {prompt}:")
    for i, o in enumerate(options, 1):
        print(f"     {i}. {o}")
    while True:
        raw = ask("Choose number", default)
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        print(red(f"    ✗ Enter 1-{len(options)}"))


def ask_skills():
    print("  Skills (enter numbers separated by commas, e.g. 1,3,5 — blank for none):")
    for i, s in enumerate(config.SKILLS, 1):
        print(f"     {i:>2}. {s}")
    while True:
        raw = ask("Skills", "")
        if not raw:
            return []
        try:
            idx = sorted({int(x) for x in raw.replace(" ", "").split(",") if x})
            if all(1 <= i <= len(config.SKILLS) for i in idx):
                return [config.SKILLS[i - 1] for i in idx]
        except ValueError:
            pass
        print(red(f"    ✗ Use numbers 1-{len(config.SKILLS)} separated by commas"))


def collect_employee_details():
    return {
        "experience_years": ask_number("Years of experience", 0, 40, float, 3),
        "education": ask_choice("Highest education", config.EDUCATION_LEVELS, 2),
        "job_role": ask_choice("Job role", config.JOB_ROLES, 1),
        "location": ask_choice("Work location", config.LOCATIONS, 1),
        "company_size": ask_choice("Company type", config.COMPANY_SIZES, 2),
        "certifications": ask_number("Professional certifications", 0, 20, int, 1),
        "performance_rating": ask_number("Performance rating", 1, 5, int, 3),
        "skills": ask_skills(),
    }


# ------------------------------------------------------------------ helpers ---
def require_model():
    model = SalaryModel.load()
    if model is None:
        print(yellow("\n  No trained model found — training one now ..."))
        model = train_and_save(get_db())
    return model


def show_prediction(model, inp, name=""):
    db = get_db()
    res = model.predict(inp)

    header("PREDICTED SALARY" + (f" — {name}" if name else ""))
    print()
    print(f"  {bold('Annual CTC'):<28} {green(bold(fmt_inr(res['predicted'])))}  ({fmt_lpa(res['predicted'])})")
    print(f"  {bold('Monthly (gross)'):<28} {fmt_inr(res['monthly'])}")
    print(f"  {bold('Likely range (80%)'):<28} {fmt_inr(res['lower'])}  –  {fmt_inr(res['upper'])}")
    print(f"  {bold('Model used'):<28} {res['model_name']}  (R² {model.metrics['r2']:.3f})")

    section("Input summary")
    print(f"  {inp['job_role']} · {inp['experience_years']} yrs · {inp['education']} · "
          f"{inp['location']} · {inp['company_size']}")
    print(f"  Certifications: {inp['certifications']}   Rating: {'★' * inp['performance_rating']}"
          f"{'☆' * (5 - inp['performance_rating'])}")
    print(f"  Skills: {', '.join(inp['skills']) or '—'}")

    # Peer comparison (same role in database)
    df = db.load_training_frame()
    peers = df[df.job_role == inp["job_role"]]["salary"]
    if len(peers):
        pct = (peers < res["predicted"]).mean() * 100
        section(f"Compared with {len(peers)} {inp['job_role']}s in the database")
        bar_chart([("Role minimum", peers.min()), ("Role median", peers.median()),
                   ("Your prediction", res["predicted"]), ("Role maximum", peers.max())],
                  value_fmt=fmt_lpa, highlight="Your prediction")
        print(f"\n  You would earn more than {bold(f'{pct:.0f}%')} of {inp['job_role']}s.")

    section("Salary growth projection (same profile, more experience)")
    curve = model.experience_curve(inp, 25)
    pts = [p for p in curve if p["years"] in (0, 2, 4, 6, 8, 10, 12, 15, 20, 25)]
    cur_year = int(round(inp["experience_years"]))
    if cur_year not in [p["years"] for p in pts] and cur_year <= 25:
        pts.append(curve[cur_year])
        pts.sort(key=lambda p: p["years"])
    bar_chart([(f"{p['years']:>2} yrs", p["salary"]) for p in pts], value_fmt=fmt_lpa,
              highlight=f"{cur_year:>2} yrs")

    section("Skill impact — what each skill is worth for this profile")
    impact = model.skill_impact(inp)
    bar_chart([(("✓ " if d["has"] else "+ ") + d["skill"], d["delta"]) for d in impact],
              value_fmt=lambda v: f"{'+' if v >= 0 else ''}{fmt_inr(v)}")
    print(dim("  ✓ = skill you have (current contribution)   + = skill you could add (gain)"))
    best_new = [d for d in impact if not d["has"] and d["delta"] > 0][:3]
    if best_new:
        tips = ", ".join(f"{d['skill']} (+{d['pct']:.1f}%)" for d in best_new)
        print(f"\n  {yellow('Tip:')} learning {tips} gives the biggest raise.")

    section("Education what-if")
    edu = model.category_impact(inp, "education", config.EDUCATION_LEVELS)
    bar_chart([(e["option"], e["salary"]) for e in edu], value_fmt=fmt_lpa, highlight=inp["education"])

    db.save_prediction(inp, res, "cli", name)
    print(dim("\n  ✔ Prediction saved to the predictions table."))
    return res


# ------------------------------------------------------------- menu actions ---
def action_predict():
    header("PREDICT EMPLOYEE SALARY")
    model = require_model()
    name = ask("Employee name (optional)", "")
    inp = validate_input(collect_employee_details())
    show_prediction(model, inp, name)


def action_stats():
    db = get_db()
    df = db.load_training_frame()
    header("DATASET SUMMARY")
    if df.empty:
        print(red("  No employee records. Run: python setup_db.py"))
        return
    s = df.salary
    print(f"  Records: {bold(len(df))}    Average: {bold(fmt_lpa(s.mean()))}    Median: {bold(fmt_lpa(s.median()))}")
    print(f"  Min: {fmt_lpa(s.min())}    Max: {fmt_lpa(s.max())}    "
          f"Avg experience: {df.experience_years.mean():.1f} yrs")

    section("Salary distribution")
    histogram(s, bins=10)

    section("Average salary by job role")
    g = df.groupby("job_role").salary.mean().sort_values(ascending=False)
    bar_chart(list(g.items()), value_fmt=fmt_lpa)

    section("Average salary by education")
    g = df.groupby("education").salary.mean().reindex(config.EDUCATION_LEVELS).dropna()
    bar_chart(list(g.items()), value_fmt=fmt_lpa)

    section("Average salary by experience band")
    bands = [0, 2, 5, 8, 12, 16, 40]
    labels = ["0-2 yrs", "2-5 yrs", "5-8 yrs", "8-12 yrs", "12-16 yrs", "16+ yrs"]
    df["band"] = pd_cut(df.experience_years, bands, labels)
    g = df.groupby("band", observed=True).salary.mean()
    bar_chart(list(g.items()), value_fmt=fmt_lpa)

    section("Skill premium (avg salary with skill vs without)")
    rows = []
    for sk in config.SKILLS:
        col = config.skill_col(sk)
        w, wo = df[df[col] == 1].salary.mean(), df[df[col] == 0].salary.mean()
        rows.append((sk, (w / wo - 1) * 100))
    rows.sort(key=lambda r: r[1], reverse=True)
    bar_chart(rows, value_fmt=lambda v: f"{v:+.1f}%")

    section("Average salary by location and company type")
    bar_chart(list(df.groupby("location").salary.mean().reindex(config.LOCATIONS).items()), value_fmt=fmt_lpa)
    print()
    bar_chart(list(df.groupby("company_size").salary.mean().reindex(config.COMPANY_SIZES).items()),
              value_fmt=fmt_lpa)


def pd_cut(series, bins, labels):
    import pandas as pd
    return pd.cut(series, bins=bins, labels=labels, right=False)


def action_model():
    model = require_model()
    b = model.bundle
    header("MODEL PERFORMANCE")
    print(f"  Selected model : {bold(model.name)}  (best cross-validated R²)")
    print(f"  Trained at     : {b['trained_at']}   on {b['train_rows']} rows, tested on {b['test_rows']} rows")

    section("Model comparison (test set)")
    table([(("★ " if r["model_name"] == model.name else "  ") + r["model_name"], f"{r['r2']:.4f}",
            f"{r['cv_r2']:.4f} ±{r['cv_std']:.3f}", fmt_inr(r["mae"]), fmt_inr(r["rmse"]), f"{r['mape']:.2f}%")
           for r in b["all_results"]],
          ["Model", "R²", "CV R²", "MAE", "RMSE", "MAPE"], [23, 7, 14, 10, 10, 7])
    print()
    bar_chart([(r["model_name"], r["r2"]) for r in b["all_results"]], value_fmt=lambda v: f"{v:.4f}",
              highlight=model.name, mark_text=" ◄ selected")

    section("Feature importance (drop in R² when the feature is shuffled)")
    bar_chart([(d["label"], d["importance"]) for d in b["importance"]], value_fmt=lambda v: f"{v:.4f}")

    section("Sample predictions on unseen test employees")
    pts = b["test_points"][:10]
    table([(p["job_role"], f"{p['experience_years']:.1f}", fmt_inr(p["actual"]), fmt_inr(p["predicted"]),
            f"{(p['predicted'] / p['actual'] - 1) * 100:+.1f}%") for p in pts],
          ["Role", "Exp", "Actual", "Predicted", "Error"], [20, 5, 12, 12, 8])


def action_train():
    header("RETRAIN MODEL")
    model = train_and_save(get_db())
    print(green(f"\n  ✔ Trained. Best model: {model.name}  R² = {model.metrics['r2']:.4f}"))


def action_add_employee():
    header("ADD EMPLOYEE RECORD")
    name = ""
    while not name:
        name = ask("Full name")
    details = validate_input(collect_employee_details())
    salary = ask_number("Actual annual salary in ₹", 50000, 50000000, float)
    emp_id = get_db().insert_employee({**details, "full_name": name, "salary": salary})
    print(green(f"\n  ✔ Employee #{emp_id} saved. Retrain the model (option 4) to learn from it."))


def action_search():
    header("SEARCH EMPLOYEES")
    term = ask("Name contains (blank = all)", "")
    rows = get_db().fetch_employees(search=term, limit=25)
    if not rows:
        print(yellow("  No matching employees."))
        return
    table([(r["employee_id"], r["full_name"], r["job_role"], f"{r['experience_years']:.1f}", r["education"],
            fmt_lpa(r["salary"]), r["skills"] or "") for r in rows],
          ["ID", "Name", "Role", "Exp", "Education", "Salary", "Skills"], [5, 18, 18, 5, 10, 10, 30])
    total = get_db().count_employees(search=term)
    if total > len(rows):
        print(dim(f"\n  showing {len(rows)} of {total}"))


def action_history():
    header("PREDICTION HISTORY")
    rows = get_db().fetch_predictions(20)
    if not rows:
        print(yellow("  No predictions yet."))
        return
    table([(r["prediction_id"], str(r["created_at"])[:16], r["employee_name"] or "—", r["job_role"],
            f"{float(r['experience_years']):.1f}", fmt_lpa(r["predicted_salary"]), r["source"])
           for r in rows],
          ["#", "When", "Name", "Role", "Exp", "Predicted", "Via"], [4, 16, 16, 18, 5, 11, 4])


def action_db_info():
    db = get_db()
    header("DATABASE INFO")
    print(f"  Backend     : {db.description}")
    if getattr(db, "fallback_reason", None):
        print(yellow(f"  MySQL error : {db.fallback_reason}"))
    print(f"  employees   : {db.employee_count()} rows")
    print(f"  predictions : {db.scalar('SELECT COUNT(*) AS n FROM predictions')} rows")
    print(f"  model_runs  : {db.scalar('SELECT COUNT(*) AS n FROM model_runs')} rows")


MENU = [
    ("Predict salary for an employee", action_predict),
    ("Dataset summary & charts", action_stats),
    ("Model performance & feature importance", action_model),
    ("Retrain model from MySQL data", action_train),
    ("Add employee record", action_add_employee),
    ("Search employees", action_search),
    ("Prediction history", action_history),
    ("Database info", action_db_info),
]


def interactive():
    db = get_db()
    if db.employee_count() == 0:
        print(yellow("Database is empty — seeding sample data (one time) ..."))
        from data_generator import generate_employees
        db.bulk_insert_employees(generate_employees())
    while True:
        header("EMPLOYEE SALARY PREDICTION SYSTEM")
        print(dim(f"  Python + MySQL · Regression   |   DB: {db.backend.upper()}").center(WIDTH))
        print()
        for i, (label, _) in enumerate(MENU, 1):
            print(f"   {cyan(i)}. {label}")
        print(f"   {cyan(0)}. Exit")
        choice = ask("\n  Select option", "")
        if choice == "0":
            print("  Goodbye!")
            return
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            try:
                MENU[int(choice) - 1][1]()
            except ValueError as e:
                print(red(f"  ✗ {e}"))
            except KeyboardInterrupt:
                print(yellow("\n  cancelled"))
            ask(dim("\n  Press Enter to return to the menu"), "")
        else:
            print(red("  ✗ Invalid option"))


def main():
    ap = argparse.ArgumentParser(description="Employee Salary Prediction System (CLI)")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("predict", help="predict salary non-interactively")
    p.add_argument("--name", default="")
    p.add_argument("--experience", type=float, required=True)
    p.add_argument("--education", required=True, choices=config.EDUCATION_LEVELS)
    p.add_argument("--role", required=True, choices=config.JOB_ROLES)
    p.add_argument("--location", default="Tier-1 City", choices=config.LOCATIONS)
    p.add_argument("--company", default="Mid-size", choices=config.COMPANY_SIZES)
    p.add_argument("--certifications", type=int, default=0)
    p.add_argument("--rating", type=int, default=3)
    p.add_argument("--skills", default="", help='comma separated, e.g. "Python,SQL"')
    sub.add_parser("stats", help="dataset summary")
    sub.add_parser("model", help="model performance")
    sub.add_parser("train", help="retrain model")
    sub.add_parser("history", help="prediction history")
    args = ap.parse_args()

    try:
        if args.cmd == "predict":
            inp = validate_input({
                "experience_years": args.experience, "education": args.education, "job_role": args.role,
                "location": args.location, "company_size": args.company,
                "certifications": args.certifications, "performance_rating": args.rating,
                "skills": args.skills})
            show_prediction(require_model(), inp, args.name)
        elif args.cmd == "stats":
            action_stats()
        elif args.cmd == "model":
            action_model()
        elif args.cmd == "train":
            action_train()
        elif args.cmd == "history":
            action_history()
        else:
            interactive()
    except ValueError as e:
        print(red(f"Error: {e}"))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    config.fix_console()
    main()
