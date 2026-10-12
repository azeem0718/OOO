"""
One-time setup: creates the MySQL tables, seeds the employee dataset and
trains the first model.

    python setup_db.py            # create + seed (if empty) + train
    python setup_db.py --reset    # drop everything and start fresh
"""
import argparse

import config
from data_generator import generate_employees
from db import get_db
from ml_model import train_and_save


def main():
    ap = argparse.ArgumentParser(description="Set up the Employee Salary Prediction database")
    ap.add_argument("--reset", action="store_true", help="drop all tables and re-seed")
    ap.add_argument("--rows", type=int, default=config.DATASET_SIZE, help="number of employees to generate")
    args = ap.parse_args()

    db = get_db()
    print(f"Database : {db.description}")
    if getattr(db, "fallback_reason", None):
        print(f"           (MySQL not reachable: {db.fallback_reason})")

    if args.reset:
        print("Resetting all tables ...")
        db.reset()

    if db.employee_count() == 0:
        print(f"Seeding {args.rows} employee records ...")
        db.bulk_insert_employees(generate_employees(args.rows))
    print(f"Employees in database: {db.employee_count()}")

    print("Training models ...")
    model = train_and_save(db)
    m = model.metrics
    print(f"\nDone. Best model: {model.name}")
    print(f"  R² = {m['r2']:.4f}   MAE = {config.fmt_inr(m['mae'])}   MAPE = {m['mape']:.2f}%")
    print(f"  Saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    config.fix_console()
    main()
