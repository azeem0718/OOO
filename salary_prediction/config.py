"""
Central configuration for the Employee Salary Prediction System.

Database settings come from db_settings.ini (written by configure_db.py /
START_HERE.bat) and can be overridden with environment variables. By default the
app tries MySQL first; if MySQL is not reachable it falls back to a local SQLite
file so the project can still be demonstrated.

    DB_BACKEND   = mysql | sqlite        (default: mysql)
    MYSQL_HOST   = localhost
    MYSQL_PORT   = 3306
    MYSQL_USER   = root
    MYSQL_PASSWORD = (empty)
    MYSQL_DATABASE = salary_prediction_db
"""
import configparser
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "db_settings.ini")

# ---------------------------------------------------------------- database ---
_ini = configparser.ConfigParser(interpolation=None)
_ini.read(SETTINGS_PATH, encoding="utf-8")


def _setting(env, key, default):
    """Environment variable > db_settings.ini > default."""
    return os.getenv(env) or _ini.get("database", key, fallback=default)


DB_BACKEND = _setting("DB_BACKEND", "backend", "mysql").lower()
MYSQL_CONFIG = {
    "host": _setting("MYSQL_HOST", "host", "localhost"),
    "port": int(_setting("MYSQL_PORT", "port", "3306")),
    "user": _setting("MYSQL_USER", "user", "root"),
    "password": os.getenv("MYSQL_PASSWORD", _ini.get("database", "password", fallback="")),
    "database": _setting("MYSQL_DATABASE", "database", "salary_prediction_db"),
}
SQLITE_PATH = os.path.join(BASE_DIR, "salary_prediction.sqlite3")

# ------------------------------------------------------------------- model ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "salary_model.joblib")
RANDOM_STATE = 42
DATASET_SIZE = 1200   # synthetic employees seeded into the database

# -------------------------------------------------------- domain constants ---
EDUCATION_LEVELS = ["Diploma", "Bachelor's", "Master's", "PhD"]

JOB_ROLES = [
    "Software Developer",
    "Data Analyst",
    "Data Scientist",
    "DevOps Engineer",
    "QA Engineer",
    "Project Manager",
    "HR Executive",
    "Sales Executive",
]

LOCATIONS = ["Tier-1 City", "Tier-2 City", "Tier-3 City"]
COMPANY_SIZES = ["Startup", "Mid-size", "MNC"]

SKILLS = [
    "Python",
    "Java",
    "SQL",
    "JavaScript",
    "Machine Learning",
    "Cloud",
    "Docker",
    "Excel",
    "Communication",
    "Leadership",
]

# Feature columns used by the model (skills become one 0/1 column each)
NUMERIC_FEATURES = ["experience_years", "certifications", "performance_rating"]
CATEGORICAL_FEATURES = ["education", "job_role", "location", "company_size"]


def skill_col(skill: str) -> str:
    """Column name for a skill flag, e.g. 'Machine Learning' -> 'skill_machine_learning'."""
    return "skill_" + skill.lower().replace(" ", "_")


SKILL_FEATURES = [skill_col(s) for s in SKILLS]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + SKILL_FEATURES
TARGET = "salary"


def fmt_inr(amount: float) -> str:
    """Format a rupee amount the Indian way: 1234567 -> '₹12,34,567'."""
    amount = int(round(amount))
    s = str(abs(amount))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return ("-₹" if amount < 0 else "₹") + s


def fmt_lpa(amount: float) -> str:
    """Rupees per annum -> 'x.xx LPA' (lakhs per annum)."""
    return f"{amount / 100000:.2f} LPA"


def fix_console():
    """Make ₹, ★ and box-drawing characters and colours work in the Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if os.name == "nt":
        os.system("")  # enables ANSI colour codes in cmd.exe / PowerShell
