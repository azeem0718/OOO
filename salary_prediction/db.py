"""
Database layer: all SQL lives here.

Primary backend is MySQL (mysql-connector-python). If MySQL is not installed or
not reachable, the app automatically falls back to SQLite so it still runs.
Queries are written once with '%s' placeholders and translated for SQLite.
"""
import sqlite3
from contextlib import contextmanager

import pandas as pd

import config

try:
    import mysql.connector
    HAVE_MYSQL_DRIVER = True
except ImportError:  # pragma: no cover
    HAVE_MYSQL_DRIVER = False


MYSQL_DDL = [
    """CREATE TABLE IF NOT EXISTS skills (
        skill_id INT AUTO_INCREMENT PRIMARY KEY,
        skill_name VARCHAR(50) NOT NULL UNIQUE)""",
    """CREATE TABLE IF NOT EXISTS employees (
        employee_id INT AUTO_INCREMENT PRIMARY KEY,
        full_name VARCHAR(100) NOT NULL,
        experience_years DECIMAL(4,1) NOT NULL,
        education VARCHAR(30) NOT NULL,
        job_role VARCHAR(50) NOT NULL,
        location VARCHAR(30) NOT NULL,
        company_size VARCHAR(30) NOT NULL,
        certifications INT NOT NULL DEFAULT 0,
        performance_rating INT NOT NULL DEFAULT 3,
        salary DECIMAL(12,2) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS employee_skills (
        employee_id INT NOT NULL,
        skill_id INT NOT NULL,
        PRIMARY KEY (employee_id, skill_id),
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
        FOREIGN KEY (skill_id) REFERENCES skills(skill_id) ON DELETE CASCADE)""",
    """CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INT AUTO_INCREMENT PRIMARY KEY,
        employee_name VARCHAR(100),
        experience_years DECIMAL(4,1) NOT NULL,
        education VARCHAR(30) NOT NULL,
        job_role VARCHAR(50) NOT NULL,
        location VARCHAR(30) NOT NULL,
        company_size VARCHAR(30) NOT NULL,
        certifications INT NOT NULL,
        performance_rating INT NOT NULL,
        skills VARCHAR(300),
        predicted_salary DECIMAL(12,2) NOT NULL,
        lower_bound DECIMAL(12,2),
        upper_bound DECIMAL(12,2),
        model_name VARCHAR(60),
        source VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS model_runs (
        run_id INT AUTO_INCREMENT PRIMARY KEY,
        model_name VARCHAR(60) NOT NULL,
        r2_score DECIMAL(6,4),
        mae DECIMAL(12,2),
        rmse DECIMAL(12,2),
        cv_r2 DECIMAL(6,4),
        train_rows INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
]

# SQLite flavour of the same schema
SQLITE_DDL = [
    ddl.replace("INT AUTO_INCREMENT PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
    for ddl in MYSQL_DDL
]


class Database:
    def __init__(self):
        self.backend = None
        self._choose_backend()

    # ------------------------------------------------------------ connect ---
    def _choose_backend(self):
        if config.DB_BACKEND == "mysql" and HAVE_MYSQL_DRIVER:
            try:
                cfg = dict(config.MYSQL_CONFIG)
                dbname = cfg.pop("database")
                conn = mysql.connector.connect(connection_timeout=3, **cfg)
                cur = conn.cursor()
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}`")
                conn.commit()
                cur.close()
                conn.close()
                self.backend = "mysql"
                return
            except Exception as exc:  # MySQL down / wrong password / not installed
                self.fallback_reason = str(exc)
        self.backend = "sqlite"

    @property
    def description(self) -> str:
        if self.backend == "mysql":
            c = config.MYSQL_CONFIG
            return f"MySQL  {c['user']}@{c['host']}:{c['port']}/{c['database']}"
        return f"SQLite (fallback)  {config.SQLITE_PATH}"

    @contextmanager
    def connect(self):
        if self.backend == "mysql":
            conn = mysql.connector.connect(**config.MYSQL_CONFIG)
        else:
            conn = sqlite3.connect(config.SQLITE_PATH)
            conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _q(self, sql: str) -> str:
        return sql if self.backend == "mysql" else sql.replace("%s", "?")

    def execute(self, sql, params=()):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(self._q(sql), params)
            last_id = cur.lastrowid
            cur.close()
            return last_id

    def query(self, sql, params=()) -> list:
        """Run a SELECT and return a list of dicts."""
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(self._q(sql), params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            cur.close()
        # MySQL returns Decimal for DECIMAL columns -> convert to float
        for r in rows:
            for k, v in r.items():
                if v.__class__.__name__ == "Decimal":
                    r[k] = float(v)
        return rows

    def scalar(self, sql, params=()):
        rows = self.query(sql, params)
        return list(rows[0].values())[0] if rows else None

    # ------------------------------------------------------------- schema ---
    def init_schema(self):
        ddl = MYSQL_DDL if self.backend == "mysql" else SQLITE_DDL
        with self.connect() as conn:
            cur = conn.cursor()
            for stmt in ddl:
                cur.execute(stmt)
            for skill in config.SKILLS:
                if self.backend == "mysql":
                    cur.execute("INSERT IGNORE INTO skills (skill_name) VALUES (%s)", (skill,))
                else:
                    cur.execute("INSERT OR IGNORE INTO skills (skill_name) VALUES (?)", (skill,))
            cur.close()

    def reset(self):
        with self.connect() as conn:
            cur = conn.cursor()
            for t in ["employee_skills", "predictions", "model_runs", "employees", "skills"]:
                cur.execute(f"DROP TABLE IF EXISTS {t}")
            cur.close()
        self.init_schema()

    # ---------------------------------------------------------- employees ---
    def skill_ids(self) -> dict:
        return {r["skill_name"]: r["skill_id"] for r in self.query("SELECT skill_id, skill_name FROM skills")}

    def employee_count(self) -> int:
        return int(self.scalar("SELECT COUNT(*) AS n FROM employees") or 0)

    def insert_employee(self, emp: dict) -> int:
        ids = self.skill_ids()
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(self._q(
                """INSERT INTO employees (full_name, experience_years, education, job_role,
                       location, company_size, certifications, performance_rating, salary)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""),
                (emp["full_name"], emp["experience_years"], emp["education"], emp["job_role"],
                 emp["location"], emp["company_size"], emp["certifications"],
                 emp["performance_rating"], emp["salary"]))
            emp_id = cur.lastrowid
            rows = [(emp_id, ids[s]) for s in emp.get("skills", []) if s in ids]
            if rows:
                cur.executemany(self._q("INSERT INTO employee_skills (employee_id, skill_id) VALUES (%s,%s)"), rows)
            cur.close()
        return emp_id

    def bulk_insert_employees(self, employees: list):
        """Fast path used for seeding the dataset."""
        ids = self.skill_ids()
        with self.connect() as conn:
            cur = conn.cursor()
            for emp in employees:
                cur.execute(self._q(
                    """INSERT INTO employees (full_name, experience_years, education, job_role,
                           location, company_size, certifications, performance_rating, salary)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""),
                    (emp["full_name"], emp["experience_years"], emp["education"], emp["job_role"],
                     emp["location"], emp["company_size"], emp["certifications"],
                     emp["performance_rating"], emp["salary"]))
                emp_id = cur.lastrowid
                rows = [(emp_id, ids[s]) for s in emp["skills"]]
                if rows:
                    cur.executemany(self._q("INSERT INTO employee_skills (employee_id, skill_id) VALUES (%s,%s)"), rows)
            cur.close()

    def fetch_employees(self, search: str = "", role: str = "", limit: int = None, offset: int = 0) -> list:
        sql = """SELECT e.employee_id, e.full_name, e.experience_years, e.education, e.job_role,
                        e.location, e.company_size, e.certifications, e.performance_rating, e.salary,
                        GROUP_CONCAT(s.skill_name) AS skills
                 FROM employees e
                 LEFT JOIN employee_skills es ON es.employee_id = e.employee_id
                 LEFT JOIN skills s ON s.skill_id = es.skill_id
                 WHERE 1=1"""
        params = []
        if search:
            sql += " AND e.full_name LIKE %s"
            params.append(f"%{search}%")
        if role:
            sql += " AND e.job_role = %s"
            params.append(role)
        sql += """ GROUP BY e.employee_id, e.full_name, e.experience_years, e.education, e.job_role,
                          e.location, e.company_size, e.certifications, e.performance_rating, e.salary
                   ORDER BY e.employee_id DESC"""
        if limit:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        return self.query(sql, params)

    def count_employees(self, search: str = "", role: str = "") -> int:
        sql = "SELECT COUNT(*) AS n FROM employees WHERE 1=1"
        params = []
        if search:
            sql += " AND full_name LIKE %s"
            params.append(f"%{search}%")
        if role:
            sql += " AND job_role = %s"
            params.append(role)
        return int(self.scalar(sql, params) or 0)

    def delete_employee(self, employee_id: int):
        self.execute("DELETE FROM employee_skills WHERE employee_id = %s", (employee_id,))
        self.execute("DELETE FROM employees WHERE employee_id = %s", (employee_id,))

    def load_training_frame(self) -> pd.DataFrame:
        """Employees joined with skills, skills expanded into 0/1 columns."""
        df = pd.DataFrame(self.fetch_employees())
        if df.empty:
            return df
        df["skills"] = df["skills"].fillna("")
        for skill in config.SKILLS:
            df[config.skill_col(skill)] = df["skills"].apply(
                lambda s, sk=skill: int(sk in s.split(",")))
        df["experience_years"] = df["experience_years"].astype(float)
        df["salary"] = df["salary"].astype(float)
        return df

    # -------------------------------------------------------- predictions ---
    def save_prediction(self, inp: dict, result: dict, source: str, name: str = ""):
        return self.execute(
            """INSERT INTO predictions (employee_name, experience_years, education, job_role, location,
                   company_size, certifications, performance_rating, skills, predicted_salary,
                   lower_bound, upper_bound, model_name, source)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (name or None, inp["experience_years"], inp["education"], inp["job_role"], inp["location"],
             inp["company_size"], inp["certifications"], inp["performance_rating"],
             ",".join(inp["skills"]), round(result["predicted"], 2), round(result["lower"], 2),
             round(result["upper"], 2), result["model_name"], source))

    def fetch_predictions(self, limit: int = 100) -> list:
        return self.query(f"SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT {int(limit)}")

    def clear_predictions(self):
        self.execute("DELETE FROM predictions")

    # --------------------------------------------------------- model runs ---
    def log_model_run(self, m: dict):
        self.execute(
            """INSERT INTO model_runs (model_name, r2_score, mae, rmse, cv_r2, train_rows)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (m["model_name"], round(m["r2"], 4), round(m["mae"], 2), round(m["rmse"], 2),
             round(m["cv_r2"], 4), m["train_rows"]))

    def fetch_model_runs(self, limit: int = 20) -> list:
        return self.query(f"SELECT * FROM model_runs ORDER BY run_id DESC LIMIT {int(limit)}")


_db = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
        _db.init_schema()
    return _db
