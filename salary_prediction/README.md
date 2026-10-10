# Employee Salary Prediction System

**Python + MySQL + Machine Learning (Regression). Mini project #12.**

The system predicts an employee's **annual salary (CTC in ₹)** from their experience, education,
skills, job role, location, company type, certifications and performance rating.
Employee records live in **MySQL**. Several regression models are trained on that data, and the best one
is used for predictions.

The project comes in **two versions** that share the same database and ML core:

| Version | File | What it offers |
|---|---|---|
| Command line | `cli_app.py` | Menu-driven terminal app with ASCII bar charts and histograms |
| Web UI | `web_app.py` | Flask website with a dashboard, prediction page, model insights, data management and 20+ interactive charts |

---

## 1. Project structure

```
salary_prediction/
├── config.py            # DB settings, feature lists, ₹ formatting helpers
├── schema.sql           # MySQL schema (5 tables)
├── db.py                # All SQL: MySQL with automatic SQLite fallback
├── data_generator.py    # Generates a realistic sample dataset (1,200 employees)
├── ml_model.py          # Training, model comparison, prediction, what-if analysis
├── setup_db.py          # One-time setup: create tables → seed data → train
├── cli_app.py           # VERSION 1: command-line app
├── web_app.py           # VERSION 2: Flask web app
├── templates/           # HTML pages (dashboard, predict, model, employees, history)
├── static/              # CSS, chart helpers, Chart.js (bundled, works offline)
└── requirements.txt
```

## 2. Setup

### Step 1: Install Python packages
```bash
cd salary_prediction
pip install -r requirements.txt
```

### Step 2: Configure MySQL
Install MySQL Server (or XAMPP/MariaDB) and make sure it is running. Then set your credentials
(they default to `root` with an empty password on `localhost:3306`):

**Windows (cmd)**
```cmd
set MYSQL_USER=root
set MYSQL_PASSWORD=yourpassword
```
**Linux / macOS**
```bash
export MYSQL_USER=root
export MYSQL_PASSWORD=yourpassword
```
Optional: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE` (default `salary_prediction_db`).

The database and tables are created automatically. You can also create them by hand with
`mysql -u root -p < schema.sql`.

> If MySQL is not running, the app **falls back to SQLite** automatically so it still works.
> The sidebar and the CLI's *Database info* option show which backend is in use.
> To force SQLite, set `DB_BACKEND=sqlite`.

### Step 3: Create tables, load data, train the model
```bash
python setup_db.py           # first time
python setup_db.py --reset   # wipe everything and start again
```

## 3. Running the two versions

### Version 1: Command line
```bash
python cli_app.py
```
```
   1. Predict salary for an employee
   2. Dataset summary & charts
   3. Model performance & feature importance
   4. Retrain model from MySQL data
   5. Add employee record
   6. Search employees
   7. Prediction history
   8. Database info
   0. Exit
```
Direct commands (no menu):
```bash
python cli_app.py predict --name "Priya" --experience 4 --education "Master's" \
       --role "Data Scientist" --location "Tier-1 City" --company MNC \
       --certifications 2 --rating 4 --skills "Python,SQL,Machine Learning"
python cli_app.py stats      # dataset charts
python cli_app.py model      # model comparison
python cli_app.py train      # retrain
python cli_app.py history    # saved predictions
```

The prediction output shows:
* the predicted CTC, monthly salary and an 80% likely range
* a comparison with others in the same role (min / median / max, percentile)
* projected salary growth over 0–25 years of experience
* what each skill is worth and which skills to learn next
* the effect of a different education level

### Version 2: Web UI
```bash
python web_app.py
```
Open **http://127.0.0.1:5000**

| Page | Visualisations |
|---|---|
| **Dashboard** | KPI tiles, salary-vs-experience scatter with a role filter, salary histogram, experience trend (average and median), salary by role, salary by education, skill premium, location × company grouped bars |
| **Predict Salary** | Input form (slider, chips, star rating). The result shows the predicted CTC with a range bar, percentile, growth projection, where the salary falls in the role's distribution, a scatter against every employee in the role, skill impact, "how to earn more" tips, and what-if charts for education, location, company type and rating |
| **Model Insights** | Comparison table for the 5 models, R² and MAPE charts, actual-vs-predicted scatter, error distribution, permutation feature importance, training history, **Retrain** button |
| **Employee Data** | Search, filter, page through, add and delete MySQL records, export CSV |
| **History** | Every prediction from the CLI and the web, with a chart |

Light and dark themes are supported, and the layout works on phone screens.

## 4. Database design (MySQL)

| Table | Purpose |
|---|---|
| `employees` | Training data: experience, education, role, location, company size, certifications, rating, salary |
| `skills` | Master list of the 10 skills |
| `employee_skills` | Many-to-many link between employees and skills |
| `predictions` | Every prediction made (inputs, predicted salary, range, model, source cli/web) |
| `model_runs` | Metrics logged for every training run |

## 5. Machine learning

* **Problem type:** supervised **regression** (target = annual salary)
* **Features:** experience, certifications and performance rating (numeric, scaled);
  education, job role, location and company size (one-hot encoded); 10 skills (0/1 flags)
* **Target transform:** the model learns `log(salary)`, because pay grows by percentages rather than fixed amounts
* **Models compared:** Linear Regression, Ridge Regression, Polynomial Regression, Random Forest, Gradient Boosting
* **Evaluation:** 80/20 train-test split plus 5-fold cross-validation. Metrics: R², MAE, RMSE, MAPE
* **Selection:** the model with the highest cross-validated R² is saved with `joblib`
* **Prediction range:** an 80% interval taken from the spread of errors on unseen test data
* **Explainability:** permutation feature importance, plus what-if analysis that changes one input at a time

Results on the sample data:

| Model | R² (test) | CV R² | MAPE |
|---|---|---|---|
| **Gradient Boosting** ★ | **0.957** | **0.945** | **8.9%** |
| Polynomial Regression | 0.944 | 0.934 | 9.2% |
| Linear Regression | 0.932 | 0.914 | 10.2% |
| Ridge Regression | 0.932 | 0.913 | 10.2% |
| Random Forest | 0.868 | 0.853 | 14.3% |

## 6. About the dataset
`data_generator.py` creates 1,200 realistic employee records for the Indian job market, with salaries
from about ₹2.4 L to ₹51 L. Salaries depend on role, experience, education, skills, city tier, company
type, certifications and rating, plus random noise. The ML model never sees this formula; it has to learn
the pattern from the MySQL rows.
To use real data, insert it into the `employees` / `employee_skills` tables
(or through *Add employee*), then retrain.
