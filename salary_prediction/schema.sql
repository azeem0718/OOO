-- ============================================================
--  Employee Salary Prediction System  -  MySQL schema
--  Run manually with:  mysql -u root -p < schema.sql
--  (setup_db.py creates the same tables automatically)
-- ============================================================

CREATE DATABASE IF NOT EXISTS salary_prediction_db;
USE salary_prediction_db;

-- Master list of skills
CREATE TABLE IF NOT EXISTS skills (
    skill_id    INT AUTO_INCREMENT PRIMARY KEY,
    skill_name  VARCHAR(50) NOT NULL UNIQUE
);

-- Historical employee records (training data)
CREATE TABLE IF NOT EXISTS employees (
    employee_id         INT AUTO_INCREMENT PRIMARY KEY,
    full_name           VARCHAR(100) NOT NULL,
    experience_years    DECIMAL(4,1) NOT NULL,
    education           VARCHAR(30)  NOT NULL,
    job_role            VARCHAR(50)  NOT NULL,
    location            VARCHAR(30)  NOT NULL,
    company_size        VARCHAR(30)  NOT NULL,
    certifications      INT          NOT NULL DEFAULT 0,
    performance_rating  INT          NOT NULL DEFAULT 3,
    salary              DECIMAL(12,2) NOT NULL,          -- annual CTC in INR
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Many-to-many: which employee has which skill
CREATE TABLE IF NOT EXISTS employee_skills (
    employee_id INT NOT NULL,
    skill_id    INT NOT NULL,
    PRIMARY KEY (employee_id, skill_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id)    REFERENCES skills(skill_id)       ON DELETE CASCADE
);

-- Every prediction made from the CLI or the web UI
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       INT AUTO_INCREMENT PRIMARY KEY,
    employee_name       VARCHAR(100),
    experience_years    DECIMAL(4,1) NOT NULL,
    education           VARCHAR(30)  NOT NULL,
    job_role            VARCHAR(50)  NOT NULL,
    location            VARCHAR(30)  NOT NULL,
    company_size        VARCHAR(30)  NOT NULL,
    certifications      INT          NOT NULL,
    performance_rating  INT          NOT NULL,
    skills              VARCHAR(300),
    predicted_salary    DECIMAL(12,2) NOT NULL,
    lower_bound         DECIMAL(12,2),
    upper_bound         DECIMAL(12,2),
    model_name          VARCHAR(60),
    source              VARCHAR(10),                     -- 'cli' or 'web'
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Log of every training run and its evaluation metrics
CREATE TABLE IF NOT EXISTS model_runs (
    run_id      INT AUTO_INCREMENT PRIMARY KEY,
    model_name  VARCHAR(60) NOT NULL,
    r2_score    DECIMAL(6,4),
    mae         DECIMAL(12,2),
    rmse        DECIMAL(12,2),
    cv_r2       DECIMAL(6,4),
    train_rows  INT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
