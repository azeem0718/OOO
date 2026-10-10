"""
Generates a realistic synthetic employee dataset (Indian IT/corporate market,
annual CTC in INR) so the project works without any external CSV.

The salary is built from known effects (role, experience, education, skills,
location, company size, certifications, rating) plus random noise. The ML model
does not see this formula; it has to learn it back from the rows in MySQL.
"""
import numpy as np

import config

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan", "Rohan",
    "Ananya", "Diya", "Aadhya", "Saanvi", "Pari", "Ira", "Meera", "Kavya", "Sneha", "Priya",
    "Rahul", "Karthik", "Nikhil", "Varun", "Manoj", "Suresh", "Harsha", "Teja", "Ravi", "Kiran",
    "Lakshmi", "Divya", "Pooja", "Swathi", "Keerthi", "Bhavana", "Nandini", "Harini", "Shreya", "Anjali",
    "Imran", "Faisal", "Ayesha", "Fatima", "John", "Mary", "David", "Sarah", "Gurpreet", "Simran",
]
LAST_NAMES = [
    "Sharma", "Verma", "Reddy", "Naidu", "Rao", "Iyer", "Nair", "Menon", "Patel", "Shah",
    "Gupta", "Singh", "Kumar", "Das", "Bose", "Mukherjee", "Khan", "Joshi", "Kulkarni", "Desai",
    "Chowdary", "Varma", "Pillai", "Fernandes", "Mehta", "Agarwal", "Mishra", "Pandey", "Yadav", "Goud",
]

# Base annual CTC (INR) for a fresher with a Bachelor's degree
ROLE_BASE = {
    "Software Developer": 420000,
    "Data Analyst": 380000,
    "Data Scientist": 600000,
    "DevOps Engineer": 480000,
    "QA Engineer": 340000,
    "Project Manager": 650000,   # usually hired with experience, but base is higher
    "HR Executive": 300000,
    "Sales Executive": 280000,
}

# How strongly each role's pay grows per year of experience
ROLE_GROWTH = {
    "Software Developer": 0.13, "Data Analyst": 0.11, "Data Scientist": 0.14,
    "DevOps Engineer": 0.13, "QA Engineer": 0.10, "Project Manager": 0.10,
    "HR Executive": 0.08, "Sales Executive": 0.09,
}

# Probability that someone in a role has a given skill
ROLE_SKILL_PROB = {
    "Software Developer": {"Python": .55, "Java": .6, "SQL": .6, "JavaScript": .6, "Machine Learning": .12,
                           "Cloud": .35, "Docker": .35, "Excel": .2, "Communication": .45, "Leadership": .15},
    "Data Analyst": {"Python": .6, "Java": .1, "SQL": .85, "JavaScript": .1, "Machine Learning": .25,
                     "Cloud": .2, "Docker": .08, "Excel": .85, "Communication": .55, "Leadership": .12},
    "Data Scientist": {"Python": .95, "Java": .15, "SQL": .75, "JavaScript": .1, "Machine Learning": .85,
                       "Cloud": .45, "Docker": .3, "Excel": .4, "Communication": .5, "Leadership": .15},
    "DevOps Engineer": {"Python": .55, "Java": .25, "SQL": .35, "JavaScript": .2, "Machine Learning": .05,
                        "Cloud": .9, "Docker": .9, "Excel": .15, "Communication": .4, "Leadership": .15},
    "QA Engineer": {"Python": .45, "Java": .55, "SQL": .55, "JavaScript": .35, "Machine Learning": .05,
                    "Cloud": .15, "Docker": .2, "Excel": .45, "Communication": .5, "Leadership": .1},
    "Project Manager": {"Python": .15, "Java": .2, "SQL": .3, "JavaScript": .1, "Machine Learning": .1,
                        "Cloud": .3, "Docker": .05, "Excel": .8, "Communication": .9, "Leadership": .85},
    "HR Executive": {"Python": .02, "Java": .01, "SQL": .08, "JavaScript": .01, "Machine Learning": .01,
                     "Cloud": .02, "Docker": .01, "Excel": .8, "Communication": .9, "Leadership": .35},
    "Sales Executive": {"Python": .02, "Java": .01, "SQL": .08, "JavaScript": .01, "Machine Learning": .01,
                        "Cloud": .05, "Docker": .01, "Excel": .65, "Communication": .95, "Leadership": .35},
}

# Salary premium for having a skill (multiplicative)
SKILL_PREMIUM = {
    "Python": 0.06, "Java": 0.05, "SQL": 0.04, "JavaScript": 0.04, "Machine Learning": 0.14,
    "Cloud": 0.10, "Docker": 0.06, "Excel": 0.02, "Communication": 0.04, "Leadership": 0.03,
}

EDU_MULT = {"Diploma": 0.82, "Bachelor's": 1.00, "Master's": 1.18, "PhD": 1.38}
LOC_MULT = {"Tier-1 City": 1.18, "Tier-2 City": 1.00, "Tier-3 City": 0.84}
SIZE_MULT = {"Startup": 0.94, "Mid-size": 1.00, "MNC": 1.17}


def salary_formula(exp, education, role, location, size, certs, rating, skills, rng=None):
    """Ground-truth salary used only for generating data."""
    g = ROLE_GROWTH[role]
    exp_mult = 1 + g * exp + 0.0025 * exp ** 2 - 0.00012 * exp ** 3  # fast early, flattening late
    salary = ROLE_BASE[role] * exp_mult * EDU_MULT[education] * LOC_MULT[location] * SIZE_MULT[size]
    for s in skills:
        prem = SKILL_PREMIUM[s]
        if s == "Leadership":
            prem += 0.012 * min(exp, 15)      # leadership pays more as you get senior
        salary *= 1 + prem
    salary *= 1 + 0.025 * certs
    salary *= 1 + 0.045 * (rating - 3)
    if rng is not None:
        salary *= rng.lognormal(mean=0.0, sigma=0.09)
    return float(salary)


def generate_employees(n: int = config.DATASET_SIZE, seed: int = config.RANDOM_STATE) -> list:
    rng = np.random.default_rng(seed)
    role_weights = np.array([0.26, 0.13, 0.10, 0.10, 0.12, 0.09, 0.10, 0.10])
    employees = []
    for _ in range(n):
        role = rng.choice(config.JOB_ROLES, p=role_weights / role_weights.sum())

        if role == "Project Manager":
            exp = float(np.clip(rng.normal(11, 4), 4, 28))
        else:
            exp = float(np.clip(rng.gamma(2.0, 3.0), 0, 28))
        exp = round(exp, 1)

        edu_p = [0.08, 0.60, 0.29, 0.03]
        if role == "Data Scientist":
            edu_p = [0.0, 0.35, 0.52, 0.13]
        education = rng.choice(config.EDUCATION_LEVELS, p=edu_p)

        location = rng.choice(config.LOCATIONS, p=[0.5, 0.33, 0.17])
        size = rng.choice(config.COMPANY_SIZES, p=[0.28, 0.37, 0.35])
        certs = int(min(rng.poisson(0.6 + exp / 8), 6))
        rating = int(rng.choice([1, 2, 3, 4, 5], p=[0.04, 0.14, 0.42, 0.30, 0.10]))

        skills = [s for s, p in ROLE_SKILL_PROB[role].items() if rng.random() < p]
        if not skills:
            skills = [max(ROLE_SKILL_PROB[role], key=ROLE_SKILL_PROB[role].get)]

        salary = round(salary_formula(exp, education, role, location, size, certs, rating, skills, rng), -2)
        employees.append({
            "full_name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "experience_years": exp,
            "education": str(education),
            "job_role": str(role),
            "location": str(location),
            "company_size": str(size),
            "certifications": certs,
            "performance_rating": rating,
            "salary": salary,
            "skills": skills,
        })
    return employees
