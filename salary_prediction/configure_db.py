"""
Asks for MySQL connection details once, tests them and saves them to
db_settings.ini. START_HERE.bat runs this automatically on first launch.

    python configure_db.py
"""
import configparser
import getpass
import sys

import config


def ask(prompt, default=""):
    try:
        val = input(f"  {prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    except EOFError:
        print()
        sys.exit(1)
    return val or default


def test_mysql(cfg):
    import mysql.connector
    conn = mysql.connector.connect(host=cfg["host"], port=int(cfg["port"]), user=cfg["user"],
                                   password=cfg["password"], connection_timeout=5)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{cfg['database']}`")
    cur.execute("SELECT VERSION()")
    version = cur.fetchone()[0]
    conn.close()
    return version


def save(settings):
    ini = configparser.ConfigParser(interpolation=None)
    ini["database"] = settings
    with open(config.SETTINGS_PATH, "w", encoding="utf-8") as f:
        f.write("; Database settings for the Employee Salary Prediction System\n")
        f.write("; Edit this file or run  python configure_db.py  to change them.\n")
        ini.write(f)
    print(f"\n  Saved to {config.SETTINGS_PATH}")


def main():
    print("=" * 64)
    print("  DATABASE SETUP")
    print("=" * 64)
    print("  This project stores its data in MySQL.")
    print("  Works with MySQL Server, XAMPP or WAMP (start MySQL first).")
    print("  No MySQL? Answer N and it will use a built-in SQLite file instead.\n")

    if ask("Use MySQL? (Y/N)", "Y").lower().startswith("n"):
        save({"backend": "sqlite"})
        return

    cfg = {"host": "localhost", "port": "3306", "user": "root", "password": "",
           "database": "salary_prediction_db"}
    while True:
        cfg["host"] = ask("MySQL host", cfg["host"])
        cfg["port"] = ask("MySQL port", cfg["port"])
        cfg["user"] = ask("MySQL username", cfg["user"])
        try:
            cfg["password"] = getpass.getpass("  MySQL password (typing is hidden; just press Enter if none, e.g. XAMPP): ")
        except EOFError:
            cfg["password"] = ""
        try:
            version = test_mysql(cfg)
            print(f"\n  Connected to MySQL {version}. Database '{cfg['database']}' is ready.")
            save({"backend": "mysql", **cfg})
            return
        except Exception as exc:
            print(f"\n  Could not connect: {exc}")
            print("  Check that MySQL is running and the username/password are right.")
            choice = ask("[R]etry, or use [S]QLite instead", "R").lower()
            if choice.startswith("s"):
                save({"backend": "sqlite"})
                return
            print()


if __name__ == "__main__":
    config.fix_console()
    main()
