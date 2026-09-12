"""Run every query in sql/queries.sql against data/churn.db and print results.

Usage:
    python scripts/run_queries.py
"""
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "churn.db"
SQL_PATH = PROJECT_ROOT / "sql" / "queries.sql"

SEPARATOR = "-- ----------------------------------------------------------------"


def load_statements():
    content = SQL_PATH.read_text()
    blocks = content.split(SEPARATOR)
    statements = []
    for block in blocks:
        lines = [l for l in block.split("\n") if not l.strip().startswith("--") and l.strip()]
        stmt = "\n".join(lines).strip()
        if stmt:
            statements.append(stmt)
    return statements


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found -- run scripts/load_data.py first.")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    statements = load_statements()

    for i, stmt in enumerate(statements, 1):
        print(f"\n{'='*70}")
        print(f"QUERY {i}")
        print('='*70)
        try:
            cur.execute(stmt)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            print(" | ".join(cols))
            print("-" * 70)
            for r in rows[:15]:
                print(" | ".join(str(v) for v in r))
            if len(rows) > 15:
                print(f"... ({len(rows) - 15} more rows)")
        except Exception as e:
            print(f"FAILED: {e}")

    conn.close()


if __name__ == "__main__":
    main()