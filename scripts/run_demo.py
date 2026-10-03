from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the team-formation research prototype on synthetic demo data.")
    parser.add_argument("--clean", action="store_true", help="Remove the previous demo output before running.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = root / "outputs" / "demo"
    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(root / "config.demo.json", run_dir / "config.json")
    shutil.copy2(root / "data" / "demo" / "jira_issues.csv", run_dir / "jira_issues.csv")

    script = root / "src" / "team_formation_pipeline.py"
    print(f"Running demo in: {run_dir}")
    completed = subprocess.run([sys.executable, str(script)], cwd=run_dir)
    if completed.returncode == 0:
        print(f"\nDone. Generated artifacts are in: {run_dir}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
