from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DOCS_DIR = PROJECT_ROOT / "docs"
SQL_DIR = PROJECT_ROOT / "sql"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"


def ensure_directories() -> None:
    """Create standard project directories if they do not exist."""
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
