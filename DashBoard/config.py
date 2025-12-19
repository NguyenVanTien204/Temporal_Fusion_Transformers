from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "test.csv"
RESULTS_DIR = DATA_DIR / "results"

PAGE_TITLE = "Deep Learning Forecast Hub"
PAGE_ICON = ":bar_chart:"
DEFAULT_TOP_N = 15
MAX_TOP_N = 50
