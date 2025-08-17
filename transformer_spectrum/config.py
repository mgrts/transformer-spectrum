from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f'PROJ_ROOT path is: {PROJ_ROOT}')

DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

MODELS_DIR = PROJ_ROOT / 'models'

REPORTS_DIR = PROJ_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)
except ModuleNotFoundError:
    pass

DATA_URL = 'https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv'

TRACKING_URI = PROJ_ROOT / 'mlruns'

SEED = 927

SEQUENCE_LENGTH = 300

N_RUNS = 5

SGT_LOSS_LAMBDAS = [-0.1, -0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01, 0.1]

SYNTHETIC_DATA_CONFIGS = [
    {'lam': 0.0, 'q': 100.0, 'sigma': 0.707, 'experiment_name': 'normal'},
    {'lam': 0.0, 'q': 1.001, 'sigma': 15.0, 'experiment_name': 'heavy-tailed'},
    {'lam': 0.0, 'q': 100.0, 'sigma': 0.707, 'kernel_size': 15, 'experiment_name': 'normal-short-kernel'},
    {'lam': 0.0, 'q': 1.001, 'sigma': 15.0, 'kernel_size': 15, 'experiment_name': 'heavy-tailed-short-kernel'},
]

TRAINING_CONFIGS = [
    {'loss_type': 'sgt', 'sgt_loss_lambda': 0.0, 'sgt_loss_q': 1.001, 'sgt_loss_sigma': 15.0},
    {'loss_type': 'sgt', 'sgt_loss_lambda': 0.0, 'sgt_loss_q': 100.0, 'sgt_loss_sigma': 0.707},
    {'loss_type': 'sgt', 'sgt_loss_lambda': 0.0, 'sgt_loss_q': 5.0, 'sgt_loss_sigma': 0.707},
    {'loss_type': 'sgt', 'sgt_loss_lambda': 0.0, 'sgt_loss_q': 20.0, 'sgt_loss_sigma': 0.707},
    {'loss_type': 'sgt', 'sgt_loss_lambda': 0.0, 'sgt_loss_q': 75.0, 'sgt_loss_sigma': 0.707},
    {'loss_type': 'mse'},
    {'loss_type': 'mae'},
    {'loss_type': 'cauchy'},
]
