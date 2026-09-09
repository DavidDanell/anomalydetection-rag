import yaml
from pathlib import Path
import pandas as pd
import numpy as np



#generates a basline, no errors

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / 'config.yaml', encoding='UTF-8') as f:
        return yaml.safe_load(f)


def generate_baseline()