from os import path

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))

DATASET_DIR = path.join(PROJECT_ROOT, "dataset/")
PARAM_DIR = path.join(DATASET_DIR, "parameters/")

IMAGE_SIZE = 28

