from os import path

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))

# Directory path
DATASET_DIR = path.join(PROJECT_ROOT, "dataset/")
IMAGE_DIR = path.join(DATASET_DIR, "bin-images/")
METADATA_DIR = path.join(DATASET_DIR, "metadata/")

# File path
METADATA_FILE = path.join(DATASET_DIR, "metadata.json")
RAW_METADATA_FILE = path.join(DATASET_DIR, "raw_metadata.json")
ASIN_INDEX_FILE = path.join(DATASET_DIR, "asin_index_dic.json")
INDEX_ASIN_FILE = path.join(DATASET_DIR, "index_asin_dic.json")

TOTAL_DATA_SIZE = 535234
IMAGE_SIZE = 224
