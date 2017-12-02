from os import path

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))

# Directory path
DATASET_DIR = path.join(PROJECT_ROOT, "dataset/")
IMAGE_DIR = path.join(DATASET_DIR, "bin-images/")
METADATA_DIR = path.join(DATASET_DIR, "metadata/")
PARAM_DIR = path.join(DATASET_DIR, "parameters/")

# File path
METADATA_FILE = path.join(DATASET_DIR, "metadata.json")
RAW_METADATA_FILE = path.join(DATASET_DIR, "raw_metadata.json")
ASIN_INDEX_FILE = path.join(DATASET_DIR, "asin_index_dic.json")
INDEX_ASIN_FILE = path.join(DATASET_DIR, "index_asin_dic.json")
VALID_IMAGES_FILE = path.join(DATASET_DIR, "valid_images.json")

TOTAL_DATA_SIZE = 535234
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.02

IMAGE_SIZE = 224
TOTAL_CLASS_SIZE = 459558
CLASS_SIZE = 43053
MINIMUM_REPEAT = 4
MAXIMUM_COUNT = 10
MAXIMUM_IMAGE_NUM = 10000
BALANCE_RATE = 0.09
