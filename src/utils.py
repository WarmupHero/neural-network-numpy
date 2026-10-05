import os

# Calculate paths based on where utils.py is located
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

REPORT_DIR = os.path.join(ROOT_DIR, "report")
PREPROCESSING_GRAPHS_DIR = os.path.join(REPORT_DIR, "preprocessing_graphs")

# Global Configurations
RANDOM_SEED = 42