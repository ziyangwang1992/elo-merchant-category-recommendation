import sys

import pandas as pd
import numpy as np

from data.data_processor import *
from feature.extractor import extract_features
from config.config import *


if __name__ == '__main__':
    # get data
    # gen features
    result = extract_features()
    if not result:
        sys.stderr.write("[FATAL]extract_features failed.\n")
        sys.exit(1)

    # get models