import os
import sys
import time
from optparse import OptionParser
from tqdm import tqdm
sys.path.append("..")
from module.feature import FeatureManager
from module.utils import do_multiprocess
from module.utils import load_func_data, store_func_data
import logging
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

def extract_features(bin_name):
    global feature_funcs
    bin_name, func_data_list = load_func_data(bin_name)
    fm = FeatureManager()
    for func_data in func_data_list:
        features = fm.get_all(func_data)
        func_data["feature"] = features
    store_func_data(bin_name, func_data_list)

if __name__ == "__main__":
    op = OptionParser()
    op.add_option(
        "--input_list",
        type="str",
        action="store",
        dest="input_list",
        help="a file containing a list of input binaries",
    )
    op.add_option(
        "--threshold",
        type="int",
        action="store",
        dest="threshold",
        default=1,
        help="number of binaries to process in parallel",
    )
    op.add_option(
        "--chunk_size",
        type="int",
        action="store",
        dest="chunk_size",
        default=1,
        help="number of binaries to process in each process",
    )
    op.add_option("--debug", action="store_true", dest="debug")
    (opts, args) = op.parse_args()

    assert opts.input_list and os.path.isfile(opts.input_list)
    with open(opts.input_list, "r") as f:
        bins = f.read().splitlines()
    if opts.debug:
        bins = [bins[0]]
    t0 = time.time()
    logger.info("Processing %d binaries ...", len(bins))
    do_multiprocess(
        extract_features, bins, chunk_size=opts.chunk_size, threshold=opts.threshold
    )
    logger.info("done. (%0.3fs)", (time.time() - t0))
