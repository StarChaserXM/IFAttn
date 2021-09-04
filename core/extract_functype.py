import os
import sys
import glob
import time
from optparse import OptionParser
from tqdm import tqdm
sys.path.append("..")
from module.debug.functype import create_ctags, get_default_type_map
from module.debug.functype import update_known_types, update_type_map
from module.debug.functype import fetch_type
from module.utils import do_multiprocess
from module.utils import load_func_data, store_func_data
from module.utils import load_cache, store_cache
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

def extract_func_types(args):
    type_map, bin_name = args
    bin_name, func_data_list = load_func_data(bin_name)
    for func in func_data_list:
        ret_type = fetch_type(type_map, func["ret_type"])
        arg_types = []
        for idx, var_name, t, _ in func["args"]:
            arg_types.append(fetch_type(type_map, t))
        func["abstract_args_type"] = arg_types
        func["abstract_ret_type"] = ret_type
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
        "--source_list",
        type="str",
        action="store",
        dest="source_list",
        help="a file containing a list of source directories",
    )
    op.add_option(
        "--ctags_dir",
        type="str",
        action="store",
        dest="ctags_dir",
        help="root directory name of the ctags",
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
    op.add_option("--force", action="store_true", dest="force")
    (opts, args) = op.parse_args()

    assert opts.input_list and opts.ctags_dir
    os.makedirs(opts.ctags_dir, exist_ok=True)

    # create ctags
    if opts.source_list:
        logger.info("creating ctags ...")
        t0 = time.time()
        create_ctags(opts.source_list, opts.ctags_dir)
        logger.info("done. (%0.3fs)", time.time() - t0)

    # create type map
    type_map = load_cache(fname="ctags_cache", cache_dir=".module_cache")
    if opts.force or not type_map:
        logger.info("creating type map ...")
        t0 = time.time()
        type_map = get_default_type_map()
        update_known_types(type_map)

        for ctags_fname in glob.glob(os.path.join(opts.ctags_dir, "include*.tags")):
            update_type_map(type_map, ctags_fname)

        for ctags_fname in glob.glob(os.path.join(opts.ctags_dir, "[!include]*.tags")):
            update_type_map(type_map, ctags_fname)

        logger.info("done ... %0.3fs", time.time() - t0)
        store_cache(type_map, fname="ctags_cache", cache_dir=".module_cache")

    with open(opts.input_list, "r") as f:
        bins = f.read().splitlines()

    t0 = time.time()
    logger.info("Processing %d binaries ...", len(bins))
    bins = list(map(lambda x: (type_map, x), bins))
    do_multiprocess(
        extract_func_types, bins, chunk_size=opts.chunk_size, threshold=opts.threshold
    )
    logger.info("done. (%0.3fs)", (time.time() - t0))
