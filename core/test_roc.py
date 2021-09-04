import time
import random
import itertools
import os
import sys
import datetime
import numpy as np
import yaml
import copy
from tqdm import tqdm
import pandas as pd

from operator import itemgetter
from optparse import OptionParser
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(sys.path[0], ".."))
from module.utils import do_multiprocess, parse_fname
from module.utils import load_func_data
from module.utils import flatten
from module.utils import store_cache
from module.CodeAttention.DrawPic import DrawROC,DrawRecall_Pre_F1

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)
coloredlogs.install(level=logging.DEBUG)
np.seterr(divide="ignore", invalid="ignore")


def get_package(func_key):
    return func_key[0]


def get_binary(func_key):
    return func_key[1]


def get_func(func_key):
    return func_key[2]


def get_opti(option_key):
    return option_key[0]


def get_arch(option_key):
    return option_key[1]


def get_arch_nobits(option_key):
    return option_key[1].split("_")[0]


def get_bits(option_key):
    return option_key[1].split("_")[1]


def get_compiler(option_key):
    return option_key[2]


def get_others(option_key):
    return option_key[3]


def parse_other_options(bin_path):
    other_options = ["lto", "pie", "noinline"]
    for opt in other_options:
        if opt in bin_path:
            return opt
    return "normal"


def get_optionidx_map(options):
    return {opt: idx for idx, opt in enumerate(sorted(options))}


def is_valid(dictionary, s):
    return s in dictionary and dictionary[s]


def calc_ap(X, y):
    return average_precision_score(y, X)


def calc_roc(X, y):
    fpr, tpr, tresholds = roc_curve(y, X, pos_label=1)
    return auc(fpr, tpr)


def calc_topK(topk_funcs,feature_indices,k_list):
    for func in topk_funcs.keys():
        for opt in topk_funcs[func].keys():
            select_feature = [topk_funcs[func][opt][idx] for idx in feature_indices]
            topk_funcs[func][opt] = np.array(select_feature)
    query = []
    data = []
    y_score = []
    for func in topk_funcs.keys():
        if len(topk_funcs[func]) < 2:
            continue
        else:
            features = list(topk_funcs[func].values())
            query.append(features[0])
            data.append(features[1])
    num_list = list(np.zeros(len(k_list)))
    total = float(len(query))
    score_list = []
    for i in tqdm(range(len(query))):
        q = query[i]
        pred_list = [cosine_similarity([q],[d])[0][0] for d in data]
        pred_dict = {}
        for idx,item in enumerate(pred_list):
            pred_dict[idx] = item
        pred_dict = dict(sorted(pred_dict.items(), key=lambda d: d[1], reverse=True))
        for idx,k in enumerate(k_list):
            pred = list(pred_dict.keys())[:k]
            if i in pred:
                num_list[idx] += 1
    for idx in range(len(k_list)):
        score_list.append(num_list[idx]/total)
    return score_list




def calc_tptn_gap(tps, tns):
    return np.mean(np.abs(tps - tns), axis=0)


def relative_difference(a, b):
    max_val = np.maximum(np.absolute(a), np.absolute(b))
    d = np.absolute(a - b) / max_val
    d[np.isnan(d)] = 0  
    d[np.isinf(d)] = 1  
    return d


def relative_distance(X, feature_indices):
    return 1 - (np.sum(X[:, feature_indices], axis=1)) / len(feature_indices)


def calc_metric_helper(func_key):
    global g_funcs, g_func_keys, g_dst_options
    func_data = g_funcs[func_key]
    option_candidates = list(func_data.keys())
    tp_results = []
    tn_results = []
    target_opts = []
    for src_opt, src_func in func_data.items():
        candidates = []
        for opt in func_data:
            if opt == src_opt:
                continue
            if src_opt not in g_dst_options:
                continue
            if opt not in g_dst_options[src_opt]:
                continue
            candidates.append(opt)
        if not candidates:
            continue
        dst_opt = random.choice(candidates)
        tp_func = func_data[dst_opt]

        while True:
            func_tn_key = random.choice(g_func_keys)
            if get_func(func_tn_key) != get_func(func_key):
                if dst_opt in g_funcs[func_tn_key]:
                    tn_func = g_funcs[func_tn_key][dst_opt]
                    break
        assert not np.isnan(src_func).any()
        assert not np.isnan(tp_func).any()
        assert not np.isnan(tn_func).any()
        tp_results.append(relative_difference(src_func, tp_func))
        tn_results.append(relative_difference(src_func, tn_func))
        target_opts.append((src_opt, dst_opt))
    # merge results into one numpy array
    if tp_results:
        tp_results = np.vstack(tp_results)
    if tn_results:
        tn_results = np.vstack(tn_results)
    return func_key, tp_results, tn_results, target_opts


# inevitably use globals since it is fast.
def _init_calc(funcs, dst_options):
    global g_funcs, g_func_keys, g_dst_options
    g_funcs = funcs
    g_func_keys = sorted(funcs.keys())
    g_dst_options = dst_options


def calc_metric(funcs, dst_options):
    metric_results = do_multiprocess(
        calc_metric_helper,
        funcs.keys(),
        chunk_size=1,
        threshold=1,
        initializer=_init_calc,
        initargs=(funcs, dst_options),
    )
    func_keys, tp_results, tn_results, target_opts = zip(*metric_results)
    tp_results = np.vstack([x for x in tp_results if len(x)])
    tn_results = np.vstack([x for x in tn_results if len(x)])
    assert len(tp_results) == len(tn_results)
    return func_keys, tp_results, tn_results, target_opts


def train(tp_results, tn_results, features):
    max_roc = None
    num_features = len(features)
    selected_feature_indices = []
    for idx in range(num_features):
        tmp_results = {}
        for feature_idx in range(num_features):
            if feature_idx in selected_feature_indices:
                continue
            tmp_feature_indices = selected_feature_indices.copy()
            tmp_feature_indices.append(feature_idx)
            roc, ap = calc_results(tp_results, tn_results, tmp_feature_indices,'','',train=True)
            tmp_results[feature_idx] = (roc, ap)
        feature_idx, (roc, ap) = max(tmp_results.items(), key=itemgetter(1, 0))
        if max_roc and roc < max_roc:
            break
        max_roc = roc
        selected_feature_indices.append(feature_idx)
        logger.debug(
            "%d/%d: %d selected. roc: %0.4f (include %s)",
            idx + 1,
            len(features),
            len(selected_feature_indices),
            roc,
            features[feature_idx],
        )
    return selected_feature_indices

def train_by_topk(train_funcs, features):
    k = [1]
    max_topk = None
    num_features = len(features)
    selected_feature_indices = []
    for idx in range(num_features):
        tmp_results = {}
        for feature_idx in range(num_features):
            if feature_idx in selected_feature_indices:
                continue
            tmp_feature_indices = selected_feature_indices.copy()
            tmp_feature_indices.append(feature_idx)
            topk_list = calc_topK(copy.deepcopy(train_funcs), tmp_feature_indices, k)
            tmp_results[feature_idx] = topk_list[0]
            logger.debug("idx-{}:{}_top-{}:{:.4f}".format(feature_idx,features[feature_idx],k[0],topk_list[0]))
        feature_idx, topk = max(tmp_results.items(), key=itemgetter(1, 0))
        if max_topk and topk < max_topk:
            break
        max_topk = topk
        selected_feature_indices.append(feature_idx)
        logger.debug(
            "%d/%d: %d selected. top-%d: %0.4f (include %s)",
            idx + 1,
            len(features),
            len(selected_feature_indices),
            k[0],
            topk,
            features[feature_idx],
        )
    return selected_feature_indices

def calc_results(tps, tns, feature_indices, outdir, out_curve, train=True):
    feature_indices = np.array(feature_indices)
    num_data = len(tps)
    num_features = len(feature_indices)
    X = np.concatenate(
        [
            relative_distance(tps, feature_indices),
            relative_distance(tns, feature_indices),
        ]
    )
    y = np.concatenate(
        [np.ones(num_data, dtype=np.bool), np.zeros(num_data, dtype=np.bool)]
    )
    if not train:
        os.makedirs(os.path.join(outdir, 'curve'), exist_ok=True)
        DrawROC(y, X, os.path.join(outdir, 'curve'))
        DrawRecall_Pre_F1(y, X, os.path.join(outdir, 'curve'))

        os.makedirs(out_curve, exist_ok=True)
        DrawROC(y, X, out_curve)
        DrawRecall_Pre_F1(y, X, out_curve)
    return calc_roc(X, y), calc_ap(X, y)

def load_options(config):
    options = ["opti", "arch", "compiler", "others"]
    src_options = []
    dst_options = []
    fixed_options = []
    for idx, opt in enumerate(options):
        src_options.append(config["src_options"][opt])
        dst_options.append(config["dst_options"][opt])
        if is_valid(config, "fixed_options") and opt in config["fixed_options"]:
            fixed_options.append(idx)
    src_options = set(itertools.product(*src_options))
    dst_options = set(itertools.product(*dst_options))
    options = sorted(src_options.union(dst_options))
    optionidx_map = get_optionidx_map(options)
    dst_options_filtered = {}
    for src_option in src_options:

        def _check_option(opt):
            if opt == src_option:
                return False
            for idx in fixed_options:
                if opt[idx] != src_option[idx]:
                    return False
            return True

        candidates = list(filter(_check_option, dst_options))

        if "arch_bits" in config["fname"]:

            def _check_arch_without_bits(opt):
                return get_arch_nobits(opt) == get_arch_nobits(src_option)
            candidates = list(filter(_check_arch_without_bits, candidates))
        elif "arch_endian" in config["fname"]:

            def _check_bits(opt):
                return get_bits(opt) == get_bits(src_option)

            candidates = list(filter(_check_bits, candidates))
        candidates = list(set([optionidx_map[opt] for opt in candidates]))
        dst_options_filtered[optionidx_map[src_option]] = candidates
    logger.info("total %d options.", len(options))
    logger.info("%d src options.", len(src_options))
    logger.info("%d dst options.", len(dst_options))
    logger.info("%d filtered dst options.", len(dst_options_filtered))
    return options, dst_options_filtered


def group_binaries(input_list):
    with open(input_list, "r") as f:
        bin_paths = f.read().splitlines()
    bins = {}
    packages = set()
    for bin_path in bin_paths:
        package, compiler, arch, opti, bin_name = parse_fname(bin_path)
        others = parse_other_options(bin_path)
        key = (package, bin_name)
        if key not in bins:
            bins[key] = []
        bins[key].append(bin_path)
        packages.add(package)
    logger.info(
        "%d packages, %d unique binaries, total %d binaries",
        len(packages),
        len(bins),
        len(bin_paths),
    )
    return bins, packages


def load_func_features_helper(bin_paths):
    global g_options, g_features
    func_features = {}
    num_features = len(g_features)
    optionidx_map = get_optionidx_map(g_options)
    for bin_path in bin_paths:
        package, compiler, arch, opti, bin_name = parse_fname(bin_path)
        others = parse_other_options(bin_path)
        _, func_data_list = load_func_data(bin_path)
        try:
            for func_data in func_data_list:
                if func_data["seg_name"] != ".text":
                    continue
                if func_data["name"].startswith("sub_"):
                    continue
                func_key = (package, bin_name, func_data["name"])
                option_key = (opti, arch, compiler, others)
                if option_key not in optionidx_map:
                    continue
                option_idx = optionidx_map[option_key]
                if func_key not in func_features:
                    func_features[func_key] = {}
                if option_key not in func_features[func_key]:
                    func_features[func_key][option_idx] = np.zeros(
                        num_features, dtype=np.float64
                    )
                for feature_idx, feature in enumerate(g_features):
                    if feature not in func_data["feature"]:
                        continue
                    val = func_data["feature"][feature]
                    func_features[func_key][option_idx][feature_idx] = val
        except Exception as e:
            print(bin_path)
            print(e)

    return func_features

def _init_load(options, features):
    global g_options, g_features
    g_options = options
    g_features = features

def load_func_features(input_list, options, features):
    grouped_bins, packages = group_binaries(input_list)
    func_features_list = do_multiprocess(
        load_func_features_helper,
        grouped_bins.values(),
        chunk_size=1,
        threshold=1,
        initializer=_init_load,
        initargs=(options, features),
    )
    funcs = {}
    for func_features in func_features_list:
        funcs.update(func_features)
    return funcs

def save_result_csv(save_dict,outdir):
    os.makedirs(outdir, exist_ok=True)
    dataframe = pd.DataFrame(save_dict)
    dataframe.to_csv(os.path.join(outdir, "result.csv"), index=False, sep=',')
    print('save result data csv')

def do_test(opts):
    binary_list = []
    options_list = []
    ROC_list = []
    AP_list = []
    TOP1_list = []
    TOP5_list = []
    train_pairs_list = []
    test_pairs_list = []
    train_funcs_list = []
    test_funcs_list  =[]
    config_fname = opts.config
    config_folder = "/XXX/config/gnu_type/"
    config_fname_list = [
        config_folder + "config_gnu_normal_all_type.yml",
        config_folder + "config_gnu_normal_arch_all_type.yml",
        config_folder + "config_gnu_normal_arch_arm_mips_type.yml",
        config_folder + "config_gnu_normal_arch_x86_arm_type.yml",
        config_folder + "config_gnu_normal_arch_x86_mips_type.yml",
        config_folder + "config_gnu_normal_obfus_all_type.yml",
        config_folder + "config_gnu_normal_obfus_bcf_type.yml",
        config_folder + "config_gnu_normal_obfus_fla_type.yml",
        config_folder + "config_gnu_normal_obfus_sub_type.yml",
        config_folder + "config_gnu_normal_opti_O0-O3_type.yml",
        config_folder + "config_gnu_normal_opti_O0toO3_type.yml",
        config_folder + "config_gnu_normal_opti_O1-O2_type.yml",
        config_folder + "config_gnu_normal_opti_O2-O3_type.yml",
    ]

    for config_fname in tqdm(config_fname_list):
        with open(config_fname, "r") as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        config["fname"] = config_fname

        if "outdir" in config and config["outdir"]:
            outdir = config["outdir"]
        else:
            base_name = os.path.splitext(os.path.basename(config_fname))[0]
            outdir = os.path.join(opts.log_out, base_name)
        out_curve = os.path.join(outdir, 'curve')
        date = datetime.datetime.now()
        outdir = os.path.join(outdir, str(date).replace(':','-'))
        os.makedirs(outdir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(outdir, "log.txt"))
        logger.addHandler(file_handler)
        logger.info("input file name: %s", opts.input_list)
        logger.info("config file name: %s", config["fname"])
        logger.info("output directory: %s", outdir)

        options, dst_options = load_options(config)
        features = sorted(config["features"])
        logger.info("%d features", len(features))

        t0 = time.time()
        logger.info("Feature loading ...")
        funcs = load_func_features(opts.input_list, options, features)
        logger.info(
            "%d functions (%d unique).", sum([len(x) for x in funcs.values()]), len(funcs)
        )
        logger.info("Feature loading done. (%0.3fs)", time.time() - t0)



        num_folds = 5
        k_list = [1,5]
        kf = KFold(n_splits=num_folds)
        assert len(funcs) > num_folds

        # ===============================================
        # start test
        # ===============================================
        func_keys = sorted(funcs.keys())
        # shuffle for 10-fold test ===============
        if config["debug"]:
            random.seed(config["seed"])
        random.shuffle(func_keys)

        data_all = []
        binary_list.append(opts.input_list.split('done_list_')[1].split('.elf.txt')[0])
        options_list.append(config_fname.split('config_')[1].split('.yml')[0])

        for trial_idx, (train_func_keys, test_func_keys) in enumerate(kf.split(func_keys)):
            logger.info("[+] TRIAL %d/%d ================", trial_idx + 1, num_folds)

            train_func_keys = [func_keys[i] for i in train_func_keys]
            train_funcs = {key: funcs[key] for key in train_func_keys}
            test_func_keys = [func_keys[i] for i in test_func_keys]
            test_funcs = {key: funcs[key] for key in test_func_keys}
            logger.info(
                "Train: %d unique funcs, Test: %d unique funcs",
                len(train_func_keys),
                len(test_func_keys),
            )

            # ===================== training ======================
            t0 = time.time()
            logger.info("selecting train funcs ...")
            train_keys, train_tps, train_tns, train_opts = calc_metric(
                train_funcs, dst_options
            )
            logger.info("selecting train funcs done. (%0.3fs)", time.time() - t0)

            logger.info("selecting features ...")
            t0 = time.time()
            if config["do_train"]:
                # start training to select features
                selected_feature_indices = train(train_tps, train_tns, features) # train_by_roc
                # selected_feature_indices = train_by_topk(train_funcs,features) # train_by_topk
            else:
                # use given pre-defined featureset by commandline option
                logger.info("using pre-defined features ...")
                selected_feature_indices = list(range(features))
            train_time = time.time() - t0

            selected_features = [features[x] for x in selected_feature_indices]
            train_roc, train_ap = calc_results(
                train_tps, train_tns, selected_feature_indices,outdir,out_curve,train=True)
            logger.info(
                "selected %d features. roc: %0.4f, ap: %0.4f",
                len(selected_features),
                train_roc,
                train_ap,
            )
            logger.info("selecting features done. (%0.3fs)", train_time)

            # ===================== testing ======================
            t0 = time.time()
            logger.info("testing ...")
            test_keys, test_tps, test_tns, test_opts = calc_metric(test_funcs, dst_options)
            test_roc, test_ap = calc_results(test_tps, test_tns, selected_feature_indices,outdir,out_curve,train=False)

            test_topk_list = calc_topK(copy.deepcopy(test_funcs),selected_feature_indices,k_list)
            test_time = time.time() - t0
            for idx,k in enumerate(k_list):
                logger.info(
                    "test results: %d features, roc: %0.4f, ap: %0.4f top-%d: %0.4f",
                    len(selected_features),
                    test_roc,
                    test_ap,
                    k,
                    test_topk_list[idx],
                )
            logger.info("testing done. (%0.3fs)", test_time)

            data = [
                train_func_keys,
                train_tps,
                train_tns,
                train_opts,
                train_roc,
                train_ap,
                train_time,
                test_func_keys,
                test_tps,
                test_tns,
                test_opts,
                test_roc,
                test_ap,
                test_time,
                features,
                selected_feature_indices,
                test_topk_list,
            ]
            data_all.append(data)

        avg_roc,avg_ap,avg_top1,avg_top5,train_pairs,test_pairs,train_funcs,test_funcs = analyze_results(data_all,k_list)
        ROC_list.append(avg_roc)
        AP_list.append(avg_ap)
        TOP1_list.append(avg_top1)
        TOP5_list.append(avg_top5)
        train_pairs_list.append(train_pairs)
        test_pairs_list.append(test_pairs)
        train_funcs_list.append(train_funcs)
        test_funcs_list.append(test_funcs)
        logger.removeHandler(file_handler)
    result_dict = {'binary': binary_list,
                   'options': options_list,
                   'ROC': ROC_list,
                   'AP': AP_list,
                   'TOP1': TOP1_list,
                   'TOP5': TOP5_list,
                   'train_pairs':train_pairs_list,
                   'test_pairs':test_pairs_list,
                   'train_funcs':train_funcs_list,
                   'test_funcs':test_funcs_list}
    date = datetime.datetime.now()
    savedir = os.path.join(opts.log_out, 'result_csv', str(date).replace(':', '-').replace(' ', '-'))
    save_result_csv(result_dict, savedir)


def analyze_results(data_all,k_list):
    rocs = []
    aps = []
    topks_list = [[] for _ in range(len(k_list))]
    train_times = []
    test_times = []
    num_train_pairs = []
    num_test_pairs = []
    num_train_funcs = []
    num_test_funcs = []
    num_features = []
    features_inter = set()
    features_union = set()
    tptn_gaps = []
    for data in data_all:
        feature_indices = data[15]
        if not features_inter:
            features_inter = set(feature_indices)
        else:
            features_inter.intersection_update(feature_indices)
        features_union.update(feature_indices)

    for data in data_all:
        train_func_keys, train_tps, train_tns = data[:3]
        train_roc, train_ap, train_time = data[4:7]
        test_func_keys, test_tps, test_tns = data[7:10]
        test_roc, test_ap, test_time = data[11:14]
        features, feature_indices = data[14:16]
        test_topk_list = data[-1]

        rocs.append(test_roc)
        aps.append(test_ap)
        train_times.append(train_time)
        test_times.append(test_time)
        num_train_pairs.append(len(train_tps) + len(train_tns))
        num_test_pairs.append(len(test_tps) + len(test_tns))
        num_train_funcs.append(len(train_func_keys))
        num_test_funcs.append(len(test_func_keys))
        num_features.append(len(feature_indices))
        for idx,k in enumerate(k_list):
            topks_list[idx].append(test_topk_list[idx])

        tptn_gap = calc_tptn_gap(test_tps, test_tns)
        tptn_gaps.append(tptn_gap)
    tptn_gap = np.mean(tptn_gaps, axis=0)

    logger.info("Features: ")
    for idx in features_union:
        if idx in features_inter:
            logger.info("%s (inter): %.4f", features[idx], tptn_gap[idx])
        else:
            logger.info("%s: %.4f", features[idx], tptn_gap[idx])
    logger.info("Avg # of selected features: %.4f", np.mean(num_features))
    logger.info("Avg. TP-TN Gap: %0.4f", np.mean(tptn_gap))
    logger.info(
        "Avg. TP-TN Gap of Grey: %0.4f", np.mean(tptn_gap[list(features_inter)])
    )
    logger.info("Avg. ROC: %0.4f", np.mean(rocs))
    logger.info("Std. of ROC: %0.4f", np.std(rocs))
    logger.info("Avg. AP: %0.4f", np.mean(aps))
    logger.info("Std. of AP: %0.4f", np.std(aps))
    for idx,k in enumerate(k_list):
        logger.info("Avg. Top-%d: %0.4f", k,np.mean(topks_list[idx]))
        logger.info("Std. of Top-%d: %0.4f", k,np.std(topks_list[idx]))
    logger.info("Avg. Train time: %0.4f", np.mean(train_times))
    logger.info("AVg. Test time: %0.4f", np.mean(test_times))
    logger.info("Avg. # of Train Pairs: %d", np.mean(num_train_pairs))
    logger.info("Avg. # of Test Pairs: %d", np.mean(num_test_pairs))
    logger.info("Avg. # of Train Funcs: %d", np.mean(num_train_funcs))
    logger.info("Avg. # of Test Funcs: %d", np.mean(num_test_funcs))
    return np.mean(rocs),np.mean(aps),np.mean(topks_list[0]),np.mean(topks_list[1]),np.mean(num_train_pairs),np.mean(num_test_pairs),np.mean(num_train_funcs),np.mean(num_test_funcs)


if __name__ == "__main__":
    op = OptionParser()
    op.add_option(
        "--config",
        action="store",
        dest="config",
        help="give config file (ex) config/config_default.yml",
    )
    op.add_option(
        "--type",
        action="store_true",
        dest="type",
        help="test type features after loading features",
    )
    op.add_option(
        "--input_list",
        type="str",
        action="store",
        dest="input_list",
        help="a file containing a list of input binaries",
    )

    op.add_option('--log_out', type=str, default="results/openssl")

    (opts, args) = op.parse_args()

    if not opts.config:
        op.print_help()
        exit(1)
    do_test(opts)
