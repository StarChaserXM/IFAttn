# coding:utf-8
import time
import random
import itertools

# import gc
import os
import sys
import datetime
import numpy as np
import pandas as pd
import yaml

from tqdm import tqdm
import heapq

from operator import itemgetter
from optparse import OptionParser
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(sys.path[0], ".."))
from tiknib.utils import do_multiprocess, parse_fname
from tiknib.utils import load_func_data
from tiknib.utils import flatten
from tiknib.utils import store_cache
from tiknib.CodeAttention.datasets import CreateDataLoader
from tiknib.CodeAttention.model import SiameseAttentionNet,MyLoss
from tiknib.CodeAttention.Optim import ScheduledOptim
from tiknib.CodeAttention.DrawPic import DrawROC,DrawRecall_Pre_F1
from torch import optim
import torch
import torch.nn.functional as F

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

def get_optionidx_map_re(options):
    return {idx: opt for idx, opt in enumerate(sorted(options))}


def is_valid(dictionary, s):
    return s in dictionary and dictionary[s]


def calc_ap(X, y):
    return average_precision_score(y, X)


def calc_roc(X, y):
    fpr, tpr, tresholds = roc_curve(y, X, pos_label=1)
    return auc(fpr, tpr)


def calc_tptn_gap(tps, tns):
    return np.mean(np.abs(tps - tns), axis=0)


def relative_difference(a, b):
    max_val = np.maximum(np.absolute(a), np.absolute(b))
    d = np.absolute(a - b) / max_val
    d[np.isnan(d)] = 0  # 0 / 0 = nan -> 0
    d[np.isinf(d)] = 1  # x / 0 = inf -> 1 (when x != 0)
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
    # Testing all functions takes too much time, so we select one true
    # positive and one true negative function for each function.
    for src_opt, src_func in func_data.items():
        # select one tp function.
        ## below random.choice may work faster than list filtering.
        # while True:
        #    dst_opt = random.choice(option_candidates)
        #    if dst_opt != src_opt:
        #        if dst_opt in g_dst_options[src_opt]:
        #            break
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

        # select one tn function
        while True:
            func_tn_key = random.choice(g_func_keys)
            # Since difference binaries may have an equal function, pick a
            # function having a different name for precise comparison
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
    # now select for features. this find local optimum value using hill
    # climbing.
    metric_results = do_multiprocess(
        calc_metric_helper,
        funcs.keys(),
        chunk_size=1,
        threshold=1,
        initializer=_init_calc,
        initargs=(funcs, dst_options),
    )
    func_keys, tp_results, tn_results, target_opts = zip(*metric_results)
    # merge results into one numpy array
    tp_results = np.vstack([x for x in tp_results if len(x)])
    tn_results = np.vstack([x for x in tn_results if len(x)])
    assert len(tp_results) == len(tn_results)
    return func_keys, tp_results, tn_results, target_opts

# 生成正负样本对
# tp_pairs:[[{func_src:feature},{func_dst:feature}]]
def create_train_pairs(funcs, dst_options, optionidx_map):
    g_funcs = funcs
    g_func_keys = sorted(funcs.keys())
    g_dst_options = dst_options
    tp_pairs = []
    tn_pairs = []
    for func_key in funcs.keys():
        func_data = g_funcs[func_key]
        option_candidates = list(func_data.keys())
        # Testing all functions takes too much time, so we select one true
        # positive and one true negative function for each function.
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

            # select one tn function
            while True:
                func_tn_key = random.choice(g_func_keys)
                # Since difference binaries may have an equal function, pick a
                # function having a different name for precise comparison
                if get_func(func_tn_key) != get_func(func_key):
                    if dst_opt in g_funcs[func_tn_key]:
                        tn_func = g_funcs[func_tn_key][dst_opt]
                        break
            assert not np.isnan(src_func).any()
            assert not np.isnan(tp_func).any()
            assert not np.isnan(tn_func).any()
            src_bin = "{}#{}".format('_'.join(list(func_key)),'_'.join(list(optionidx_map[src_opt])))
            tp_dst_bin = "{}#{}".format('_'.join(list(func_key)), '_'.join(list(optionidx_map[dst_opt])))
            tn_dst_bin = "{}#{}".format('_'.join(list(func_tn_key)), '_'.join(list(optionidx_map[dst_opt])))

            tp_pairs.append([{src_bin:src_func},{tp_dst_bin:tp_func}])
            tn_pairs.append([{src_bin:src_func},{tn_dst_bin:tn_func}])

    return tp_pairs,tn_pairs

def load_model(opts, device):
    checkpoint = torch.load(opts.model_path, map_location=device)
    model_opt = checkpoint['settings']

    model = SiameseAttentionNet(model_opt.feature_dim,
                                model_opt.hidden_dim,
                                model_opt.n_layers,
                                model_opt.n_head,
                                model_opt.d_k,
                                model_opt.d_v,
                                model_opt.att_type,
                                model_opt.dropout).to(device)  # 定义模型且移至GPU

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model

# Select features in greedy way
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
            # check roc for training functions
            roc, ap = calc_results(tp_results, tn_results, tmp_feature_indices)
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


def calc_results(pred, label):
    return calc_roc(pred, label), calc_ap(pred, label)

# funcs:{(package, bin_name, func_name):{option_idx:[feature]}}
# feature_indices:[select_feature]
# 从相同函数的两个编译选项选择一个作为查询query，另一个放入数据集被匹配data
def calc_topK(src_all, dst_all, lable_all, k_list, func_num):
    query = []
    data = []
    y_score = []
    choose_num = 0
    pos_true = []
    # 抽样函数个数值计算topk
    for i,x in enumerate(lable_all):
        if x == 1.0:
            pos_true.append(i)
            if len(pos_true) >= func_num:
                break
    # pos_true = [i for i,x in enumerate(lable_all) if x == 1.0]
    for i in pos_true:
        query.append(src_all[i])
        data.append(dst_all[i])
    # print('num-{} top'.format(len(query)))
    # exit(0)

    num_list = list(np.zeros(len(k_list)))
    total = float(len(pos_true))
    score_list = []
    for i in tqdm(range(len(query))):
        q = query[i]
        # y_score.append([cosine_similarity([q],[d])[0][0] for d in data])
        pred_list = [cosine_similarity([q],[d])[0][0] for d in data]
        pred_dict = {}
        for idx,item in enumerate(pred_list):
            pred_dict[idx] = item
        pred_dict = dict(sorted(pred_dict.items(), key=lambda d: d[1], reverse=True)) # 降序,sorted返回的是列表，要先转字典
        for idx, k in enumerate(k_list):
            pred = list(pred_dict.keys())[:k]
            # pred = np.argmax(np.array([cosine_similarity([q],[d])[0][0] for d in data]))
            if i in pred:
                num_list[idx] += 1
    for idx in range(len(k_list)):
        score_list.append(num_list[idx]/total)
    return score_list


# preprocess possible target options for src option
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
    # Filtering dst options
    for src_option in src_options:

        def _check_option(opt):
            if opt == src_option:
                return False
            for idx in fixed_options:
                if opt[idx] != src_option[idx]:
                    return False
            return True

        candidates = list(filter(_check_option, dst_options))

        # arch needs more filtering ...
        # - 32 vs 64 bits
        # - little vs big endian
        # need to have same archs without bits
        # TODO: move this file name checking into config option.
        if "arch_bits" in config["fname"]:

            def _check_arch_without_bits(opt):
                return get_arch_nobits(opt) == get_arch_nobits(src_option)

            candidates = list(filter(_check_arch_without_bits, candidates))
        # need to have same bits
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

# {函数名:{编译选项索引:{函数特征}}}
def load_func_features_helper(bin_paths):
    # returns {function_key: {option_idx: np.array(feature_values)}}
    global g_options, g_features
    func_features = {}
    num_features = len(g_features)
    optionidx_map = get_optionidx_map(g_options)
    for bin_path in bin_paths:
        package, compiler, arch, opti, bin_name = parse_fname(bin_path)
        others = parse_other_options(bin_path)
        _, func_data_list = load_func_data(bin_path)
        for func_data in func_data_list:
            # Use only .text functions for testing
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
            if option_idx not in func_features[func_key]:
                func_features[func_key][option_idx] = np.zeros(
                    num_features, dtype=np.float64
                )
            for feature_idx, feature in enumerate(g_features):
                if feature not in func_data["feature"]:
                    continue
                val = func_data["feature"][feature]
                func_features[func_key][option_idx][feature_idx] = val

    return func_features


# inevitably use globals since it is fast.
def _init_load(options, features):
    global g_options, g_features
    g_options = options
    g_features = features


def load_func_features(input_list, options, features):
    grouped_bins, packages = group_binaries(input_list)
    # _init_load(options, features)
    # func_features_list = []
    # for bin in tqdm(grouped_bins.values()):
    #     func_features_list.append(load_func_features_helper(bin))
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

# 保存binary-func-opt-feature
def save_funcdatalist_csv(funcs,options,features,outdir):
    print('start save func_data list ...')
    func_list = []
    opts_list = []
    features_list = []
    features_dict = {}
    for func in funcs.keys():
        for opts in funcs[func].keys():
            func_list.append(func)
            opts_list.append(options[opts])
            features_list.append('-'.join(map(str,funcs[func][opts])))
            for idx,feature in enumerate(features):
                if feature not in features_dict:
                    features_dict[feature] = [funcs[func][opts][idx]]
                else:
                    features_dict[feature].append(funcs[func][opts][idx])
    # dataframe = pd.DataFrame({'func_name': func_list, 'options': opts_list, '-'.join(features): features_list})
    data_dict = {'func_name': func_list, 'options': opts_list}
    data_dict.update(features_dict)
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(os.path.join(outdir,"funcdatalist.csv"), index=False, sep=',')
    print('save func_data list csv in {}'.format(os.path.join(outdir,"funcdatalist.csv")))

# 保存经过神经网络前后的函数特征
def save_origin_attn_feature_csv(src_options, src_origin, src_all, dst_options, dst_origin, dst_all, features, outdir):
    print('start save origin_attn_feature ...')
    func_list = []
    origin_att = []
    features_dict = {}
    for i in range(1000):
        func_list.extend([src_options[i],dst_options[i],src_options[i],dst_options[i]])
        origin_att.extend(['origin', 'origin', 'att', 'att'])

        for idx, feature in enumerate(features):
            if feature not in features_dict:
                features_dict[feature] = [src_origin[i][idx]]
                features_dict[feature].append(dst_origin[i][idx])
                features_dict[feature].append(src_all[i][idx])
                features_dict[feature].append(dst_all[i][idx])
            else:
                features_dict[feature].append(src_origin[i][idx])
                features_dict[feature].append(dst_origin[i][idx])
                features_dict[feature].append(src_all[i][idx])
                features_dict[feature].append(dst_all[i][idx])

    data_dict = {'func_option': func_list, 'origin_att': origin_att}
    data_dict.update(features_dict)
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(os.path.join(outdir,"origin_attn_feature.csv"), index=False, sep=',')
    print('save origin_attn_feature csv in {}'.format(os.path.join(outdir,"origin_attn_feature.csv")))





# binary-options-ROC-AP-TOP1-TOP5
def save_result_csv(save_dict,outdir):
    os.makedirs(outdir, exist_ok=True)
    dataframe = pd.DataFrame(save_dict)
    dataframe.to_csv(os.path.join(outdir, "result.csv"), index=False, sep=',')
    print('save result data csv')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# att:[50,50]
# features:[50]
def att_weight_analysis(att,features):
    att = torch.sum(att, dim=0)
    soft_att = F.softmax(att)
    att = soft_att.cpu().detach().numpy()
    fea_ids = heapq.nlargest(10, range(len(att)), att.take) # 取权值最大的前10个特征
    for i in range(len(fea_ids)):
        print("att:{}-{}".format(i,features[fea_ids[i]]))


def do_train(opts):
    binary_list = []
    options_list = []
    train_pairs_list = []
    valid_pairs_list = []
    test_pairs_list = []
    train_keys_list = []
    valid_keys_list = []
    test_keys_list = []
    ROC_list = []
    AP_list = []
    TOP1_list = []
    TOP5_list = []
    config_fname = opts.config
    # config_folder = "D:\program_jiang\Pro\BCA\BCSA\TikNib\config\gnu_type/"
    config_folder = "/mnt/JiangS/BCA/BCSA/TikNib/config/gnu_type/"

    config_fname_list = [
        config_folder + "config_gnu_normal_all_type.yml",
        config_folder + "config_gnu_normal_arch_all_type.yml",
        config_folder + "config_gnu_normal_arch_arm_mips_type.yml",
        config_folder + "config_gnu_normal_arch_x86_arm_type.yml",
        config_folder + "config_gnu_normal_arch_x86_mips_type.yml",
        # config_folder + "config_gnu_normal_obfus_all_type.yml",
        # config_folder + "config_gnu_normal_obfus_bcf_type.yml",
        # config_folder + "config_gnu_normal_obfus_fla_type.yml",
        # config_folder + "config_gnu_normal_obfus_sub_type.yml",
        config_folder + "config_gnu_normal_opti_O0-O3_type.yml",
        config_folder + "config_gnu_normal_opti_O0toO3_type.yml",
        config_folder + "config_gnu_normal_opti_O1-O2_type.yml",
        config_folder + "config_gnu_normal_opti_O2-O3_type.yml",
        # opts.config
    ]
    for config_fname in tqdm(config_fname_list):
        with open(config_fname, "r") as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        config["fname"] = config_fname

        # setup output directory
        if "outdir" in config and config["outdir"]:
            outdir = config["outdir"]
        else:
            base_name = os.path.splitext(os.path.basename(config_fname))[0]
            outdir = os.path.join(opts.log_out, base_name)
        out_curve = os.path.join(outdir, 'curve')
        date = datetime.datetime.now()
        outdir = os.path.join(outdir, str(date).replace(':','-').replace(' ','-'))

        model_save = os.path.join(opts.log_out, opts.model_save)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(model_save, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(outdir, "log.txt"))
        logger.addHandler(file_handler)
        logger.info("config file name: %s", config["fname"])
        logger.info("output directory: %s", outdir)

        # 设置函数对的编译选项，为生成正样本对做准备dst_options{src:[dst]}
        options, dst_options = load_options(config)
        features = sorted(config["features"])
        logger.info("%d features", len(features))

        t0 = time.time()
        logger.info("Feature loading ...")

        funcs = load_func_features(opts.input_list, options, features)
        # 保存函数特征
        # save_funcdatalist_csv(funcs,options,features,outdir)

        logger.info(
            "%d functions (%d unique).", sum([len(x) for x in funcs.values()]), len(funcs)
        )
        logger.info("Feature loading done. (%0.3fs)", time.time() - t0)


        # We revised the code and now NUM_TRAIN_LIMIT is not used.
        # NUM_TRAIN_LIMIT = 2000000

        k_list = [1,5]
        logger.info("[+] Model Parameter: ")
        logger.info("{}".format(opts))

        # set_seed(7)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        optionidx_map = get_optionidx_map_re(options)
        logger.info("Split Datasets and Create Dataloader...")
        # 划分数据集 8:1:1
        func_keys = sorted(funcs.keys())
        # shuffle for 10-fold test ===============
        if opts.debug:
            set_seed(1226)
        # elif config["debug"]:
        #     set_seed(config["seed"])
        random.shuffle(func_keys)
        train_num = int(len(func_keys) * 0.8)
        test_num = int(len(func_keys) * 0.1)
        train_func_keys = func_keys[:train_num]
        valid_func_keys = func_keys[train_num:train_num + test_num]
        test_func_keys = func_keys[train_num + test_num:]

        train_funcs = {key: funcs[key] for key in train_func_keys}
        valid_funcs = {key: funcs[key] for key in valid_func_keys}
        test_funcs = {key: funcs[key] for key in test_func_keys}

        train_tp_pairs, train_tn_pairs = create_train_pairs(train_funcs, dst_options, optionidx_map)
        train_data_loader = CreateDataLoader(train_tp_pairs, train_tn_pairs, opts.batch_size, device)

        valid_tp_pairs, valid_tn_pairs = create_train_pairs(valid_funcs, dst_options, optionidx_map)
        valid_data_loader = CreateDataLoader(valid_tp_pairs, valid_tn_pairs, opts.batch_size, device)

        test_tp_pairs, test_tn_pairs = create_train_pairs(test_funcs, dst_options, optionidx_map)
        test_data_loader = CreateDataLoader(test_tp_pairs, test_tn_pairs, opts.batch_size, device)

        logger.info(
            "Train: %d unique funcs, Valid: %d unique funcs , Test: %d unique funcs",
            len(train_func_keys),
            len(valid_func_keys),
            len(test_func_keys),
        )
        train_keys_list.append(len(train_func_keys))
        valid_keys_list.append(len(valid_func_keys))
        test_keys_list.append(len(test_func_keys))
        model_name = 'binary_{}_config_{}_att-{}_fea{}_hid{}_kv{}_head{}_layer{}.chkpt'.format(
            opts.input_list.split('done_list_')[1].split('.elf.txt')[0],
            config_fname.split('config_')[1].split('.yml')[0],
            opts.att_type,
            opts.feature_dim,
            opts.hidden_dim,
            opts.d_k,
            opts.n_head,
            opts.n_layers)
        binary_list.append(opts.input_list.split('done_list_')[1].split('.elf.txt')[0])
        options_list.append(config_fname.split('config_')[1].split('.yml')[0])

        if opts.train:
            net = SiameseAttentionNet(opts.feature_dim,
                                      opts.hidden_dim,
                                      opts.n_layers,
                                      opts.n_head,
                                      opts.d_k,
                                      opts.d_v,
                                      opts.att_type,
                                      opts.dropout).to(device)  # 定义模型且移至GPU
            # optimizer = optim.Adam(net.parameters(), lr=0.0005)  # 定义优化器
            optimizer = ScheduledOptim(
                optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-09),
                opts.lr_mul, opts.feature_dim, opts.warmup_steps)
            logger.info("create train model ...")

            valid_loss = []

            logger.info("train...")
            # ===================== training ======================
            t0 = time.time()


            if opts.use_tb:
                logger.info("Use Tensorboard")
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter(log_dir=os.path.join(outdir, 'tensorboard'))
                # 绘制网络结构图
                # tb_writer.add_graph(net,
                #                     input_to_model=[torch.ones(1, opts.feature_dim).float().to(device), torch.ones(1, opts.feature_dim).float().to(device)],
                #                     verbose=False)

            for epoch in range(opts.epoch):
                pred_all = []
                lable_all = []
                loss_all = []
                for i,data in enumerate(train_data_loader):
                    _, src, _, dst, label = data
                    optimizer.zero_grad()
                    src_out,dst_out,similarity,slf_attn1,slf_attn2 = net(src, dst)
                    # print(slf_attn1[0][0][0].shape,len(slf_attn1))
                    # exit(0)

                    loss_contrastive = MyLoss(similarity, label)
                    loss_contrastive.backward()
                    # optimizer.step()
                    optimizer.step_and_update_lr()
                    loss_all.append(loss_contrastive.cpu().detach().numpy())
                    pred_all.extend(similarity.cpu().detach().numpy())
                    lable_all.extend(label.cpu().detach().numpy())
                lr = optimizer._optimizer.param_groups[0]['lr']
                epoch_train_roc, epoch_train_ap = calc_results(pred_all, lable_all)
                epoch_train_loss = np.mean(loss_all)
                logger.info(" -Train- Epoch number:{} , AUC:{:.4f} , Loss:{:.4f} , Lr:{}".format(epoch, epoch_train_roc, epoch_train_loss,lr))

                # ===================== validing ======================
                pred_all = []
                lable_all = []
                loss_all = []
                for i, data in enumerate(valid_data_loader):
                    _, src, _, dst, label = data
                    src_out, dst_out, similarity, slf_attn1, slf_attn2 = net(src, dst)
                    loss_contrastive = MyLoss(similarity, label)
                    pred_all.extend(similarity.cpu().detach().numpy())
                    lable_all.extend(label.cpu().detach().numpy())
                    loss_all.append(loss_contrastive.cpu().detach().numpy())

                epoch_valid_roc, epoch_valid_ap = calc_results(pred_all, lable_all)
                epoch_valid_loss = np.mean(loss_all)
                logger.info(" -Valid- Epoch number:{} , AUC:{:.4f} , Loss:{:.4f}".format(epoch, epoch_valid_roc, epoch_valid_loss))


                valid_loss += [epoch_valid_loss]
                checkpoint = {'epoch': epoch, 'settings': opts, 'model': net.state_dict()}

                if epoch_valid_loss <= min(valid_loss):
                    torch.save(checkpoint, os.path.join(model_save, model_name))
                    logger.info('-The checkpoint file has been updated.')

                if opts.use_tb:
                    tb_writer.add_scalars('roc_auc', {'train': epoch_train_roc*100, 'val': epoch_valid_roc*100}, epoch)
                    tb_writer.add_scalars('avg_ap', {'train': epoch_train_ap*100, 'val': epoch_valid_ap*100}, epoch)
                    tb_writer.add_scalars('loss', {'train': epoch_train_loss, 'val': epoch_valid_loss}, epoch)
                    tb_writer.add_scalar('learning_rate', lr, epoch)

            train_time = time.time() - t0
            logger.info("train down. (%0.3fs)", train_time)


        # ===================== testing ======================
        opts.model_path = os.path.join(model_save, model_name)
        net = load_model(opts,device)
        t0 = time.time()
        logger.info("testing ...")
        pred_all = []
        lable_all = []
        src_options = []
        src_all = []
        src_origin = []
        dst_options = []
        dst_all = []
        dst_origin = []
        for i, data in enumerate(test_data_loader):
            src_option, src, dst_option, dst, label = data
            src_origin.extend(src.cpu().detach().numpy())
            dst_origin.extend(dst.cpu().detach().numpy())
            src_options.extend(src_option)
            dst_options.extend(dst_option)
            src_out,dst_out,similarity,slf_attn1,slf_attn2 = net(src, dst)
            src_all.extend(src_out.cpu().detach().numpy())
            dst_all.extend(dst_out.cpu().detach().numpy())
            pred_all.extend(similarity.cpu().detach().numpy())
            lable_all.extend(label.cpu().detach().numpy())

        #
        #     # att_weight_analysis(slf_attn1[0][0][0],features) # 查看注意力权重特点
        save_origin_attn_feature_csv(src_options, src_origin, src_all, dst_options, dst_origin, dst_all, features, outdir)

        topk_list = calc_topK(src_all, dst_all, lable_all, k_list, len(test_funcs)+len(valid_funcs)) # 采样test+val个函数计算topk
        test_roc, test_ap = calc_results(pred_all, lable_all)
        # topk_list = [0.0,0.0]
        # test_roc = 0.0
        # test_ap  = 0.0
        # 没有经过网络结构直接测试基础特征效果
        # for i, data in enumerate(test_data_loader):
        #     src_option, src, dst_option, dst, label = data
        #     src_all.extend(src.cpu().detach().numpy())
        #     dst_all.extend(dst.cpu().detach().numpy())
        #     similarity = F.cosine_similarity(src, dst, dim=1, eps=1e-8)
        #     pred_all.extend(similarity.cpu().detach().numpy())
        #     lable_all.extend(label.cpu().detach().numpy())
        # topk_list = calc_topK(src_all, dst_all, lable_all, k_list, len(test_funcs) + len(valid_funcs))  # 采样test+val个函数计算topk
        # test_roc, test_ap = calc_results(pred_all, lable_all)



        os.makedirs(os.path.join(outdir, 'curve'), exist_ok=True)
        DrawROC(lable_all,pred_all,os.path.join(outdir, 'curve'))
        DrawRecall_Pre_F1(lable_all,pred_all,os.path.join(outdir, 'curve'))

        os.makedirs(out_curve, exist_ok=True)
        DrawROC(lable_all, pred_all, out_curve)
        DrawRecall_Pre_F1(lable_all, pred_all, out_curve)



        logger.info(" -Test- AUC:{:.4f} , AP:{:.4f}".format(test_roc, test_ap))
        for idx, k in enumerate(k_list):
            logger.info(" -Test- top{}:{:.4f}".format(k,topk_list[idx]))

        test_time = time.time() - t0
        logger.info("testing done. (%0.3fs)", test_time)
        logger.info("# of Train Pairs: %d", len(train_tp_pairs) + len(train_tn_pairs))
        logger.info("# of Valid Pairs: %d", len(valid_tp_pairs) + len(valid_tn_pairs))
        logger.info("# of Test Pairs: %d", len(test_tp_pairs) + len(test_tn_pairs))
        logger.removeHandler(file_handler)
        train_pairs_list.append(len(train_tp_pairs) + len(train_tn_pairs))
        valid_pairs_list.append(len(valid_tp_pairs) + len(valid_tn_pairs))
        test_pairs_list.append(len(test_tp_pairs) + len(test_tn_pairs))

        ROC_list.append(test_roc)
        AP_list.append(test_ap)
        TOP1_list.append(topk_list[0])
        TOP5_list.append(topk_list[1])
    result_dict = {'binary':binary_list,
                   'options':options_list,
                   'ROC':ROC_list,
                   'AP':AP_list,
                   'TOP1':TOP1_list,
                   'TOP5':TOP5_list,
                   'train_keys':train_keys_list,
                   'valid_keys':valid_keys_list,
                   'test_keys':test_keys_list,
                   'train_pairs':train_pairs_list,
                   'valid_pairs':valid_pairs_list,
                   'test_pairs':test_pairs_list}
    date = datetime.datetime.now()
    savedir = os.path.join(opts.log_out,'result_csv',str(date).replace(':', '-').replace(' ', '-'))
    save_result_csv(result_dict,savedir)


def analyze_results(data_all,k_list):
    rocs = []
    aps = []
    topks_list = [[] for _ in range(len(k_list))]
    train_times = []
    test_times = []


    for data in data_all:
        train_roc, train_ap, train_time = data[:3]
        test_roc, test_ap, test_time = data[3:6]
        test_topk_list = data[-1]

        rocs.append(test_roc)
        aps.append(test_ap)
        train_times.append(train_time)
        test_times.append(test_time)
        for idx, k in enumerate(k_list):
            topks_list[idx].append(test_topk_list[idx])

    logger.info("Avg. ROC: %0.4f", np.mean(rocs))
    logger.info("Std. of ROC: %0.4f", np.std(rocs))
    logger.info("Avg. AP: %0.4f", np.mean(aps))
    logger.info("Std. of AP: %0.4f", np.std(aps))
    for idx,k in enumerate(k_list):
        logger.info("Avg. Top-%d: %0.4f", k,np.mean(topks_list[idx]))
        logger.info("Std. of Top-%d: %0.4f", k,np.std(topks_list[idx]))
    logger.info("Avg. Train time: %0.4f", np.mean(train_times))
    logger.info("AVg. Test time: %0.4f", np.mean(test_times))


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
    op.add_option("--batch_size",type=int,default=64)
    op.add_option("--feature_dim",type=int,default=50)
    op.add_option('--hidden_dim', type=int, default=512)
    op.add_option('--n_layers', type=int, default=6)
    op.add_option('--epoch', type=int, default=50)
    op.add_option('--n_head', type=int, default=8)
    op.add_option('--d_k', type=int, default=64)
    op.add_option('--d_v', type=int, default=64)
    op.add_option('--dropout', type=float, default=0.3)
    op.add_option('--warmup_steps', type=int, default=4000)
    op.add_option('--lr_mul', type=float, default=0.5)
    op.add_option('--num_folds', type=int, default=5)
    op.add_option('--train_ratio', type=float, default=0.8)
    op.add_option('--model_save', type=str, default='model_save')
    op.add_option('--log_out', type=str, default="results/dl/a2ps/")
    op.add_option('--use_tb', action='store_true')
    op.add_option('--debug', action='store_true')
    op.add_option('--train', action='store_true')
    op.add_option('--att_type', choices=['SelfAttention','ExternalAttention','NoAttention'], default='SelfAttention')

    (opts, args) = op.parse_args()

    if not opts.config:
        op.print_help()
        exit(1)

    do_train(opts)
    # do_test(opts)
