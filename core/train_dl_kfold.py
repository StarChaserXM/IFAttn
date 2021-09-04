import time
import random
import itertools
import os
import sys
import datetime
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
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
from module.CodeAttention.datasets import CreateDataLoader
from module.CodeAttention.model import SiameseAttentionNet,MyLoss
from torch import optim
import torch

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
    if tp_results:
        tp_results = np.vstack(tp_results)
    if tn_results:
        tn_results = np.vstack(tn_results)
    return func_key, tp_results, tn_results, target_opts

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

def create_train_pairs(funcs, dst_options, optionidx_map):
    g_funcs = funcs
    g_func_keys = sorted(funcs.keys())
    g_dst_options = dst_options
    tp_pairs = []
    tn_pairs = []
    for func_key in tqdm(funcs.keys()):
        func_data = g_funcs[func_key]
        option_candidates = list(func_data.keys())
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
            src_bin = "{}#{}".format('_'.join(list(func_key)),'_'.join(list(optionidx_map[src_opt])))
            tp_dst_bin = "{}#{}".format('_'.join(list(func_key)), '_'.join(list(optionidx_map[dst_opt])))
            tn_dst_bin = "{}#{}".format('_'.join(list(func_tn_key)), '_'.join(list(optionidx_map[dst_opt])))

            tp_pairs.append([{src_bin:src_func},{tp_dst_bin:tp_func}])
            tn_pairs.append([{src_bin:src_func},{tn_dst_bin:tn_func}])

    return tp_pairs,tn_pairs

def load_model(opts, device):
    checkpoint = torch.load(opts.model, map_location=device)
    model_opt = checkpoint['settings']

    model = SiameseAttentionNet(model_opt.feature_dim,
                              model_opt.hidden_dim,
                              model_opt.n_layers,
                              model_opt.n_head,
                              model_opt.d_k,
                              model_opt.d_v,
                              model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model

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
def calc_topK(src_all, dst_all, lable_all, k_list):
    query = []
    data = []
    y_true = []
    y_score = []
    pos_true = [i for i,x in enumerate(lable_all) if x == 1.0 and i <= 500]
    for i in pos_true:
        query.append(src_all[i])
        data.append(dst_all[i])
        y_true.append(len(query)-1)

    num_list = list(np.zeros(len(k_list)))
    total = float(len(y_true))
    score_list = []
    for i in tqdm(range(len(query))):
        q = query[i]
        pred_list = [cosine_similarity([q],[d])[0][0] for d in data]
        pred_dict = {}
        for idx,item in enumerate(pred_list):
            pred_dict[idx] = item
        pred_dict = dict(sorted(pred_dict.items(), key=lambda d: d[1], reverse=True))
        for idx, k in enumerate(k_list):
            pred = list(pred_dict.keys())[:k]
            if i in pred:
                num_list[idx] += 1
    for idx in range(len(k_list)):
        score_list.append(num_list[idx]/total)
    return score_list

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

def save_funcdatalist_csv(funcs,options,features,outdir):
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
    data_dict = {'func_name': func_list, 'options': opts_list}
    data_dict.update(features_dict)
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(os.path.join(outdir,"funcdatalist.csv"), index=False, sep=',')
    print('save func_data list csv')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def do_train(opts):
    config_fname = opts.config
    config_folder = "/XXX/config/gnu_type/"
    config_fname_list = [
        config_folder + "config_gnu_normal_obfus_all_type.yml",
        config_folder + "config_gnu_normal_obfus_bcf_type.yml",
        config_folder + "config_gnu_normal_obfus_fla_type.yml",
        config_folder + "config_gnu_normal_obfus_sub_type.yml",
        config_folder + "config_gnu_normal_opti_O0-O3_type.yml",
        config_folder + "config_gnu_normal_opti_O0toO3_type.yml",
        config_folder + "config_gnu_normal_opti_O1-O2_type.yml"
    ]
    for config_fname in tqdm(config_fname_list):
        with open(config_fname, "r") as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        config["fname"] = config_fname

        if "outdir" in config and config["outdir"]:
            outdir = config["outdir"]
        else:
            base_name = os.path.splitext(os.path.basename(config_fname))[0]
            outdir = os.path.join("results/dl", base_name)
        date = datetime.datetime.now()
        outdir = os.path.join(outdir, str(date).replace(':','-'))
        os.makedirs(outdir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(outdir, "log.txt"))
        logger.addHandler(file_handler)
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


        num_folds = opts.num_folds
        kf = KFold(n_splits=num_folds)
        assert len(funcs) > num_folds
        k_list = [1,5]
        logger.info("[+] Model Parameter: ")
        logger.info("{}".format(opts))

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        net = SiameseAttentionNet(opts.feature_dim,
                                  opts.hidden_dim,
                                  opts.n_layers,
                                  opts.n_head,
                                  opts.d_k,
                                  opts.d_v,
                                  opts.dropout).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.0005)
        logger.info("create train model ...")

        # ===============================================
        # start test
        # ===============================================
        func_keys = sorted(funcs.keys())
        # shuffle for 10-fold test ===============
        if config["debug"]:
            set_seed(config["seed"])
        random.shuffle(func_keys)

        data_all = []
        test_topk = []
        for trial_idx, (train_func_keys, test_func_keys) in enumerate(kf.split(func_keys)):
            logger.info("[+] TRIAL %d/%d ================", trial_idx + 1, num_folds)

            train_func_keys = [func_keys[i] for i in train_func_keys]
            train_funcs = {key: funcs[key] for key in train_func_keys}
            test_func_keys = [func_keys[i] for i in test_func_keys]
            test_funcs = {key: funcs[key] for key in test_func_keys}
            logging.info(
                "Train: %d unique funcs, Test: %d unique funcs",
                len(train_func_keys),
                len(test_func_keys),
            )

            # ===================== training ======================
            t0 = time.time()

            optionidx_map = get_optionidx_map_re(options)
            tp_pairs, tn_pairs = create_train_pairs(train_funcs, dst_options, optionidx_map)
            data_loader = CreateDataLoader(tp_pairs, tn_pairs, opts.batch_size, device)
            train_roc_list = []
            train_ap_list = []
            for epoch in range(opts.epoch):
                pred_all = []
                lable_all = []
                loss_all = []
                for i,data in enumerate(data_loader):
                    src, dst, label = data
                    optimizer.zero_grad()
                    src_out,dst_out,similarity,slf_attn1,slf_attn2 = net(src, dst)
                    loss_contrastive = MyLoss(similarity, label)
                    loss_contrastive.backward()
                    optimizer.step()
                    loss_all.append(loss_contrastive.cpu().detach().numpy())
                    pred_all.extend(similarity.cpu().detach().numpy())
                    lable_all.extend(label.cpu().detach().numpy())
                epoch_train_roc, epoch_train_ap = calc_results(pred_all, lable_all)
                logger.info("Epoch number:{} , AUC:{:.4f} , Loss:{:.4f}".format(epoch, epoch_train_roc, np.mean(loss_all)))
                train_roc_list.append(epoch_train_roc)
                train_ap_list.append(epoch_train_ap)
            train_roc = np.mean(train_roc_list)
            train_ap = np.mean(train_ap_list)
            logger.info("train_roc:{:.4f} , train_ap:{:.4f}".format(train_roc, train_ap))
            train_time = time.time() - t0
            logger.info("train down. (%0.3fs)", train_time)


            # ===================== testing ======================
            t0 = time.time()
            logger.info("testing ...")
            pred_all = []
            lable_all = []
            src_all = []
            dst_all = []
            tp_pairs, tn_pairs = create_train_pairs(test_funcs, dst_options, optionidx_map)
            data_loader = CreateDataLoader(tp_pairs, tn_pairs, opts.batch_size, device)
            for i, data in enumerate(data_loader):
                src, dst, label = data
                src_out,dst_out,similarity,slf_attn1,slf_attn2 = net(src, dst)
                src_all.extend(src_out.cpu().detach().numpy())
                dst_all.extend(dst_out.cpu().detach().numpy())
                pred_all.extend(similarity.cpu().detach().numpy())
                lable_all.extend(label.cpu().detach().numpy())

            topk_list = calc_topK(src_all, dst_all, lable_all, k_list)
            test_roc, test_ap = calc_results(pred_all, lable_all)
            for idx, k in enumerate(k_list):
                logger.info("AUC:{:.4f} , AP:{:.4f} , top{}:{:.4f}".format(test_roc,test_ap,k,topk_list[idx]))
            test_topk += [topk_list[0]]

            checkpoint = {'trial_idx': trial_idx, 'settings': opts, 'model': net.state_dict()}
            model_name = 'binary_{}_config_{}_fea{}_hid{}_kv{}_head{}_layer{}.chkpt'.format(opts.input_list.split('done_list_')[1].split('.elf.txt')[0],
                                                                                            config_fname.split('config_')[1].split('.yml')[0],
                                                                                            opts.feature_dim,
                                                                                            opts.hidden_dim,
                                                                                            opts.d_k,
                                                                                            opts.n_head,
                                                                                            opts.n_layers)
            if topk_list[0] >= max(test_topk):
                torch.save(checkpoint, os.path.join(outdir, model_name))
                logger.info('-The checkpoint file has been updated.')

            test_time = time.time() - t0
            logger.info("testing done. (%0.3fs)", test_time)
            data = [
                train_roc,
                train_ap,
                train_time,
                test_roc,
                test_ap,
                test_time,
                topk_list,
            ]
            data_all.append(data)
        analyze_results(data_all,k_list)
        logger.removeHandler(file_handler)


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
    op.add_option('--dropout', type=float, default=0.2)
    op.add_option('--num_folds', type=int, default=5)
    (opts, args) = op.parse_args()

    if not opts.config:
        op.print_help()
        exit(1)

    do_train(opts)
