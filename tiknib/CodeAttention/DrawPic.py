# coding:utf-8
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# 根据tsv文件绘制曲线
def Draw_By_tsv(tsv_files,curve_label,curve_name,logdir):
    os.makedirs(logdir, exist_ok=True)
    colors = ['r','b']
    if curve_name == 'best_test_roc':
        fig = plt.figure()
        plt.title(curve_name)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        for i in range(len(tsv_files)):
            tsv = tsv_files[i]
            thresholds = []
            tprs = []
            fprs = []
            with open(tsv, "r") as the_file:
                content = the_file.readlines()
                for line in content[1:]:
                    line.strip()
                    threshold = float(line.split('\t')[0])
                    tpr = float(line.split('\t')[1])
                    fpr = float(line.split('\t')[2])
                    thresholds.append(threshold)
                    tprs.append(tpr)
                    fprs.append(fpr)
            plt.plot(fprs, tprs, color=colors[i], label=curve_label[i])
        plt.legend()
        fig.savefig(str(logdir) + "/" + curve_name + ".pdf")
        plt.close(fig)


    elif curve_name == 'F1score-CDF':
        fig = plt.figure()
        plt.title(curve_name)
        plt.xlabel('F1_score')
        plt.ylabel('percentage')
        for i in range(len(tsv_files)):
            tsv = tsv_files[i]
            F1_scores = []
            percentages = []
            with open(tsv, "r") as the_file:
                content = the_file.readlines()
                for line in content[1:]:
                    line.strip()
                    F1_score = float(line.split('\t')[0])
                    percentage = float(line.split('\t')[1])
                    F1_scores.append(F1_score)
                    percentages.append(percentage)
            plt.plot(F1_scores, percentages, color=colors[i], label=curve_label[i])
        plt.legend()
        fig.savefig(str(logdir) + "/" + curve_name + ".pdf")
        plt.close(fig)

    elif curve_name == 'pre_recall':
        fig = plt.figure()
        plt.title(curve_name)
        plt.xlabel('recall')
        plt.ylabel('precision')
        for i in range(len(tsv_files)):
            tsv = tsv_files[i]
            precisions = []
            recalls = []
            with open(tsv, "r") as the_file:
                content = the_file.readlines()
                for line in content[1:]:
                    line.strip()
                    threshold = float(line.split('\t')[0])
                    precision = float(line.split('\t')[1])
                    recall = float(line.split('\t')[2])
                    precisions.append(precision)
                    recalls.append(recall)
            plt.plot(recalls, precisions, color=colors[i], label=curve_label[i])
        plt.legend()
        fig.savefig(str(logdir) + "/" + curve_name + ".pdf")
        plt.close(fig)

# 根据csv文件绘制曲线，直接读取对应列
def Draw_By_csv(csv_files,curve_label,curve_name,logdir):
    os.makedirs(logdir, exist_ok=True)
    colors = ['r','b']
    if curve_name == 'best_test_roc':
        fig = plt.figure()
        plt.title(curve_name)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        for i in range(len(csv_files)):
            csv = csv_files[i]
            dataframe = pd.read_csv(csv)
            tprs = dataframe['tprs']
            fprs = dataframe['fprs']
            plt.plot(fprs, tprs, color=colors[i], label=curve_label[i])
        plt.legend()
        fig.savefig(str(logdir) + "/" + curve_name + ".pdf")
        plt.close(fig)


    elif curve_name == 'F1score-CDF':
        fig = plt.figure()
        plt.title(curve_name)
        plt.xlabel('F1_score')
        plt.ylabel('percentage')
        for i in range(len(csv_files)):
            csv = csv_files[i]
            dataframe = pd.read_csv(csv)
            F1_scores = dataframe['CDF_X']
            percentages = dataframe['f1_scores_percents']
            plt.plot(F1_scores, percentages, color=colors[i], label=curve_label[i])
        plt.legend()
        fig.savefig(str(logdir) + "/" + curve_name + ".pdf")
        plt.close(fig)

    elif curve_name == 'pre_recall':
        fig = plt.figure()
        plt.title(curve_name)
        plt.xlabel('recall')
        plt.ylabel('precision')
        for i in range(len(csv_files)):
            csv = csv_files[i]
            dataframe = pd.read_csv(csv)
            precisions = dataframe['precision']
            recalls = dataframe['recall']
            plt.plot(recalls, precisions, color=colors[i], label=curve_label[i])
        plt.legend()
        fig.savefig(str(logdir) + "/" + curve_name + ".pdf")
        plt.close(fig)


def DrawROC(test_y,test_pred,logdir):
    test_fpr, test_tpr, test_thresholds = metrics.roc_curve(test_y, test_pred, pos_label=1)
    # write ROC raw data
    # with open(str(logdir) + "/best_test_roc.tsv", "w") as the_file:
    #     the_file.write("#thresholds\ttpr\tfpr\n")
    #     for t, tpr, fpr in zip(test_thresholds, test_tpr, test_fpr):
    #         the_file.write("{}\t{}\t{}\n".format(t, tpr, fpr))

    data_dict = {'threshold': test_thresholds, 'tprs': test_tpr, 'fprs': test_fpr}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/best_test_roc.csv", index=False, sep=',')

    test_auc = metrics.auc(test_fpr, test_tpr)
    fig = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(test_fpr, test_tpr, 'b',label='AUC = %0.2f' % test_auc)
    fig.savefig(str(logdir) + "/best_test_roc.pdf")
    plt.close(fig)
    return test_auc


def DrawRecall_Pre_F1(test_y,test_pred,logdir):
    precision, recall, thresholds = metrics.precision_recall_curve(test_y, test_pred,pos_label=1)
    # write P-R raw data
    # with open(str(logdir) + "/pre_recall.tsv", "w") as the_file:
    #     the_file.write("#thresholds\tprecision\trecall\n")
    #     for t, pre, rec in zip(thresholds, precision, recall):
    #         the_file.write("{}\t{}\t{}\n".format(t, pre, rec))

    data_dict = {'threshold': thresholds, 'precision': precision[0:-1], 'recall': recall[0:-1]}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/pre_recall.csv", index=False, sep=',')

    fig = plt.figure()
    plt.title('Precision-Recall')
    plt.plot(recall, precision, 'b')
    fig.savefig(str(logdir) + "/pre_recall.pdf")
    plt.close(fig)

    fig = plt.figure()
    plt.title('thresholds-TPR')
    plt.plot(thresholds, recall[0:-1], 'b')
    fig.savefig(str(logdir) + "/thresholds_tpr.pdf")
    plt.close(fig)

    f1_scores = []
    for i in range(len(precision)):
        f1_socre = (2*precision[i]*recall[i])/(precision[i]+recall[i])
        f1_scores.append(f1_socre)

    # with open(str(logdir) + "/thresholds_f1_score.tsv", "w") as the_file:
    #     the_file.write("#thresholds\tf1_score\n")
    #     for t, f1 in zip(thresholds, f1_scores):
    #         the_file.write("{}\t{}\n".format(t, f1))

    data_dict = {'threshold': thresholds, 'f1_scores': f1_scores[0:-1]}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/thresholds_f1_score.csv", index=False, sep=',')

    fig = plt.figure()
    plt.title('thresholds_f1_score')
    plt.plot(thresholds, f1_scores[0:-1], 'b')
    fig.savefig(str(logdir) + "/thresholds_f1_score.pdf")
    plt.close(fig)

    DrawF1score_CDF(precision, recall, logdir)


def DrawF1score_CDF(precision,recall,logdir):
    f1_scores = []
    f1_scores_percents = []
    CDF_X = list(np.linspace(0, 1, num=100))  # f1-score-cdf的横坐标
    for i in range(len(precision)):
        f1_socre = (2*precision[i]*recall[i])/(precision[i]+recall[i])
        f1_scores.append(f1_socre)
    for CDF in CDF_X:
        f1_scores_percents.append(GetPercent_Of_F1_score(f1_scores,CDF))
    fig = plt.figure()
    plt.title('F1score-CDF')
    plt.plot(CDF_X, f1_scores_percents, 'b')
    fig.savefig(str(logdir) + "/F1score-CDF.pdf")
    plt.close(fig)
    # with open(logdir + "/F1score-CDF.tsv", "w") as the_file:
    #     the_file.write("#F1_score\tpercentage\n")
    #     for c, per in zip(CDF_X, f1_scores_percents):
    #         the_file.write("{}\t{}\n".format(c, per))

    data_dict = {'CDF_X': CDF_X, 'f1_scores_percents': f1_scores_percents}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/F1score-CDF.csv", index=False, sep=',')


def GetPercent_Of_F1_score(f1_scores,CDF):
    num = 0
    for f1_score in f1_scores:
        if f1_score <= CDF:
            num += 1
    percent = float(num)/len(f1_scores)
    return percent

def Draw_ROC_K(similar_rate,truth,logdir):
    sort_similar,sort_truth = similar_truth_sort(similar_rate,truth)
    keylist = [i for i in range(5, len(truth), 5)]
    fpr_my,tpr_my = myself_roc(sort_similar,sort_truth,keylist)
    auc_my = metrics.auc(fpr_my,tpr_my)

    # with open(str(logdir) + "/roc_k.tsv", "w") as the_file:
    #     the_file.write("#k\ttpr\tfpr\n")
    #     for k, tpr, fpr in zip(keylist, tpr_my, fpr_my):
    #         the_file.write("{}\t{}\t{}\n".format(k, tpr, fpr))
    data_dict = {'keylist': keylist, 'tpr_my': tpr_my, 'fpr_my': fpr_my}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/roc_k.csv", index=False, sep=',')

    fig = plt.figure()
    plt.title('roc_k')
    plt.plot(fpr_my, tpr_my, 'b')
    fig.savefig(str(logdir) + "/roc_k.pdf")
    plt.close(fig)


    # with open(logdir + "/k_recall.tsv", "w") as the_file:
    #     the_file.write("#k\trecall\n")
    #     for k, recall in zip(keylist, tpr_my):
    #         the_file.write("{}\t{}\n".format(k, recall))
    data_dict = {'keylist': keylist, 'tpr_my': tpr_my}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/k_recall.csv", index=False, sep=',')

    fig = plt.figure()
    plt.title('k_recall')
    plt.plot(keylist, tpr_my, 'b')
    fig.savefig(str(logdir) + "/k_recall.pdf")
    plt.close(fig)



def similar_truth_sort(similar,truth):
    sort_similar = []
    sort_truth = []
    sort_index = np.argsort(-similar) # from max to small
    for i in sort_index:
        sort_similar.append(similar[i])
        sort_truth.append(truth[i])
    return sort_similar,sort_truth

def myself_roc(similar,truth,keylist):
    fpr = []
    tpr = []
    for key in keylist:
        tp = float(0)
        fp = float(0)
        tn = float(0)
        fn = float(0)
        for i in range(key):
            if truth[i] == True:
                tp += 1
            else:
                fp += 1
        for i in range(key,len(similar)):
            if truth[i] == True:
                fn += 1
            else:
                tn += 1
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    return fpr,tpr

if __name__ == "__main__":
    # 欧几里得距离，余弦相似度计算，实质就是余弦夹角
    # input1 = torch.abs(torch.randn(1,5))
    # input2 = torch.abs(torch.randn(1,5))
    # output = F.pairwise_distance(input1, input2, p=2)
    # output2 = F.cosine_similarity(input1, input2)
    # output3 = cosine_similarity(input1.numpy(),input2.numpy())[0][0]
    # print(input1,input2,output,output2,output3)
    # exit(0)

    main_folder = '/mnt/JiangS/BCA/BCSA/TikNib/helper/results/'
    config_fname_list = [
        # "config_gnu_normal_all_type",
        # "config_gnu_normal_arch_all_type",
        # "config_gnu_normal_arch_arm_mips_type",
        # "config_gnu_normal_arch_x86_arm_type",
        # "config_gnu_normal_arch_x86_mips_type",
        "config_gnu_normal_obfus_all_type",
        # "config_gnu_normal_obfus_bcf_type",
        # "config_gnu_normal_obfus_fla_type",
        # "config_gnu_normal_obfus_sub_type",
        # "config_gnu_normal_opti_O0-O3_type",
        # "config_gnu_normal_opti_O0toO3_type",
        # "config_gnu_normal_opti_O1-O2_type",
        # "config_gnu_normal_opti_O2-O3_type"
    ]
    curve_name_list = [
        'best_test_roc',
        'F1score-CDF',
        'pre_recall',
        'thresholds_f1_score'
    ]
    for config in config_fname_list:
        dl_folder = main_folder + 'dl/a2ps/' + config + '/curve/'
        tiknib_folder = main_folder + 'a2ps/' + config + '/curve/'
        logdir = main_folder + 'curve/' + config
        for curve_name in curve_name_list:
            print('draw:{}-{}-curve'.format(config,curve_name))
            Draw_By_csv([dl_folder+curve_name+'.csv',tiknib_folder+curve_name+'.csv'],['dl','tiknib'],curve_name,logdir)
            # Draw_By_tsv([dl_folder+curve_name+'.tsv',tiknib_folder+curve_name+'.tsv'],['dl','tiknib'],curve_name,logdir)

