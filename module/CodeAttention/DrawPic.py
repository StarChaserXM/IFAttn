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
    CDF_X = list(np.linspace(0, 1, num=100))
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
    data_dict = {'keylist': keylist, 'tpr_my': tpr_my, 'fpr_my': fpr_my}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/roc_k.csv", index=False, sep=',')

    fig = plt.figure()
    plt.title('roc_k')
    plt.plot(fpr_my, tpr_my, 'b')
    fig.savefig(str(logdir) + "/roc_k.pdf")
    plt.close(fig)

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
    sort_index = np.argsort(-similar)
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
