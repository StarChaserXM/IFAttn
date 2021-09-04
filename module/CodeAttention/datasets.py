import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset


def pairs_parse(pairs,label):
    src_funcs_option = []
    src_funcs_feature = []
    dst_funcs_option = []
    dst_funcs_feature = []
    labels = []
    for pair in pairs:
        pair_src = pair[0]
        pair_dst = pair[1]
        src_funcs_option.append(list(pair_src.keys())[0])
        src_funcs_feature.append(list(pair_src.values())[0])
        dst_funcs_option.append(list(pair_dst.keys())[0])
        dst_funcs_feature.append(list(pair_dst.values())[0])
        labels.append(label)
    return src_funcs_option,src_funcs_feature,dst_funcs_option,dst_funcs_feature,labels


class MyDataset(Dataset):
    def __init__(self, tp_pairs, tn_pairs, device):
        src_funcs_option = []
        src_funcs_feature = []
        dst_funcs_option = []
        dst_funcs_feature = []
        labels = []

        tp_src_funcs_option,tp_src_funcs_feature,tp_dst_funcs_option,tp_dst_funcs_feature,tp_labels = pairs_parse(tp_pairs,1.0)
        tn_src_funcs_option,tn_src_funcs_feature,tn_dst_funcs_option,tn_dst_funcs_feature,tn_labels = pairs_parse(tn_pairs,0.0)

        src_funcs_option.extend(tp_src_funcs_option)
        src_funcs_feature.extend(tp_src_funcs_feature)
        dst_funcs_option.extend(tp_dst_funcs_option)
        dst_funcs_feature.extend(tp_dst_funcs_feature)
        labels.extend(tp_labels)

        src_funcs_option.extend(tn_src_funcs_option)
        src_funcs_feature.extend(tn_src_funcs_feature)
        dst_funcs_option.extend(tn_dst_funcs_option)
        dst_funcs_feature.extend(tn_dst_funcs_feature)
        labels.extend(tn_labels)

        self.src_option = src_funcs_option
        self.dst_option = dst_funcs_option
        self.src = torch.from_numpy(np.array(src_funcs_feature)).float().to(device)
        self.dst = torch.from_numpy(np.array(dst_funcs_feature)).float().to(device)
        self.labels = torch.from_numpy(np.array(labels)).float().to(device)


    def __getitem__(self, index):
        return self.src_option[index],self.src[index],self.dst_option[index],self.dst[index],self.labels[index]

    def __len__(self):
        return len(self.labels)


def CreateDataLoader(tp_pairs, tn_pairs, batch_size, device):
    dataset = MyDataset(tp_pairs, tn_pairs, device)
    data_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    return data_loader