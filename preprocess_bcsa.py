# coding:utf-8

import os
from tqdm import tqdm
import pandas as pd
import argparse
from sklearn.utils import shuffle



"""
filter : {bin_name,version,compiler,arch,opt,others}
"""
def create_input_list(src_folder,outDir,filter,not_include): # 生成需要IDA PRO分析的文件列表 txt,过滤出需要的文件,不包含的文件夹
    # 递归获取目录下所有文件 os.walk本身就是递归
    file_list = []
    save_name = 'input_list_'
    for key in filter:
        save_name += '_'.join(filter[key]) + '_'
    save_name = save_name[:-1] + '.txt'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    f_w = open(outDir + save_name,'w')
    # root 表示当前正在访问的文件夹路径
    # dirs 表示该文件夹下的子目录名list
    # files 表示该文件夹下的文件list
    for root, dirs, files in os.walk(src_folder):
        for f in files:
            if judge_not_include(root,not_include):
                if judge_filter(f,filter) and f.split('.')[-1] == 'elf':
                    file_list.append(os.path.join(root, f))
                    print(os.path.join(root, f))
                    f_w.write(os.path.join(root, f) + '\n')
    return file_list

def judge_filter(file_name,filter_dict): # 判断文件是否需要
    for key in filter_dict:
        if filter_dict[key][0] != 'all' and all(option not in file_name for option in filter_dict[key]): #
            return False
    return True

def judge_not_include(file_path,not_include): # 判断不需要的文件夹
    if any(option in file_path for option in not_include): #
        return False
    else:
        return True


def get_done_list(src_folder,outDir,filter): # 获取所有IDA分析完成的文件
    save_name = 'done_list_'
    for key in filter:
        save_name += '_'.join(filter[key]) + '_'
    save_name = save_name[:-1] + '.txt'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    f_w = open(outDir + save_name,'w')
    for root, dirs, files in os.walk(src_folder):
        for f in files:
            if judge_filter(f,filter) and f.split('.')[-1] == 'pickle':
                print(os.path.join(root, f.split('.pickle')[0]))
                f_w.write(os.path.join(root, f.split('.pickle')[0]) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='preprocess_BCSA')
    parser.add_argument('--data_folder', type=str, default='/mnt/JiangS/CodeTransformer/BCSA/DataSet/', help='sentences folder')
    parser.add_argument('--out', type=str, default='helper/input/linux/', help='save folder')
    args = parser.parse_args()

    filter = {'bin_name':['all'],
              'version':['all'],
              'compiler':['clang-4.0'],
              'arch':['all'],
              'opt':['all'],
              'others':['all']}
    not_include = ['lto','noinline','obfus_2loop','pie','sizeopt']
    # IDA 分析原始二进制获得函数特征
    # file_list = create_input_list(args.data_folder, args.out, filter, not_include)
    # 获取已分析完成的二进制列表
    get_done_list(args.data_folder, args.out, filter)

