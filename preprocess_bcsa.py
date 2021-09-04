import os
from tqdm import tqdm
import pandas as pd
import argparse
from sklearn.utils import shuffle

def create_input_list(src_folder,outDir,filter,not_include):
    file_list = []
    save_name = 'input_list_'
    for key in filter:
        save_name += '_'.join(filter[key]) + '_'
    save_name = save_name[:-1] + '.txt'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    f_w = open(outDir + save_name,'w')
    for root, dirs, files in os.walk(src_folder):
        for f in files:
            if judge_not_include(root,not_include):
                if judge_filter(f,filter) and f.split('.')[-1] == 'elf':
                    file_list.append(os.path.join(root, f))
                    print(os.path.join(root, f))
                    f_w.write(os.path.join(root, f) + '\n')
    return file_list

def judge_filter(file_name,filter_dict):
    for key in filter_dict:
        if filter_dict[key][0] != 'all' and all(option not in file_name for option in filter_dict[key]): #
            return False
    return True

def judge_not_include(file_path,not_include):
    if any(option in file_path for option in not_include):
        return False
    else:
        return True


def get_done_list(src_folder,outDir,filter):
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
    parser.add_argument('--data_folder', type=str, default='/XXX/DataSet/', help='sentences folder')
    parser.add_argument('--out', type=str, default='XXX/', help='save folder')
    args = parser.parse_args()

    filter = {'bin_name':['all'],
              'version':['all'],
              'compiler':['clang-4.0'],
              'arch':['all'],
              'opt':['all'],
              'others':['all']}
    not_include = ['lto','noinline','obfus_2loop','pie','sizeopt']
    get_done_list(args.data_folder, args.out, filter)

