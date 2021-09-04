# CodeTransformer
## PEP8 ctl+alt+l
### Requirements:

* angr
* networkx
* dill
* python==3.8.8
* pytorch==1.8.0
* torchtext==0.6.0
* tqdm
* transformers # 提供了自然语言理解（NLU）和自然语言生成（NLG）的通用体系结构（BERT，GPT-2，RoBERTa，XLM，DistilBert，XLNet等）

### Create virtual environment
```python
conda create -n env_name python=3.8 # create
Linux:  source activate env_nam # activate
Win: activate env_name
conda install -n env_name [package] # install pkg
pip install [package] # 先要确定pip的路径是否在虚拟环境中
Win: deactivate env_name # deactivate
Linux: source deactivate env_name
conda remove -n env_name --all # delete
```
### Run the tool
```python
# 路径不需要加引号~~~~~~~~
# 解析src和trg二进制func-tokens并依据函数名生成tokens-pairs
python angr_method.py -src_bin src_path -trg_bin trg_path -angr_out out/ -article_out data/article/
# 结合上一步生成的token-pairs预处理数据集，用于Transformer翻译训练
python preprocess_spacy.py -src_article src_file -trg_article trg_file -save_data data/prepare/openssl_origin_bcf.pkl
# 设置超参数训练Transformer
python train.py -data_pkl data/prepare/openssl_origin_bcf.pkl -epoch 500 -b 16 -d_model 256 -d_inner_hid 512 -d_k 64 -d_v 64 -n_head 8 -n_layers 6 -dropout 0.2 -output_dir model_save -use_tb (-no_cuda)
# 基于Transformer模型翻译测试
python translate.py -model model_path -data_pkl use_dataset (-no_cuda)
```

### Scheme
1. across the Architecture/Optimization/Obfuscation base Function-level
2. instruction->word;basic-block->sentence;function-paragraph;program->article base Transformer
3. serial info / structural info

### Tensorboard
```python
tensorboard --logdir model_save/model_name # 注意路径的问题
```

### Bianry Obfuscation
1. 模拟控制流Bogus Control Flow(BCF):此模糊技术通过添加大量不相关的随机基本块和分支来修改CFG。它还分割、合并和重新排序原始基本块，将随机选择的垃圾指令插入到原始基本块中。这种混淆技术会破坏了CFG和基本块的完整性，使控制流分支复杂化，并增加节点数量、垃圾指令和常数。
2. 控制流量扁平化Control Flow Flattening(FLA):这种模糊技术是一种扁平化控制流程策略。通过添加交换语句和循环，程序中与条件转换和嵌套循环相关的控制结构将被扁平。这种技术分割基本块，从而产生冗余的垃圾指令和伪分支。混淆的CFG几乎完全失去了它原来的结构。
3. 指令替换Instructions Substitution(SUB):这种混淆技术使用身份操作混淆策略，旨在用语义相同但更复杂的指令替换简单的指令。特别是，该技术主要取代二进制运算符，如加法、减法、乘法和除法。这种混淆技术将修改基本块的内容，增加算术指令、逻辑指令、常数等的数量。


### BCSA
1. 创建数据集
```python
# create_input_list:生成要IDA分析的二进制列表
# get_done_list:生成已完成IDA分析的二进制列表
python preprocess_bcsa.py --data_folder "/mnt/JiangS/CodeTransformer/BCSA/DataSet/" --out "helper/input/linux/"
```
2. IDA特征提取
```python
# IDA分析在win下进行
python do_idascript.py --idapath "D:\program_jiang\tool\IDA_Pro_v7.5_Portable" --idc "D:\program_jiang\Pro\CodeTransformer\BCSA\TikNib\tiknib\ida\fetch_funcdata_v7.5.py" --input_list "input/input_list_a2ps_all_all_all_all_a2ps.elf.txt" --log
# source_list:包含源码的绝对路径
python extract_functype.py --input_list D:\program_jiang\Pro\CodeTransformer\BCSA\TikNib\helper\input\done_list_a2ps_all_all_all_all_a2ps.elf.txt --source_list "D:\program_jiang\Pro\CodeTransformer\BCSA\TikNib\helper\input\source_list_a2ps.txt" --ctags_dir "data/ctags" --threshold 1
python extract_features.py --input_list "D:\program_jiang\Pro\CodeTransformer\BCSA\TikNib\helper\input\done_list_a2ps_all_all_all_all_a2ps.elf.txt" --threshold 1
# 贪心特征选择
# 评价指标保存在log_out路径下的result_save文件夹
python test_roc.py --input_list "D:\program_jiang\Pro\BCA\BCSA\TikNib\helper\input\linux\done_list_openssl_all_clang-7.0_x86_64_all_libssl.so.elf.txt" --config None
```
3. 模型训练
```python
# 深度学习模型在linux下进行
# 先将已完成IDA特征提取的.pickle文件上传到对应二进制文件夹
# 选择不同的config文件进行不同测试
# --config已硬编码
# --train设置训练还是测试
# 模型保存到log_out路径下的model_save文件夹
# 评价指标保存在log_out路径下的result_save文件夹
# save_funcdatalist_csv函数用于保存当前数据集下所有函数特征
python train_dl.py --input_list "/mnt/JiangS/CodeTransformer/BCSA/TikNib/helper/input/linux/done_list_a2ps_all_clang-4.0_clang-obfus_all_all_a2ps.elf.txt" --config "/mnt/JiangS/CodeTransformer/BCSA/TikNib/config/gnu_type/config_gnu_normal_opti_O0-O3_type.yml" --config None --use_tb --debug --train
```
4. 曲线绘制
```python
python BCA/BCSA/TikNib/tiknib/CodeAttention/DrawPic.py # 整合tiknib和attention在不同实验任务和不同指标上的曲线
```

5. Top-K计算
> 从测试集中取样test_funcs个正样本对，分成querys(带查询函数)和datas(被查询范围)，根据余弦相似度降序排序，目标值就是querys的索引值
> 1. 贪心算法querys取样test_funcs个
> 2. 注意力网络querys取样test_funcs+valid_funcs个，和贪心算法个数相同