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
* transformers # �ṩ����Ȼ������⣨NLU������Ȼ�������ɣ�NLG����ͨ����ϵ�ṹ��BERT��GPT-2��RoBERTa��XLM��DistilBert��XLNet�ȣ�

### Create virtual environment
```python
conda create -n env_name python=3.8 # create
Linux:  source activate env_nam # activate
Win: activate env_name
conda install -n env_name [package] # install pkg
pip install [package] # ��Ҫȷ��pip��·���Ƿ������⻷����
Win: deactivate env_name # deactivate
Linux: source deactivate env_name
conda remove -n env_name --all # delete
```
### Run the tool
```python
# ·������Ҫ������~~~~~~~~
# ����src��trg������func-tokens�����ݺ���������tokens-pairs
python angr_method.py -src_bin src_path -trg_bin trg_path -angr_out out/ -article_out data/article/
# �����һ�����ɵ�token-pairsԤ�������ݼ�������Transformer����ѵ��
python preprocess_spacy.py -src_article src_file -trg_article trg_file -save_data data/prepare/openssl_origin_bcf.pkl
# ���ó�����ѵ��Transformer
python train.py -data_pkl data/prepare/openssl_origin_bcf.pkl -epoch 500 -b 16 -d_model 256 -d_inner_hid 512 -d_k 64 -d_v 64 -n_head 8 -n_layers 6 -dropout 0.2 -output_dir model_save -use_tb (-no_cuda)
# ����Transformerģ�ͷ������
python translate.py -model model_path -data_pkl use_dataset (-no_cuda)
```

### Scheme
1. across the Architecture/Optimization/Obfuscation base Function-level
2. instruction->word;basic-block->sentence;function-paragraph;program->article base Transformer
3. serial info / structural info

### Tensorboard
```python
tensorboard --logdir model_save/model_name # ע��·��������
```

### Bianry Obfuscation
1. ģ�������Bogus Control Flow(BCF):��ģ������ͨ����Ӵ�������ص����������ͷ�֧���޸�CFG�������ָ�ϲ�����������ԭʼ�����飬�����ѡ�������ָ����뵽ԭʼ�������С����ֻ����������ƻ���CFG�ͻ�����������ԣ�ʹ��������֧���ӻ��������ӽڵ�����������ָ��ͳ�����
2. ����������ƽ��Control Flow Flattening(FLA):����ģ��������һ�ֱ�ƽ���������̲��ԡ�ͨ����ӽ�������ѭ����������������ת����Ƕ��ѭ����صĿ��ƽṹ������ƽ�����ּ����ָ�����飬�Ӷ��������������ָ���α��֧��������CFG������ȫʧȥ����ԭ���Ľṹ��
3. ָ���滻Instructions Substitution(SUB):���ֻ�������ʹ����ݲ����������ԣ�ּ����������ͬ�������ӵ�ָ���滻�򵥵�ָ��ر��ǣ��ü�����Ҫȡ�����������������ӷ����������˷��ͳ��������ֻ����������޸Ļ���������ݣ���������ָ��߼�ָ������ȵ�������


### BCSA
1. �������ݼ�
```python
# create_input_list:����ҪIDA�����Ķ������б�
# get_done_list:���������IDA�����Ķ������б�
python preprocess_bcsa.py --data_folder "/mnt/JiangS/CodeTransformer/BCSA/DataSet/" --out "helper/input/linux/"
```
2. IDA������ȡ
```python
# IDA������win�½���
python do_idascript.py --idapath "D:\program_jiang\tool\IDA_Pro_v7.5_Portable" --idc "D:\program_jiang\Pro\CodeTransformer\BCSA\TikNib\tiknib\ida\fetch_funcdata_v7.5.py" --input_list "input/input_list_a2ps_all_all_all_all_a2ps.elf.txt" --log
# source_list:����Դ��ľ���·��
python extract_functype.py --input_list D:\program_jiang\Pro\CodeTransformer\BCSA\TikNib\helper\input\done_list_a2ps_all_all_all_all_a2ps.elf.txt --source_list "D:\program_jiang\Pro\CodeTransformer\BCSA\TikNib\helper\input\source_list_a2ps.txt" --ctags_dir "data/ctags" --threshold 1
python extract_features.py --input_list "D:\program_jiang\Pro\CodeTransformer\BCSA\TikNib\helper\input\done_list_a2ps_all_all_all_all_a2ps.elf.txt" --threshold 1
# ̰������ѡ��
# ����ָ�걣����log_out·���µ�result_save�ļ���
python test_roc.py --input_list "D:\program_jiang\Pro\BCA\BCSA\TikNib\helper\input\linux\done_list_openssl_all_clang-7.0_x86_64_all_libssl.so.elf.txt" --config None
```
3. ģ��ѵ��
```python
# ���ѧϰģ����linux�½���
# �Ƚ������IDA������ȡ��.pickle�ļ��ϴ�����Ӧ�������ļ���
# ѡ��ͬ��config�ļ����в�ͬ����
# --config��Ӳ����
# --train����ѵ�����ǲ���
# ģ�ͱ��浽log_out·���µ�model_save�ļ���
# ����ָ�걣����log_out·���µ�result_save�ļ���
# save_funcdatalist_csv�������ڱ��浱ǰ���ݼ������к�������
python train_dl.py --input_list "/mnt/JiangS/CodeTransformer/BCSA/TikNib/helper/input/linux/done_list_a2ps_all_clang-4.0_clang-obfus_all_all_a2ps.elf.txt" --config "/mnt/JiangS/CodeTransformer/BCSA/TikNib/config/gnu_type/config_gnu_normal_opti_O0-O3_type.yml" --config None --use_tb --debug --train
```
4. ���߻���
```python
python BCA/BCSA/TikNib/tiknib/CodeAttention/DrawPic.py # ����tiknib��attention�ڲ�ͬʵ������Ͳ�ָͬ���ϵ�����
```

5. Top-K����
> �Ӳ��Լ���ȡ��test_funcs���������ԣ��ֳ�querys(����ѯ����)��datas(����ѯ��Χ)�������������ƶȽ�������Ŀ��ֵ����querys������ֵ
> 1. ̰���㷨querysȡ��test_funcs��
> 2. ע��������querysȡ��test_funcs+valid_funcs������̰���㷨������ͬ