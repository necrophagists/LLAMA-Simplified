#这个文件用来读取数据文件并用sentencepice分词,假设数据为文本行的txt
import torch

import sentencepiece as spm



#https://github.com/brightmart/nlp_chinese_corpus 数据集 选择了json社区问答语料
#选取前三万行来进行训练
def training():
    file_path ='chinese_data_tiny.txt'
    cn_vocab_size=5000
    model_name ='cn_vocab'
    model_type ='bpe'
    character_coverage =0.9995

    input_argument ='--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s '\
                    '--character_coverage=%s --pad_id=0 --eos_id=2 --bos_id=1 --unk_id=3'


    #character_coverage 句子分词之后有多少落在此表里? 适用中文这种。
    cmd =input_argument%(file_path,model_name,cn_vocab_size,model_type,character_coverage)
    print("start trainging...")
    spm.SentencePieceTrainer.Train(cmd)
    print('finshed training')

def ceshi():
    sp =spm.SentencePieceProcessor()
    text ='我不想上班'
    sp.Load('cn_vocab.model')
    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))


training()
ceshi()
