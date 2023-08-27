import os
from logging import getLogger
from typing import List


from sentencepiece import SentencePieceProcessor
##这个文件是用来读取已经训练好的sentencepiece分词器,训练分词器请见Dataset/data_process.py

logger =getLogger()


class Tokenizer:
    def __init__(self,model_path:str):
        self.sp_model =SentencePieceProcessor(model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        self.n_words =self.sp_model.vocab_size()  #词表大小
        self.bos_id =self.sp_model.bos_id()
        self.eos_id =self.sp_model.eos_id()
        self.pad_id =self.sp_model.pad_id()

        logger.info(f"#words:{self.n_words},bos:{self.bos_id},eos:{self.eos_id},pad:{self.pad_id}")

    def encode(self,s:str,bos:bool,eos:bool): #编码 将字符串转为词索引并视情况加上bos和eos

        t =self.sp_model.encode(s)
        if bos:
            t =[self.bos_id]+t
        if eos:
            t =t+[self.eos_id]

        return t

    def decode(self,t:List[int]):           #解码 将词索引序列解码回字符序列
        return self.sp_model.decode(t)