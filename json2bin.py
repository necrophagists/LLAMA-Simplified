from LLAMA.chatglm2_tokenizer.tokenization_chatglm import ChatGLMTokenizer

import json
import numpy as np
from tqdm import tqdm
import pandas as pd

tokenizer =ChatGLMTokenizer(vocab_file='./chatglm2_tokenizer/tokenizer.model')
print(tokenizer.vocab_size)
def process_wiki_clean():
    with open('./Dataset/wikipedia-cn-20230720-filtered.json','r',encoding='utf-8') as f:
        data =json.load(f)
    data_list=[]
    for line in tqdm(data):
        text =line['completion']

        text_ids =tokenizer.encode(text,add_special_tokens=False)
        text_ids.append(tokenizer.special_tokens['<eos>'])  #每句话后面加eos
        #滤除长度太短的样本
        if len(text_ids)>5:
            data_list+=text_ids
    print(f"wiki dataset gather { len(data_list)} samples")
    #转为二进制文件 先转成数组 为什么要存为uint16？
    arr =np.array(data_list,dtype=np.uint16)
    print(arr.shape)

    with open('./Dataset/wiki.bin','wb') as f:
        f.write(arr.tobytes())
#point: (1)每句话encode之后加eos (2)因为glm2的词表小于65535 所以使用uint16就行了 能节省内存 (3)tobytes() 转为二进制数据


def process_baidu():
    data_list=[]
    with open("./Dataset/web_text_zh_train.json", 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            temp = json.loads(line)
            text =temp['title']+temp['content']
            text_ids =tokenizer.encode(text,add_special_tokens=False)
            text_ids.append(tokenizer.special_tokens['<eos>'])  #每句话后面加eos作为样本间的分隔符
            #滤除长度太短的样本
            if len(text_ids)>20:
                data_list+=text_ids
    print(f"baidu dataset gather {len(data_list)} samples")
    #转为二进制文件 先转成数组 为什么要存为uint16？
    arr =np.array(data_list,dtype=np.uint16)
    print(arr.shape)

    with open('./Dataset/baidu.bin','wb') as f:
        f.write(arr.tobytes())
#process_wiki_clean()
# process_baidu()