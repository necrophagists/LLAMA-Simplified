import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

#Literal 字面量 一看就知道是什么
#TypedDict 创建特定键和值的字典

import torch
import torch.nn.functional as F

from LLAMA.llama.model import ModelArgs,Transformer
from LLAMA.llama.tokenizer import Tokenizer

Role =Literal["system",'user','assistant'] #设定角色

class Message(TypedDict):
    role:Role
    content:str

class CompletionPrediction(TypedDict,total=False): #引入非必填项 3.11有NOTREQURED
    generation :str
    tokens: List[str]  #非必需
    logprobs: List[float] #非必需

class ChatPredication(TypedDict,total=False):
    generation:Message
    tokens:List[str]
    logprobs: List[float]  # 非必需

Dialog =List[Message]

class TiebaGPT:

    @staticmethod  #静态方法不用self
    def build(
            ckpt_dir:str,
            tokenizer_path:str,
            max_seq_len:int,
            max_batch_size:int,
            model_paralle_size:Optional[int] =None #是否模型并行
    ) ->"TiebaGPT": #怪

         start =time.time()
         #load ckpt model_args and tokenzier
         checkpoint =torch.load(ckpt_dir,map_location='cuda')

         model_args =ModelArgs(
             max_batch_size=max_batch_size,
             max_seq_len=max_seq_len
         )

         tokenizer =Tokenizer(model_path=tokenizer_path)
         model_args.vocab_size =tokenizer.n_words
         torch.set_default_tensor_type(torch.cuda.HalfTensor) #半精度


         #load model
         model =Transformer(model_args)
         model.load_state_dict(checkpoint,strict=False)
         print(f"加载模型花了{time.time()-start:.2f} 秒")

         return TiebaGPT(model,tokenizer)  #build函数用来初始化模型和那个tokenzier

    def __init__(self,model,tokenzier):
         self.model =model
         self.tokenizer =tokenzier

    @torch.inference_mode() #推理模式
    def generate(self, prompt_tokens:List[List[int]], #批推理 送进来的已经是tokenizer后的序列了
        max_gen_len:int,
        temperature:float =0.5,        #温度 softmax分数的缩放值
        top_p:float =0.9,              #贪婪和随机搜索之间的一个阈值
        logprobs:bool =False,          #对数概率
        echo:bool =False,
    ) ->Tuple[List[List[int]],Optional[List[List[int]]]]:
        # 返回一个tuple 有生成的batch数据 以及 可能返回logits
        ##思路 先确定输出长度 做好padding和mask
        ##然后输出模型 对输出logits进行介于贪心和束搜索之间的

        params =self.model.params
        bsz = len(prompt_tokens)

        min_prompt_len =min([len(t) for t in prompt_tokens])
        max_prompt_len =max([len(t) for t in prompt_tokens])

        total_len =max(max_prompt_len+max_gen_len,params.max_seq_len) #最长的序列

        pad_id =self.tokenizer.pad_id
        #要把batch数据转为输入的token了
        tokens =torch.full((bsz,total_len),pad_id,dtype=torch.long,device='cuda')
        for k,t in enumerate(prompt_tokens):
            tokens[k,:len(t)] =torch.tensor(t,dtype=torch.long,device='cuda')

        #如果要记录logorobs
        if logprobs:
            token_logprobs =torch.zeros_like(tokens,dtype=torch.float)

        #生成mask
        #因为要用到kv_cache所以要记录当前送入模型进行推理的最后一个token1的位置(start_pos)还有上一时刻的起点

        eos_reached =torch.tensor([False]*bsz,device='cuda') #记录batch内数据是否都生成完毕
        input_text_mask =tokens !=pad_id
        prev_pos =0

        #生成的话也是从prompt的最后一个开始生成 那么就需要把0-n-1长度给传入模型做推理,并且要从最小的prompt长度这儿进行推理
        for cur_pos in range(min_prompt_len,total_len):
            logits =self.model.forward(tokens[:,prev_pos:cur_pos],prev_pos)
            if logprobs:
                #logits --bs *seq_len *vocab_size
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1], #目标是整数
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature>0:
                #temperature是对输出logits进行softmax时缩放的一个操作
                #并且由于是生成 一次只生成一个token 对-1位置的进行计算就行 变为2d  bs *1
                probs =torch.softmax(logits[:,-1]/temperature,dim=-1)
                next_token =sample_top_p(probs,top_p)
            else:
                #贪心
                next_token =torch.argmax(logits[:,-1],dim=-1)
            next_token =next_token.reshape(-1)  #size :bsz
            #然后要考虑下一个元素是不是pad了 torch.where (condition,x,y)
            next_token =torch.where(input_text_mask[:,cur_pos],tokens[:,cur_pos],next_token)

            #把当前生成的和tokens覆盖掉
            tokens[:,cur_pos] =next_token
            #更新eos_reached
            #要保证模型在生成eostoken时当前预测的位置为pad 不能是prompt
            eos_reached =eos_reached | ((next_token==self.tokenizer.eos_id) &(~input_text_mask[:,cur_pos]))
            prev_pos = cur_pos
            if all(eos_reached) ==True:
                break
        if logprobs:
            token_logprobs =token_logprobs.tolist()
        out_tokens,out_logprobs=[],[]

        #for循环后的tokens即为生成的序列
        for i,toks in enumerate(tokens.tolist()):#bs seq
            #因为是生成,所以只要生成的部分就行，要裁剪
            start =0 if echo else len(prompt_tokens[i]) #echo就是从头
            toks =toks[start:max_gen_len+len(prompt_tokens[i])]
            probs =None
            if logprobs:
                probs =token_logprobs[i][start:max_gen_len+len(prompt_tokens[i])]
            #删去eos 有可能不用删 因为这个时候生成超过最长了
            if self.tokenizer.eos_id in toks:
                eos_idx =toks.index(self.tokenizer.eos_id)
                toks =toks[:eos_idx]
                probs =probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)

        return (out_tokens, out_logprobs if logprobs else None)


    #一个generate 一个chat接口
    def batch_inference(self,
        text:List[str],
        temperature:float=0.6,
        top_p:float=0.9,
        max_gen_len:Optional[int]=None,
        logprobs:bool=False,
        echo:bool=False):

        if max_gen_len is None:
            max_gen_len =self.model.max_seq_len -1 #留一个给bos 但是我觉得没必要

        prompt_tokens =[self.tokenizer.encode(t,bos=True,eos=False) for t in text]

        generation_tokens,generation_logprobs =self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo)
        if logprobs:
           return   [{"generation":self.tokenizer.decode(gt),'logprobs':gl} for gt,gl in zip(generation_tokens,generation_logprobs)]
        return  [{"generation":self.tokenizer.decode(gt)} for gt in generation_tokens]
    def chat(self,
             text: str,
             history:Optional[str]="",
             temperature: float = 0.6,
             top_p: float = 0.9,
             max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len =self.model.max_seq_len -1 #留一个给bos 但是我觉得没必要

        text =history+text
        prompt_tokens =self.tokenizer.encode(text,bos=True,eos=False)

        response,generation_logprobs =self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=False,
            echo=False)

        return (response[0],text+self.tokenizer.decode(response[0]))
def sample_top_p(probs,p):
    #介于贪心和书搜索之间
    #思路 先排序 然后求累计和 然后再按top_p来mask掉不符合要求的 然后剩下的按权重来抽取1个
    probs_sort,probs_idx =torch.sort(probs,dim=-1,descending=True) #降序
    probs_sum =torch.cumsum(probs_sort,dim=-1) #类似于求cdf
    mask =probs_sum -probs_sort >p  #cdf-pdf 就是如果自己的概率小于(1-p)就舍去
    probs_sort[mask] =0.0
    #接下来归一化
    probs_sort.div_(probs_sort.sum(dim=-1,keepdim=True))

    #按权重采样下标 支持1d 2d数据
    next_token =torch.multinomial(probs_sort,num_samples=1) #这个是sort的顺序下表 要到idx里去找
    next_token =torch.gather(probs_idx,-1,next_token) #idx里才是每个单词的idx
    return next_token