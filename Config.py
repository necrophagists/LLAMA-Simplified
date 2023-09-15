from dataclasses import dataclass
from typing import Optional,List,Tuple


class Args:
    # model args
        dim: int = 64                     #d_model
        n_layers: int = 2                 #n_block
        n_heads: int = 2                  #n_head
        n_kv_heads: Optional[int] = None  # GQA  KV分为几组?
        vocab_size: int = 64789               #词表大小
        norm_eps: float = 1e-5              # 平滑
        dropout: int = 0.1
        use_kv_cache: bool = False          #是否使用kv缓存？推理的时候用
        use_flashAttn: bool = False         #是否使用flashattention
        is_train: bool = True               #现在是训练还是推理？

        # trainer args
        tokenizer_path: str ="./chatglm2_tokenizer/tokenizer.model"
        train_data_path_list: List[str]  = ["./Dataset/wiki.bin"]    #训练数据路径
        test_data_path_list: List[str] = None                        #测试数据路径
        memmap: str = False
        init_mode: str = "new"                                       #从头开始训练还是加载ckpt再训练？
        last_ckpt_path: str = ""                                     #上一次ckpt的文件夹
        lr: int = 1e-3                                               #初始lr
        epochs: int = 500
        max_batch_size: int = 48
        max_seq_len: int = 256                                       #最长句子长
        save_steps: int = 10000000000                                #每多少步保存一次ckpt
        logging_steps: int = 100                                     #每多少步输出一次日志
        gradient_checkpoint_step: int = 20                           #梯度累计多少部一次

        is_overwrite: bool = False                                   #保存ckpt的时候要不要覆盖？
        ckpt_path: str = "./ckpt"                                    #ckpt路径
        early_stop: bool = False                                     #是否早停？
        patience: int = 5                                            #早停的步数