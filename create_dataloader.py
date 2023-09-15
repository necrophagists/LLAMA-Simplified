import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


#参考baby-llama2-chinese
#由于是decoder-only的模型,不适用MLM,此外也没使用GLM的方式进行span mask(后续打算实现)，仅采用了最常规的自回归预测任务: input = X[:-1] label =X[1:]
#之前制作二进制数据时利用了<eos>分割的方式来保留样本内部的语义边界，从而避免了在预训练时的额外padding。
#预训练仍然是基于自回归的方式进行的，因为在每个块内，模型可以依据已知的部分来预测下一个部分，只是每个块内可能包含多个句子。
#这样做的好处:1.不需要padding了 2.这样子分割比较符合实际建模。通过按照实际语义分割样本，模型可以更好地学习到不同句子之间的上下文关系。
class PretrainedDataset(Dataset):
    def __init__(self,data_path,max_seq_len=256,memmap=False):
        super(PretrainedDataset,self).__init__()
        self.data=[]
        if memmap:                                         #f.seek(offset,idx)  idx={0:从头开始,1:从当前读取点开始,2:从末尾开始} offset-读取offset个字节
            for dp in data_path:
                with open(dp,'r') as f:      #不要使用二进制读取
                    f.seek(0,2)                    #这里是直接把指针移到文件末尾了
                    flen = f.tell()//np.dtype('uint16').itemsize  #.tell()返回文件现在的位置 然后要除以当前保存的类型的字节大小才能得token总数
                self.data.append(np.memmap(dp,dtype=np.dtype('uint16'),shape=(flen//max_seq_len,max_seq_len))) #
                    #然后使用np.memmap读取进虚拟内存中 这是内存映射的技术。
                    #内存映射（memmap）适用于处理大于物理内存容量的数据文件。由于内存映射只加载部分数据块到内存中，通过虚拟内存进行访问，可以避免一次性加载整个大文件到内存
                    #从而节省内存资源。如果数据文件较小，可以完全加载到内存中，这时使用普通的读取方式更加方便和高效。
        else:
            data_list =[]
            for dp in data_path:
                with open(dp,'rb') as f:
                    data =np.fromfile(f,dtype=np.uint16)
                    data_list.append(data)
                data =np.concatenate(data_list)
                data =data[:max_seq_len*((len(data)//15)//max_seq_len)]
                #data = data[:max_seq_len * (len(data)// max_seq_len)]
                self.data.append(data.reshape(-1,max_seq_len))

        self.data=np.concatenate(self.data,axis=0)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
         #是不是需要加bos?
         sample =self.data[idx]#我不想上班<eos>你好
         input = np.array(sample[:-1]).astype(np.int64)   #我不想上班<eos>你
         label =np.array(sample[1:]).astype(np.int64)    #不想上班<eos>你好

         return torch.from_numpy(input),torch.from_numpy(label)

class SFTDataset(Dataset):
    def __init__(self):
        super(SFTDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

class RMDataset(Dataset):
    def __init__(self):
        super(RMDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

# data_path0 =r"C:\LCY\AICode\LLAMA\Dataset\wiki.bin"
# data_path1=r"C:\LCY\AICode\LLAMA\Dataset\baidu.bin"
#
# train_dataset =PretrainedDataset(data_path=[data_path0,data_path1])
#
# train_loader =DataLoader(train_dataset,pin_memory=False,batch_size=2,shuffle=False)
#
# for x,y in train_loader:
#     print(x.shape)
#     print(y.shape)
#     break