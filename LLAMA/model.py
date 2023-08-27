import math

from dataclasses import dataclass
from typing import Any,Optional,Tuple

import torch
import torch.nn.functional as F

from torch import nn

device ='cuda' if torch.cuda.is_available() else 'cpu'

#BS *6*5
@dataclass  #修饰函数
class ModelArgs:
    dim:int =64
    n_layers:int =2
    n_heads: int =2
    n_kv_heads: Optional[int] =None        #GQA  KV分为几组?
    vocab_size:int = 5
    multiple_of: int =32  # swiGLU的设置
    norm_eps:float =1e-5  #平滑

    max_batch_size:int =8
    max_seq_len:int =256
    use_kv_cache =False
    use_flashAttn =False
    dropout=0.1
    train=True
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float =1e-6):
       super(RMSNorm,self).__init__()
       self.eps =eps
       self.weight =nn.Parameter(torch.ones(dim)) #1*4096维度的weight

    def _norm(self,x):
        return x *torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x):
        output =self._norm(x.float()).type_as(x)
        return output *self.weight


#旋转编码 g(xm,xn，m-n) 等价于 <fq(xm,m),fk(xn,n)>
#其中 设fq(xm,m) =WqXm*exp(imθ),fq(xm,m) =WkXn*exp(inθ)
#g(xm,xn,m-n) =Re[(WqXm)((WkXn)^*)exp(i(m-n)θ)]
#WqXm =qm  WkXn =kn
#并且可证fq(xm,m)=[[cosmθ,-sinmθ],[sinmθ,cosmθ]][qm1,qm2] #旋转矩阵 *（qm拆成两个）
#可以先求出所有的θj  对于batch*seq_len*d来说 对于d维 每两个元素为一组赋一个旋转角度θj
#是按头的维度来算的d_k


#对于位置m的q 以及位置n的k  乘以对应的d*d的旋转矩阵 其中里面的m就为当前位置，然后θ是每2特征维分配一个 即d维的输入 有d//2个theta

#θj =10000.pow(-2(j-1)/d) =1/10000.pow(2(j-1)/d)

#这里是直接先计算e^imθ

def precompute_freqs_cis(dim:int,end:int,theta:float=10000.0):
    freqs = 1.0/theta **(torch.arange(0,dim,2)[:(dim//2)].float()/dim).cuda()   #θj   #dim//2
    t =torch.arange(end,device=device)  #max_seq_len*2 t可以看做是公式里的m,n
    freqs =torch.outer(t,freqs).float() # 向量t 和向量freqs做内积 得到t *treqs的矩阵 算是计算了公式中的mθ or nθ  #max_seq_len *d//2
    #print(freqs.shape)     #max_seq_len*2,d_k
    #极坐标转换为笛卡尔坐标 A*cos(B) +i(A*sin(B))   #e^imθ   m-0,M
    freqs_cis =torch.polar(torch.ones_like(freqs),freqs) #float32->complex64 float64->complex128
    print(freqs_cis.shape)
    return freqs_cis
                         # seq_len *d_k
#GQA
def reshape_for_broadcast(freqs_cis:torch.Tensor,x: torch.Tensor): #x bs *seq_len *num_head *d_k/2 complex
    ndim =x.ndim  #求x的维度数量  这时freqs_cis的维度应该是 seq_len *d_model
    shape =[d if i==1 or i ==ndim-1 else 1 for i,d in enumerate(x.shape)]
    #加上batch维度以及多头的维度 这样就可以广播了

    return freqs_cis.view(*shape)  #*收集参数 **收集关键字参数,返回字典

def apply_rotary_emb(
        xq:torch.Tensor, #b * seq_len *num_head *d_k
        xk:torch.Tensor,#b *seq_len *num_kv_head *d_k
        freqs_cis:torch.Tensor,
) -> Tuple[torch.Tensor,torch.Tensor]:

#torch.view_as_complex 最后一维必须为2
#torch.view_as_real 负数转re
    #先转为最后一维为2的矩阵形式才能转为复数
    #torch.view_as_complex最后一维要是2才让输入 出来那个维度没了

    #把q k转为复数
    xq_ =torch.view_as_complex(xq.float().reshape(*xq.shape[:-1],-1,2)) #b *s *n_h *d_h/2
    xk_ =torch.view_as_complex(xk.float().reshape(*xq.shape[:-1],-1,2))
    #现在需要扩展freqs_cis的维度以便广播
    freqs_cis =reshape_for_broadcast(freqs_cis,xq_) #1,seq_len,1,d_h/2  #加入了batch维度以及head的维度
    xq_out =torch.view_as_real(xq_*freqs_cis).flatten(3) #b *s *n_h *d_h  #做的是点乘 会自动广播 就是这个旋转编码只在d_k和seq_len维度上有区别 和batch n_head维度无关
    xk_out =torch.view_as_real(xk_*freqs_cis).flatten(3)
    return xq_out.type_as(xq),xk_out.type_as(xk)

def repeat_kv(x:torch.Tensor,n_rep:int) ->torch.Tensor:
    bs,seq_len,n_kv_head,head_dim =x.shape
    if n_rep ==1:
        return x
    else:
        return (x[:,:,:,None,:].expand(bs,seq_len,n_kv_head,n_rep,head_dim)
                .reshape(bs,seq_len,n_kv_head*n_rep,head_dim))



class Attention(nn.Module):
    def __init__(self,args:ModelArgs):
        super(Attention, self).__init__()
        #kv头为什么还要再分   #空间换时间
        self.n_kv_heads =args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_local_heads =args.n_heads         #就是头的数量
        self.n_local_kv_heads =self.n_kv_heads   #kv头的数量
        self.n_rep =self.n_local_heads//self.n_local_kv_heads   #有几个组 ？到时候要广播n_rep次
        self.head_dim =args.dim //args.n_heads   #d_head

        assert  self.n_local_heads%self.n_local_kv_heads ==0

        #如果n_kv_heads =n_local_heads ==MHA   n_kv_heads=1 ==MQA else GQA
        self.wq =nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wk =nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wv =nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, self.n_kv_heads*self.head_dim, bias=False)

        self.attn_dropout=nn.Dropout(args.dropout)
        self.residual_dropout =nn.Dropout(args.dropout)


        self.train =args.train  #如果是训练模式 就使用self.mask 如果是推理模式 就不使用self.mask 改为根据seq_len可变的mask
        #是否使用flash attention
        self.use_flashAttn =args.use_flashAttn

        if self.use_flashAttn:
               if hasattr(torch.nn.functional,'scaled_dot_product_attention') ==False:
                     print("Pytorch should >=2.0!!!Still use origin MHA")
                     self.use_flashAttn =False

        #KV缓存? #推理的时候在用
        #因此第i+1轮推理时必然包含了第轮的部分计算。KVCache的出发点就在这里，缓存当前轮可重复利用的计算结果，下一轮计算时直接读取缓存结果，
        self.cache_k =torch.zeros(  #bs *seq_len *n_kv_head *d_head
            args.max_batch_size,
            args.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )
        self.cache_v=torch.zeros(
            args.max_batch_size,
            args.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )


    #源代码是自回归式的生成 不需要padding 训练时需要与padding代码结合。
    def forward(self,x:torch.Tensor,freqs_cis:torch.Tensor,mask:torch.Tensor,use_kv_cahce=False,start_pos=None):  #推理的时候才用到start_pos 用于kv cache
        bs,seq_len =x.shape[0],x.shape[1]

        xq,xk,xv =self.wq(x),self.wk(x),self.wv(x)
        xq =xq.view(bs,seq_len,self.n_local_heads,self.head_dim)
        xk =xk.view(bs,seq_len,self.n_local_kv_heads,self.head_dim)
        xv =xv.view(bs,seq_len,self.n_local_kv_heads,self.head_dim)

        xq,xk =apply_rotary_emb(xq,xk,freqs_cis=freqs_cis)


        if use_kv_cahce == True:  #for inference  推理 kv cache
           self.cache_k =self.cache_k.to(xk)
           self.cache_v =self.cache_v.to(xv)

           #先存 再取出来计算
           self.cache_k[:bs,start_pos:start_pos+seq_len] =xk
           self.cache_v[:bs,start_pos:start_pos+seq_len] =xv  #save_current kv

           keys   =self.cache_k[:bs,:start_pos+seq_len]
           values =self.cache_v[:bs,:start_pos+seq_len]

        keys =repeat_kv(xk,n_rep=self.n_rep) #广播
        values=repeat_kv(xv,n_rep=self.n_rep)

        xq = xq.transpose(1,2)
        keys =keys.transpose(1,2)
        values =values.transpose(1,2)

        weights = torch.matmul(xq,keys.transpose(-2,-1))/math.sqrt(self.head_dim)
        print(mask)
        weights =weights+mask  #会广播

        scores  = F.softmax(weights.float(),dim=-1).type_as(xq)
        scores  =self.attn_dropout(scores)
        output =torch.matmul(scores,values)
        output =output.transpose(1,2).contiguous().view(bs,seq_len,-1)
        return self.residual_dropout(self.wo(output))


class FeedForward(nn.Module):
    def __init__(self,args:ModelArgs):
        super(FeedForward, self).__init__()
        self.w1 =nn.Linear(args.dim,args.dim*4,bias=False)
        self.w2 =nn.Linear(args.dim*4,args.dim,bias=False)
        self.w3 =nn.Linear(args.dim,args.dim*4,bias=False)

        self.ffn_dropout =nn.Dropout()
    def forward(self,x:torch.Tensor):
        #silu   x*sigmoid(beta*x)
        #here is x*silu(x)
        return self.w2(F.silu(self.w1(x))*self.w3(x))
class TransformerBlock(nn.Module):
    def __init__(self,layer_id:int,args:ModelArgs):
        super(TransformerBlock, self).__init__()
        self.n_heads =args.n_heads
        self.dim =args.dim
        self.head_dim =args.dim //args.n_heads


        self.attention =Attention(args)
        self.ffn =FeedForward(
              args=args
        )
        self.layer_id =layer_id
        self.attention_norm = RMSNorm(dim=args.dim,eps=args.norm_eps)
        self.ffn_norm =RMSNorm(args.dim,eps=args.norm_eps)

    #KV 缓存这一块再说
    def forward(self,x,freqs_cis:torch.Tensor,mask=Optional[torch.Tensor],use_kv_cahce=False,start_pos=None):
        #pre RMSnorm +residual connection
        h =x +self.attention(self.attention_norm(x),freqs_cis,mask,use_kv_cahce,start_pos)
        out =h +self.ffn(self.ffn_norm(h))

        return out
class Transformer(nn.Module):
    def __init__(self,args:ModelArgs):
      super(Transformer, self).__init__()
      self.params =args
      self.vocab_size =args.vocab_size
      self.n_layers =args.n_layers

      self.token_embedding =nn.Embedding(args.vocab_size,args.dim)

      self.layers =torch.nn.ModuleList()
      for layer_id in range(args.n_layers):
          self.layers.append(TransformerBlock(layer_id,args))

      self.norm =RMSNorm(args.dim,args.norm_eps)
      self.output =nn.Linear(args.dim,args.vocab_size,bias=False)

      freqs_cis =precompute_freqs_cis(self.params.dim//self.params.n_heads,self.params.max_seq_len*2)
      self.register_buffer('freqs_cis',freqs_cis)

      self.train =args.train
      self.use_kv_cache =True
      if self.train:
        self.use_kv_cache =False
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), -float('inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)  # mask不用推理

    def init_weights(self,model):
        if isinstance(model,nn.Linear):
            nn.init.xavier_normal_(model.weight)
            if model.bias is not None:
                nn.init.xavier_normal_(model.bias)
        elif isinstance(model,nn.Embedding):
            nn.init.xavier_normal_(model.weight)
    #进来的token都是bs *seq_len 还没有embedding
    def forward(self,tokens:torch.Tensor,start_pos=None,pad_mask=None):
       bsz,seq_len, =tokens.shape

       h =self.token_embedding(tokens)
       #生成的时候 第一步因为从0-min_prompt_len 所以seq_len>1 需要mask 后面是一个一个生成的就不需要mask了

       for i, layer in enumerate(self.layers):
               h = layer(h, self.freqs_cis[:seq_len,:], self.mask[:,:,:seq_len,:seq_len],self.use_kv_cache,start_pos)
       h =self.norm(h)
       output =self.output(h)
       return output

##推理用kv缓存就没必要用flashattention了  训练阶段需要flash attention

m =ModelArgs()

from torchinfo import summary

model =Transformer(m).cuda()
model.train =False
input =torch.randint(low=0,high=5,size=(1,1,)).long().cuda()
model(input).cuda()

#现在的问题是self.mask要不要呢？




