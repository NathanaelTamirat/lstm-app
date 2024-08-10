from typing import Optional ,Tuple
import torch
from torch import nn
from labml_helpers.module import Module
'''
c is like the long-term memory, and h is like the short term memory.
We use the input x and h to update the long term memory. In the update, 
some features of c are cleared with a forget gate f, and some features 
i are added through a gate g.
'''
class LSTMCell(Module):
    def __init__(self,input_size: int, hidden_size: int, layer_norm: bool=False):
        super().__init__()
        '''
        the cell doesn't look at long term memory c when doing the update. It only modifies it. 
        Also c never goes through a linear transformation. This is what solves vanishing and exploding gradients.
        '''
        self.hidden_lin=nn.Linear(hidden_size,4*hidden_size)
        self.input_lin=nn.Linear(input_size,4*hidden_size,bias=False)

        #applying layer norm
        if layer_norm: #Applying layer normalization gives better results. i, f, g and o embeddings are normalized
            self.layer_norm=nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c=nn.LayerNorm(hidden_size) # ct normalized in ht=ot * tanh(LN(ct))
        else:
            self.layer_norm=nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c=nn.Identity()

    def forward(self,x: torch.Tensor,h: torch.Tensor,c: torch.Tensor):
        ifgo=self.hidden_lin(h)+self.input_lin(x)
        ifgo=ifgo.chunk(4,dim=-1)
        ifgo=[self.layer_norm[i](ifgo[i])  for i in range(4)]
        i, f, g, o = ifgo
        
        c_next=torch.sigmoid(f)*c+torch.sigmoid(i)*torch.tanh(g)
        h_next=torch.sigmoid(o)*torch.tanh(self.layer_norm_c(c_next))


        return h_next,c_next
        

class LSTM(Module):
    def __init__(self,input_size: int,hidden_size: int, n_layers: int):
        super().__init__()
        self.n_layers=n_layers
        self.hidden_size=hidden_size

        self.cells=nn.ModuleList([LSTMCell(input_size,hidden_size)] +
                                 [LSTMCell(hidden_size,hidden_size) for _ in range(n_layers-1)])
        
    def forward(self,x: torch.Tensor,state: Optional[Tuple[torch.Tensor,torch.Tensor]]=None):
        
        n_steps,batch_size=x.shape[:2]
        if state is None:
            h=[x.new_zeros(batch_size,self.hidden_size) for _ in range(self.n_layers)]
            c=[x.new_zeros(batch_size,self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h,c)=state
            h,c=list(torch.unbind(h)),list(torch.unbind(c))

        out=[]
        for t in range(n_steps):
            inp=x[t]
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
            
            out.append(h[-1])   

        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)
        return out, (h,c)

