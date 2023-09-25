import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
    

class Attention(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim//2),
            nn.Tanh(),
            nn.Linear(hid_dim//2, 1)
        )

    def forward(self, x):#[N,*,c]
        attn = self.attention(x)#[N,*,1]
        attn = torch.softmax(attn, dim=1) 
        x = torch.bmm(torch.transpose(attn, 1, 2), x)
        x = x.squeeze(dim=1) #[N, h]        
        return x#,attn
    
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=240):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False).to('cuda')
        return self.dropout(x)


class STTransformer(nn.Module):
    def __init__(self, joint_in_channels=3, joint_hidden_channels=64, time_window=9, time_step=3):
        super().__init__()
        #x
        self.cnn = nn.Conv2d(joint_in_channels, joint_hidden_channels, [1,16],[1,1])
        self.tcn = nn.Conv2d(joint_hidden_channels, joint_hidden_channels, [time_window,1],[time_step,1])
        encoder_layer = nn.TransformerEncoderLayer(d_model=joint_hidden_channels, nhead=4, dim_feedforward=4*joint_hidden_channels, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(0.6)  
        self.attention = Attention(joint_hidden_channels, joint_hidden_channels)
        self.pe = PositionalEncoding(joint_hidden_channels, dropout=0.6)
        self.linear = nn.Linear(joint_hidden_channels, 2)
        #instance to bag
        self.attention2 = Attention(2, joint_hidden_channels//2)
        #info
        self.info_linear1 = nn.Linear(4, joint_hidden_channels//2)
        self.info_linear2 = nn.Linear(joint_hidden_channels//2, 2)
        
        

        
    def forward(self, x, info=None): #x:[N,v,v-1,c,t] #info:[2,4]
        #x
        x = x.permute(0,4,3,1,2).contiguous() #x:[N,t,c,v,v-1]
        N,t,c,v,v_ = x.size()
        x = x.view(N*t,c,v,v_)
        x = self.cnn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(N,t,-1,v)
        x = x.permute(0,2,1,3).contiguous() #x:[N,c,t,v]
        x = self.tcn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.tcn(x)
        x = F.relu(x)
        x = self.dropout(x) #x:[N,c,t,v]
        ##spatial-temporal attention feature fusion
        N, c, t, v = x.size()
        x = x.permute(0,2,3,1).contiguous() #[N,t,v,c]
        x = x.view(N*t, v, c)
        x = self.transformer_encoder(x) #[N*t,v,c]
        x = self.attention(x) #[N*t,c]
        x = x.view(N, t, c) #[N,t,c]
        x = self.pe(x)
        x = self.transformer_encoder(x) #[N,t,c]
        x = self.attention(x) #[N,c]
        ##clip classifaction feature
        clip_cls = self.linear(x)
        ##bag classifaction feature
        x_train = clip_cls[:N//2].reshape(2,N//4,-1)
        x_train = torch.softmax(x_train,2)
        x_train_MI_cls = self.attention2(x_train)


        if not info is None:
            #info
            info = self.info_linear1(info) #[2,c]
            info = F.relu(info)
            info = self.dropout(info)
            info = self.info_linear2(info) #[2,2]
            info = torch.softmax(info,1)
            #fuse train data and persons info
            x_train_MI_cls = torch.softmax(torch.add(x_train_MI_cls, info),1)

        return x, clip_cls, x_train_MI_cls  
    
    
    
    
    
    
    
    
    
    
    