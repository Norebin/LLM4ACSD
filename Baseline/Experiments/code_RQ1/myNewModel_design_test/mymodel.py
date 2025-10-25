
from numpy import matmul
import torch
from torch import nn
import math
from torch.nn import *
from torch import Tensor
from typing import Optional, Any
from torch_geometric.nn import GCNConv
from torch_geometric.nn.glob import GlobalAttention
import torch.nn.functional as F
from selfattentionlayer import SelfAttention
from global_self_att import GlobalSelfAttentionLayer
from bi_lstm import LSTMModel


def _no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)

class MyDynamicGCN(nn.Module):
    def __init__(self, num_gcn_layers, hidden):
        super(MyDynamicGCN, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.gcn_layers = nn.ModuleList()
        self.mlp_gates = nn.ModuleList()
        self.gpools = nn.ModuleList()
        self.device = 'cuda'

        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(hidden, hidden))
            self.mlp_gates.append(nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid()))
            self.gpools.append(GlobalAttention(gate_nn=self.mlp_gates[-1]))

        self.aaaaa = nn.Parameter(torch.empty(size=(13, 1))).to('cuda')
        nn.init.xavier_uniform_(self.aaaaa.data, gain=1.414)

    def reset_parameters(self):
        for gcn_layer in self.gcn_layers:
            gcn_layer.reset_parameters()
        for gpool in self.gpools:
            gpool.reset_parameters()

    def forward(self, x, edge_index):
        #edge_weight = torch.matmul(edge_attr, F.softmax(self.aaaaa, dim=0))
        #edge_weight = edge_weight.reshape(len(edge_weight))

        readout = []

        for i in range(self.num_gcn_layers):
            #x = self.gcn_layers[i](x, edge_index, edge_weight)
            x = self.gcn_layers[i](x, edge_index)
            x = F.relu(x)
            batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
            readout.append(self.gpools[i](x, batch=batch))

        output = torch.stack(readout, dim=1)
        return output

class MyTransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None) -> None:
        super(MyTransformerEncoder, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            src_mask: the additive mask for the src sequence (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - src_mask: :math:`(S, S)`.
            - src_key_padding_mask: :math:`(N, S)`.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        if src.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return memory

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FullModel(nn.Module):
    def __init__(self, vocablen, metriclen, hidden, d_model, nhead, num_gcn_layers, num_encoder_layers, dim_feedforward, dropout, alpha):
        super(FullModel, self).__init__()
        self.device = 'cuda'
        self.gcn = MyDynamicGCN(num_gcn_layers, hidden)
        self.transformer = MyTransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        
        self.embed=nn.Embedding(vocablen,hidden)
        self.metric_embed=nn.Embedding(metriclen,hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        # self.W1 = nn.Parameter(torch.empty(size=(num_gcn_layers*hidden, hidden)).to('cuda'))
        # nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        # self.W2 = nn.Parameter(torch.empty(size=(11*hidden, hidden)).to('cuda'))
        # nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.mlp_gate1 = nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool1 = GlobalAttention(gate_nn=self.mlp_gate1)

        self.mlp_gate2 = nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool2 = GlobalAttention(gate_nn=self.mlp_gate2)

        self.attention = GlobalSelfAttentionLayer(2*hidden, hidden, dropout, alpha)

        self.fc = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, edge_attr, metrics):
        x = self.embed(x)
        metrics = self.pos_encoder(self.metric_embed(metrics))
        out1 = self.gcn(x, edge_index).squeeze(0)
        out2 = self.transformer(metrics).squeeze(1)
        # print('out',out1.shape,out2.shape)
        # exit()
        batch1 = torch.zeros(out1.size(0),dtype=torch.long).to(self.device)
        out1 = self.gpool1(out1, batch1)
        batch2 = torch.zeros(out2.size(0),dtype=torch.long).to(self.device)
        out2 = self.gpool2(out2, batch2)
        out = torch.cat([out1,out2],dim=1)
        out = self.attention(out)
        out = F.softmax(self.fc(out), dim=-1)
        return out

        
        

class FullModel_DeepFusion(nn.Module):
    def __init__(self, vocablen, metriclen, hidden, d_model, nhead, num_gcn_layers, num_encoder_layers, dim_feedforward, dropout, alpha):
        super(FullModel_DeepFusion, self).__init__()
        self.device = 'cuda'
        self.gcn = MyDynamicGCN(num_gcn_layers, hidden)
        self.transformer = MyTransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        
        self.embed=nn.Embedding(vocablen,hidden)
        self.metric_embed=nn.Embedding(metriclen,hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.fc_l = nn.Linear(num_gcn_layers*hidden, hidden)
        self.fc_r = nn.Linear(11*hidden, hidden)
        self.fc_o_1 = nn.Linear(2*hidden, hidden)
        self.fc_o_2 = nn.Linear(hidden, 2)

        self.W1 = nn.Parameter(torch.empty(size=(3*hidden, hidden)).to('cuda'))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W2 = nn.Parameter(torch.empty(size=(11*hidden, hidden)).to('cuda'))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.Wo = nn.Parameter(torch.empty(size=((3+11)*hidden, 2*hidden)).to('cuda'))
        nn.init.xavier_uniform_(self.Wo.data, gain=1.414)

    def forward(self, x, edge_index, edge_attr, metrics):
        x = self.embed(x)
        metrics = self.pos_encoder(self.metric_embed(metrics))
        out1 = self.gcn(x, edge_index, edge_attr).reshape(1,-1)
        #out1 = F.elu(torch.mm(out1, self.W1))
        #out1 = self.fc_l(out1)
        out2 = self.transformer(metrics).squeeze(1).reshape(1,-1)
        #out2 = F.elu(torch.mm(out2, self.W2))
        #out2 = self.fc_r(out2)
        deep_fusion = torch.cat((out1,out2),dim=1)
        deep_fusion = F.elu(torch.mm(deep_fusion, self.Wo))
        #fusion = torch.cat((out1,out2),dim=1)
        output =  self.fc_o_2(F.elu(self.fc_o_1(deep_fusion)))
        result = F.softmax(output,dim=-1)
        return result

#  -------------------------------------baselines----------------------------------------

class myDNN(nn.Module):
    def __init__(self, hidden):
        super(myDNN, self).__init__()
        hidden_size = hidden
        metriclen = 11

        self.dense_layer1 = nn.Linear(metriclen, hidden_size)
        self.dense_layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.dense_layer3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.dense_layer4 = nn.Linear(hidden_size//4, hidden_size//8)
        self.dense_layer5 = nn.Linear(hidden_size//8, hidden_size//16)
        self.fc = nn.Linear(hidden_size//16, 2)


    def forward(self, src_metrics):
        src_metrics = src_metrics.reshape(1,-1)
        src_metrics = F.normalize(src_metrics,dim=0)
        #print("src_metrics",src_metrics.shape,src_metrics)
        h = self.dense_layer1(src_metrics)
        h = F.relu(h)
        h = self.dense_layer2(h)
        h = F.relu(h)
        h = self.dense_layer3(h)
        h = F.relu(h)
        h = self.dense_layer4(h)
        h = F.relu(h)
        h = self.dense_layer5(h)
        h = F.relu(h)
        out = F.softmax(self.fc(h), dim=-1)

        return out

class myBiLSTM(nn.Module):
    def __init__(self, vocablen, metriclen, hidden, dropout, alpha):
        super(myBiLSTM, self).__init__()
        self.device = 'cuda'

        self.token_embed=nn.Embedding(vocablen,hidden)
        self.metric_embed=nn.Embedding(metriclen,hidden)

        self.bi_lstm1 = LSTMModel(hidden, 128, 2)
        self.bi_lstm2 = LSTMModel(hidden, 128, 2)

        self.mlp_gate1 = nn.Sequential(nn.Linear(2*hidden,1),nn.Sigmoid())
        self.gpool1 = GlobalAttention(gate_nn=self.mlp_gate1)

        self.mlp_gate2 = nn.Sequential(nn.Linear(2*hidden,1),nn.Sigmoid())
        self.gpool2 = GlobalAttention(gate_nn=self.mlp_gate2)

        self.attention = GlobalSelfAttentionLayer(4*hidden, 2*hidden, dropout, alpha)
        
        self.fc = Linear(hidden*2, 2)

    def forward(self, token_list, metrics):
        
        token_list = token_list.reshape(-1,1)
        
        token_list = self.token_embed(token_list)
        metrics = self.metric_embed(metrics)
        
        out1 = self.bi_lstm1(token_list)
        out2 = self.bi_lstm2(metrics)

        batch1 = torch.zeros(out1.size(0),dtype=torch.long).to(self.device)
        out1 = self.gpool1(out1, batch1)
        batch2 = torch.zeros(out2.size(0),dtype=torch.long).to(self.device)
        out2 = self.gpool2(out2, batch2)
        out = torch.cat([out1,out2],dim=1)
        out = self.attention(out)
        out = F.softmax(self.fc(out), dim=-1)
        
        return out


#  -------------------------------------baselines----------------------------------------

class MyModel(nn.Module):
    def __init__(self, vocablen, metriclen, hidden, d_model, nhead, num_gcn_layers, num_encoder_layers, dim_feedforward, dropout, alpha):
        super(MyModel, self).__init__()
        self.device = 'cuda'

        self.FullModel = FullModel(vocablen, metriclen, hidden, d_model, nhead, num_gcn_layers, num_encoder_layers, dim_feedforward, dropout, alpha)
        #self.FullModel_DeepFusion = FullModel_DeepFusion(vocablen, metriclen, hidden, d_model, nhead, num_gcn_layers, num_encoder_layers, dim_feedforward, dropout, alpha)
        

    def forward(self, x, edge_index, edge_attr, metrics, token_list, src_metrics):
        
        output = self.FullModel(x, edge_index, edge_attr, metrics)
        #output = self.FullModel_DeepFusion(x, edge_index, edge_attr, metrics)
        
        return output