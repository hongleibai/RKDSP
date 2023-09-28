import numpy as np
import pandas as pd
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class MeanAggregator(nn.Module):
    def __init__(self, dim,activation=None):
        super(MeanAggregator,self).__init__()
        self.dim=dim
        self.self_W= nn.Parameter(torch.zeros(size=(dim, dim//2)))
        nn.init.xavier_uniform_(self.self_W.data)
        self.neigh_W = nn.Parameter(torch.zeros(size=(dim, dim // 2)))
        nn.init.xavier_uniform_(self.neigh_W.data)
        self.activate=activation

    def forward(self,self_emb,neigh_emb):
        agg_emb=torch.mean(neigh_emb,dim=1)
        from_self=torch.matmul(self_emb,self.self_W)
        from_neigh = torch.matmul(agg_emb,self.neigh_W)
        if self.activate:
            from_self = self.activate(from_self)
            from_neigh=self.activate(from_neigh)

        return torch.cat([from_self,from_neigh],dim=1)


class SageEncoder(nn.Module):
    def __init__(self,nlayer,feature_dim,alpha,dim,fanouts):
        super(SageEncoder,self).__init__()
        self.nlayer=nlayer
        self.aggregator=[]
        for layer in range(self.nlayer):
            activation=nn.ReLU() if layer<self.nlayer-1 else None
            mean_aggregator=MeanAggregator(dim,activation=activation).cuda()
            self.aggregator.append(mean_aggregator)
            self.add_module(f'mean_aggregator_{layer}',mean_aggregator)
        self.dims=[feature_dim]+[dim]*self.nlayer
        self.fanouts=fanouts

    def sample(self,features,sample_nodes):

        feature_list=[]
        for sample_node_list in sample_nodes:
            feature_list.append(features[sample_node_list,:])

        return feature_list

    def forward(self,features,sample_nodes):
        hidden=self.sample(features,sample_nodes)
        for layer in range(self.nlayer):
            aggregator=self.aggregator[layer]
            next_hidden=[]
            for hop in range(self.nlayer-layer):
                neigh_shape=[-1,self.fanouts[hop],self.dims[layer]]
                h=aggregator(hidden[hop],torch.reshape(hidden[hop+1],neigh_shape))
                next_hidden.append(h)
            hidden=next_hidden

        return hidden[0]


class GCN_lnc(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN_lnc, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act


        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):  # 3025, 21, 100    3025, 21, 21
        seq = seq.type(torch.float32)
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            adj = adj.type(torch.float32)
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        if self.act is not None:
            out = self.act(out)
            # out = torch.squeeze(out, 0)
        return out

class GCN_drug(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN_drug, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act


        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):  # 3025, 21, 100    3025, 21, 21
        seq = seq.type(torch.float32)
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            adj = adj.type(torch.float32)
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        if self.act is not None:
            out = self.act(out)
            # out = torch.squeeze(out, 0)
        return out

class GCN_dis(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN_dis, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act


        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):  # 3025, 21, 100    3025, 21, 21
        seq = seq.type(torch.float32)
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            adj = adj.type(torch.float32)
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        if self.act is not None:
            out = self.act(out)
            # out = torch.squeeze(out, 0)
        return out

class GCN_se(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN_se, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):  # 3025, 21, 100    3025, 21, 21
        seq = seq.type(torch.float32)
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            adj = adj.type(torch.float32)
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        if self.act is not None:
            out = self.act(out)
            # out = torch.squeeze(out, 0)
        return out


class CKD_lnc(nn.Module):
    def __init__(self, in_ft, out_ft, layers, act='prelu', bias=True, idx=0):  # 100, 100, 2
        super(CKD_lnc, self).__init__()
        self.layers = layers  # 2
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.gcn_list = []
        self.dim = [in_ft]+[out_ft]*layers  # 300

        # self.node_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        # nn.init.xavier_uniform_(self.node_trans.data)
        # self.graph_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        # nn.init.xavier_uniform_(self.graph_trans.data)

        for layer in range(self.layers):
            gcn = GCN_lnc(self.dim[layer], self.dim[layer+1], act='prelu' if layer != self.layers-1 else None)
            self.gcn_list.append(gcn)
            self.add_module(f'gcn_{idx}_{layer}', gcn)

    def readout(self, node_emb):
        return torch.sigmoid(torch.mean(node_emb, dim=1, keepdim=True).squeeze(dim=1))

    def forward(self, seq, adj):  # feature 3025, 21, 100    adj 3025, 21, 21
        out = seq
        for layer in range(self.layers):
            gcn = self.gcn_list[layer]
            out = out.type(torch.double)
            out = gcn(out, adj)
        graph_emb = self.readout(out)
        return out, graph_emb  # 3025, 21, 100 hi     3025, 100 li

class CKD_drug(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate, layers, act='prelu', bias=True, idx=0):  # 100, 100, 2
        super(CKD_drug, self).__init__()
        self.layers = layers  # 2
        # self.in_ft = in_dim
        # self.out_ft = out_ft
        # self.trans_list = nn.ModuleList()
        # self.dim = [in_ft]+[out_ft]*layers  # 300

        # self.node_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        # nn.init.xavier_uniform_(self.node_trans.data)
        # self.graph_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        # nn.init.xavier_uniform_(self.graph_trans.data)

        self.Attention1 = Attention(in_dim, hidden_dim, dropout_rate)
        self.Attention2 = Attention(hidden_dim, out_dim, dropout_rate)

        # for layer in range(self.layers):
        #     Attention1 = Attention(in_dim, hidden_dim, dropout_rate)
        #     Attention2 = Attention(hidden_dim, out_dim, dropout_rate)
        #     self.trans_list.append(Attention1)
        #     self.trans_list.append(Attention2)
            # self.add_module(f'trans_{idx}_{layer}', gcn)
            # gcn = GCN_drug(self.dim[layer], self.dim[layer+1], act='prelu' if layer != self.layers-1 else None)
            # self.gcn_list.append(gcn)
            # self.add_module(f'gcn_{idx}_{layer}', gcn)

    def readout(self, node_emb):
        return torch.sigmoid(torch.mean(node_emb, dim=1, keepdim=True).squeeze(dim=1))

    def forward(self, features, x, y):  # feature 3025, 21, 100    adj 3025, 21, 21
        # out = seq
        # for layer in range(self.layers):
        #     gcn = self.gcn_list[layer]
        #     out = out.type(torch.double)
        #     out = gcn(out, adj)
        Att1 = self.Attention1(features)
        Att2 = self.Attention2(Att1)
        out_d = Att2[x.tolist()]
        out_s = Att2[y.tolist()]

        graph_emb_d = self.readout(out_d)
        graph_emb_s = self.readout(out_s)

        return out_d, out_s, graph_emb_d, graph_emb_s  # 3025, 21, 100 hi     3025, 100 li

class CKD_dis(nn.Module):
    def __init__(self, in_ft, out_ft, layers, act='prelu', bias=True, idx=0):  # 100, 100, 2
        super(CKD_dis, self).__init__()
        self.layers = layers  # 2
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.gcn_list = []
        self.dim = [in_ft]+[out_ft]*layers  # 300

        # self.node_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        # nn.init.xavier_uniform_(self.node_trans.data)
        # self.graph_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        # nn.init.xavier_uniform_(self.graph_trans.data)

        for layer in range(self.layers):
            gcn = GCN_dis(self.dim[layer], self.dim[layer+1], act='prelu' if layer != self.layers-1 else None)
            self.gcn_list.append(gcn)
            self.add_module(f'gcn_{idx}_{layer}', gcn)

    def readout(self, node_emb):
        return torch.sigmoid(torch.mean(node_emb, dim=1, keepdim=True).squeeze(dim=1))

    def forward(self, seq, adj):  # feature 3025, 21, 100    adj 3025, 21, 21
        out = seq
        for layer in range(self.layers):
            gcn = self.gcn_list[layer]
            out = out.type(torch.double)
            out = gcn(out, adj)
        graph_emb = self.readout(out)
        return out, graph_emb  # 3025, 21, 100 hi     3025, 100 li

class CKD_se(nn.Module):
    def __init__(self, in_ft, out_ft, layers, act='prelu', bias=True, idx=0):  # 100, 100, 2
        super(CKD_se, self).__init__()
        self.layers = layers  # 2
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.gcn_list = []
        self.dim = [in_ft]+[out_ft]*layers  # 300

        # self.node_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        # nn.init.xavier_uniform_(self.node_trans.data)
        # self.graph_trans = torch.nn.Parameter(torch.zeros(size=(self.out_ft, self.out_ft)))
        # nn.init.xavier_uniform_(self.graph_trans.data)

        for layer in range(self.layers):
            gcn = GCN_se(self.dim[layer], self.dim[layer+1], act='prelu' if layer != self.layers-1 else None)
            self.gcn_list.append(gcn)
            self.add_module(f'gcn_{idx}_{layer}', gcn)

    def readout(self, node_emb):
        return torch.sigmoid(torch.mean(node_emb, dim=1, keepdim=True).squeeze(dim=1))

    def forward(self, seq, adj):  # feature 3025, 21, 100    adj 3025, 21, 21
        out = seq
        for layer in range(self.layers):
            gcn = self.gcn_list[layer]
            out = out.type(torch.double)
            out = gcn(out, adj)
        graph_emb = self.readout(out)
        return out, graph_emb  # 3025, 21, 100 hi     3025, 100 li

class FC_d(nn.Module):
    def __init__(self):
        super(FC_d, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4900, 500, bias=True),
            nn.ReLU(),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, data):
        data = data.to(torch.float32)  # 4298*4560
        # res = []
        temp_res = self.fc(data)
        # res.append(temp_res)

        return temp_res

class FC_d_n(nn.Module):
    def __init__(self):
        super(FC_d_n, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4900, 500, bias=True),
            nn.ReLU(),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, data):
        data = data.to(torch.float32)  # 4298*4560
        # res = []
        temp_res = self.fc(data)
        # res.append(temp_res)

        return temp_res

class FC_s(nn.Module):
    def __init__(self):
        super(FC_s, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4900, 500, bias=True),
            nn.ReLU(),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, data):
        data = data.to(torch.float32)  # 4298*4560
        # res = []
        temp_res = self.fc(data)
        # res.append(temp_res)

        return temp_res

class FC_s_n(nn.Module):
    def __init__(self):
        super(FC_s_n, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4900, 500, bias=True),
            nn.ReLU(),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, data):
        data = data.to(torch.float32)  # 4298*4560
        # res = []
        temp_res = self.fc(data)
        # res.append(temp_res)

        return temp_res


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2000, 1000, bias=True),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1000, 2, bias=True)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, data):
        data = data.to(torch.float32)  # 4298*4560
        res = []
        # 将 lncRNA 和疾病的嵌入表示通过两个全连接层，降维到（1，2）
        # for i in range(data.shape[0]):
        temp_res = self.fc(data)
        temp_res = temp_res.view(-1, 2)
        res.append(temp_res)

        return res

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate, num_heads=8):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(out_dim / num_heads)
        self.all_head_size = out_dim

        self.query = nn.Linear(in_dim, self.all_head_size)
        self.key = nn.Linear(in_dim, self.all_head_size)
        self.value = nn.Linear(in_dim, self.all_head_size)

        self.drop = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1)

    def forward(self, x):
        mixed_query = self.query(x)
        mixed_key = self.key(x)
        mixed_value = self.value(x)

        query = self.transpose_for_scores(mixed_query)
        key = self.transpose_for_scores(mixed_key)
        value = self.transpose_for_scores(mixed_value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.drop(attention_scores)

        res = torch.matmul(attention_scores, value)
        res = res.permute(0, 2, 1).contiguous()
        res_shape = res.size()[:-2] + (self.all_head_size,)
        res = res.view(*res_shape)

        return res

class global_EncodingModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate):
        super(global_EncodingModule, self).__init__()

        self.Attention1 = Attention(in_dim, hidden_dim, dropout_rate)
        self.Attention2 = Attention(hidden_dim, out_dim, dropout_rate)

    def forward(self, features, x, y):
        Att1 = self.Attention1(features)
        Att2 = self.Attention2(Att1)

        return Att2[x.tolist()], Att2[y.tolist()]

