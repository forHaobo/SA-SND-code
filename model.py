"""
@FileName：model.py
@Description： SA-SDN中医药模型
@Author：HaoBoli
"""


from torch.nn import Embedding, Linear
from torch_geometric.nn import SAGEConv, GATConv
import torch
import torch.nn.functional as F
from attention import ScaledDotProductAttention, Attention
from torch import nn, optim, Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing

class LightGCN_ST(MessagePassing):
    def __init__(self, num_src, num_dst, embedding_dim=128, in_src_dim = 384, in_dst_idm = 103, K=2, add_self_loops=False):

        super().__init__()
        self.num_src, self.num_dst = num_src, num_dst
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.src_fc = Linear(in_src_dim, embedding_dim)
        self.dst_fc = Linear(in_dst_idm, embedding_dim)

        self.src_emb = nn.Embedding(num_embeddings=self.num_src, embedding_dim=self.embedding_dim) # e_u^0
        self.dst_emb = nn.Embedding(num_embeddings=self.num_dst, embedding_dim=self.embedding_dim) # e_i^0

        nn.init.normal_(self.src_emb.weight, std=0.1)
        nn.init.normal_(self.dst_emb.weight, std=0.1)

    def forward(self,z_src, z_dst, edge_index: SparseTensor):

        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        src_e = self.src_fc(z_src)
        dst_e = self.dst_fc(z_dst)
        src_e = self.src_emb.weight + src_e
        dst_e = self.dst_emb.weight + dst_e
        emb_0 = torch.cat([src_e, dst_e])

        embs = [emb_0]
        emb_k = emb_0

        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)
        src_emb_final, dst_emb_final = torch.split(emb_final, [self.num_src, self.num_dst])
        return src_emb_final, dst_emb_final

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)

class self_gate(nn.Module):
    def __init__(self, embedding_dim):
        super(self_gate, self).__init__()
        self.embedding_dim = embedding_dim
        self.Linear = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, embedding):
        gate = self.Linear(embedding)
        gate = torch.sigmoid(gate)
        return embedding * gate

class SymAgg(torch.nn.Module):
    def __init__(self, hidden_channels, args, device):
        super().__init__()
        self.syn_agg = Attention(hidden_channels, device=device)
        self.sym_agg = Attention(hidden_channels, device=device)
        self.sAtt = ScaledDotProductAttention(d_model=hidden_channels, d_k=hidden_channels, d_v=hidden_channels, h=1)
        self.hidden_channels = hidden_channels
        self.attention = torch.nn.Parameter(torch.randn(1, self.hidden_channels))
        self.device = device
        self.num_sym = args.numSym
        self.lins = torch.nn.ModuleList()
        for i in range(self.num_sym):
            self.lins.append(nn.Linear(self.hidden_channels, self.hidden_channels))

    def forward(self, dis_embed_sym, z_syn, dis_indexs, syn_indexs):  # z_src dis (N,64)
        b_syn = torch.empty((0, 1, self.hidden_channels)).to(self.device)
        b_sym = torch.empty((0, self.num_sym, self.hidden_channels)).to(self.device)
        for dis_index, syn_index in zip(dis_indexs, syn_indexs):
            select_dis_list = []
            for i in range(self.num_sym):  # 分解出 s 个症状
                select_dis = torch.masked_select(dis_embed_sym[i], dis_index.unsqueeze(1).bool())
                select_dis = select_dis.view(-1, self.hidden_channels)
                select_dis_list.append(select_dis)
            select_dis_stacked = torch.stack(select_dis_list, dim=0)
            # 对每个疾病分解出的多个s，进行聚合
            H = select_dis_stacked * self.attention.unsqueeze(0)
            H_sum = H.sum(-1) # (s,n)
            weight = torch.softmax(H_sum,dim=1)
            aggregation = (weight.unsqueeze(-1) * select_dis_stacked).sum(1)
            b_sym = torch.cat((b_sym, aggregation.unsqueeze(0)), dim = 0)

            select_syn = torch.masked_select(z_syn, syn_index.unsqueeze(1).bool())
            select_syn = self.syn_agg(select_syn.view(1,-1, self.hidden_channels))
            # 将这个batch的放入最终集合
            b_syn = torch.cat((b_syn, select_syn.unsqueeze(0)), dim=0)

        # 将s个症状分别与当前证候聚合
        b_sym_syn = torch.empty((len(dis_indexs), 0, self.hidden_channels)).to(self.device)
        for i in range(self.num_sym):
            sym_syn = self.sAtt(b_syn, b_sym[:, i:i+1, :], b_sym[:, i:i+1, :])
            sym_syn = sym_syn + self.lins[i](sym_syn)
            b_sym_syn = torch.cat((b_sym_syn, sym_syn), dim=1)
        syn_out = self.sym_agg(b_sym_syn).unsqueeze(1)
        return syn_out

class Model(torch.nn.Module):
    def __init__(self,args, data_args, device):
        super().__init__()
        self.args = args
        self.data_args = data_args
        self.num_dis,  self.num_tcm, self.num_syn = data_args['dis_num'], data_args['tcm_num'], data_args['syn_num']
        self.dis_syn_encoder = LightGCN_ST(self.num_dis, self.num_syn, args.emb_dim, data_args['dis_dim'], data_args['syn_dim'], args.layer_num)
        self.syn_tcm_encoder = LightGCN_ST(self.num_syn, self.num_tcm, args.emb_dim, data_args['syn_dim'], data_args['tcm_dim'], args.layer_num)
        self.num_sym = args.numSym
        self.embedding_dim = args.emb_dim
        self.SymAgg = SymAgg(self.embedding_dim, args, device)
        self.syms = torch.nn.ModuleList()
        for i in range(self.num_sym):
            self.syms.append(self_gate(self.embedding_dim))
        self.device = device

        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

        self.threshold = args.st
        self.reg = args.reg
        self.is_reg = args.isReg

    def forward(self, x_dict, edge_index_dict, dis_index, syn_index):
        z_dict = {}

        dis_syn_sparse_edge = SparseTensor(row=edge_index_dict['dis','to','syn'][0], col=edge_index_dict['dis','to','syn'][1]+self.num_dis,
                                           sparse_sizes=(self.num_dis + self.num_syn, self.num_dis + self.num_syn))

        syn_tcm_sparse_edge = SparseTensor(row=edge_index_dict['syn','to','tcm'][0], col=edge_index_dict['syn','to','tcm'][1]+self.num_syn,
                                           sparse_sizes=(self.num_tcm + self.num_syn, self.num_tcm + self.num_syn))

        z_dict['dis'], syn1_emm = self.dis_syn_encoder(x_dict['dis'], x_dict['syn'], dis_syn_sparse_edge)
        syn2_emm, tcm_emm = self.syn_tcm_encoder(x_dict['syn'], x_dict['tcm'], syn_tcm_sparse_edge)
        z_dict['syn'] = (syn1_emm + syn2_emm) / 2

        # 症状分解
        dis_embed_sym = {}
        for i in range(len(self.syms)):
            dis_embed_sym[i] = self.syms[i](z_dict['dis'])
        self.dis_embed_sym = dis_embed_sym

        z_dict['syn'] = self.SymAgg(dis_embed_sym, z_dict['syn'], dis_index, syn_index)
        z_dict['tcm'] = tcm_emm.unsqueeze(0).repeat(len(dis_index), 1, 1)

        scores = torch.bmm(z_dict['syn'],z_dict['tcm'].transpose(-1, -2)).squeeze(-2)

        return scores


    def sym_regular(self):
        if self.is_reg:
            div_metric = None
            # N x K x dim
            for i in range(self.num_sym):
                for j in range(i + 1, self.num_sym):
                    sim_matrix = F.cosine_similarity(self.dis_embed_sym[i], self.dis_embed_sym[j])
                    mask = torch.abs(sim_matrix) > self.threshold
                    if i == 0 and j == 1:
                        div_metric = (torch.abs(torch.masked_select(sim_matrix, mask))).sum()
                    else:
                        div_metric += (torch.abs(torch.masked_select(sim_matrix, mask))).sum()

            div_reg =  div_metric
            return div_reg
        else:
            return 0