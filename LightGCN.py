"""
@FileName：model.py
@Description： 中医药测试模型
@Author：HaoBoli
"""


from torch.nn import Embedding, Linear
from torch_geometric.nn import SAGEConv, GATConv
import torch
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

class EdgeDecoder_lgcn(torch.nn.Module):
    def __init__(self, hidden_channels, device):
        super().__init__()
        self.sym_agg = Attention(hidden_channels, device=device)
        self.hidden_channels = hidden_channels
        self.device = device
        # self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        # self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, sym_indexs):  # z_src (n,64)  z_dst (n,64)
        select_src_emm = torch.empty((0, 1, self.hidden_channels)).to(self.device) # (b,1,64)
        b_dst = torch.empty((0, z_dst.shape[0], self.hidden_channels)) # (b,n,64)
        for sym_index in sym_indexs:
            select_emm = torch.masked_select(z_src,sym_index.unsqueeze(1).bool())
            select_emm = select_emm.view(1,-1, self.hidden_channels) # (b,n,64)
            select_emm_agg = self.sym_agg(select_emm) # (b,64)
            # print(select_src_emm.device,select_emm_agg.device)
            select_src_emm = torch.cat((select_src_emm, select_emm_agg.unsqueeze(0)), dim=0)
            b_dst = torch.cat((b_dst.to(self.device),z_dst.unsqueeze(0)),dim=0)

        scores = torch.bmm(select_src_emm,b_dst.transpose(-1, -2)).squeeze(-2) # (b,n)
        # scores = torch.sigmoid(scores)
        return scores

class Model_lgcn(torch.nn.Module):
    def __init__(self,args, data_args, device):
        super().__init__()
        self.args = args
        self.data_args = data_args
        self.num_dis,  self.num_tcm = data_args['dis_num'], data_args['tcm_num']
        self.syn_tcm_encoder = LightGCN_ST(self.num_dis, self.num_tcm, args.emb_dim, data_args['dis_dim'], data_args['tcm_dim'], args.layer_num)
        self.decoder = EdgeDecoder_lgcn(args.emb_dim, device)
        self.device = device

    def forward(self, x_dict, edge_index_dict, dis_index):
        z_dict = {}

        dis_tcm_sparse_edge = SparseTensor(row=edge_index_dict['dis','to','tcm'][0], col=edge_index_dict['dis','to','tcm'][1]+self.num_dis,
                                           sparse_sizes=(self.num_dis + self.num_tcm, self.num_dis + self.num_tcm))

        z_dict['dis'], z_dict['tcm'] = self.syn_tcm_encoder(x_dict['dis'], x_dict['tcm'], dis_tcm_sparse_edge)

        return self.decoder(z_dict['dis'], z_dict['tcm'], dis_index)