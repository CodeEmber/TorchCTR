'''
Author       : wyx-hhhh
Date         : 2024-07-09
LastEditTime : 2024-09-09
Description  : 
'''
from typing import List

from utils.utilities import get_values_by_keys
import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.middleware import time_middleware
from utils.utilities import get_values_by_keys


class NCL(nn.Module):

    def __init__(
        self,
        config: dict,
        g,
    ):
        super(NCL, self).__init__()
        self.config = config
        self.g = g
        self.user_num = self.config['user_num']
        self.item_num = self.config['item_num']
        self.embedding_dim = self.config['embedding_dim']
        self.hidden_units = self.config['hidden_units']
        self.num_layers = len(self.hidden_units)
        self.lmbd_reg = self.config['lmbd_reg']
        self.lmbd_ssl = self.config['lmbd_ssl']
        self.lmbd_proto = self.config['lmbd_proto']
        self.cl_tau = self.config['cl_tau']
        self.alpha_ssl = self.config['alpha_ssl']
        self.alpha_proto = self.config['alpha_proto']
        self.hyper_layers = self.config['hyper_layers']
        self.k = config['num_clusters']
        assert self.hyper_layers * 2 <= self.num_layers
        self.user_embedding_layer = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embedding_layer = nn.Embedding(self.item_num, self.embedding_dim)

        self.f = nn.Sigmoid()
        self.user_centroids = torch.zeros(self.k, self.embedding_dim).to(self.config['device'])
        self.user_2cluster = torch.zeros(self.user_num).long().to(self.config['device'])
        self.item_centroids = torch.zeros(self.k, self.embedding_dim).to(self.config['device'])
        self.item_2cluster = torch.zeros(self.item_num).long().to(self.config['device'])

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    @time_middleware("kmeans聚类")
    def e_step(self):
        user_embeddings = self.user_embedding_layer.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding_layer.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        kmeans = faiss.Kmeans(d=self.embedding_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        centroids = torch.Tensor(cluster_cents).to(self.config['device'])
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.config['device'])
        return centroids, node2cluster

    def computer(self):
        user_embedding = self.user_embedding_layer.weight
        item_embedding = self.item_embedding_layer.weight

        all_emb = torch.cat([user_embedding, item_embedding])
        embs_list = [all_emb]

        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(self.g, all_emb)
            embs_list.append(all_emb)
        embs = torch.stack(embs_list, dim=1)
        light_out = torch.mean(embs, dim=1)
        final_user_embedding, final_item_embedding = torch.split(light_out, [self.user_num, self.item_num])
        return final_user_embedding, final_item_embedding, embs_list

    def create_bpr_loss(self, embedding_dict):
        user_embedding, pos_item_embedding, neg_item_embedding = embedding_dict["user_embedding"], embedding_dict["pos_item_embedding"], embedding_dict["neg_item_embedding"]
        pos_scores = (user_embedding * pos_item_embedding).sum(dim=1)
        neg_scores = (user_embedding * neg_item_embedding).sum(dim=1)

        bpr_loss = -torch.log(1e-10 + torch.sigmoid(pos_scores - neg_scores)).mean()
        # bpr_loss = -torch.sum(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def create_regularization_loss(self, embedding_dict):
        user_embedding_ego, pos_item_embedding_ego, neg_item_embedding_ego = embedding_dict["user_embedding_ego"], embedding_dict["pos_item_embedding_ego"], embedding_dict["neg_item_embedding_ego"]
        reg_loss = (torch.norm(user_embedding_ego)**2 + torch.norm(pos_item_embedding_ego)**2 + torch.norm(neg_item_embedding_ego)**2) / (2 * user_embedding_ego.shape[0])
        return reg_loss

    def create_ssl_loss(self, data, sub_embedding_dict):
        previous_user_embeddings_all = sub_embedding_dict["final_user_embedding1"]
        previous_item_embeddings_all = sub_embedding_dict["final_item_embedding1"]
        current_user_embeddings = sub_embedding_dict["final_user_embedding2"]
        current_item_embeddings = sub_embedding_dict["final_item_embedding2"]

        # current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        # previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding, [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[data["user_id"]]
        previous_user_embeddings = previous_user_embeddings_all[data["user_id"]]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.cl_tau)
        ttl_score_user = torch.exp(ttl_score_user / self.cl_tau).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[data["item_id"]]
        previous_item_embeddings = previous_item_embeddings_all[data["item_id"]]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.cl_tau)
        ttl_score_item = torch.exp(ttl_score_item / self.cl_tau).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = ssl_loss_user + self.alpha_ssl * ssl_loss_item
        return ssl_loss

    # def create_ssl_loss(self, data, sub_embedding_dict):
    #     user_embedding1, item_embedding1, user_embedding2, item_embedding2 = get_values_by_keys(
    #         data=sub_embedding_dict,
    #         keys=[
    #             "final_user_embedding1",
    #             "final_item_embedding1",
    #             "final_user_embedding2",
    #             "final_item_embedding2",
    #         ],
    #     )

    #     user_embeddings1 = F.normalize(user_embedding1, dim=1)
    #     item_embeddings1 = F.normalize(item_embedding1, dim=1)
    #     user_embeddings2 = F.normalize(user_embedding2, dim=1)
    #     item_embeddings2 = F.normalize(item_embedding2, dim=1)

    #     user_embs1 = F.embedding(data['user_id'], user_embeddings1)
    #     item_embs1 = F.embedding(data['item_id'], item_embeddings1)
    #     user_embs2 = F.embedding(data['user_id'], user_embeddings2)
    #     item_embs2 = F.embedding(data['item_id'], item_embeddings2)

    #     pos_ratings_user = torch.sum(user_embs1 * user_embs2, dim=1)
    #     pos_ratings_item = torch.sum(item_embs1 * item_embs2, dim=1)
    #     tot_ratings_user = torch.matmul(user_embs1, torch.transpose(user_embeddings2, 0, 1))
    #     tot_ratings_item = torch.matmul(item_embs1, torch.transpose(item_embeddings2, 0, 1))

    #     ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]
    #     ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]
    #     clogits_user = torch.logsumexp(ssl_logits_user / self.cl_tau, dim=1)
    #     clogits_item = torch.logsumexp(ssl_logits_item / self.cl_tau, dim=1)
    #     infonce_loss = torch.sum(clogits_user + clogits_item * self.alpha_ssl)
    #     return infonce_loss

    def create_proto_loss(self, data, sub_embedding_dict):
        user_embeddings_all, item_embeddings_all = sub_embedding_dict["final_user_embedding1"], sub_embedding_dict["final_item_embedding1"]

        user_embeddings = user_embeddings_all[data['user_id']]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[data['user_id']]
        user2centroids = self.user_centroids[user2cluster]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.cl_tau)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.cl_tau).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[data['item_id']]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[data['item_id']]
        item2centroids = self.item_centroids[item2cluster]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.cl_tau)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.cl_tau).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = torch.sum(proto_nce_loss_user + proto_nce_loss_item * self.alpha_proto)
        return proto_nce_loss

    def total_loss(self, data, embedding_dict, sub_embedding_dict):
        # embedding_dict = dict()
        # import pickle

        # with open('embeddings_list.pkl', 'rb') as file:
        #     embeddings_list = pickle.load(file)
        # with open('item_all_embeddings.pkl', 'rb') as file:
        #     item_all_embeddings = pickle.load(file).to(self.config['device'])
        # with open('user_all_embeddings.pkl', 'rb') as file:
        #     user_all_embeddings = pickle.load(file).to(self.config['device'])
        # with open('user.pkl', 'rb') as file:
        #     user = pickle.load(file).to(self.config['device'])
        # with open('pos_item.pkl', 'rb') as file:
        #     pos_item = pickle.load(file).to(self.config['device'])
        # with open('neg_item.pkl', 'rb') as file:
        #     neg_item = pickle.load(file).to(self.config['device'])

        # import torch

        # # 加载状态字典
        # state_dicta = torch.load('item_embedding.pth')

        # # 创建嵌入层实例
        # item_embeddings = nn.Embedding(num_embeddings=1219865, embedding_dim=64)

        # # 加载权重
        # item_embeddings.load_state_dict(state_dicta)

        # # 移动到所需设备
        # item_embeddings = item_embeddings.to(self.config['device'])

        # state_dictb = torch.load('user_embedding.pth')
        # user_embeddings = nn.Embedding(num_embeddings=57933, embedding_dim=64)
        # user_embeddings.load_state_dict(state_dictb)
        # user_embeddings = user_embeddings.to(self.config['device'])

        # embedding_dict = {
        #     "user_embedding": user_all_embeddings[user],
        #     "pos_item_embedding": item_all_embeddings[pos_item],
        #     "neg_item_embedding": item_all_embeddings[neg_item],
        #     "user_embedding_ego": item_embeddings(user),
        #     "pos_item_embedding_ego": user_embeddings(pos_item),
        #     "neg_item_embedding_ego": user_embeddings(neg_item),
        # }

        # sub_embedding_dict = {
        #     "final_user_embedding1": embeddings_list[0][:57933],
        #     "final_item_embedding1": embeddings_list[0][57933:],
        #     "final_user_embedding2": embeddings_list[self.hyper_layers * 2][:57933],
        #     "final_item_embedding2": embeddings_list[self.hyper_layers * 2][57933:],
        # }

        bpr_loss = self.create_bpr_loss(embedding_dict)
        reg_loss = self.create_regularization_loss(embedding_dict)

        ssl_loss = self.create_ssl_loss(data, sub_embedding_dict)
        proto_loss = self.create_proto_loss(data, sub_embedding_dict)
        return bpr_loss + reg_loss * self.lmbd_reg + ssl_loss * self.lmbd_ssl + proto_loss * self.lmbd_proto

    def get_embedding(self, data, final_user_embedding, final_item_embedding):
        embedding_dict = dict()
        embedding_dict["user_embedding"] = final_user_embedding[data['user_id']]
        embedding_dict["pos_item_embedding"] = final_item_embedding[data['item_id']]
        embedding_dict["neg_item_embedding"] = final_item_embedding[data['neg_item_id']]
        embedding_dict["user_embedding_ego"] = self.user_embedding_layer(data['user_id'])
        embedding_dict["pos_item_embedding_ego"] = self.item_embedding_layer(data['item_id'])
        embedding_dict["neg_item_embedding_ego"] = self.item_embedding_layer(data['neg_item_id'])
        return embedding_dict

    def forward(self, data, is_train=True):

        final_user_embedding, final_item_embedding, embs_list = self.computer()

        output_dict = dict()

        if is_train:
            final_user_embedding1, final_item_embedding1 = torch.split(embs_list[0], [self.user_num, self.item_num])
            final_user_embedding2, final_item_embedding2 = torch.split(embs_list[self.hyper_layers * 2], [self.user_num, self.item_num])

            sub_embedding_dict = {
                "final_user_embedding1": final_user_embedding1,
                "final_item_embedding1": final_item_embedding1,
                "final_user_embedding2": final_user_embedding2,
                "final_item_embedding2": final_item_embedding2,
            }
            embedding_dict = self.get_embedding(
                data,
                final_user_embedding,
                final_item_embedding,
            )

            loss = self.total_loss(data, embedding_dict, sub_embedding_dict)
            output_dict['loss'] = loss
        else:
            output_dict['user_embedding'] = final_user_embedding
            output_dict['item_embedding'] = final_item_embedding
        return output_dict
