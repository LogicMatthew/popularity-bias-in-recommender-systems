"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import torch
import numpy as np
from torch import nn
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from utils import column_prop
from collections import Counter


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = self.config['device']
        self._cached_embs = None
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers   = self.config['n_layers']
        self.A_split    = self.config['A_split']

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        # print('Using Xavier distribution initilizer.')
        # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

        # Random normal init seems to be a better choice when LightGCN actually don't use any non-linear activation function
        print('Using Normal distribution initilizer.')
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        self.embedding_item_start = self.embedding_item.weight.clone().to(self.device)
        self.embedding_user_start = self.embedding_user.weight.clone().to(self.device)
        
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.Graph
        print("LightGCN is ready to go!")
        if self.config['node_dropout'] != 0 or self.config['message_dropout'] != 0:
            if self.config['node_dropout'] != 0:
                print(f"Node dropout: {self.config['node_dropout']}")
            if self.config['message_dropout'] != 0:
                print(f"Message dropout: {self.config['message_dropout']}")
        else:
            print("No dropout.")

    def train(self, mode: bool = True):
        if mode:
            self._cached_embs = None
        return super(LightGCN, self).train(mode)
    
    def __mdropout_sub_g(self, sub_g, p):
        """Message dropout for sparse tensor."""
        
        keep_prob = 1 - p
        size = sub_g.size()
        indices = sub_g.indices().t()
        values = sub_g.values()
        mask = torch.rand(len(values)) + keep_prob
        mask = mask.int().bool()
        indices = indices[mask].t()
        values = values[mask] / keep_prob
        
        g = torch.sparse.FloatTensor(
            indices,
            values,
            sub_g.size()
        )
        return g
    
    def __ndropout_sub_g(self, sub_g, p):
        """Node dropout for sparse tensor."""
        
        keep_prob = 1 - p
        num_nodes = sub_g.size(0)
        indices = sub_g.indices()
        values = sub_g.values()
        mask = torch.empty(num_nodes, device=sub_g.device).bernoulli_(keep_prob).bool()
        mask_indices = torch.nonzero(mask).squeeze()
        
        edge_mask = mask[sub_g.indices()[0]] & mask[sub_g.indices()[1]]
        indices = indices[:, edge_mask]
        values = values[edge_mask] / keep_prob
        
        g = torch.sparse.FloatTensor(
            indices,
            values,
            sub_g.size(),
        )
        
        return g
    
    def __mdropout(self, p):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__mdropout_sub_g(g, p))
        else:
            graph = self.__mdropout_sub_g(self.Graph, p)
        return graph
    
    def __ndropout(self, p):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__ndropout_sub_g(g, p))
        else:
            graph = self.__ndropout_sub_g(self.Graph, p)
        return graph

    def computer(self):
        """
        Propagate methods for LightGCN
        """
        if not self.training and self._cached_embs is not None:
            return self._cached_embs

        if self.config['message_dropout'] != 0 or self.config['node_dropout'] != 0:
            if self.training:
                if self.config['message_dropout'] != 0:
                    g_dropped = self.__mdropout(self.config['message_dropout'])
                elif self.config['node_dropout'] != 0:
                    g_dropped = self.__ndropout(self.config['node_dropout'])
            else:
                g_dropped = self.Graph        
        else:
            g_dropped = self.Graph
              
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        if self.training:
            noise_disrupt = self.config['noise_disrupt']
            
            if noise_disrupt != 0:
                items_emb = items_emb + torch.randn_like(items_emb) * noise_disrupt

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(torch.sparse.mm(g_dropped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_dropped, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        if not self.training:
            self._cached_embs = (users, items)

        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (
            users_emb, pos_emb, neg_emb, 
            userEmb0,  posEmb0, negEmb0
        ) = self.getEmbedding(users.long(), pos.long(), neg.long())

        reg_loss = (1/2) * (
            userEmb0.norm(2).pow(2) + 
            posEmb0.norm(2).pow(2)  +
            negEmb0.norm(2).pow(2)
        ) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)

        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        bpr_loss_per_sample = torch.nn.functional.softplus(neg_scores - pos_scores)
        
        if self.config['ips_clip'] > 0.0:
            batch_weights_pos = self.dataset.ips_weights[pos]
            bpr_loss = torch.mean(batch_weights_pos * bpr_loss_per_sample)
        else:
            bpr_loss = torch.mean(bpr_loss_per_sample)
        
        return bpr_loss, reg_loss
       
    def forward(self, users, items):
        # Compute the embedding of users and items
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class FastLightGCN(LightGCN):
    def __init__(self, config: dict, dataset):
        super(FastLightGCN, self).__init__(config, dataset)
        
        self.layers_size = config['layers_size']
        self.unique_layers = config['unique']
        self.alpha = config.get('sis_alpha', None)
        
        if self.alpha is not None:
            print(f"Using SIS sampling with alpha={self.alpha}")
            item_degrees = dataset.item_degrees.astype(np.float32)
            user_degrees = dataset.user_degrees.astype(np.float32)
            self.raw_degrees = torch.tensor(np.concatenate([user_degrees, item_degrees])).float().to(self.device)
        else:
            self.p = column_prop(self.Graph)
            
        self.sampled_nodes = Counter()
        
        print("FastLightGCN is ready to go!")
        if self.config['node_dropout'] != 0 or self.config['message_dropout'] != 0:
            if self.config['node_dropout'] != 0:
                print(f"Node dropout: {self.config['node_dropout']}")
            if self.config['message_dropout'] != 0:
                print(f"Message dropout: {self.config['message_dropout']}")
        else:
            print("No dropout.")
         
    def getEmbedding(self, users, pos_items, neg_items):
        users_emb, pos_emb, neg_emb = self.computer(users, pos_items, neg_items)
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def computer(self, users=None, pos_items=None, neg_items=None):
        if self.training:
            if self.config['message_dropout'] != 0 or self.config['node_dropout'] != 0:
                if self.config['message_dropout'] != 0:
                    g = self.__mdropout(self.config['message_dropout'])
                elif self.config['node_dropout'] != 0:
                    g = self.__ndropout(self.config['node_dropout'])
            else:
                g = self.Graph
            
            # Counting number of times node is choosen
            self.sampled_nodes.update(pos_items.cpu().numpy())
            self.sampled_nodes.update(neg_items.cpu().numpy())
            
            # Global node embedding, which is used for sampling
            all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
            
            # Shiftting item indices to the right to fit the global node embedding
            pos_items = pos_items + self.num_users
            neg_items = neg_items + self.num_users
            
            # Target batch nodes (Top layer sampling) - Without duplicates
            batch_nodes_raw = torch.cat([users, pos_items, neg_items])
            batch_nodes, inverse_indices = torch.unique(batch_nodes_raw, return_inverse=True)
            if self.unique_layers:
                used_nodes = batch_nodes.clone()
            num_batch_nodes = len(batch_nodes)
            
            layer_nodes = [batch_nodes]
            layer_A_sub = []
            
            current_nodes = batch_nodes
            g_indices = g.indices()
            g_values = g.values()
            num_all_nodes = self.num_users + self.num_items
            done_layers = 0
            
            for layer in range(len(self.layers_size)):
                t = self.layers_size[layer]
                
                # Finding neighbors of current nodes
                edge_mask = torch.isin(g_indices[1], current_nodes)
                neighbours = torch.unique(g_indices[0][edge_mask]) # Neighbors of current nodes (source nodes of masked edges)
                if self.unique_layers:
                    unseen_mask = ~torch.isin(neighbours, used_nodes) 
                    neighbours = neighbours[unseen_mask]
                
                if self.alpha is not None:
                    # Smoothed degree-based sampling probability: |N(u)|^alpha / sum(|N(v)|^alpha) where v is all nodes and u is some node
                    p = torch.pow(self.raw_degrees[neighbours], self.alpha)
                else:
                    # Probability function: ||A(:, u)||^2 / sum(||A(:, v)||^2) where v is all nodes and u is some node
                    p = self.p[neighbours]

                p_sum = p.sum()
                
                if p_sum > 0:
                    # Normalizing to get a sum of 1, which is required for np.random.choice
                    p = p / p_sum 
                else:
                    # If there is no neighbours, we end sampling for this batch
                    done_layers = layer
                    break
                
                if t >= len(neighbours):
                    # If the number of neighbors is less than the required sample size, we take all neighbors
                    # This also means that there is no stochastic sampling, which is equivalent to the original LightGCN aggregation for this layer
                    # So we can set the estimator scale to 1.0 for all edges, which means that we are not scaling the edges at all (no importance sampling correction needed)                    
                    q_nodes = neighbours
                    estimator_scale = torch.ones_like(q_nodes, dtype=torch.float32, device=self.device)
                else:                    
                    chosen_indices = torch.multinomial(p, t, replacement=False)
                    q_nodes = neighbours[chosen_indices]
                    q_prob = p[chosen_indices]
                    
                    # Monte Carlo Estimator applied directly to the collected edges
                    # Estimator: 1 / (t * q_prob) where t is the number of sampled neighbors and q_prob is the probability of sampling that neighbor
                    # We are scaling the edges dependent on the probabilty in THIS particular layer
                    estimator_scale = 1.0 / (t * q_prob)
                                   
                if self.unique_layers:
                    used_nodes = torch.cat([used_nodes, q_nodes])
                    
                # Updating node counter
                q_nodes_items = q_nodes[q_nodes >= self.num_users] - self.num_users
                self.sampled_nodes.update(q_nodes_items.cpu().numpy())
                
                # Getting edges where source node is in q and destination node is in current_nodes
                final_edge_mask = torch.isin(g_indices[0], q_nodes) & edge_mask # Masking edges where source node is in q and destination node is in current_nodes
                sub_indices = g_indices[:, final_edge_mask]
                sub_values = g_values[final_edge_mask]

                # Mapping estimator_scale on edges (applying q_prob for q_nodes)
                scale_map = torch.zeros(num_all_nodes, dtype=torch.float32, device=self.device)
                scale_map[q_nodes] = estimator_scale
                scaled_sub_values = sub_values * scale_map[sub_indices[0]]
                
                # Getting a mapping for global nodes indices to local sub-matrix indices
                # Rows (TO == current_nodes) & Columns (FROM == q_nodes) seperated!
                map_dst = torch.zeros(num_all_nodes, dtype=torch.long, device=self.device)
                map_dst[current_nodes] = torch.arange(len(current_nodes), device=self.device)
                map_src = torch.zeros(num_all_nodes, dtype=torch.long, device=self.device)
                map_src[q_nodes] = torch.arange(len(q_nodes), device=self.device)
                sub_indices_mapped = torch.stack([
                    map_dst[sub_indices[1]], # Matrix rows are targets (current_nodes) - destination nodes to aggregate into
                    map_src[sub_indices[0]]  # Matrix columns are sources (q_nodes) - knowledge nodes to propagate from
                ])
                
                # Building A_sub matrix for propagation
                A_sub = torch.sparse_coo_tensor(
                    indices=sub_indices_mapped,
                    values=scaled_sub_values,
                    size=(len(current_nodes), len(q_nodes)),
                    device=self.device
                )
                
                layer_nodes.append(q_nodes)
                layer_A_sub.append(A_sub) # Adding transition matrix from L layer to L-1 layer
                current_nodes = q_nodes
            else:
                done_layers = len(self.layers_size)
                
            embs_at_batch = [all_emb[batch_nodes]] # 0-hop knowledge
            
            # Aggregating embeddings through layers (hops 1, 2, 3...)
            for hop in range(done_layers):
                # Initialize with embeddings of sampled nodes for current hop
                h = all_emb[layer_nodes[hop + 1]]
                
                # Propagate embeddings through the computed transition matrices up
                # to the batch (O(K) matrix-vector multiplication)
                for j in reversed(range(hop + 1)):
                    h = torch.sparse.mm(layer_A_sub[j], h)
                    
                embs_at_batch.append(h)
                
            embs = torch.stack(embs_at_batch, dim=1)
            light_out = torch.mean(embs, dim=1)
            
            # Reconstruct original duplicates using the inverse mapping
            light_out_expanded = light_out[inverse_indices]
            
            # Split back into user, positive item, and negative item tensors
            users_final, pos_items_final, neg_items_final = torch.split(
                light_out_expanded, 
                [len(users), len(pos_items), len(neg_items)]
            )
            
            return users_final, pos_items_final, neg_items_final
        else:
            return super(FastLightGCN, self).computer()