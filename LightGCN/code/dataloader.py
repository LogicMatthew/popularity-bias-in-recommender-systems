"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn), Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import torch
import numpy as np
import pandas as pd
import world
import scipy.sparse as sp
import dgl
import matplotlib
from os.path import join
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from time import time
from collections import defaultdict
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BasicDataset(Dataset):
    def __init__(self):
        print("Init dataset.")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        # It isn't necessary for large dataset.
        # Returning all neg items in super large dataset is 
        # time-consuming and memory-consuming.
        raise NotImplementedError
    
    def getSparseGraph(self):
        raise NotImplementedError
    
class Loader(BasicDataset):
    """
    Dataset type for Pytorch
    Incldue graph information
    Gowalla, Yelp2018, Amazon-book datasets
    """

    def __init__(self, config = world.config, path="../data/gowalla"):
        # Train or Test
        print(f'Loading [{path}]')
        self.config = config
        self.dgl_graph = None
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        
        self.n_user = 0
        self.m_item = 0
        
        if config['data_disrupt'] != 0:
            from utils import data_disrupt
            data_disrupt(config['dataset'], config['data_disrupt'])
            train_file = path + f"/train_{config['data_disrupt']}.txt"
        else:
            train_file = path + f"/train.txt"
        test_file = path + '/test.txt'
        
        self.path = path
        user_degrees_dict = defaultdict(int)
        item_degrees_dict = defaultdict(int)
        
        trainItem, trainUser = [], []
        testItem, testUser = [], []
        self.trainDataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    for item in items:
                        item_degrees_dict[item] += 1
                    uid = int(l[0])
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.trainDataSize += len(items)
                    user_degrees_dict[uid] += len(items)
        
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.m_item += 1
        self.n_user += 1
        
        # Histogram of user degrees and item degrees
        self.generate_degrees_histogram(user_degrees_dict, item_degrees_dict)
        
        # Mapping users/items to decile groups
        user_decile = self.map_node_to_decile(user_degrees_dict)
        item_decile = self.map_node_to_decile(item_degrees_dict)
        
        # Mapping degrees to 20-60-20 groups for metrics evaluation
        # and
        # Mapping node to groups
        print("User groups:")
        self.user_decile_groups, self.user_to_decile_group = self.map_node_to_decile_group(user_degrees_dict, user_decile)
        print("Item groups:")
        self.item_decile_groups, self.item_to_decile_group = self.map_node_to_decile_group(item_degrees_dict, item_decile)
        
        self.Graph = None
        self.getSparseGraph()
        
        print(f"{self.trainDataSize} interactions for training.")
        print(f"{self.testDataSize} interactions for testing.")
        print(f"{config['dataset']} Sparsity: {round((self.trainDataSize + self.testDataSize) / self.n_users / self.m_items, 10)}")

        # (users, items) - bipartite graph
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        self.item_degrees = np.array(self.UserItemNet.sum(axis=0)).squeeze().astype(np.float32)
        self.user_degrees = np.array(self.UserItemNet.sum(axis=1)).squeeze().astype(np.float32)
        
        # Smoothed BPR sampling distribution for negative sampling in training
        if self.config['smoothed_bpr'] is not None:
            smoothed_degrees = np.power(self.item_degrees, self.config['smoothed_bpr'])
            smoothed_degrees = smoothed_degrees / smoothed_degrees.sum()
            self.neg_item_p = smoothed_degrees
        else:
            self.neg_item_p = np.array([])
        
        # IPS weights for unbiased evaluation
        if self.config['ips_clip'] > 0.0:
            ips_w = self.item_degrees.max() / self.item_degrees
            ips_w = np.clip(ips_w, a_min=1.0, a_max=self.config['ips_clip'])
            self.ips_weights = torch.tensor(ips_w, dtype=torch.float32, device=self.config['device'])
        else:
            self.ips_weights = None
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{config['dataset']} is ready to go!")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos
    
    def getSparseGraph(self):
        print("Loading adjacency matrix.")
        try:
            # s_pre_adj_mat.npz is the preprocessed adjacency matrix, which is 
            # symmetric normalized and can be directly used for graph convolution.
            A = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
            print("Successfully loaded.")
        except:
            print("Generating adjacency matrix.")
            start = time()
            
            """
            We convert to dok_matrix (or lil_matrix) first for efficient incremental 
            construction of the graph structure, as they support fast element insertions. 
            However, for efficient arithmetic operations (like .sum() or matrix multiplication), 
            we MUST finally convert the matrix to csr_matrix, because dok_matrix 
            is highly unoptimized and extremely slow for mathematical computations.
            """
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()

            """
            The adjacency matrix is symmetric, so we need to fill in both the upper right and lower left blocks
                            self.n_items:  :self.n_users
            self.n_items:  |    0               R     |
            :self.n_users  |    R^T             0     |
            """
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.tocsr()

            """
            We can add self-connections to the adjacency matrix, 
            which can help to stabilize the training process and
            improve the performance of the model. However, in 
            LightGCN, we do not add self-connections because it
            simplifies the model and focuses on the essential part of
            graph convolution, which is the aggregation of
            neighbor information. Adding self-connections may introduce
            noise and reduce the effectiveness of the model.
            """
            # sp.eye is an identity matrix, which has 1s on the diagonal and 0s elsewhere.
            # A = A + sp.eye(A.shape[0])

            """
            In LightGCN, we use symmetric normalization for the adjacency matrix, which is 
            defined as D^(-1/2) * A * D^(-1/2), where D is the degree matrix of A. 
            This normalization can help to prevent the scale of the features from growing too 
            large or too small during the graph convolution process, and it 
            can also help to improve the convergence of the model.
            """
            degree_arr = np.array(adj_mat.sum(axis=1))
            degree_inv = np.power(degree_arr, -0.5).flatten()
            degree_inv[np.isinf(degree_inv)] = 0.0
            degree_mat = sp.diags(degree_inv)
            
            A = degree_mat.dot(adj_mat)
            A = A.dot(degree_mat)
    
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', A)
            print(f"Costing {time() - start}s, saving norm_mat...")

        if self.split:
            self.Graph = self._split_A_hat(A)
            print("Done splitting the matrix.")
        else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(A)
            self.Graph = self.Graph.coalesce().to(self.config['device'])
            print("Didn't split the matrix.")
    
    def get_dgl_graph(self):
        coo = self.UserItemNet.tocoo()
        u = torch.from_numpy(coo.row)
        v = torch.from_numpy(coo.col)
        v = v + self.n_users 
        src = torch.cat([u, v])
        dst = torch.cat([v, u])
        graph = dgl.graph((src, dst), num_nodes=self.n_users + self.m_items)
        self.dgl_graph = graph
        return graph

    def get_pin_graph(self, dgl_graph, n_traces, n_hops, top_k, weighted=True):
        if top_k <= 0:
            top_k = 1

        n_total = self.n_users + self.m_items
        all_nodes = torch.arange(n_total)
        
        start_nodes = all_nodes.repeat_interleave(n_traces).to(dgl_graph.idtype)
        traces, _ = dgl.sampling.random_walk(dgl_graph, start_nodes, length=n_hops)
        
        src = start_nodes.repeat_interleave(n_hops)
        dst = traces[:, 1:].flatten()
        
        mask = dst != -1
        src = src[mask]
        dst = dst[mask]
        
        count_mat = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(n_total, n_total))
        count_mat = count_mat.tolil()
        
        new_data, new_rows, new_cols = [], [], []
        for i, (rows, data) in enumerate(zip(count_mat.rows, count_mat.data)):
            if not rows: continue
            paired = sorted(zip(data, rows), key=lambda x: x[0], reverse=True)[:top_k]
            d, r = zip(*paired)
            new_data.extend(d)
            new_rows.extend([i]*len(d))
            new_cols.extend(r)
            
        count_mat = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=count_mat.shape)
        adj_mat = count_mat.copy()

        adj_mat.data = np.ones_like(adj_mat.data)
        adj_mat = (adj_mat + adj_mat.T) / 2.0
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
            
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = norm_adj.tocsr()
        
        return self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.config['device'])

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.config['device']))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def generate_degrees_histogram(self, user_degrees_dict, item_degrees_dict):
        # Getting the degree of each user and item
        item_degrees: list[int] = list(item_degrees_dict.values())
        user_degrees: list[int] = list(user_degrees_dict.values())
        
        # Counting the number of users and items for each degree
        item_degree_count = np.bincount(item_degrees)
        user_degree_count = np.bincount(user_degrees)
        
        # Creating DataFrames for plotting
        user_deg_range = np.arange(len(user_degree_count))
        item_deg_range = np.arange(len(item_degree_count))
        
        user_df = pd.DataFrame({
            'Degree': user_deg_range,
            'Count': user_degree_count
        })
        item_df = pd.DataFrame({
            'Degree': item_deg_range,
            'Count': item_degree_count
        })
        
        # Plotting the degree distributions
        plt.figure(figsize=(12, 5))
        
        # User degree distribution
        plt.subplot(1, 2, 1)
        plt.bar(user_df['Degree'], user_df['Count'], color='blue')
        plt.title('Rozkład stopni użytkowników')
        plt.xlabel('Stopień')
        plt.ylabel('Ilość')
        plt.yscale('log')
        
        # Item degree distribution
        plt.subplot(1, 2, 2)
        plt.bar(item_df['Degree'], item_df['Count'], color='orange')
        plt.title('Rozkład stopni przedmiotów')
        plt.xlabel('Stopień')
        plt.ylabel('Ilość')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(join(self.path, "rozkład_stopni.png"))
        plt.close()
        
        # Saving the degree distributions to CSV files
        user_df = user_df[user_df['Count'] > 0]
        item_df = item_df[item_df['Count'] > 0]
        user_df.to_csv(join(self.path, "user_degree_distribution.csv"), index=False)
        item_df.to_csv(join(self.path, "item_degree_distribution.csv"), index=False)
        
    def map_node_to_decile(self, degrees_dict):
        """
        The nodes are categorised into ten deciles, with each decile
        contains around 10% of all interactions in the graph.
        """
        # Sorting nodes ascending by their degrees (from least active to bestsellers)
        sorted_nodes = sorted(degrees_dict.items(), key=lambda x: x[1])
        
        # Decile threshold
        threshold = len(degrees_dict) / 10
        
        node_to_decile = {}
        current_num_nodes = 0
        decile = 0
        
        for node, degree in sorted_nodes:
            node_to_decile[node] = decile
            current_num_nodes += 1
            
            # If current number of nodes cross the threshold for current decile,
            # then current decile number is increased by 1
            if current_num_nodes >= (decile + 1) * threshold and decile < 9:
                decile += 1
        
        return node_to_decile
    
    def map_node_to_decile_group(self, degrees_dict, decile_map):
        # Three groups: rare (bottom 20%) - uncommon (middle 60%) - common (Top 20%)
        groups = world.all_metrics_group_names
        decile_degree_groups = {name: [] for name in groups}
        decile_node_groups = {name: [] for name in groups}
        node_to_group = {}
        
        for node, degree in degrees_dict.items():
            decile = decile_map[node]
            
            if decile < 2:
                target = "Bottom 20%"
            elif decile < 8:
                target = "Middle 60%"
            else:
                target = "Top 20%"
            
            node_to_group[node] = target
            decile_degree_groups[target].append(degree)
            decile_node_groups[target].append(node)
        
        for group_name, degrees in decile_degree_groups.items():
            if degrees:
                print(f"Group {group_name}: Degrees from {min(degrees)} to {max(degrees)} (Number of nodes: {len(degrees)})") 
        
        return decile_node_groups, node_to_group