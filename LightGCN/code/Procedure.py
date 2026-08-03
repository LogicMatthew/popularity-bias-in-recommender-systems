'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import multiprocessing
import copy
from utils import Timer
from collections import defaultdict, Counter
from functools import partial


def BPR_train_original(dataset, RecModel, loss_class, epoch, neg_k=1):
    RecModel.train()
    bpr: utils.BPRLoss = loss_class
    
    with Timer(name="Sampling"):
        S = utils.UniformSample(dataset, neg_ratio=neg_k)
    
    if neg_k > 1:
        users = torch.from_numpy(np.repeat(S[:, 0], neg_k)).long()
        posItems = torch.from_numpy(np.repeat(S[:, 1], neg_k)).long()
        negItems = torch.from_numpy(S[:, 2:].flatten()).long()
    else:
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()
    
    users = users.to(world.config['device'])
    posItems = posItems.to(world.config['device'])
    negItems = negItems.to(world.config['device'])
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    avg_loss = 0.0
    sampled_nodes = Counter()
    
    with Timer(name="BPR Training"):
        for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])
        ):
            sampled_nodes.update(batch_pos.cpu().numpy())
            sampled_nodes.update(batch_neg.cpu().numpy())
            
            loss = bpr.stageOne(batch_users, batch_pos, batch_neg)
            avg_loss += loss
    
    avg_loss = avg_loss / total_batch
    avg_loss = round(avg_loss, 5)
    time_info = Timer.dict(["BPR Training", "Sampling"])
    Timer.zero(["BPR Training", "Sampling"])

    return avg_loss, time_info, sampled_nodes

def test_one_batch(data, groups=None):
    groundTrue = data[1]
    prediction = data[0].numpy()
    
    if groups is not None:
        group_metrics_at_k = {metric_name: np.zeros(len(world.config['topKs'])) for metric_name in world.all_metrics}
        group_metrics_at_k['n_users'] = 0.0
        group_results_at_k = {metric_group_name: copy.deepcopy(group_metrics_at_k) for metric_group_name in world.all_metrics_group_names}

        relevance_group_matrices = utils.getLabel(groundTrue, prediction, groups)
        relevance_group_matrices, groundTrue_groups = zip(*relevance_group_matrices)
    else:
        relevance_matrix = utils.getLabel(groundTrue, prediction, groups)
        results_at_k = {metric_name: np.zeros(len(world.config['topKs'])) for metric_name in world.all_metrics}
        
    for k_idx, k in enumerate(world.config['topKs']):
        if groups is None:
            result_at_k = utils.metrics_at_k(groundTrue, relevance_matrix, k)
            
            for metric_name in world.all_metrics:
                results_at_k[metric_name][k_idx] = result_at_k[metric_name]
        else:
            for group_idx, group_name in enumerate(world.all_metrics_group_names):
                group_result_at_k = utils.metrics_at_k(groundTrue_groups[group_idx], relevance_group_matrices[group_idx], k)
                
                for metric_name in world.all_metrics:
                    group_results_at_k[group_name][metric_name][k_idx] = group_result_at_k[metric_name]
                group_results_at_k[group_name]['n_users'] = len(groundTrue_groups[group_idx])
    
    if groups is None:
        return results_at_k
    return group_results_at_k
    
def Test(dataset, RecModel, epoch=None, multicore=0, groups=None):
    batch_size = world.config['test_batch_size']
    testDict: dict = dataset.testDict

    # Eval mode with no dropout
    RecModel = RecModel.eval()
    max_K = max(world.config['topKs'])
    
    if multicore == 1:
        pool = multiprocessing.Pool(world.config['CORES'])
    
    if groups is None:
        results = {metric_name: np.zeros(len(world.config['topKs'])) for metric_name in world.all_metrics}
    else:
        groups_metrics = {metric_name: np.zeros(len(world.config['topKs'])) for metric_name in world.all_metrics}
        groups_results = {metric_group_name: copy.deepcopy(groups_metrics) for metric_group_name in world.all_metrics_group_names}

    with torch.no_grad():
        # Getting users from testDict, because some users in
        # trainUser may not appear in testDict, and they shouldn't be evaluated.
        users = list(testDict.keys())
        
        try:
            assert batch_size <= len(users) // 10
        except AssertionError:
            print(f"test_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        
        users_list = []
        rating_list = []
        groundTrue_list = []

        total_batch = len(users) // batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.config['device'])

            # Masking user interacted training items in ratings
            # to avoid recommending them in evaluation
            rating = RecModel.getUsersRating(batch_users_gpu)
            
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)  # underscore ("_") is values_K
            
            rating = rating.cpu().numpy()
            del rating
            
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        
        assert total_batch == len(users_list)
        
        data = zip(rating_list, groundTrue_list)            
        
        if multicore == 1:
            test_one_batch_with_groups = partial(test_one_batch, groups=groups)
            pre_results = pool.map(test_one_batch_with_groups, data)
        else:
            pre_results = []
            for batch_data in data:
                pre_results.append(test_one_batch(batch_data, groups))
        
        if groups is not None:
            groups_n_users = defaultdict(int)
            for group_result_at_k in pre_results:
                for group_name in world.all_metrics_group_names:
                    for metric in world.all_metrics:
                        groups_results[group_name][metric] += group_result_at_k[group_name][metric]
                    groups_n_users[group_name] += group_result_at_k[group_name]['n_users']
            for metric in world.all_metrics:
                for group_name in world.all_metrics_group_names:
                    groups_results[group_name][metric] /= float(groups_n_users[group_name])
        else:
            for result_at_k in pre_results:
                for metric in world.all_metrics:
                    results[metric] += result_at_k[metric]
            for metric in world.all_metrics:
                results[metric] /= float(len(users))
                    
        if multicore == 1:
            pool.close()

        if groups is not None:
            return groups_results, groups_n_users
        return results