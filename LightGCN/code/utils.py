'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
import json
import multiprocessing
import numpy as np
import dgl
from torch import optim, nn, norm
from json import dumps


# Load sampler
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "sources"))
    import sampling
    sampling.seed(world.config['seed'])
    sample_ext = True
    if multiprocessing.current_process().name == 'MainProcess':
        print("Successfully loaded C++ extension for sampling.")
except Exception as e:
    import traceback
    traceback.print_exc()
    sample_ext = False
    if multiprocessing.current_process().name == 'MainProcess':
        print(f"C++ extension not loaded! Error: {e}")

# Samplers
class BPRLoss:
    def __init__(self, RecModel, config: dict):
        self.model = RecModel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(RecModel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

def UniformSample(dataset, neg_ratio=1):
    allPos = dataset.allPos
    if sample_ext:
        S = sampling.sample_negative(
            dataset.n_users,
            dataset.m_items,
            dataset.trainDataSize,
            allPos,
            neg_ratio,
            dataset.neg_item_p
        )
    else:
        S = UniformSample_python(dataset, neg_ratio)
    return S

def UniformSample_python(dataset, neg_ratio=1):
    allPos   = dataset.allPos
    S        = []
    samplesPerUser = dataset.trainDataSize // dataset.n_users

    for user in range(dataset.n_users):
        userPos = allPos[user]
        if len(userPos) == 0:
            continue

        for _ in range(samplesPerUser):
            pos_index = np.random.randint(0, len(userPos))
            pos_item = userPos[pos_index]
            
            for _ in range(neg_ratio):
                while True:
                    if dataset.neg_item_p.size > 0:
                        neg_item = np.random.choice(dataset.m_items, p=dataset.neg_item_p)
                    else:
                        neg_item = np.random.randint(0, dataset.m_items)
                    
                    if neg_item not in userPos:
                        break
                    
                S.append([user, pos_item, neg_item])

    return np.array(S)

# Utils
def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)

def getName():
    model_name = world.config['model_name']
    if model_name == 'lgcn':
        name = f"lgn-{world.config['dataset']}-"
        if world.config['message_dropout'] != 0:
            name += f"mdr({world.config['message_dropout']})-"
        if world.config['node_dropout'] != 0:
            name += f"ndr({world.config['node_dropout']})-"
        if world.config['smoothed_bpr'] is not None:
            name += f"sbpr({world.config['smoothed_bpr']})-"
        if world.config['ips_clip'] > 0.0:
            name += f"ips({world.config['ips_clip']})-"
        name += f"{world.config['n_layers']}-{world.config['latent_dim_rec']}"
        if world.config['data_disrupt'] > 0:
            name += f"-data-{world.config['data_disrupt']}"
        elif world.config['noise_disrupt'] > 0:
            name += f"-noise-{world.config['noise_disrupt']}"
        elif world.config['pin_sampling'] > 0:
            if world.config['pin_sampling'] == 1:
                name += f"-pin-{world.config['pin_sampling']}-0-{world.config['pin_n_traces']}"
            else:
                name += f"-pin-{world.config['pin_sampling']}-{world.config['pin_n_hops']}-{world.config['pin_n_traces']}-{world.config['pin_top_k']}"
    elif model_name == 'flgcn':
        name = f"flgcn-{world.config['dataset']}-"
        if world.config['sis_alpha'] is not None:
            name += f"sis({world.config['sis_alpha']})-"
        if world.config['smoothed_bpr'] is not None:
            name += f"sbpr({world.config['smoothed_bpr']})-"
        if world.config['ips_clip'] > 0.0:
            name += f"ips({world.config['ips_clip']})-"
        if world.config['unique'] == 1:
            name += "uniq-"
        name += f"{world.config['layers_size']}-{world.config['bpr_batch_size']}-{world.config['lr']}"
    return name

def getFileName(is_best=False):
    if is_best:
        file = f"best-{getName()}.pth"
    else:
        file = f"{getName()}.pth"
    return os.path.join(world.FILE_PATH, file)

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def data_disrupt(dataset, top_k):
    """
    Function to disrupt user-item interaction data by replacing items with their nearest (top_k) neighbors.
    """
    state_dict = torch.load(f"checkpoints/{getName()}.pth", map_location=torch.device('cpu'))
    embedding_item = state_dict["embedding_item.weight"]
    
    A = embedding_item.clone().detach()
    B = embedding_item.clone().detach()
    distances = -2 * (A @ B.T) + torch.sum(A ** 2, dim=1, keepdim=True) + torch.sum(B ** 2, dim=1)
    nearest_neighbours = torch.argsort(distances, axis=1)
    nearest_neighbours_topk = nearest_neighbours[:, 1:(top_k+1)]

    with open(f"../data/{dataset}/train.txt", "r") as train:
        with open(f"../data/{dataset}/train_{top_k}.txt", "w") as modified_train:
            for line in train.readlines():
                line = line[:-1]
                modified_line = []
                user_id, *user_interactions = map(lambda n: int(n), line.split(" "))
                modified_line.append(str(user_id))
                for item_id in user_interactions:
                    random_neighbour_id = torch.randint(0, top_k, (1, )).item()
                    new_item_id = nearest_neighbours_topk[item_id][random_neighbour_id].item()
                    modified_line.append(str(new_item_id))
                modified_train.write(" ".join(modified_line) + "\n")

def column_prop(graph):
    g_indices = graph.indices()
    g_values = graph.values()
    
    values = g_values.pow(2)
    col_sum = torch.zeros(graph.size(1), dtype=torch.float32, device=graph.device)
    col_sum.scatter_add_(0, g_indices[1], values)
    p = col_sum / col_sum.sum()
    
    return p

class Timer:
    """
    Time context manager for code block
        with Timer():
            do something
        Timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(Timer.TAPE) > 1:
            return Timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in Timer.NAMED_TAPE.items():
                hint = hint + f"{key}: {value:.2f}|"
        else:
            for key in select_keys:
                value = Timer.NAMED_TAPE[key]
                hint = hint + f"{key}: {value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in Timer.NAMED_TAPE.items():
                Timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                Timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            Timer.NAMED_TAPE[kwargs['name']] = Timer.NAMED_TAPE[
                kwargs['name']] if Timer.NAMED_TAPE.get(kwargs['name']) else 0.0
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or Timer.TAPE

    def __enter__(self):
        self.start = Timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_w):
        if self.named:
            Timer.NAMED_TAPE[self.named] += Timer.time() - self.start
        else:
            self.tape.append(Timer.time() - self.start)

# Metrics
def metrics_at_k(test_data, relevance_matrix, k):
    assert len(test_data) == len(relevance_matrix)
    recall_at_k, precision_at_k = recall_precision_at_k(test_data, relevance_matrix, k)
    
    return {
        "recall": recall_at_k,
        "ndcg": ndcg_at_k(test_data, relevance_matrix, k),
        "precision": precision_at_k
    }

def recall_precision_at_k(test_data, relevance_matrix, k):
    right_pred = relevance_matrix[:, :k].sum(1)
    precision_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    
    recall = np.sum(right_pred / recall_n)
    precision = np.sum(right_pred / precision_n)
    return recall, precision

def ndcg_at_k(test_data, relevance_matrix, k):
    pred_data = relevance_matrix[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    
    max_revelance_matrix = test_matrix
    idcg = max_revelance_matrix * 1.0 / np.log2(np.arange(2, k + 2))
    idcg = np.sum(idcg, axis=1)
    
    dcg = pred_data * (1.0 / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)

def getLabel(test_data, pred_data, groups=None):
    num_of_matrices = 1 if groups is None else 4
    start_matrix_idx = 0 if groups is None else 1
    
    relavance_matrices = [[] for _ in range(num_of_matrices)]  # unchanged revelence_matrix + filtered with groups
    groundTrue_data = [[] for _ in range(num_of_matrices)]  # unchanged test_data + filtered with groups
    names_of_groups = world.all_metrics_group_names
    
    for batch_idx in range(len(test_data)):
        for matrix_idx in range(start_matrix_idx, num_of_matrices):
            groundTrue = test_data[batch_idx]
            
            if matrix_idx > 0:  # groups is not None
                group_set = groups[names_of_groups[matrix_idx - 1]]
                groundTrue = list(set(groundTrue) & set(group_set))
                if len(groundTrue) == 0:
                    continue
                groundTrue_data[matrix_idx].append(groundTrue)
                
            prediction = pred_data[batch_idx]
            
            revelance = list(map(lambda x: x in groundTrue, prediction))
            revelance = np.array(revelance).astype("float")
            
            relavance_matrices[matrix_idx].append(revelance)
    
    if groups is None:
        return np.array(relavance_matrices[0]).astype('float')
    return [
        (np.array(relavance_matrices[matrix_idx]).astype('float'), groundTrue_data[matrix_idx])
        for matrix_idx in range(start_matrix_idx, num_of_matrices)
    ]

def print_and_update_metrics(results, best_results, loss, epoch):
    is_best = False
    wandb_results = {}
    
    if loss < best_results['loss']['best_value']:
        best_results['loss']['best_value'] = round(loss, 5)
        best_results['loss']['best_epoch'] = epoch + 1

    for k_idx, k in enumerate(world.config['topKs']):
        metrics_line = ""
        results_line = ""
        
        for metric in world.all_metrics:
            curr_metric = round(results[metric][k_idx], 5)
            
            metric_str = f"{metric}@{k}"
            metrics_line += f"{metric_str:<{len(metric_str) + 2}}"
            results_line += f"{curr_metric:<{len(metric_str) + 2}}"

            wandb_results['metrics/' + metric_str] = curr_metric
            
            if curr_metric > best_results[metric_str]['best_value']:
                is_best = True
                best_results[metric_str]['best_value'] = curr_metric
                best_results[metric_str]['best_epoch'] = epoch + 1
                
        print(metrics_line)
        print(results_line)
        
    return wandb_results, best_results, is_best

def print_and_parse_group_metrics(groups_results, groups_n_users):
    results = {}
    
    for group_name, group_result in groups_results.items():
        group_results = {
            f"{metric}@{k}": 0.0 for metric in world.all_metrics for k in world.config['topKs']
        }
        
        print(f"Group: {group_name} (Tested on {groups_n_users[group_name]} users)")
        for k_idx, k in enumerate(world.config['topKs']):
            metrics_line = ""
            results_line = ""
            
            for metric in world.all_metrics:
                metric_str = f"{metric}@{k}"
                
                group_results[metric_str] = round(group_result[metric][k_idx], 5)
                metrics_line += f"{metric_str:<{len(metric_str) + 2}}"
                
                result_str = group_results[metric_str]
                results_line += f"{result_str:<{len(metric_str) + 2}}"
            
            print(metrics_line)
            print(results_line)
        
        results[group_name] = group_results
    
    return results

def print_and_aggravate_node_dist(RecModel, decile_groups, node_type):
    l2_norm_dict = {}
    cos_drift_dict = {}
    
    cos_drift_f = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_drift_items = cos_drift_f(RecModel.embedding_item.weight, RecModel.embedding_item_start)
    l2_norm_items = norm(RecModel.embedding_item.weight, dim=1)
    
    for group_name, items in decile_groups.items():
        l2_norm_dict["L2 norm/" + group_name] = round(l2_norm_items[items].mean().item(), 5)
        cos_drift_dict["Cosine drift/" + group_name] = round(cos_drift_items[items].mean().item(), 5)
    
    node_embedding_metrics = {**l2_norm_dict, **cos_drift_dict}
    print(f"{node_type} node embedding metrics: {json.dumps(node_embedding_metrics, indent=4)}")
    
    return l2_norm_dict, cos_drift_dict

def print_and_aggravate_node_occurance(node_counter, item_to_decile_group, location):
    group_node_counter = {f"{location}/{group_name}": 0 for group_name in world.all_metrics_group_names}
    
    for node, count in node_counter.items():
        group_name = item_to_decile_group[node]
        group_node_counter[f"{location}/{group_name}"] += count
    
    print(f"{location}: {json.dumps(group_node_counter, indent=4)}")
    
    return group_node_counter