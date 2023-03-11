from src.datasets.uspto_50k import Uspto50k_torch, USPTO50K_torch_test
from src.datasets.uspto_mit import UsptoMit
from src.datasets.uspto_full import UsptoFull

from src.datasets.uspto_hard import UsptoHard_torch
from torch.autograd import Variable

import os
import networkx as nx
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from src.datasets.fast_dataloader import DataLoaderX
from torch_scatter import scatter_softmax, scatter_sum


def get_subgraph_id(edge_idx_remove_supernode, node_features, supernode_idx, padding_idx, shift):
    # find connected subgraph based on edge_idx
    G = nx.Graph()
    G.add_edges_from(edge_idx_remove_supernode.tolist())
    sub_graphs = list(nx.connected_components(G))


    batch_sub_graph = [[] for _ in range(len(shift))]
    for sub_graph in sub_graphs:
        for j in range(len(shift)):
            if shift[j] <= list(sub_graph)[0] < shift[j + 1]:
                batch_sub_graph[j].append(sub_graph)
                break
    
    
    sub_graph_ids = []
    for i in range(node_features.shape[0]):
        if i in supernode_idx or i in padding_idx:
            sub_graph_ids.append(0)
            continue
    

        batch_idx = 0
        for j in range(len(shift)):
            if shift[j] <= i < shift[j + 1]:
                batch_idx = j
                break
        
        flag = False
        for j, sub_graph in enumerate(batch_sub_graph[batch_idx]):
            if i in sub_graph:
                sub_graph_ids.append(j + 1)
                flag = True
                break
        assert flag

    sub_graph_ids = torch.tensor(sub_graph_ids)
    return sub_graph_ids


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,collate_fn=None, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,**kwargs)


class DataCollator(object):
    def __init__(self, args, feat_vocab, action_vocab):
        self.args = args
        self.feat_vocab = feat_vocab
        self.action_vocab = action_vocab

    def to_one_hot(self, x, dims: int):
        one_hot = torch.FloatTensor(*x.shape, dims).zero_() # .to(self.device)
        x = torch.unsqueeze(x, -1)
        target = one_hot.scatter_(x.dim() - 1, x.data, 1)

        target = Variable(target)
        return target

    def merge_graph(self, graph_list):
        reaction_type = torch.tensor([one["reaction_type"] for one in graph_list if one is not None])
        num_nodes = [one["node_features"].shape[0] if one is not None else 0 for one in graph_list ]
        shift = np.cumsum([0, *num_nodes]).tolist()

        graph_id = torch.arange(len(num_nodes)).repeat_interleave(torch.tensor(num_nodes), dim=0)
        node_features = torch.cat([one["node_features"]  for one in graph_list if one is not None])
        atom_action_mask = torch.cat([one['node_mask'].expand(*one["atom_action_mask"].shape) * one["atom_action_mask"] for one in graph_list if one is not None])
        
        n_bond_actions = [one["n_bond_actions"] for one in graph_list if one is not None][0]
    
        atom_target = torch.vstack([one['unflatten_target'][-1, :, :] for one in graph_list if one is not None])
        
        bond_target = torch.vstack([one['unflatten_target'][:-1, :, :n_bond_actions].view(-1, n_bond_actions)[one['adj_mask'].view(-1).bool()] for one in graph_list if one is not None]) 
        
        action_tuples = [graph['action_tuple'] for graph in graph_list if graph is not None]


        num_atom_added = torch.tensor([one["num_atoms_added"] for one in graph_list if one is not None])

        
        edge_idx = []
        edge_val = []
        local_node_idx = []
        local_edge_idx = []
        bond_action_mask = []
        bond_action = []
        atom_action = []
        attach_source = []
        attach_target = []
        
        for i in range(len(graph_list)):
            if graph_list[i] is not None:
                adj_mat = graph_list[i]['adj_mask'][:,:,0]
                local_edge = torch.nonzero(adj_mat)
                
                val = graph_list[i]["adj"][local_edge[:,0], local_edge[:,1]]
                edge_idx.append(local_edge + shift[i]) 
                edge_val.append(val)
                local_edge_idx.append(local_edge)
                local_node_idx.append(torch.arange(graph_list[i]["node_features"].shape[0]) )

                node_adj_mask = graph_list[i]['node_mask'].unsqueeze(1)
                node_adj_mask = node_adj_mask.expand(*node_adj_mask.shape)
                node_adj_mask = node_adj_mask * node_adj_mask.permute(1, 0, 2).contiguous()

                bond_action_mask.append((node_adj_mask.contiguous() * graph_list[i]['bond_action_mask'])[local_edge[:, 0], local_edge[:, 1]])

                unflatten_target = graph_list[i]['unflatten_target']
                bond_target_idx = torch.nonzero(unflatten_target[:-1,:,:])
                
                bond_action.append(unflatten_target[bond_target_idx[:,0],bond_target_idx[:,1]])
                atom_action.append(graph_list[i]['unflatten_target'][-1])
                
                action_tuple = graph_list[i]['action_tuple']
                if action_tuple[2][0]=="attach_atom":
                    attach_source.append(action_tuple[0])
                    attach_target.append(action_tuple[1])
                else:
                    attach_source.append(None)
                    attach_target.append(None)
                
                
        edge_idx = torch.cat(edge_idx)
        local_node_idx = torch.cat(local_node_idx)
        local_edge_idx = torch.cat(local_edge_idx)
        # remove supernode

        supernode_idx = torch.where(node_features[:, -2] == 2)[0]

        edge_idx_remove_supernode_idx = [a not in supernode_idx and b not in supernode_idx for a, b in edge_idx]
        edge_idx_remove_supernode = edge_idx[edge_idx_remove_supernode_idx]

        padding_idx = torch.where(node_features[:, -2] == 0)[0]


        G = nx.Graph()
        G.add_edges_from(edge_idx_remove_supernode.tolist())
        node_ids_rm_supernode = list(set(torch.arange(node_features.shape[0]).tolist())-set(supernode_idx.tolist()))
        G.add_nodes_from(node_ids_rm_supernode)
        
        sub_graphs = list(nx.connected_components(G))
        sub_graph_id = torch.zeros(node_features.shape[0])-1
        for idx, sub_id in enumerate(sub_graphs):
            sub_graph_id[list(sub_id)] = idx
            
        degree = []
        for i in range(node_features.shape[0]):
            if i in supernode_idx or i in padding_idx:
                degree.append(0)
                continue
            degree.append(G.degree[i])

        degree = torch.tensor(degree)
        
        edge_val = torch.cat(edge_val)
        bond_action_mask = torch.cat(bond_action_mask)
        bond_action = torch.cat(bond_action)
        atom_action = torch.cat(atom_action)


        action_ind = torch.stack([one["action_ind"]  for one in graph_list if one is not None])

        oh_atom_feats = []
        for i, key in enumerate(self.feat_vocab['atom_feature_keys']):
            oh_feat = self.to_one_hot(node_features[:,i], dims=len(self.feat_vocab['prop2oh']['atom'][key]) + 1)
            oh_atom_feats.append(oh_feat)
        atom_feats = torch.cat(oh_atom_feats, dim=-1)  # slight different when 
        atom_mask = torch.sign(torch.max(node_features, dim=-1)[0])
        
        oh_bond_feats = []

        for i, key in enumerate(self.feat_vocab['bond_feature_keys']):
            oh_feat = self.to_one_hot(edge_val[:,i], dims=len(self.feat_vocab['prop2oh']['bond'][key]) + 1)
            oh_bond_feats.append(oh_feat)
        bond_feats = torch.cat(oh_bond_feats, dim=-1)
        edge_idx_remove_supernode_idx = torch.tensor(edge_idx_remove_supernode_idx)
        bond_action_mask = bond_action_mask[:,:-1]# remove attach action
        bond_target = bond_target[:,:-1]
        
        # # remove supernode edges
        # bond_feats = bond_feats[edge_idx_remove_supernode_idx]
        # edge_idx = edge_idx[edge_idx_remove_supernode_idx]
        # bond_action_mask = bond_action_mask[edge_idx_remove_supernode_idx]
        # bond_target = bond_target[edge_idx_remove_supernode_idx]

        return {"graph_id": graph_id, # 
                "atom_feats": atom_feats,
                "bond_feats": bond_feats,
                "atom_mask": atom_mask,
                "padding_idx": padding_idx,
                "supernode_idx": supernode_idx,
                "degree": degree,
                "atom_action_mask": atom_action_mask,
                "bond_action_mask": bond_action_mask, 
                "sub_graph_id":sub_graph_id,
                "action_ind":action_ind,
                "action_tuples":action_tuples,
                "edge_idx": edge_idx,
                "edge_val": edge_val,
                "local_node_idx": local_node_idx,
                "local_edge_idx": local_edge_idx,
                "edge_idx_remove_supernode": edge_idx_remove_supernode_idx,
                "reaction_type": torch.tensor(reaction_type),
                "atom_target": atom_target.view(-1),
                "bond_target":  bond_target.reshape(-1), # remove attach action
                "add_atom_nums": num_atom_added,
                "shift":torch.tensor(shift),
                "attach_source":attach_source,
                "attach_target":attach_target
        }


    def collate_fn_sparse(self, batch):
        node_features_list = [one["node_features"] for one in batch] 
        node_mask_list = [one["node_mask"] for one in batch]
        adj_list = [one["adj"] for one in batch]
        adj_mask_list = [one["adj_mask"] for one in batch]
        atom_action_mask_list = [one["atom_action_mask"] for one in batch]
        bond_action_mask_list = [one["bond_action_mask"] for one in batch]
        action_ind_list = [one["action_ind"] for one in batch]
        source_smi = [one["source_smi"] for one in batch]
        target_smi = [one["target_smi"] for one in batch]
        reaction_type = [one["reaction_type"] for one in batch]
        n_bond_actions = [one["n_bond_actions"] for one in batch]
        unflatten_target = [one["unflatten_target"] for one in batch]
        action_tuples = [one["action_tuples"] for one in batch]



        max_path = max([one.shape[0] for one in node_features_list])
        
        
        graphs = {t:[None for i in range(len(batch))] for t in range(max_path)}
        
        for i in range(len(batch)):  # 一个 batch 中第几个数据
            path = node_features_list[i].shape[0]
            for t in range(path):  # 第几步的数据
                num_atoms_added = sum((node_features_list[i][t].sum(1) == 0).tolist())
                mask = [True] * len((node_features_list[i][t].sum(1) != 0).tolist())

    
                graphs[t][i] = {
                    "node_features": node_features_list[i][t][mask],   
                    "node_mask":node_mask_list[i][t][mask],   # [num_atom, 1]
                    'adj':adj_list[i][t][mask][:, mask, :],  # [num_atom, num_atom, num_bond_feature]
                    'adj_mask': adj_mask_list[i][t][mask][:, mask, :],  # [num_atom, num_atom, 1]
                    'atom_action_mask': atom_action_mask_list[i][t][mask], # [num_atom, num_actions]
                    'bond_action_mask': bond_action_mask_list[i][t][mask][:, mask, :],  # [num_atom, num_atom, num_actions]
                    'n_bond_actions': n_bond_actions[i],
                    'action_ind': action_ind_list[i][t],  # tensor(0) 
                    "n_steps": torch.tensor([one["n_steps"] for one in batch]),
                    "num_atoms_added": torch.tensor(num_atoms_added),
                    "reaction_type": reaction_type[i],
                    "unflatten_target":unflatten_target[i][t][mask + [True]][:, mask, :],
                    "action_tuple": action_tuples[i][t]}
        

        graphs = [self.merge_graph(graphs[t]) for t in range(max_path)]
        return graphs, source_smi, target_smi

    def to_one_hot(self, x, dims: int):
        one_hot = torch.FloatTensor(*x.shape, dims).zero_()
        x = torch.unsqueeze(x, -1)
        target = one_hot.scatter_(x.dim() - 1, x.data, 1)

        target = Variable(target)
        return target

def collate_fn_test(batch):
    source_smi = [one['source_smi'] for one in batch]
    target_smi = [one['target_smi'] for one in batch]
    return {
        "source_smi": source_smi,
        "target_smi": target_smi,
    }

def get_dataset(args, data_name, featurizer_key, data_path, keep_action = None, use_reaction_type = False, vocab_path=None, num_workers=8, batch_size=4, mode="train", distributed=False):
    if mode != 'test':
        if data_name == "uspto_50k":
            dataset = Uspto50k_torch(
                                    args,
                                    data_path, 
                                    featurizer_key, 
                                    keep_action,
                                    use_reaction_type,
                                    vocab_path,
                                    mode=mode)
        if data_name == "uspto_hard":
            dataset = UsptoHard_torch(
                                    args,
                                    data_path, 
                                    featurizer_key, 
                                    use_reaction_type,
                                    mode=mode)

        if data_name == "uspto_mit":
            dataset = UsptoMit()
        
        if data_name == "uspto_full":
            dataset = UsptoFull()

        args.atom_feature_keys = dataset.feat_vocab['atom_feature_keys']
        args.bond_feature_keys = dataset.feat_vocab['bond_feature_keys']
        collator = DataCollator(args, dataset.feat_vocab, dataset.action_vocab)

        if mode == 'valid':
            shuffle = False
        else:
            shuffle = True
        
        
        if distributed:
            dataset_sample = torch.utils.data.distributed.DistributedSampler(dataset)
        
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator.collate_fn_sparse, num_workers=num_workers, drop_last=False, sampler=dataset_sample, pin_memory=True)
        else:
            dataloader = DataLoaderX(local_rank=0,dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator.collate_fn_sparse, num_workers=num_workers, pin_memory=True, prefetch_factor=4)

    else:
        if data_name == "uspto_50k":
            dataset = USPTO50K_torch_test(data_path)
        
        if data_name == "uspto_hard":
            dataset = USPTO50K_torch_test(data_path)
        
        if data_name == "uspto_mit":
            dataset = UsptoMit()
        
        if data_name == "uspto_full":
            dataset = UsptoFull()
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_test, num_workers=1, pin_memory=False)
        # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_test, num_workers=num_workers, pin_memory=False)
        
        
    return dataloader


if __name__ == '__main__':
    
    dataloader = get_dataset("uspto_50k", "megan_16_bfs_randat", batch_size=128, mode='valid')
    
    total_error = 0 # train: 143 (0.36%) valid:25 (0.5%)
    for batch in tqdm(dataloader):
        total_error += sum([one['error'] for one in batch]) 
        # print(batch)
        # print()
    print()