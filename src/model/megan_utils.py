# utilities for using MEGAN in training and evaluation

from typing import List, Tuple, Optional, Set

import logging
import numpy as np
import pandas as pd
import torch

from src import USE_MOTIF_FEATURES_KEYS, USE_MOTIF_ACTION_KEYS
# from src.feat import ORDERED_ATOM_OH_KEYS
from rdkit import Chem
from rdkit.Chem import rdchem, Atom, Mol, BondStereo, BondType

logger = logging.getLogger(__name__)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def set_atom_feat(a: Atom, key: str, val: int):
    if key == 'atomic_num':
        a.SetAtomicNum(val)
    elif key == 'formal_charge':
        a.SetFormalCharge(val)
    elif key == 'chiral_tag':
        a_chiral = rdchem.ChiralType.values[val]
        a.SetChiralTag(a_chiral)
    elif key == 'num_explicit_hs':
        a.SetNumExplicitHs(val)
    elif key == 'is_aromatic':
        a.SetIsAromatic(bool(val))
    return a


class RdkitCache(object):
    def __init__(self, feat_vocab: dict):
        self.cache = {}
        self.feat_vocab = feat_vocab
        self.props = feat_vocab['prop2oh']

        self.atom_value2key = {key: {v: k for k, v in self.props['atom'][key].items()} for key in self.props['atom'].keys()}
        self.bond_value2key = {key: {v: k for k, v in self.props['bond'][key].items()} for key in self.props['bond'].keys()}

    def get_atom(self, features: Tuple[int]) -> Atom:
        if features not in self.cache:
            atom = Chem.Atom(6)
            feat_i = 0
            for key in self.feat_vocab['atom_feature_keys']:
                if key in self.props['atom']:
                    if features[feat_i] < 1:
                        raise ValueError(f"Atom {key} feat value must be >= 1")
                        # # --------- 临时修复
                        # if key == 'num_explicit_hs':  # 临时修复 for num_explicit_hs = 3 的情况
                        #     val = 3
                        # else:
                        #     raise ValueError(f"Atom {key} feat value must be >= 1")
                    else:
                        val = self.atom_value2key[key][features[feat_i]]
                    atom = set_atom_feat(atom, key, val)
                    feat_i += 1
            self.cache[features] = atom
            return atom
        return self.cache[features]

    def get_bond(self, features: Tuple[int]) -> Tuple[int, int]:
        if features not in self.cache:
            b_type = self.bond_value2key['bond_type'][features[0]]
            b_stereo = self.bond_value2key['bond_stereo'][features[1]]
            b_type = BondType.values[b_type]
            b_stereo = BondStereo.values[b_stereo]
            bond = b_type, b_stereo
            self.cache[features] = bond
            return bond
        return self.cache[features]


def mols_from_graph(rdkit_cache: RdkitCache, input_mol: Mol, input_bond_dirs: dict,
                    adj: np.ndarray, nodes: np.ndarray, changed_atoms: Set[int], only_edited: bool = False) \
        -> List[Mol]:
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms
    n_atoms = 0
    is_edited = []

    for node_i, node in enumerate(nodes[1:]):
        if node[0] != 0:
            map_num = node_i + 1
            try:
                new_a = rdkit_cache.get_atom(tuple(node))
            except ValueError as e:
                logger.warning(f'Exception while RdkitCache.get_atom: {str(e)}')
                # in case of an unknown atom type, copy atom from input mol
                new_a = input_mol.GetAtomWithIdx(map_num - 1)
            new_a.SetAtomMapNum(map_num)
            mol.AddAtom(new_a)
            n_atoms += 1
            is_edited.append(map_num in changed_atoms)

    # add bonds between adjacent atoms
    nonzero_x, nonzero_y = np.nonzero(np.triu(adj[1:, 1:, 0], k=1))

    neighbors = {i: set() for i in range(n_atoms)}

    for i in range(len(nonzero_x)):
        ix, iy = int(nonzero_x[i]), int(nonzero_y[i])
        bond = adj[ix + 1, iy + 1]
        neighbors[ix].add(iy)
        neighbors[iy].add(ix)

        b_type, b_stereo = rdkit_cache.get_bond(tuple(bond))
        bond_ind = mol.AddBond(ix, iy, order=b_type) - 1
        new_bond = None
        if b_stereo != BondStereo.STEREONONE:
            new_bond = mol.GetBondWithIdx(bond_ind)
            new_bond.SetStereo(b_stereo)

        bond_atoms = ix, iy
        if bond_atoms in input_bond_dirs:
            if new_bond is None:
                new_bond = mol.GetBondWithIdx(bond_ind)
            new_bond.SetBondDir(input_bond_dirs[bond_atoms])

    # remove compounds that have not been edited (needed for properly evaluating on some datasets, ie USPTO-MIT)
    if only_edited:
        def mark_edited(atom_i):
            is_edited[atom_i] = True
            for atom_j in neighbors[atom_i]:
                if not is_edited[atom_j]:
                    mark_edited(atom_j)

        for i in range(n_atoms):
            if is_edited[i]:
                mark_edited(i)

        for i in reversed(range(n_atoms)):
            if not is_edited[i]:
                mol.RemoveAtom(i)

    return [mol]


def get_base_action_masks(n_max_nodes: int, action_vocab: dict) -> dict:
    node_mask = torch.ones((n_max_nodes,), device=device, dtype=torch.long)

    atom_action_mask = torch.ones((*node_mask.shape, action_vocab['n_atom_actions']),
                                  dtype=torch.long, device=device)
    bond_action_mask = torch.ones((*node_mask.shape, n_max_nodes, action_vocab['n_bond_actions']),
                                  dtype=torch.long, device=device)

    # supernode (first node) can only predict "stop" action
    # (and "stop" action can be predicted only by supernode)
    # (this masking is always applied)
    bond_action_mask[0, :] = 0
    bond_action_mask[:, 0] = 0
    atom_action_mask[0] = 0
    atom_action_mask[0, action_vocab['atom_action2ind'][("stop", None)]] = 1
    atom_action_mask[1:, action_vocab['atom_action2ind'][("stop", None)]] = 0

    # mask out bond actions for diagonal ('self' node)
    # mask out bond actions for upper half (matrix is symmetric)
    triu = torch.triu(torch.ones((n_max_nodes, n_max_nodes), dtype=torch.int, device=device), diagonal=1)
    triu = triu.unsqueeze(-1)
    bond_action_mask *= triu

    # noinspection PyArgumentList
    return {
        'node_mask': node_mask.unsqueeze(-1),
        'atom_action_mask': atom_action_mask,
        'bond_action_mask': bond_action_mask, # remove attach
        'vocab': action_vocab
    }


def generate_eval_batch(old_mol_graphs: List, base_action_masks: dict, reaction_types: Optional[np.ndarray] = None) -> dict:
    results = []
    for mol_graph in old_mol_graphs:
        mol_graphs = [mol_graph]
        n_max_nodes = 0
        for adj, nodes in mol_graphs:
            n_max_nodes = max(n_max_nodes, len(nodes))

        batch_adj = torch.zeros((len(mol_graphs), n_max_nodes, n_max_nodes, mol_graphs[0][0].shape[-1]),
                                dtype=torch.long, device=device)
        node_feats = torch.zeros((len(mol_graphs), n_max_nodes, mol_graphs[0][1].shape[-1]),
                                dtype=torch.long, device=device)

        for i, (adj, nodes) in enumerate(mol_graphs):
            if not torch.is_tensor(adj):
                adj = torch.tensor(adj, dtype=torch.long, device=device)
                nodes = torch.tensor(nodes, dtype=torch.long, device=device)

            batch_adj[i, :adj.shape[0], :adj.shape[1]] = adj
            node_feats[i, :len(nodes)] = nodes

        node_mask = torch.sign(torch.max(node_feats, dim=-1)[0])
        adj_mask = torch.sign(torch.max(batch_adj, dim=-1)[0]).unsqueeze(-1)
        atom_action_mask = base_action_masks['atom_action_mask'][:n_max_nodes].unsqueeze(0).clone()
        bond_action_mask = base_action_masks['bond_action_mask'][:n_max_nodes, :n_max_nodes].unsqueeze(0)
        bond_action_mask = bond_action_mask.expand((node_mask.shape[0], -1, -1, -1)).clone()

        # only bonds between existing atoms can be edited
        atom_exists = node_mask.unsqueeze(-1).unsqueeze(-1)
        bond_action_mask *= atom_exists

        # noinspection PyArgumentList
        result = {
            'node_features': node_feats,
            'node_mask': node_mask.unsqueeze(-1),
            'adj': batch_adj,
            'adj_mask': adj_mask,
            'atom_action_mask': atom_action_mask,
            'bond_action_mask': bond_action_mask
        }
        if reaction_types is not None:
            result['reaction_type'] = torch.tensor(reaction_types, dtype=torch.long, device=device) - 1  # starts at 1
            result['reaction_type'] = torch.clamp(result['reaction_type'], min=0)
        results.append(result)
    return results


def generate_batch(graph_ind: np.ndarray, metadata: pd.DataFrame, featurizer, data, action_vocab: dict, 
                    use_motif_action: bool = False, use_motif_feature: bool = False, use_hierachical_action: bool = False) -> dict:
    sample_ind = []
    reac_n_steps = []
    n_paths = 0
    reaction_class = []

    if 'n_paths' in metadata:
        paths_per_reaction = []
        for ind in graph_ind:
            n_p = 1
            paths_per_reaction.append(n_p)
            n_paths += n_p
            for path_i in range(n_p):
                path_ind = ind + path_i
                start_ind = metadata['start_ind'][path_ind]
                n_steps = metadata['n_samples'][path_ind]
                sample_ind.append(np.arange(start_ind, start_ind + n_steps))
                reac_n_steps.append(n_steps)
                if 'class' in metadata:
                    reaction_class.append(metadata['class'][path_ind])
    else:
        paths_per_reaction = [1 for _ in range(len(graph_ind))]  # [1] * batch_size
        n_paths = len(graph_ind)  # batch size
        for ind in graph_ind:
            start_ind = metadata['start_ind'][ind]  # 从数据集中提取 action 对应的 data 开始的位置
            n_steps = metadata['n_samples'][ind]  # 从数据集中提取 action 的个数
            sample_ind.append(np.arange(start_ind, start_ind + n_steps))  # 记录 action 对应 data 的所有位置 （使用了 arange 函数）
            reac_n_steps.append(n_steps)  # 记录 actions 的个数
            if 'class' in metadata:
                reaction_class.append(metadata['class'][ind])  # 记录 reaction type， if exists

    paths_per_reaction = torch.tensor(paths_per_reaction, dtype=torch.long, device=device)
    reac_n_steps = torch.tensor(reac_n_steps, dtype=torch.long, device=device)
    n_max_steps = max(reac_n_steps)  # 一个 batch 中 最大的反应步数 （可能用于 padding）
    sample_ind = np.concatenate(sample_ind)  # 将所有 action 对应的 data 的起始位置拼接起来
    sample_data = data['sample_data'][sample_ind]  # 从数据集中提取所有 action 对应的 data

    if hasattr(sample_data, 'toarray'):
        sample_data = sample_data.toarray().astype(int)

    action_ind, atom_map1, atom_map2, n_nodes, is_hard, reaction_type = \
        sample_data[:, 0], sample_data[:, 1], sample_data[:, 2], sample_data[:, 3], sample_data[:, 4], sample_data[:, 5]
    if use_motif_feature:
        n_motif_nodes = sample_data[:, 6]
        n_max_motif_nodes = int(max(n_motif_nodes))  # (data['motif_node'][sample_ind].toarray() != 0).sum(1).max()  # int(max(n_motif_node))  # 记录一个batch中最大的 motif node 个数  # TODO: 需要从 featurize 改这里的 bug

    n_max_nodes = int(max(n_nodes))  # 记录一个batch中最大的 node 个数

    tensor_data = {
        'atom': data['atom'][sample_ind],   # batch 内每个 sample 每一步下的 atom information  # [sample_size, max_node]
        'bond': data['bond'][sample_ind],   # batch 内每个 sample 每一步下的 bond information  # [sample_size, max_node, max_node]
        'max_n_nodes': n_max_nodes,  # batch 内最大的 node 个数
    }


    tensor_data = featurizer.to_tensor_batch(tensor_data, action_vocab)  # tensorize data
    if use_motif_feature:
        tensor_data['motif2atom'] = torch.from_numpy(tensor_data['motif2atom'].toarray()).long().to(device)  # [sample_size, max_node]
        assert tensor_data['motif2atom'][:, n_max_nodes:].sum() == 0
    # tensor_data['atom']  # [sample_size, max_nodes_in_batch, len(node_oh_dim) = 8]
    # tensor_data['bond']  # [sample_size, max_nodes_in_batch, max_nodes_in_batch, len(bond_oh_dim) = 3]

    node_feats = torch.zeros((n_paths, n_max_steps, n_max_nodes, tensor_data['atom'].shape[-1]),  # [batch_size, max_steps_in_batch, max_nodes_in_batch, len(node_oh_dim)]
                             dtype=torch.long, device=device)

    adj = torch.zeros((n_paths, n_max_steps, n_max_nodes, n_max_nodes, tensor_data['bond'].shape[-1]),  # [batch_size, max_steps_in_batch, max_nodes_in_batch, max_nodes_in_batch, len(edge_oh_dim)]
                      dtype=torch.long, device=device)


    is_hard_matrix = torch.zeros((n_paths, n_max_steps), dtype=torch.long, device=device)  # [batch_size, max_steps_in_batch]
    reaction_type = torch.zeros((n_paths, n_max_steps), dtype=torch.long, device=device)   # [batch_size, max_steps_in_batch]

    k = 0
    # 遍历每个 mini-batch 中的 sample 的 reaction_steps (即 action 的个数)
    for i, n_steps in enumerate(reac_n_steps):
        for j in range(n_steps):
            node_feats[i, j] = tensor_data['atom'][k]
            adj[i, j] = tensor_data['bond'][k]

            if is_hard[k]:
                is_hard_matrix[i, j] = 1
            if 'reaction_type' in tensor_data:
                reaction_type[i, j] = tensor_data['reaction_type'][k] - 1  # reaction types are numbered from 1
            k += 1

    node_mask = torch.sign(torch.max(node_feats, dim=-1)[0])  # [batch_size, max_steps_in_batch, max_nodes_in_batch]
    adj_mask = torch.sign(torch.max(adj, dim=-1)[0]).unsqueeze(-1)  # [batch_size, max_steps_in_batch, max_nodes_in_batch, max_nodes_in_batch, 1] 


    atom_action_mask = torch.ones((*node_mask.shape, action_vocab['n_atom_actions']),  # [batch_size, max_steps_in_batch, max_nodes_in_batch, 47]
                                  dtype=torch.float, device=device)
    bond_action_mask = torch.ones((*node_mask.shape, n_max_nodes, action_vocab['n_bond_actions']),  # [batch_size, max_steps_in_batch, max_nodes_in_batch, max_nodes_in_batch, 7]
                                  dtype=torch.float, device=device)

    # supernode (first node) can only predict "stop" action
    # (and "stop" action can be predicted only by supernode)
    # (this masking is always applied)
    bond_action_mask[:, :, 0, :] = 0
    bond_action_mask[:, :, :, 0] = 0
    atom_action_mask[:, :, 0] = 0
    atom_action_mask[:, :, 0, action_vocab['stop_action_num']] = 1
    atom_action_mask[:, :, 1:, action_vocab['stop_action_num']] = 0

    # mask out bond actions for diagonal ('self' node)
    # mask out bond actions for upper half (matrix is symmetric)
    triu = torch.triu(torch.ones((n_max_nodes, n_max_nodes), dtype=torch.int, device=device), diagonal=1)
    triu = triu.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    bond_action_mask *= triu

    # only bonds between existing atoms can be edited
    atom_exists = node_mask.unsqueeze(-1).unsqueeze(-1)
    bond_action_mask *= atom_exists

    target = torch.zeros((n_paths, n_max_steps, n_max_nodes + 1, n_max_nodes,  # # [batch_size, max_steps_in_batch, max_nodes_in_batch + 1, max_nodes_in_batch, 47]
                          action_vocab['n_target_actions']), dtype=torch.float, device=device)

    node_mask = node_mask.unsqueeze(-1)
    if use_motif_feature:
        motif_node_mask = motif_node_mask.unsqueeze(-1)

    result = {
        'node_features': node_feats,
        'node_mask': node_mask,
        'adj': adj,
        'adj_mask': adj_mask,
        'target': target,
        'atom_action_mask': atom_action_mask,
        'bond_action_mask': bond_action_mask,
        'n_steps': reac_n_steps,
        'is_hard': is_hard_matrix,
        'n_paths': paths_per_reaction,
    }
    if 'reaction_type' in tensor_data:
        result['reaction_type'] = reaction_type

    k = 0

    batch_action_ind = torch.zeros((n_paths, n_max_steps), dtype=torch.long, device=device)  # [batch_size, max_steps_in_batch]

    if use_motif_action:
        motif_target = torch.zeros(n_paths, n_max_steps, n_max_nodes, 
                        len(action_vocab['non_empty_atom2motif_ind']), action_vocab['max_atom2motif'], dtype=torch.float, device=device)

    for i, n_steps in enumerate(reac_n_steps):  # tensor of number of steps
        for j in range(n_steps):  # reaction_steps
            this_action_ind, a1, a2 = action_ind[k], atom_map1[k], atom_map2[k]
            if a1 < a2:
                a2, a1 = a1, a2
            
            if use_motif_action and use_hierachical_action:
                if a2 == -1 and this_action_ind in action_vocab['motif2atom']:  # 如果是motif action 且有对应 atom action，把motif action 转化为 atom action，然后再在对应 atom action 的二级预测上加入 label
                    old_this_action_ind = this_action_ind
                    this_action_ind = action_vocab['motif2atom'][this_action_ind]
                    motif_target[i, j, a1, action_vocab['non_empty_atom2motif_ind'].index(this_action_ind), action_vocab['atom2motif'][this_action_ind]['motif_actions_ind'].index(old_this_action_ind)] = 1
                elif a2 == -1 and this_action_ind in action_vocab['non_empty_atom2motif_ind']:  # 如果是atom action，且其有二级预测，那么在对应 atom action 的二级预测上加入 label = 0
                    motif_target[i, j, a1, action_vocab['non_empty_atom2motif_ind'].index(this_action_ind), 0] = 1

            action_num = action_vocab['atom_action_num'][this_action_ind] if a2 == -1 \
                else action_vocab['bond_action_num'][this_action_ind]
            batch_action_ind[i, j] = this_action_ind

            target[i, j, a2, a1, action_num] = 1
            k += 1
        for j in range(n_steps, n_max_steps):
            bond_action_mask[i, j] = 0
            atom_action_mask[i, j] = 0

    # padding motif_target  # [batch_size, n_reaction_steps, num_nodes, len(action_vocab['non_empty_atom2motif_ind']), action_vocab['max_atom2motif']]
    # for i, ind in enumerate(action_vocab['non_empty_atom2motif_ind']):
    #     if motif_target[:, :, :, i, len(action_vocab['atom2motif'][ind]['motif_actions_ind'])]

    # noinspection PyArgumentList
    result['action_ind'] = batch_action_ind
    result['unflatten_target'] = target

    target = target.reshape(n_paths, n_max_steps, -1)
    result['target'] = target

    if use_motif_action:
        result['motif_target'] = motif_target

    return result
