# -*- coding: utf-8 -*-
"""
Graph representation of the source compound for MEGAN reaction generation model
"""
import logging
from typing import Tuple, List

import numpy as np
import torch
import copy
from rdkit import Chem
from src.feat.graph_features import get_atom_features, get_bond_features#, BOND_PROP2OH, ATOM_PROP2OH
# from src.feat import ORDERED_ATOM_OH_KEYS, ORDERED_BOND_OH_KEYS, ORDERED_MOTIF_OH_KEYS
# from src.feat.ps.mol_bpe import Tokenizer

logger = logging.getLogger(__name__)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# noinspection PyDefaultArgument
def ravel_atom_features(atom_features: np.array, node_oh_dim) -> int:
    raveled = np.ravel_multi_index(atom_features, node_oh_dim)
    return int(raveled)


# noinspection PyDefaultArgument
def unravel_atom_features(atom_features: np.array, node_oh_dim) -> np.array:
    return np.array(np.unravel_index(atom_features, node_oh_dim))


# noinspection PyDefaultArgument
def ravel_bond_features(bond_features: np.array, edge_oh_dim) -> int:
    raveled = np.ravel_multi_index(bond_features, edge_oh_dim)
    return int(raveled)


# noinspection PyDefaultArgument
def unravel_bond_features(bond_features: np.array, edge_oh_dim) -> np.array:
    return np.array(np.unravel_index(bond_features, edge_oh_dim))


def get_graph(mol, ravel: bool = True, to_array: bool = False, to_torch: bool = False,
              atom_feature_keys: List[str] = None,
              bond_feature_keys: List[str] = None,
              atom_prop2oh: dict = None,
              bond_prop2oh: dict = None,
              max_node = None) -> Tuple:
    
    del_idxs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol()=="*":
            del_idxs.append(atom.GetIdx())
    
    for del_idx in sorted(del_idxs, reverse=True):
        mol.RemoveAtom(del_idx)
    
    self_bond_features = np.zeros(len(bond_feature_keys), dtype=int)
    edge_oh_dim = [len(bond_prop2oh[feat_key]) + 1 for feat_key in bond_feature_keys if feat_key in bond_prop2oh]
    node_oh_dim = [len(atom_prop2oh[feat_key]) + 1 for feat_key in atom_feature_keys if feat_key in atom_prop2oh]

    supernode_bond_features = np.zeros(len(edge_oh_dim), dtype=int)
    supernode_atom_features = np.zeros(len(node_oh_dim), dtype=int)

    for i, feat_key in enumerate(bond_feature_keys):
        if feat_key == 'bond_type':
            self_bond_features[i] = bond_prop2oh['bond_type']['self']
            supernode_bond_features[i] = bond_prop2oh['bond_type']['supernode']

    for i, feat_key in enumerate(atom_feature_keys):
        if feat_key == 'is_supernode':
            supernode_atom_features[i] = atom_prop2oh['is_supernode'][1]


    adj_vals, adj_rows, adj_cols = [], [], []
    
    if max_node is None:
        n_nodes = max(a.GetAtomMapNum() for a in mol.GetAtoms()) + 1
    else:
        n_nodes = max_node
    # n_nodes = mol.GetNumAtoms() + 1

    nodes = np.zeros((n_nodes, len(supernode_atom_features)), dtype=int)
    nodes[0] = supernode_atom_features  # supernode is always the first node

    # find separate compounds in "mol" object and number them randomly
    compounds = {}
    
    added_bonds = [] # 存储已经记录的边，避免相同边被多次记录
    
    def add_bond(i1, i2, val):
        if (i1,i2) not in added_bonds:
            adj_vals.append(val)
            adj_rows.append(i1)
            adj_cols.append(i2)
            added_bonds.append((i1,i2))

        if i1 != i2:
            if (i2,i1) not in added_bonds:
                adj_vals.append(val)
                adj_rows.append(i2)
                adj_cols.append(i1)
                added_bonds.append((i2,i1))

    compounds[0] = {0}  # supernode has always mol_id=0
    # is_reactant[0] = 0  # feature irrelevant for supernode

    add_bond(0, 0, self_bond_features)

    for i, a in enumerate(mol.GetAtoms()):
        i = a.GetAtomMapNum()
        compounds[i] = {i}
        # if a.HasProp('in_target') and a.GetBoolProp('in_target'):
        #     is_reactant[i] = IN_REACTANT

        nodes[i] = get_atom_features(a, atom_feature_keys, atom_prop2oh=atom_prop2oh)

        # add special bond to 'supernode'
        add_bond(0, i, supernode_bond_features)
        add_bond(i, i, self_bond_features)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetAtomMapNum()
        a2 = bond.GetEndAtom().GetAtomMapNum()
        bond_features = get_bond_features(bond, bond_feature_keys, bond_prop2oh=bond_prop2oh)
        add_bond(a1, a2, bond_features)

        if compounds[a1] is not compounds[a2]:
            common_set = compounds[a1].union(compounds[a2])
            for map_id in common_set:
                compounds[map_id] = common_set

    # number separate compounds in graph
    cur_mol = 1  # 0 is for supernode
    mol_inds = np.zeros((n_nodes,), dtype=int)

    for i in range(1, n_nodes):
        if i in compounds and mol_inds[i] == 0:
            for m in compounds[i]:
                mol_inds[m] = cur_mol

            # if any(is_reactant[m] == IN_REACTANT for m in compounds[i]):
            #     for m in compounds[i]:
            #         is_reactant[m] = IN_REACTANT

            cur_mol += 1

    if 'is_reactant' in atom_feature_keys and 'is_reactant' in atom_prop2oh:
        is_reactant_feat_ind = atom_feature_keys.index('is_reactant')
        # nodes[:, is_reactant_feat_ind] = is_reactant

    if ravel:
        nodes = np.array([ravel_atom_features(node, node_oh_dim) for node in nodes])
        adj_vals2 = [ravel_bond_features(val, edge_oh_dim) for val in adj_vals] # TODO:edge_oh_dim和训练时不一致 
        
        adj_vals = adj_vals2

    if to_array:
        if ravel:
            adj = np.zeros((n_nodes, n_nodes), dtype=int)
        else:
            adj = np.zeros((n_nodes, n_nodes, len(edge_oh_dim)), dtype=int)

        for val, row, col in zip(adj_vals, adj_rows, adj_cols):
            adj[row, col] = val

        if to_torch:
            adj = torch.tensor(adj, device=device, dtype=torch.long)
            nodes = torch.tensor(nodes, device=device, dtype=torch.long)

        return adj, nodes

    if to_torch:
        raise NotImplementedError('"to_torch" option can be used only when to_array=True')

    return (adj_vals, adj_rows, adj_cols), nodes


