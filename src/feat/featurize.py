"""
Main functions for featurization for MEGAN reaction generator
"""
import itertools
import logging
import os
import random
from typing import List, Tuple

import copy
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol, Mol
from scipy import sparse

from src.feat.mol_graph import get_graph, get_motif_graph
from src.feat.ps.mol_bpe import Tokenizer
from src.feat.reaction_actions import ReactionAction, AddRingAction, AddAtomAction, BondEditAction, AtomEditAction, AddMotifAction, \
    StopAction
from src.feat.ring_actions import find_added_benzene_rings
from src.feat.utils import atom_to_edit_tuple, get_bond_tuple, fix_incomplete_mappings, reac_to_canonical, \
    fix_explicit_hs
from src.feat.utils import renumber_atoms_for_mapping, mark_reactants
from src.feat.graph_features import get_atom_features
# from src.feat import ORDERED_ATOM_OH_KEYS, ORDERED_BOND_OH_KEYS, ATOM_EDIT_TUPLE_KEYS
import numpy as np


logger = logging.getLogger(__name__)





def smi2mol(smiles: str, kekulize=False, sanitize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol):
    return Chem.MolToSmiles(mol)


def get_submol(mol, atom_indices, kekulize=False):
    if len(atom_indices) == 1:
        return smi2mol(mol.GetAtomWithIdx(atom_indices[0]).GetSymbol(), kekulize)
    aid_dict = { i: True for i in atom_indices }
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid = bond.GetBeginAtomIdx()
        end_aid = bond.GetEndAtomIdx()
        if begin_aid in aid_dict and end_aid in aid_dict:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol


class MotifGenerator(object):
    def __init__(self, vocab_path):
        self.tokenizer = Tokenizer(vocab_path)  # TODO: 
    
    def generate_motif(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        idx2am = dict()
        am2idx = dict()

        for atom in mol.GetAtoms():
            idx2am[atom.GetIdx()] = atom.GetAtomMapNum()
            am2idx[atom.GetAtomMapNum()] = atom.GetIdx()
            atom.SetAtomMapNum(0)
            atom.SetNumExplicitHs(0)
        
        group_idxs, group_smis = self.tokenizer(mol)  # TODO: 这里 Idx 会改变

        all_motifs_info = {'motifs': [], 'smiles': [], 'smiles_with_mapping': []}  # , 'motif_bonds': [], 'mol': mol}
        for group, _smiles in zip(group_idxs, group_smis):
            motif_info = []

            for idx in group:
                motif_info.append(idx2am[idx])

            # 这段代码是用来找这个motif与其他的motif的连接点的 =>
            editable_mol = Chem.EditableMol(Chem.MolFromSmiles(smiles))
            removed_atoms_idx = list(set(idx2am.keys()) - set(group))
            removed_atoms_idx.reverse()
            removed_atoms_idx = sorted(removed_atoms_idx, reverse=True)
            for _idx in removed_atoms_idx:
                editable_mol.RemoveAtom(_idx)
            smiles_with_mapping = Chem.MolToSmiles(editable_mol.GetMol())

            all_motifs_info['motifs'].append(tuple(motif_info))
            all_motifs_info['smiles'].append(_smiles)
            all_motifs_info['smiles_with_mapping'].append(smiles_with_mapping)
            # <= 这段代码是用来找这个motif与其他的motif的连接点的

        return all_motifs_info, am2idx, idx2am


class ReactionSampleGenerator(object):
    def __init__(self, source_mol: RWMol, target_mol: Mol, action_vocab: dict,
                 forward: bool = False, action_order: str = 'dfs', use_motif_action: bool = False, use_motif_feature: bool = False, vocab_path: str = None):
        self.source_mol = source_mol
        self.target_mol = target_mol
        self.vocab_path = vocab_path
        
        if use_motif_action:
            from src.feat.motif_utils import MotifGenerator
            vocab_path = "/gaozhangyang/experiments/MotifRetro/data/uspto_50k/vocab_500.txt"
            self.motif_generator = MotifGenerator(vocab_path)

        self.randomize_action_types = 'random' in action_order
        self.randomize_map_atom_order = action_order == 'random' or 'randat' in action_order
        self.randomize_next_atom = action_order == 'random'

        self.action_order = action_order
        self.use_motif_action = use_motif_action
        self.use_motif_feature = use_motif_feature

        self.target_mol_am2idx = dict()
        for atom in self.target_mol.GetAtoms():
            self.target_mol_am2idx[atom.GetAtomMapNum()] = atom.GetIdx()

        self.atoms_stack = []  # 关于原子的栈，用 mapping number 记录
        if 'bfs' in self.action_order:
            for a in target_mol.GetAtoms():
                self.atoms_stack.append(a.GetAtomMapNum())
            self.atoms_stack = list(sorted(self.atoms_stack))

        mark_reactants(source_mol, target_mol)

        self.edited_atoms = set()
        self.forward = forward
        self.action_vocab = action_vocab
        self.prop_dict = action_vocab['prop2oh']
        
        

        self.added_rings = {
            'benzene': find_added_benzene_rings(source_mol=source_mol, target_mol=target_mol)
        }


        self.current_step = 0
        self.current_mol_graph = get_graph(
            self.source_mol, ravel=False, to_array=True, 
            atom_feature_keys = list(self.prop_dict["atom"].keys()), 
            bond_feature_keys = list(self.prop_dict["bond"].keys()),
            atom_prop2oh=self.prop_dict['atom'], bond_prop2oh=self.prop_dict['bond']
        )

    def generate_gen_action(self) -> ReactionAction:
        atoms_only_in_target = {}
        source_atoms = {}
        target_atoms = {}
        target_atomic_nums = {}

        source_bonds = {}
        target_bonds = {}

        map2target_atom = {}

        changed_atoms = {}
        changed_bonds = {}
        added_bonds = {}
        deleted_bonds = {}
        new_rings = {}
        new_motifs = {}
        new_atoms = {}

        def find_add_new_atom_actions(source_a: int, new_a: int, bond_type: Tuple[int, int]):
            if new_a in atoms_only_in_target and source_a not in atoms_only_in_target:
                motif_added = False

                # # TODO: add new vocabulary version
                # if self.use_motif_action:
                #     for cluster, motif_info in self.added_motifs.items():
                #         if source_a not in cluster and all(m in atoms_only_in_target for m in cluster) and new_a in cluster:
                #             new_motifs[source_a] = AddMotifAction(
                #                 atom_map1=source_a,
                #                 atom_map2=new_a,
                #                 bond_type=bond_type,
                #                 new_atoms_map_nums=cluster,  #  的 atom map number
                #                 motif_info=motif_info,
                #                 action_vocab=self.action_vocab,
                #                 is_hard=source_a not in self.edited_atoms)
                #             motif_added = True
                #             break

                if not motif_added:
                    ring_added = False
                    if not self.use_motif_action:
                        for ring_key, all_ring_atom_maps in self.added_rings.items():
                            for ring_atom_maps in all_ring_atom_maps:
                                if source_a in ring_atom_maps and \
                                        all(m == source_a or m in atoms_only_in_target for m in ring_atom_maps):  # 如果这个环的原子要么是 source_a, 要么不在 source 里，加入
                                    new_rings[source_a] = AddRingAction(
                                        atom_map1=source_a,
                                        new_atoms_map_nums=ring_atom_maps,  #  的 atom map number
                                        ring_key=ring_key,
                                        action_vocab=self.action_vocab,
                                        is_hard=source_a not in self.edited_atoms)
                                    ring_added = True
                                    break
                            if ring_added:
                                break

                    if not ring_added:
                        atomic_num = target_atomic_nums[new_a]
                        new_atoms[new_a, source_a] = AddAtomAction(source_a, new_a,
                                                                *bond_type, atomic_num, *target_atoms[new_a],
                                                                action_vocab=self.action_vocab,
                                                                is_hard=source_a not in self.edited_atoms)

        def find_new_bonds(a2: int, a1: int):
            if a1 in source_atoms and a2 in source_atoms and (a1, a2) not in source_bonds:
                added_bonds[a2, a1] = \
                    BondEditAction(a1, a2, *target_bonds[(a1, a2)], action_vocab=self.action_vocab,
                                   is_hard=a1 not in self.edited_atoms and a2 not in self.edited_atoms)

        for i, a in enumerate(self.source_mol.GetAtoms()):  # 遍历所有 source 的 atom
            source_atoms[a.GetAtomMapNum()] = atom_to_edit_tuple(a)  # 记录 source molecule 的所有原子的 property

        for a in self.target_mol.GetAtoms():  # 遍历 target 的所有 atoms
            am = a.GetAtomMapNum()
            at = atom_to_edit_tuple(a)
            if am not in source_atoms:  # 如果发现 target 的 atom 不在 source 里，记录
                atoms_only_in_target[am] = a
            elif source_atoms[am] != at:  # 如果发现 target 的 atom 的 property 出现改变，则返回 edit atom action
                changed_atoms[am] = AtomEditAction(am, *at, action_vocab=self.action_vocab,
                                                   is_hard=am not in self.edited_atoms)
            target_atoms[am] = at    # 记录 target molecule 的所有原子的 property
            target_atomic_nums[am] = a.GetAtomicNum()  # 记录 target molecule 的所有原子的原子号 (e.g., C -> 6, O -> 8)
            map2target_atom[am] = a

        for bond in self.source_mol.GetBonds():  # 得到 source molecule 的所有 bond 的 property （包括两边原子编号，BondType，Stereo）
            bond_tuple = get_bond_tuple(bond)
            source_bonds[(bond_tuple[0], bond_tuple[1])] = bond_tuple[2:]

        for bond in self.target_mol.GetBonds():  # 得到 target molecule 的所有 bond 的 property （包括两边原子编号，BondType，Stereo）
            bond_tuple = get_bond_tuple(bond)
            a1, a2, bond_type = bond_tuple[0], bond_tuple[1], bond_tuple[2:]
            target_bonds[(a1, a2)] = bond_type

            find_add_new_atom_actions(a1, a2, bond_type)
            find_add_new_atom_actions(a2, a1, bond_type)
            find_new_bonds(a2, a1)

        for bond_atoms, bond_type in source_bonds.items():
            if bond_atoms not in target_bonds:
                if bond_atoms[0] in target_atoms or bond_atoms[1] in target_atoms:
                    # find deleted bonds
                    deleted_bonds[(bond_atoms[1], bond_atoms[0])] = \
                        BondEditAction(bond_atoms[0], bond_atoms[1], None, None,
                                       action_vocab=self.action_vocab,
                                       is_hard=bond_atoms[0] not in self.edited_atoms and
                                               bond_atoms[1] not in self.edited_atoms)

            elif target_bonds[bond_atoms] != bond_type:
                # find edited bonds
                changed_bonds[(bond_atoms[1], bond_atoms[0])] = \
                    BondEditAction(bond_atoms[0], bond_atoms[1], *target_bonds[bond_atoms],
                                   action_vocab=self.action_vocab,
                                   is_hard=bond_atoms[0] not in self.edited_atoms and
                                           bond_atoms[1] not in self.edited_atoms)

        # for forward synthesis, bond addition has the highest priority regardless of atoms
        # for retrosynthesis, bond deletion has the highest priority regardless of atoms
        if self.forward:
            action_type_priorities = [
                ('double', added_bonds),
                ('double', deleted_bonds),
                ('double', changed_bonds),
                ('single', changed_atoms),
                ('single', new_rings),
                ('double', new_motifs),
                ('double', new_atoms),
            ]
        else:
            action_type_priorities = [
                ('double', deleted_bonds),
                ('double', added_bonds),
                ('double', changed_bonds),
                ('single', changed_atoms),
                ('single', new_rings),
                ('double', new_motifs),
                ('double', new_atoms),
            ]

        if self.randomize_action_types:
            random.shuffle(action_type_priorities)

        target_atom_keys = list(target_atoms.keys())

        if self.randomize_map_atom_order:
            random.shuffle(target_atom_keys)
        else:
            target_atom_keys = list(sorted(target_atom_keys))

        if self.randomize_next_atom:
            atom_maps1 = target_atom_keys
            atom_maps2 = target_atom_keys
        elif 'bfs' in self.action_order:
            atom_maps1 = self.atoms_stack
            atom_maps2 = self.atoms_stack
        else:  # dfs
            atom_maps1 = reversed(self.atoms_stack)
            atom_maps2 = itertools.chain(reversed(self.atoms_stack), target_atom_keys)

        for atom_map1 in atom_maps1:
            for action_type, actions_dict in action_type_priorities:
                if action_type == 'double':
                    for atom_map2 in atom_maps2:
                        if (atom_map1, atom_map2) in actions_dict:
                            return actions_dict[(atom_map1, atom_map2)]
                        elif (atom_map2, atom_map1) in actions_dict:
                            return actions_dict[(atom_map2, atom_map1)]
                elif atom_map1 in actions_dict:
                    return actions_dict[atom_map1]

        # if no actions found in atoms stack, go with action priorities
        for action_type, actions_dict in action_type_priorities:
            if len(actions_dict) == 0:
                continue
            action_dict_keys = list(actions_dict.keys())

            if self.randomize_map_atom_order:
                atom_maps = random.choice(action_dict_keys)
            else:
                action_dict_keys = list(sorted(action_dict_keys))
                atom_maps = action_dict_keys[0]

            return actions_dict[atom_maps]

        return StopAction(action_vocab=self.action_vocab)

    def get_motif_id_by_idx(self, all_motifs_info, add_atom_idx):
        for index, id in zip(all_motifs_info['motifs'], all_motifs_info['motif_id']):
            if add_atom_idx in index:
                return id
            
    def gen_training_sample(self) -> dict:
        '''
        只需要修改这一个函数可实现MotifRetro的数据准备
        
        traning_sample includes: 'action_tuple', 'action_str', 'adj', 'nodes', 'step', 'is_hard', 'atom_map1', 'atom_map2'
        '''
        self.source_mol.UpdatePropertyCache(strict=False)

        # generate action
        reaction_action = self.generate_gen_action()

        training_sample = {
            'action_tuple': reaction_action.get_tuple(),
            'action_str': str(reaction_action)
        }

        adj, nodes = get_graph(self.source_mol, atom_prop2oh=self.prop_dict['atom'],
            atom_feature_keys = list(self.prop_dict["atom"].keys()), 
            bond_feature_keys = list(self.prop_dict["bond"].keys()),
            bond_prop2oh=self.prop_dict['bond']
        )
            


        training_sample['adj'] = adj # (adj_vals, adj_rows, adj_cols), adj_vals inlude compressed bond features
        training_sample['nodes'] = nodes # nodes inlude compressed bond features
        training_sample['step'] = self.current_step
        training_sample['is_hard'] = reaction_action.is_hard
        training_sample['atom_map1'] = reaction_action.atom_map1


        atom_map1 = reaction_action.atom_map1
        atom_map2 = reaction_action.atom_map2
        if atom_map1 > 0:
            self.atoms_stack.append(atom_map1)
            self.edited_atoms.add(atom_map1)
        if atom_map2 > 0:
            self.atoms_stack.append(atom_map2)
            self.edited_atoms.add(atom_map2)

        if not isinstance(reaction_action, BondEditAction):
            atom_map2 = -1
            
        training_sample['is_atom_action'] = 0 if atom_map2 == -1 else 1

        training_sample['atom_map2'] = atom_map2

        
        # construct source motif_id
        if self.use_motif_action:
            all_motifs_info, am2idx, idx2am = self.motif_generator.generate_motif(copy.copy(self.source_mol))
            motif_id_source = np.zeros_like(training_sample['nodes'])
            atom2motif_source = np.zeros_like(training_sample['nodes'])
            for k, index, id in zip(range(len(all_motifs_info['motifs'])), all_motifs_info['motifs'], all_motifs_info['motif_id']):
                for i in index:
                    motif_id_source[idx2am[i]] = id # 由于第一个点是supernode, 所以索引需要+1
                    atom2motif_source[idx2am[i]] = k+1
            
            training_sample['motif_id_source'] = motif_id_source
            training_sample['atom2motif_source'] = atom2motif_source
            
        
            # construct target motif_id for AddAtom action
            motif_id_target = 0
            if type(reaction_action) == AddAtomAction:
                all_motifs_info, am2idx, idx2am = self.motif_generator.generate_motif(copy.copy(self.target_mol))
                add_atom_idx = am2idx[reaction_action.atom_map2]
                motif_id_target = self.get_motif_id_by_idx(all_motifs_info, add_atom_idx)
            training_sample['motif_id_target'] = motif_id_target
            
            
        
        # execute action
        self.source_mol = reaction_action.apply(self.source_mol)                        # 基于 rdkit 的化学分子
        self.current_mol_graph = reaction_action.graph_apply(*self.current_mol_graph)   # featurized 的图
        
        self.current_step += 1
        return training_sample

    def get_motif_idx(self, mol):
        all_motifs_info, am2idx, idx2am = self.motif_generator.generate_motif(Chem.MolToSmiles(mol))
        return all_motifs_info
        
def gen_training_samples(target_mol: Mol, source_mol: Mol, n_max_steps: int,
                         action_vocab: dict, use_motif_action: bool = False, use_motif_feature: bool = False, forward: bool = False, action_order: str = 'dfs', vocab_path: str = None) -> Tuple[List, str]:
    
    training_samples = []
    final_smi = ''

    reaction_state = ReactionSampleGenerator(
        source_mol=Chem.rdchem.RWMol(source_mol),  # TODO: 这个 RWMol 的用处是什么？
        target_mol=target_mol,
        forward=forward,
        action_vocab=action_vocab,
        action_order=action_order,
        use_motif_action=use_motif_action,
        use_motif_feature=use_motif_feature,
        vocab_path=vocab_path,
    )

    for i in range(n_max_steps):
        sample = reaction_state.gen_training_sample()
        training_samples.append(sample)

        if sample['action_tuple'][0] == 'stop':
            if reaction_state.source_mol is None:
                logger.warning(f'"None" mol after {i + 1} steps')
                return [], ''
            final_smi = Chem.MolToSmiles(reaction_state.source_mol)
            break

        if i >= n_max_steps - 1:
            return [], ''

    return training_samples, final_smi



                                                            
def gen_training_samples_in_loop(source_smi, target_smi, n_max_steps, use_motif_action, use_motif_feature, forward, action_order, vocab_path, action_vocab):
    
    ############ get training samples(每一个action对应一个sample)
    if not isinstance(target_smi, str) or len(target_smi) == 0:
        target_smi = source_smi

    try:
        target_mol = Chem.MolFromSmiles(target_smi)
        source_mol = Chem.MolFromSmiles(source_smi)

        if target_mol is not None and source_mol is not None:
            target_mol, source_mol = fix_incomplete_mappings(target_mol, source_mol)  # TODO: 作用是什么？A：见 Notion
            target_mol, source_mol = reac_to_canonical(target_mol, source_mol)  # TODO：作用是什么？ A: SMILES有变化。似乎是给碳原子重新编号，以消除 information leaking 的问题。
            source_mol = fix_explicit_hs(source_mol)  # rdkit has a problem with implicit hs. By default there are only explicit hs. This is a hack to fix this error
            target_mol = fix_explicit_hs(target_mol)

            source_mol = renumber_atoms_for_mapping(source_mol)  # TODO: 作用是什么？没有看到 SMILES 有变化
            target_mol = renumber_atoms_for_mapping(target_mol)
    except Exception:
        target_mol, source_mol = None, None

    if target_mol is None or source_mol is None:
        
        return None, None
    


    try:
        # IMPORTANT! 生成输入模型的训练数据的函数. 
        # type(training_samples) = List. 应该是一个 product 到达 final_smi 的所有 actions
        # type(final_smi) = str

        training_samples, final_smi = gen_training_samples(target_mol, source_mol, n_max_steps,
                                                            action_vocab=action_vocab, use_motif_action=use_motif_action, use_motif_feature=use_motif_feature,
                                                            forward=forward, action_order=action_order, vocab_path=vocab_path)

    except Exception:
        return None, None
    
    # training_samples, final_smi = gen_training_samples(target_mol, source_mol, n_max_steps,
    #                                                         action_vocab=action_vocab, use_motif_action=use_motif_action, use_motif_feature=use_motif_feature,
    #                                                         forward=forward, action_order=action_order, vocab_path=vocab_path)
    
    

    if len(training_samples) == 0:
        return None, None
    return training_samples, final_smi
    

def get_sparse_data(training_samples, start_ind, data_x, reaction_i, max_n_nodes, samples_len, use_motif_action, use_motif_feature):
    sample_data = [], [], []  # vals, rows, cols
    action_tuples = []
    # 最后存放进入 node_mat, adj_mat 的三组列表
    adj_vals = [], [], []  # vals, rows, cols
    nodes_vals = [], [], []  # vals, rows, cols
    nodes_motif = [], [], []
    nodes_atom2motif = [], [], []

    # for each sample，遍历所有的 actions
    for sample_ind, sample in enumerate(training_samples):
        ind = start_ind + sample_ind
        action_tuples.append((ind, sample['action_tuple']))  # start_index + action_index，编辑 action 种类

        # 第零列将会是每个 action 对应的 ID (见 megan_graph.py line 354: sample_data_mat[sample_inds, 0] = action_inds)
        # 往稀疏矩阵 sample_data 中的第一列（没有第零列？）添加 atom_map1 数据
        sample_data[0].append(sample['atom_map1'])
        sample_data[1].append(ind)
        sample_data[2].append(1)

        # 往稀疏矩阵 sample_data 中的第二列（没有第零列？）添加 atom_map2 数据
        sample_data[0].append(sample['atom_map2'])
        sample_data[1].append(ind)
        sample_data[2].append(2)

        # 往稀疏矩阵 sample_data 中的第三列（没有第零列？）添加 n_nodes 数据
        nodes = sample['nodes']
        sample_data[0].append(len(nodes))
        sample_data[1].append(ind)
        sample_data[2].append(3)

        # 往稀疏矩阵 sample_data 中的第四列（没有第零列？）添加 is_hard 数据
        sample_data[0].append(int(sample['is_atom_action']))
        sample_data[1].append(ind)
        sample_data[2].append(4)

        # # 如果存在 reaction type，则往稀疏矩阵 sample_data 中的第五列（没有第零列？）添加 reaction type 数据
        # # if rt_given:  # 原本就是默认加入 reaction type 的，只有在训练的时候才选择是否使用
        # reaction_type = data_x['reaction_type'][reaction_i]
        # sample_data[0].append(reaction_type)
        # sample_data[1].append(ind)
        # sample_data[2].append(5)

        for j, node in enumerate(nodes):
            nodes_vals[0].append(node)
            nodes_vals[1].append(ind)
            nodes_vals[2].append(j)
        
        for val, row, col in zip(sample['adj'][0], sample['adj'][1], sample['adj'][2]):
            adj_vals[0].append(val)
            adj_vals[1].append(ind)
            adj_vals[2].append(row * max_n_nodes + col)
            
            
    return nodes_vals, nodes_motif, nodes_atom2motif, adj_vals, sample_data, action_tuples
    

def featurize_parallel(params) -> int:
    thread_num, samples_len, data_inds, data_x, max_n_nodes, feat_loop, \
    n_max_steps, is_train, rt_given, use_motif_action, use_motif_feature, vocab_path, forward, action_order, \
    keep_actions_list, action_vocab, chunk_save_path = params

    # assert rt_given  # 默认在预处理的时候包括 reaction type，训练的时候才选择需不需要加入这个特征

    n_reactions = len(data_x['substrates'])
    k = 1000  # to save RAM used by python lists, create sparse matrices every k reactions

    


    # 'sample_data' is a sparse matrix with 4 columns: ('action_ind', 'atom_map1', 'atom_map2', 'n_nodes')
    

    metadata = {
        'reaction_ind': [],
        'is_train': [],
        'n_samples': [],
        'start_ind': [],
        'final_smi': [],
        'source_smi': [],
        'target_smi': [],
    }
    if 'class' in data_x:
        metadata['class'] = []

    if forward:
        target_x = data_x['product']
        source_x = data_x['substrates']
    else:
        target_x = data_x['substrates']
        source_x = data_x['product']

    n_unparsed = 0
    
    nodes_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes))
    adj_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes ** 2))
    sample_data = [], [], []
    action_tuples = []

    # 遍历所有的 training samples
    for reaction_i, (reaction_ind, train, target_smi, source_smi) in feat_loop(enumerate(zip(data_inds, is_train, target_x, source_x)),desc='Thread {}: converting reactions to training samples...'.format(thread_num),total=n_reactions):
            
        
        training_samples, final_smi = gen_training_samples_in_loop(source_smi, target_smi, n_max_steps, use_motif_action, use_motif_feature, forward, action_order, vocab_path, action_vocab)
        if training_samples is None:
            n_unparsed += 1
            continue
        
        #######################
        start_ind = reaction_ind * n_max_steps

        metadata['reaction_ind'].append(reaction_ind)           # reaction_ind 是在整个数据集中的 index
        metadata['is_train'].append(train)                      # training samples or not
        metadata['final_smi'].append(str(final_smi))            # final smiles
        metadata['start_ind'].append(start_ind)                 # start_index
        metadata['n_samples'].append(len(training_samples))     # actions 个数
        metadata['source_smi'].append(source_smi)
        metadata['target_smi'].append(target_smi)

        if 'class' in data_x:
            metadata['class'].append(data_x['class'][reaction_i])   # reaction class if exists
            
        nodes_vals, nodes_motif, nodes_atom2motif, adj_vals, sample_data2, action_tuple2 = get_sparse_data(training_samples, start_ind, data_x, reaction_i, max_n_nodes, samples_len, use_motif_action, use_motif_feature)
        
        action_tuples.extend(action_tuple2)
        sample_data[0].extend(sample_data2[0])
        sample_data[1].extend(sample_data2[1])
        sample_data[2].extend(sample_data2[2])
        
        # 存入稀疏矩阵
        # (node_ind (global), local_node_ind), 
        nodes_mat += sparse.csr_matrix((nodes_vals[0], (nodes_vals[1], nodes_vals[2])),
                                    shape=(samples_len, max_n_nodes))

        adj_mat += sparse.csr_matrix((adj_vals[0], (adj_vals[1], adj_vals[2])),
                                    shape=(samples_len, max_n_nodes ** 2))

        

    sample_col2name = {1: 'atom_map1', 2: 'atom_map2', 3: 'n_nodes', 4: 'is_hard', 5: 'reaction_type'}

    # sample_data 的列数。每一列是不同的数据信息 
    n_sample_data = len(sample_col2name) + 1 # if rt_given else 6  # TODO: 还没考虑不使用 hier-motif 但 given reaction type 的情况
    sample_data = sparse.csr_matrix((sample_data[0], (sample_data[1], sample_data[2])),
                                    shape=(samples_len, n_sample_data))

    
    if not os.path.exists(chunk_save_path):
        os.makedirs(chunk_save_path)

    actions_save_path = os.path.join(chunk_save_path, 'actions.txt')
    with open(actions_save_path, 'w') as fp:
        for action in action_tuples:
            fp.write(str(action))
            fp.write('\n')

    meta_save_path = os.path.join(chunk_save_path, 'metadata.csv')
    pd.DataFrame(metadata).to_csv(meta_save_path)

    sample_data_path = os.path.join(chunk_save_path, 'sample_data.npz')
    sparse.save_npz(sample_data_path, sample_data)

    nodes_mat_path = os.path.join(chunk_save_path, 'nodes_mat.npz')
    sparse.save_npz(nodes_mat_path, nodes_mat)

    adj_mat_path = os.path.join(chunk_save_path, 'adj_mat.npz')
    sparse.save_npz(adj_mat_path, adj_mat)

    return n_unparsed
