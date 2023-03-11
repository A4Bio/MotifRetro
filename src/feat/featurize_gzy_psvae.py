"""
Main functions for featurization for MEGAN reaction generator
"""
import itertools
import logging
import os
import random
from typing import List, Tuple
from rdkit.Chem import Draw
import copy
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol, Mol
from scipy import sparse
import numpy as np
from src.feat.mol_graph import get_graph
from src.feat.reaction_actions import ReactionAction, AddRingAction, AddAtomAction, BondEditAction, AtomEditAction, AddMotifAction, DelMotifAction, StopAction, ReplaceMotifAction, AddMotifAction_with_dummySMI, AttachAtomAction
from src.feat.ring_actions import find_added_benzene_rings
from src.feat.utils import atom_to_edit_tuple, get_bond_tuple
from src.utils.retro_utils import preprocess_mols
from src.feat.utils import renumber_atoms_for_mapping, mark_reactants
from src.feat.graph_features import get_atom_features
from src.utils.chem_utils import MultiElement, get_submol, get_ams, am2inchi_order, get_am2idx_smiles_order, revise_mol_with_refsmi
from src.feat.utils import atom_to_edit_tuple, get_bond_tuple, fix_incomplete_mappings, reac_to_canonical, \
    fix_explicit_hs
import json


logger = logging.getLogger(__name__)





def smi2mol(smiles: str, kekulize=False, sanitize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol, rm_am=False):
    mol = copy.deepcopy(mol)
    if rm_am:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
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

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


get_am2idx = lambda mol: {atom.GetAtomMapNum(): i for i, atom in enumerate(mol.GetAtoms())}

def remove_am(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def match_synthons_reactants(synthons,reactants):
    assert len(synthons) == len(reactants)
    L = len(synthons)
    match = np.zeros((L,L))
    new_reactants = []
    for i in range(L):
        synthon_ams = get_ams(synthons[i]) 
        for j in range(L):
            reactant_ams = get_ams(reactants[j]) 
            if len(set(synthon_ams)&set(reactant_ams))>0:
                match[i,j] = 1
                new_reactants.append(reactants[j])
    
    assert (match.sum(axis=0) == 1).all()
    assert (match.sum(axis=1) == 1).all()
    return new_reactants

def get_neighbors(mol, ams):
    nei_ams = []
    am2idx = get_am2idx(mol)
    for am in ams:
        idx = am2idx[am]
        atom = mol.GetAtomWithIdx(idx)
        for nei in atom.GetNeighbors():
            nei_ams.append(nei.GetAtomMapNum())
    return nei_ams

def get_submol_with_ams(mol, ams):
    am2idx = get_am2idx(mol)
    idxs = []
    for am in ams:
        idxs.append(am2idx[am])

    sub_mol = get_submol(mol, idxs)
    return sub_mol


def get_frag_mol(synthon, reactant):
    frag = list(set(get_ams(reactant)) - set(get_ams(synthon)))
    nei_ams = get_neighbors(reactant, frag)
    attach = list(set(nei_ams)&set(get_ams(synthon)))
    if len(attach)>1: # 删除dummy原子之间的bond, 避免由于dummy相连而使得同一个frag里面出现多个dummy
        reactant = Chem.RWMol(reactant)
        del_bond_idx = []
        for bond in reactant.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if (a1.GetAtomMapNum() in attach) and (a2.GetAtomMapNum() in attach):
                del_bond_idx.append((a1.GetIdx(), a2.GetIdx()))
        
        for a1_idx, a2_idx in del_bond_idx:
            reactant.RemoveBond(a1_idx, a2_idx)
        reactant = reactant.GetMol()
        
    
    frag_mol = get_submol_with_ams(reactant, frag+attach)
    
    # 构造dummy atom
    frag_mol = Chem.RWMol(frag_mol)
    for atom in frag_mol.GetAtoms():
        if atom.GetAtomMapNum() in attach:
            atom.SetAtomicNum(0)
            atom.SetFormalCharge(0)
    return frag_mol


            
        
from src.feat.motif_tree import MotifTree

class ReactionSampleGenerator(object):
    def __init__(self, source_mol: RWMol, target_mol: Mol, feat_vocab: dict,  use_motif_action: bool = False,  vocab_path: str = None, only_get_synthon=False):
        self.source_mol = source_mol
        self.target_mol = target_mol
        self.vocab_path = vocab_path
        
        self.have_del_motif = False
        self.computed_motifs = False
        self.only_get_synthon = only_get_synthon


        self.use_motif_action = use_motif_action
        self.target_motifs = {}

        if self.use_motif_action:
            self.target_mol_am2idx = dict()
            for atom in self.target_mol.GetAtoms():
                self.target_mol_am2idx[atom.GetAtomMapNum()] = atom.GetIdx()


        mark_reactants(source_mol, target_mol)

        self.edited_atoms = set()
        self.feat_vocab = feat_vocab
        self.prop_dict = feat_vocab['prop2oh']
        
        # ------------- motiftree
        if vocab_path is not None:
            self.motif_generator = json.load(open(vocab_path,"r"))
        else:
            self.motif_generator = None
        # -------------------------
        
        # # ------------------和reproduce对齐
        # self.motif_generator = json.load(open("/gaozhangyang/experiments/MotifRetro/data/uspto_50k/decompose_edit_path_now.json","r"))
        # # -----------------------------

        self.current_step = 0
        self.cache_action = []
        
        self.max_node = max([atom.GetAtomMapNum() for atom in self.source_mol.GetAtoms()])+1
        
        # 1: edit atom/bond 2: add motif 3: add bond
        self.edit_pahse = 1
    
    
    def matchFrag_getMotifTree(self, raw_smi, frag_mol):
        try:
            record = copy.deepcopy(self.motif_generator[raw_smi])
            frag_mol2 = smi2mol(record['graph']['am_smi'])
            
            am_smi1, global_ams2idx = get_am2idx_smiles_order(frag_mol)
            am_smi2, local_ams2idx = get_am2idx_smiles_order(frag_mol2)
        
            global_idx2am = dict(zip(global_ams2idx.values(), global_ams2idx.keys()))
            am_mapping = {}
            for key in local_ams2idx.keys():
                am_mapping[key] = global_idx2am[local_ams2idx[key]]

            MT = MotifTree(record, am_mapping)
            assert MT.tree.graph['am_smi'] == mol2smi(smi2mol(am_smi1))
        except:
            record = copy.deepcopy(self.motif_generator[raw_smi])
            frag_mol2 = smi2mol(record['graph']['am_smi'], sanitize=False)
            
            idx_Frag2ToFrag = frag_mol.GetSubstructMatch(frag_mol2)
            global_ams2idx = get_am2idx(frag_mol)
            local_ams2idx = get_am2idx(frag_mol2)
            
            global_idx2am = dict(zip(global_ams2idx.values(), global_ams2idx.keys()))
            am_mapping = {}
            for key in local_ams2idx.keys():
                am_mapping[key] = global_idx2am[idx_Frag2ToFrag[local_ams2idx[key]]]

            MT = MotifTree(record, am_mapping)
            assert MT.tree.graph['am_smi'] == mol2smi(frag_mol)
        return MT

    
    def updateTargetMotif(self, source_mol, target_mol): # 在delete bond执行完之后运行
        synthons = MultiElement(source_mol).mols
        reactants = MultiElement(target_mol).mols
        
        
        
        reactants = match_synthons_reactants(synthons, reactants)
        frag_mol_list = []
        for k in range(len(reactants)):
            frag_mol = get_frag_mol(synthons[k], reactants[k])
            
            frag_mol_list.extend(MultiElement(frag_mol).mols) # 如果有多个不连通的frag在一起, 也就是有多个dummy atom, 则将他们拆开
        
        # 查询数据库得到fragment路径
        cache_action = []
        
        if self.only_get_synthon:
            return cache_action
        
        # -----------------------motiftree--------------------
        for frag_mol in frag_mol_list:
            frag_mol = smi2mol(mol2smi(frag_mol), sanitize=False)
            # frag_mol = revise_mol_with_refsmi(frag_mol, )
            raw_smi = mol2smi(frag_mol, rm_am=True)
            
            if raw_smi == "":
                continue
            
            
            
            MT = self.matchFrag_getMotifTree(raw_smi, frag_mol)
                
            # 根据decomposed_frags重新构造fragment, 然后执行图匹配, 以得到atom mapping映射
            for idx in MT.traversal():
                action = AddMotifAction_with_dummySMI(MT.tree.nodes[idx]['am_smi'])
                cache_action.append(action)
        #-----------------------------------------------------
            
            
            
            
        # # ------------------和reproduce对齐
        # for frag_mol in frag_mol_list:
        #     try:
        #         raw_smi = mol2smi(frag_mol)  # , rm_am=True)
        #         if raw_smi == "":
        #             continue
        #         record = self.motif_generator[raw_smi]
                
                    
                
        #         # 根据decomposed_frags重新构造fragment, 然后执行图匹配, 以得到atom mapping映射
        #         frag_mol2 = None
        #         for time_id, steps in record['path'].items():
        #             for one in steps:
        #                 step_val = list(one.values())[0]
        #                 if frag_mol2 is None:
        #                     frag_mol2 = Chem.RWMol(Chem.MolFromSmiles(step_val['am_smi']))
        #                 else:
        #                     action = AddMotifAction_with_dummySMI(step_val['am_smi'])
        #                     frag_mol2 = action.apply(frag_mol2)
                
        #         # from .chem_utils import RxnElement
        #         # from networkx.algorithms import isomorphism
        #         # frag_mol2_nx = RxnElement(frag_mol2).G_undir
        #         # frag_mol_nx = RxnElement(frag_mol).G_undir
        #         # GM = isomorphism.GraphMatcher(frag_mol_nx, frag_mol2_nx)
        #         am2toam = {}
        #         frag2Tofrag = frag_mol.GetSubstructMatch(frag_mol2)
        #         assert len(frag2Tofrag) == len(frag_mol.GetAtoms())
        #     except:
        #         # frag_mol = stand_mol(frag_mol)
                
        #         raw_smi = mol2smi(frag_mol, rm_am=True)
        #         if raw_smi == "":
        #             continue
        #         record = self.motif_generator[raw_smi]
                
                    
                
        #         # 根据decomposed_frags重新构造fragment, 然后执行图匹配, 以得到atom mapping映射
        #         frag_mol2 = None
        #         for time_id, steps in record['path'].items():
        #             for one in steps:
        #                 step_val = list(one.values())[0]
        #                 if frag_mol2 is None:
        #                     frag_mol2 = Chem.RWMol(Chem.MolFromSmiles(step_val['am_smi']))
        #                 else:
        #                     action = AddMotifAction_with_dummySMI(step_val['am_smi'])
        #                     frag_mol2 = action.apply(frag_mol2)
                
                
        #         am2toam = {}
        #         frag2Tofrag = frag_mol.GetSubstructMatch(frag_mol2)
        #         assert len(frag2Tofrag) == len(frag_mol.GetAtoms())
                
        #     for idx, atom in enumerate(frag_mol2.GetAtoms()):
        #         raw_idx = frag2Tofrag[idx]
        #         raw_am = frag_mol.GetAtomWithIdx(raw_idx).GetAtomMapNum()
        #         am2toam[atom.GetAtomMapNum()] = raw_am
                
            
        #     # 缓存megan编辑路径
        #     for time_id, steps in record['path'].items():
        #         for one in steps:
        #             step_val = list(one.values())[0]
        #             motif_mol = Chem.MolFromSmiles(step_val['am_smi'])
        #             new_atoms_map_nums = []
        #             for idx, atom in enumerate(motif_mol.GetAtoms()):
        #                 if atom.GetAtomicNum() == 0:
        #                     atom_map1 = atom.GetAtomMapNum()
        #                 else:
        #                     new_atoms_map_nums.append(am2toam[atom.GetAtomMapNum()])
        #                     atom.SetAtomMapNum(am2toam[atom.GetAtomMapNum()])
                    
        #             for atom in motif_mol.GetAtoms():
        #                 if atom.GetSymbol()=="*":
        #                     atom.SetAtomMapNum(am2toam[step_val['attach']])
        #             action = AddMotifAction_with_dummySMI(
        #                 motif_smi=Chem.MolToSmiles(motif_mol)
        #             )
        #             cache_action.append(action)
        # # -----------------------------
                    
        return cache_action
                

    def getCommonAtoms(self, target_motif):
        common = set(self.source_am2idx.keys()) & set(target_motif)
        return common
    
    def isAddMotif(self, common, target_motif, motif_info):
        neighbors_am = self.getSourceNeighbors(target_motif)
        flag = False
        # 如果包含不完整的环, 则摒弃这个motif
        for ringInfo in self.ring_motifs.values():
            ring = ringInfo['cluster_atom_mapping']
            if len(set(ring)&set(target_motif))>0 and len((set(ring)-set(target_motif)))>0:
                return False
            
        if len(common)==0 and len(neighbors_am)==1:
            flag = True
        

                
        return flag
    
    def getAddMotifAction(self, target_motif, motif_info, check=False):
        neighbors_am = self.getSourceNeighbors(target_motif)
        assert len(neighbors_am) == 1

        source_a = neighbors_am[0][0]
        new_a = neighbors_am[0][1]
        if new_a in self.source_am2idx.keys(): # 先删除残缺motif,再增加完整motif的情况
            bond = self.source_mol.GetBondBetweenAtoms(self.source_am2idx[source_a], self.source_am2idx[new_a])
        else: # 添加一个全新motif的情况, motif和source没有交集
            bond = self.target_mol.GetBondBetweenAtoms(self.target_mol_am2idx[source_a], self.target_mol_am2idx[new_a])
        bond_tuple = get_bond_tuple(bond)
        a1, a2, bond_type = bond_tuple[0], bond_tuple[1], bond_tuple[2:]
        
        action = AddMotifAction(
                        atom_map1=source_a,
                        atom_map2=new_a,
                        bond_type=bond_type[0],
                        bond_stereo = bond_type[1],
                        new_atoms_map_nums=target_motif,  # atom map number
                        motif_info=motif_info,
                        feat_vocab=self.feat_vocab,
                        is_hard=source_a not in self.edited_atoms)
        if check:
            return action
        self.action_key = source_a
        self.added_motifs[self.action_key] = action
        return action

    def isReplaceMotif(self, common, target_motif, motif_info):
        flag = False
        if all(am in target_motif for am in common) and len(common)==2: # 替换bond
            flag = True
        if all(am in target_motif for am in common) and len(common)==1: # 替换atom
            flag = True

        return flag
    
    def getReplaceMotifAction(self, atom1, atom2, motif_info, check=False):
        ring_atom_maps = motif_info['cluster_atom_mapping']
        action = ReplaceMotifAction(atom1, 
                           atom2, 
                           ring_atom_maps, 
                           motif_info, 
                           self.feat_vocab)
        if check:
            return action
        self.action_key = (atom1, atom2)
        self.replace_motifs[self.action_key] = action
        return action
    
    def isDelMotif(self, common, target_motif, motif_info):
        flag = False
        attach = []
        for am1 in common:
            atom1 = self.source_mol.GetAtomWithIdx(self.source_am2idx[am1])
            for atom2 in atom1.GetNeighbors():
                am2 = atom2.GetAtomMapNum()
                attach.append(am2)
        attach = list(set(attach) - set(common))
        

        
        if (0<len(common)) and (len(common)<len(target_motif)/2):
            if not self.have_del_motif:#不要反复删除motif,删除之后必须新增，然后才能再次删除
                if len(attach) == 1: #只考虑一个附着点的情况
                    flag = True
                    


        return flag
        
    
    def getDelMotifAction(self, common, check=False):
        attach = []
        for am1 in common:
            atom1 = self.source_mol.GetAtomWithIdx(self.source_am2idx[am1])
            for atom2 in atom1.GetNeighbors():
                am2 = atom2.GetAtomMapNum()
                attach.append(am2)
        
        attach = list(set(attach) - set(common))
        source_a = attach[0]
        del_a = []
        atom2 = self.source_mol.GetAtomWithIdx(self.source_am2idx[source_a]) 
        for atom1 in atom2.GetNeighbors():
            am1 = atom1.GetAtomMapNum()
            if am1 in common:
                del_a.append(am1)
        
        assert len(del_a) == 1
        del_a = del_a[0]
        
        action = DelMotifAction(source_a, del_a, atom_maps=common,feat_vocab=self.feat_vocab)
        if check:
            return action
        self.action_key = (action.atom_map1, action.atom_map2)
        self.deleted_motifs[self.action_key] = action
        return action

    def getSourceNeighbors(self, target_motif):
        '''
        得到新增motif的原子new_am与source mol原子之间的连接, 以(source_atom, target_atom)列表存储
        '''
        
        source_ams = [one.GetAtomMapNum() for one in self.source_mol.GetAtoms()]
        
        common = set(target_motif)&set(source_ams)
        if len(common)==0: # 如果motif和source没有交集,根据target_mol寻找邻接bond
            neighbors_am = []
            target_am2i = {one.GetAtomMapNum():i for i, one in enumerate(self.target_mol.GetAtoms())}
            for new_am in target_motif:
                target_atom = self.target_mol.GetAtomWithIdx(target_am2i[new_am])
                for nn_atom in target_atom.GetNeighbors():
                    source_am = nn_atom.GetAtomMapNum()
                    if (source_am not in target_motif) and (source_am in source_ams):
                        neighbors_am.append((source_am, new_am))
        else:# 如果motif和source有交集,根据交集部分寻找邻接bond
            neighbors_am = []
            source_am2i = {one.GetAtomMapNum():i for i, one in enumerate(self.source_mol.GetAtoms())}
            for new_am in list(common):
                common_atom = self.source_mol.GetAtomWithIdx(source_am2i[new_am])
                for nn_atom in common_atom.GetNeighbors():
                    source_am = nn_atom.GetAtomMapNum()
                    if (source_am not in common):
                        neighbors_am.append((source_am, new_am))
        
        neighbors_am = list(set(neighbors_am))
        return neighbors_am
    
    def isEditAtom(self, am):
        flag = False
        if am in self.source_atoms:
            if self.source_atoms[am] != self.target_atoms[am]:
                flag = True
                

        return flag
    
    def getEditAtomAction(self, am, check=False):
        at = self.target_atoms[am]
        action = AtomEditAction(am, *at, feat_vocab=self.feat_vocab, is_hard=am not in self.edited_atoms)
        if check:
            return action
        self.action_key = am
        self.changed_atoms[self.action_key] = action
        return action
    
    def isAddAtom(self, source_a, new_a):
        flag = False
        if (new_a not in self.source_atoms) and (source_a in self.source_atoms):
            flag = True

        return flag
    
    def getAddAtomAction(self, source_a, new_a, check=False):
        bond = self.target_mol.GetBondBetweenAtoms(self.target_mol_am2idx[source_a], self.target_mol_am2idx[new_a])
        
        bond_tuple = get_bond_tuple(bond)
        a1, a2, bond_type = bond_tuple[0], bond_tuple[1], bond_tuple[2:]
        atomic_num = self.target_atomic_nums[new_a]
        
        action = AddAtomAction(source_a, new_a,
                    *bond_type, atomic_num, *self.target_atoms[new_a],
                    feat_vocab=self.feat_vocab,
                    is_hard=source_a not in self.edited_atoms)
        if check:
            return action
        self.action_key = source_a
        self.added_atoms[self.action_key] = action
        return action
    
    def isEditBond(self, atom1, atom2):
        flag = False
        if atom1>atom2:
            atom1, atom2 = atom2, atom1
        
        if (atom1, atom2) in self.target_bonds:
            if self.source_bonds[(atom1, atom2)] != self.target_bonds[(atom1, atom2)]:
                flag = True
        

        return flag
    
    def getEditBondAction(self, atom1, atom2, check=False):
        if atom1>atom2:
            atom1, atom2 = atom2, atom1
            
        action = BondEditAction(atom1, atom2, 
                       *self.target_bonds[(atom1, atom2)],
                        feat_vocab=self.feat_vocab,
                        is_hard=atom1 not in self.edited_atoms and atom2 not in self.edited_atoms)
        if check:
            return action
        self.action_key = (atom1, atom2)
        self.changed_bonds[self.action_key] = action
        return action
            
    def isDelBond(self, atom1, atom2):
        flag = False
        if atom1>atom2:
            atom1, atom2 = atom2, atom1
        
        if (atom1, atom2) not in self.target_bonds:
            flag = True
            

        return flag
    
    def getDelBondAction(self, atom1, atom2, check=False):
        if atom1>atom2:
            atom1, atom2 = atom2, atom1
            
        action = BondEditAction(atom1, atom2, None, None,
                       feat_vocab=self.feat_vocab,
                       is_hard=atom1 not in self.edited_atoms and atom2 not in self.edited_atoms)
        if check:
            return action
        self.action_key = (atom1, atom2)
        self.deleted_bonds[self.action_key] = action
        return action
        
    
    def isAddBond(self, atom1, atom2):
        flag = False
        if atom1>atom2:
            atom1, atom2 = atom2, atom1
        
        # 化学键至少有一个原子在source中
        if atom1 in self.source_atoms or atom2 in self.source_atoms: 
            if (atom1, atom2) not in self.source_bonds:
                flag = True
        
        return flag
    
    def getAddBondAction(self, atom1, atom2, check=False):
        if atom1>atom2:
            atom1, atom2 = atom2, atom1
        action = BondEditAction(atom1, atom2, 
                                *self.target_bonds[(atom1, atom2)], 
                                feat_vocab=self.feat_vocab,
                                is_hard=atom1 not in self.edited_atoms and atom2 not in self.edited_atoms)
        if check:
            return action
        self.action_key = (atom1, atom2)
        self.added_bonds[self.action_key] = action
        return action
    
    
    def isAttachAtomAction(self, rd_atom1, rd_atom2):
        if rd_atom1.GetSymbol()=="*":
            return True
        
        if rd_atom2.GetSymbol()=="*":
            return True

        return False
    
    def getAttachAtomAction(self, rd_atom1, rd_atom2):
        if rd_atom1.GetSymbol()=="*":
            attach_atom_am = rd_atom1.GetAtomMapNum()
            base_atom_am = rd_atom2.GetAtomMapNum()
        else:
            attach_atom_am = rd_atom2.GetAtomMapNum()
            base_atom_am = rd_atom1.GetAtomMapNum()
        
        atom1, atom2 = base_atom_am, attach_atom_am
        if atom1>atom2:
            atom1, atom2 = atom2, atom1
        
        self.action_key = (atom1, atom2)
        self.attach_atoms[self.action_key] = AttachAtomAction(base_atom_am, attach_atom_am)
        
    
    def updateAtomBond(self):
        source_atoms = {}
        target_atoms = {}
        source_bonds = {}
        target_bonds = {}
        target_atomic_nums = {}
        source_am2idx = {}
        source_idx2am = {}
        max_node2 = max([atom.GetAtomMapNum() for atom in self.source_mol.GetAtoms()])+1
        if self.max_node<max_node2:
            self.max_node = max_node2
        
        # self.source_mol = Chem.rdchem.RWMol(self.custom_preprocess(self.source_mol))
        # self.target_mol = self.custom_preprocess(self.target_mol)
        
        for atom in self.source_mol.GetAtoms():
            source_idx2am[atom.GetIdx()] = atom.GetAtomMapNum()
            source_am2idx[atom.GetAtomMapNum()] = atom.GetIdx()
        
        # 记录 source molecule 的所有原子的 property
        for i, a in enumerate(self.source_mol.GetAtoms()):  # 遍历所有 source 的 atom
            source_atoms[a.GetAtomMapNum()] = atom_to_edit_tuple(a)  
            
        # 遍历 target 的所有 atoms
        for a in self.target_mol.GetAtoms():  
            target_atoms[a.GetAtomMapNum()] = atom_to_edit_tuple(a) 
            target_atomic_nums[a.GetAtomMapNum()] = a.GetAtomicNum()
            
            
        for bond in self.source_mol.GetBonds():  # 得到 source molecule 的所有 bond 的 property （包括两边原子编号，BondType，Stereo）
            bond_tuple = get_bond_tuple(bond)
            source_bonds[(bond_tuple[0], bond_tuple[1])] = bond_tuple[2:]
            
        for bond in self.target_mol.GetBonds():  # 得到 target molecule 的所有 bond 的 property （包括两边原子编号，BondType，Stereo）
            bond_tuple = get_bond_tuple(bond)
            a1, a2, bond_type = bond_tuple[0], bond_tuple[1], bond_tuple[2:]
            target_bonds[(a1, a2)] = bond_type
            
        return source_atoms, target_atoms, source_bonds, target_bonds, target_atomic_nums, source_am2idx, source_idx2am
    
    def find_actions(self):
        # Step1.2: Find AddMotif actions
        for target_motif, motif_info in self.target_motifs.items():
            common = self.getCommonAtoms(target_motif)
            if self.isAddMotif(common, target_motif, motif_info):
                action = self.getAddMotifAction(target_motif, motif_info)
            
        # Step2.1: Find AddAtom, AddBond actions from the target molecule
        for bond in self.target_mol.GetBonds():
            rd_atom1 = bond.GetBeginAtom()
            rd_atom2 = bond.GetEndAtom()
            atom1 = rd_atom1.GetAtomMapNum()
            atom2 = rd_atom2.GetAtomMapNum()
            if self.isAddBond(atom1, atom2):
                if self.isAddAtom(atom1, atom2):
                    action = self.getAddAtomAction(atom1, atom2)
                elif self.isAddAtom(atom2, atom1):
                    action = self.getAddAtomAction(atom2, atom1)
                else:
                    action = self.getAddBondAction(atom1, atom2)
        
        # Step2.2: Find getAttachAtomAction action
        for bond in self.source_mol.GetBonds():
            rd_atom1 = bond.GetBeginAtom()
            rd_atom2 = bond.GetEndAtom()
            if self.isAttachAtomAction(rd_atom1, rd_atom2):
                action = self.getAttachAtomAction(rd_atom1, rd_atom2)
       
                
        # Step3: Find EditAtom actions from the source molecule
        for atom in self.source_mol.GetAtoms():
            am = atom.GetAtomMapNum()
            if self.isEditAtom(am):
                action = self.getEditAtomAction(am)
        
        # Step4: Find EditBond, DelBond actions from the source molecule
        for bond in self.source_mol.GetBonds():
            atom1 = bond.GetBeginAtom().GetAtomMapNum()
            atom2 = bond.GetEndAtom().GetAtomMapNum()
            if self.isEditBond(atom1, atom2):
                action = self.getEditBondAction(atom1, atom2)
            
            if self.isDelBond(atom1, atom2):
                action = self.getDelBondAction(atom1, atom2)
        
    def generate_gen_action(self) -> ReactionAction:
        self.changed_atoms = {}
        self.changed_bonds = {}
        self.added_bonds = {}
        self.deleted_bonds = {}
        self.added_motifs = {}
        self.added_atoms = {}
        self.attach_atoms = {}
        
        self.source_atoms, self.target_atoms, self.source_bonds, self.target_bonds, self.target_atomic_nums, self.source_am2idx, self.source_idx2am = self.updateAtomBond()
        
        self.find_actions()
        #----------------------------action phase-------------------------
        if (self.edit_pahse==1) and (len(self.deleted_bonds) == 0) and (len(self.changed_atoms)==0) and (len(self.changed_bonds)==0):
            self.edit_pahse = 2

            self.cache_action = self.updateTargetMotif(copy.deepcopy(self.source_mol), copy.deepcopy(self.target_mol))

        
        if (self.edit_pahse==2) and (len(self.cache_action)==0):
            self.edit_pahse = 3
            self.source_mol.UpdatePropertyCache(strict=False)
            self.find_actions()
    
        if (self.edit_pahse==3) and (len(self.attach_atoms)==0):
            self.edit_pahse = 4
        #-----------------------------------------------------------------
        
        
        
        #---------------------------return action-------------------------
        if self.edit_pahse==1:
            action_type_priorities = list(self.deleted_bonds.values())+list(self.changed_bonds.values())+list(self.changed_atoms.values())
            
            for action in action_type_priorities:
                return action
                

        
        if self.edit_pahse==2:
            return self.cache_action.pop(0)
        
        if self.edit_pahse==3:
            for key, action in self.attach_atoms.items():
                return action
        
        if self.edit_pahse==4:
            return StopAction(feat_vocab=self.feat_vocab)
        


    def gen_training_sample(self) -> dict:
        self.source_mol.UpdatePropertyCache(strict=False)

        # generate action
        reaction_action = self.generate_gen_action()
        # print(reaction_action)

        

        adj, nodes = get_graph(copy.deepcopy(self.source_mol), 
                               atom_prop2oh=self.prop_dict['atom'],
                               bond_prop2oh=self.prop_dict['bond'], 
                               max_node = self.max_node, 
                               atom_feature_keys = list(self.prop_dict["atom"].keys()), bond_feature_keys = list(self.prop_dict["bond"].keys()))


        training_sample = {
            'action_tuple': reaction_action.get_tuple(),
            'action_str': str(reaction_action)
        }
        training_sample['adj'] = adj
        training_sample['nodes'] = nodes
        training_sample['step'] = self.current_step
        training_sample['atom_map1'] = reaction_action.atom_map1
        training_sample['atom_map2'] = reaction_action.atom_map2
        

        atom_map1 = reaction_action.atom_map1
        atom_map2 = reaction_action.atom_map2
        if atom_map1 > 0:
            self.edited_atoms.add(atom_map1)
        if atom_map2 > 0:
            self.edited_atoms.add(atom_map2)


        # execute action
        self.source_mol = reaction_action.apply(self.source_mol)

        self.current_step += 1
        return training_sample


def gen_training_samples(source_mol: Mol, target_mol: Mol,  n_max_steps: int, feat_vocab: dict, use_motif_action: bool = False, use_motif_feature=False, vocab_path: str = None):
    training_samples = []
    final_smi = ''

    reaction_state = ReactionSampleGenerator(
        source_mol=Chem.rdchem.RWMol(source_mol),  
        target_mol=target_mol,
        feat_vocab=feat_vocab,
        use_motif_action=use_motif_action,
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


def featurize_parallel(params) -> int:
    thread_num, samples_len, data_inds, data_x, max_n_nodes, feat_loop, \
    n_max_steps, is_train, rt_given, use_motif_action, use_motif_feature, vocab_path, forward, action_order, \
     feat_vocab, chunk_save_path = params

    n_reactions = len(data_x['substrates'])
    nodes_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes))
    adj_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes ** 2))
    # 最后存放进入 node_mat, adj_mat 的三组列表
    adj_vals = [], [], []  # vals, rows, cols
    nodes_vals = [], [], []  # vals, rows, cols
    action_tuples = []
    
    metadata = {
        'reaction_ind': [],
        'is_train': [],
        'n_samples': [],
        'start_ind': [],
        'final_smi': [],
        'source_smi': [],
        'target_smi': [],
        'reaction_type': []
    }

    if forward:
        target_x = data_x['product']
        source_x = data_x['substrates']
    else:
        target_x = data_x['substrates']
        source_x = data_x['product']
        
    reaction_types = data_x['reaction_type_id']

    n_unparsed = 0

    # 遍历所有的 training samples
    for reaction_i, (reaction_ind, reaction_type, train, target_smi, source_smi) in feat_loop(enumerate(zip(data_inds, reaction_types, is_train, target_x, source_x)),
        desc='Thread {}: converting reactions to training samples...'.format(thread_num), total=n_reactions):
        
        if reaction_ind==5:
            print()
        if not isinstance(target_smi, str) or len(target_smi) == 0:
            target_smi = source_smi

        try:
            target_mol = Chem.MolFromSmiles(target_smi)
            source_mol = Chem.MolFromSmiles(source_smi)
            
            assert target_mol.GetNumAtoms()<100, "target mol has more than 100 atoms!"
            assert target_mol.GetNumAtoms()<100, "source mol has more than 100 atoms!"

            if target_mol is not None and source_mol is not None:
                target_mol, source_mol = fix_incomplete_mappings(target_mol, source_mol)  # TODO: 作用是什么？A：见 Notion
                target_mol, source_mol = reac_to_canonical(target_mol, source_mol)  # TODO：作用是什么？ A: SMILES有变化。似乎是给碳原子重新编号，以消除 information leaking 的问题。
                source_mol = fix_explicit_hs(source_mol)  # rdkit has a problem with implicit hs. By default there are only explicit hs. This is a hack to fix this error
                target_mol = fix_explicit_hs(target_mol)

                source_mol = renumber_atoms_for_mapping(source_mol)  # TODO: 作用是什么？没有看到 SMILES 有变化
                target_mol = renumber_atoms_for_mapping(target_mol)
        except Exception:
            target_mol, source_mol = None, None
        
        if (target_mol is None) or (source_mol is None):
            n_unparsed += 1
            continue

        try:
            training_samples, final_smi = gen_training_samples(source_mol,target_mol, n_max_steps,
                                                                feat_vocab=feat_vocab, use_motif_action=use_motif_action, vocab_path=vocab_path)

            if len(training_samples) == 0:
                n_unparsed += 1
                continue

            start_ind = reaction_ind * n_max_steps
            
            #-------------------meta data----------------------
            metadata['reaction_ind'].append(reaction_ind)           # reaction_ind 是在整个数据集中的 index
            metadata['is_train'].append(train)                      # training samples or not
            metadata['final_smi'].append(str(final_smi))            # final smiles
            metadata['start_ind'].append(start_ind)                 # start_index
            metadata['n_samples'].append(len(training_samples))     # actions 个数
            metadata['source_smi'].append(source_smi)
            metadata['target_smi'].append(target_smi)
            metadata['reaction_type'].append(reaction_type)
            #---------------------------------------------------


            #-------------------graph input----------------------
            # for each sample，遍历所有的 actions
            for sample_ind, sample in enumerate(training_samples):
                ind = start_ind + sample_ind
                nodes = sample['nodes']
                for j, node in enumerate(nodes):
                    nodes_vals[0].append(node)
                    nodes_vals[1].append(ind)
                    nodes_vals[2].append(j)
                
                for val, row, col in zip(sample['adj'][0], sample['adj'][1], sample['adj'][2]): # 相同的edge可能会出现多次，最终合并在sparse matrix里面导致数值溢出
                    adj_vals[0].append(val)
                    adj_vals[1].append(ind)
                    adj_vals[2].append(row * max_n_nodes + col)

        
        
                #-------------------predictive label----------------------
                action_tuples.append((ind, sample['atom_map1'], sample['atom_map2'], len(sample['nodes']), sample['action_tuple']))  
        except:
            n_unparsed+=1
            pass

    print(f"thread {thread_num}, n_unparsed {n_unparsed}")
    nodes_mat += sparse.csr_matrix((nodes_vals[0], (nodes_vals[1], nodes_vals[2])),
                                    shape=(samples_len, max_n_nodes))
    adj_mat += sparse.csr_matrix((adj_vals[0], (adj_vals[1], adj_vals[2])),
                                    shape=(samples_len, max_n_nodes ** 2))
    
    

    if not os.path.exists(chunk_save_path):
        os.makedirs(chunk_save_path)
        
    
        
    meta_save_path = os.path.join(chunk_save_path, 'metadata.csv')
    pd.DataFrame(metadata).to_csv(meta_save_path)
    
    nodes_mat_path = os.path.join(chunk_save_path, 'nodes_mat.npz')
    sparse.save_npz(nodes_mat_path, nodes_mat)

    adj_mat_path = os.path.join(chunk_save_path, 'adj_mat.npz')
    sparse.save_npz(adj_mat_path, adj_mat)
        
    
    
    actions_save_path = os.path.join(chunk_save_path, 'actions.txt')
    with open(actions_save_path, 'w') as fp:
        for action in action_tuples:
            fp.write(str(action))
            fp.write('\n')

    return n_unparsed


