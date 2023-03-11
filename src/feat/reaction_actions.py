"""
Definitions of basic 'edits' (Actions) to transform a target into substrates
"""
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
from rdkit.Chem.rdchem import RWMol, Mol
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import CombineMols
from rdkit.Chem import rdchem, BondType, ChiralType, BondStereo, RWMol
from rdkit.Chem.rdchem import GetPeriodicTable
from rdkit.Chem import Draw
from src.feat.graph_features import get_atom_features
# from src.feat import ORDERED_ATOM_OH_KEYS, ORDERED_BOND_OH_KEYS, ATOM_EDIT_TUPLE_KEYS
from src.feat.ring_actions import add_benzene_ring
from src.feat.utils import get_bond_tuple, fix_explicit_hs, fix_endpoint
import copy
# from utils.megan_utils import get_atom_ind
get_am2idx = lambda mol: {atom.GetAtomMapNum(): i for i, atom in enumerate(mol.GetAtoms())}
from src.utils.chem_utils import smi2mol, mol2smi

def get_atom_ind(mol: Mol, atom_map: int) -> int:
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomMapNum() == atom_map:
            return i
    raise ValueError(f'No atom with map number: {atom_map}')

PERIODIC_TABLE = GetPeriodicTable()


class ReactionAction(metaclass=ABCMeta):
    def __init__(self, atom_map1: int, atom_map2: int, feat_vocab: dict, is_hard: bool = False):
        self.atom_map1 = atom_map1
        self.atom_map2 = atom_map2
        self.is_hard = is_hard
        self.feat_vocab = feat_vocab
        self.prop2oh = feat_vocab['prop2oh']

    @abstractmethod
    def get_tuple(self) -> Tuple[str, ...]:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def apply(self, mol: RWMol) -> RWMol:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError('Abstract method')


feat_key_to_str = {
    'atomic_num': 'Atomic Number',
    'formal_charge': 'Formal Charge',
    'is_aromatic': 'Is Aromatic',
    'chiral_tag': 'Chiral Type',
    'num_explicit_hs': 'Num. Explicit Hs',
    'bond_type': 'Bond Type',
    'bond_stereo': 'Bond Stereo',
}


def feat_val_to_str(feat_key, feat_val, with_key=True):
    if feat_key == 'atomic_num':
        str_val = PERIODIC_TABLE.GetElementSymbol(feat_val)
    elif feat_key == 'formal_charge':
        if with_key:
            str_val = str(feat_val)
            if feat_val > 0:
                str_val = '+' + str_val
        else:
            str_val = ''
            if feat_val == -1:
                str_val = '-'
            elif feat_val == 1:
                str_val = '+'
            elif feat_val > 1:
                str_val = f'{feat_val}+'
            elif feat_val < -1:
                str_val = f'{feat_val}-'
    elif feat_key == 'is_aromatic':
        str_val = 'Yes' if feat_val == 1 else 'No'
    elif feat_key == 'chiral_tag':
        if feat_val == ChiralType.CHI_UNSPECIFIED:
            str_val = 'None'
        elif feat_val == ChiralType.CHI_TETRAHEDRAL_CW:
            str_val = 'CW'
        elif feat_val == ChiralType.CHI_TETRAHEDRAL_CCW:
            str_val = 'CCW'
        elif feat_val == ChiralType.OTHER:
            str_val = 'Other'
        else:
            str_val = str(feat_val)
    elif feat_key == 'num_explicit_hs':
        if with_key:
            return str(feat_val)
        else:
            str_val = ''
            if feat_val == 1:
                str_val = f'H'
            elif feat_val > 1:
                str_val = f'{feat_val}H'
    elif feat_key == 'bond_type':
        str_val = str(BondType.values[feat_val]).capitalize()
    elif feat_key == 'bond_stereo':
        str_val = str(BondStereo.values[feat_val])[6:].capitalize()
    else:
        str_val = str(feat_val)

    if with_key:
        return f'{feat_key_to_str[feat_key]}={str_val}'
    return str_val


def atom_to_str(atomic_num, formal_charge, num_explicit_hs):
    symbol = feat_val_to_str('atomic_num', atomic_num, with_key=False)
    charge = feat_val_to_str('formal_charge', formal_charge, with_key=False)
    hs = feat_val_to_str('num_explicit_hs', num_explicit_hs, with_key=False)
    return symbol + hs + charge


class AtomEditAction(ReactionAction):
    def __init__(self, atom_map1: int, formal_charge: int, chiral_tag: int,
                 num_explicit_hs: int, is_aromatic: int, feat_vocab: dict,
                 is_hard: bool = False):
        super(AtomEditAction, self).__init__(atom_map1, -1, feat_vocab, is_hard)
        self.formal_charge = formal_charge
        self.chiral_tag = chiral_tag
        self.num_explicit_hs = num_explicit_hs
        self.is_aromatic = is_aromatic
        self.feat_keys = ['formal_charge', 'chiral_tag', 'num_explicit_hs', 'is_aromatic'] + ['is_edited']

    @property
    def feat_vals(self) -> Tuple[int, int, int, int, int]:
        return self.formal_charge, self.chiral_tag, self.num_explicit_hs, self.is_aromatic, 1

    def get_tuple(self) -> Tuple[str, Tuple[int, int, int, int]]:
        return 'change_atom', self.feat_vals[:-1]

    def apply(self, mol: RWMol) -> RWMol:
        atom_ind = get_atom_ind(mol, self.atom_map1)
        atom = mol.GetAtomWithIdx(atom_ind)

        atom.SetFormalCharge(self.formal_charge)
        a_chiral = rdchem.ChiralType.values[self.chiral_tag]
        atom.SetChiralTag(a_chiral)
        atom.SetNumExplicitHs(self.num_explicit_hs)
        atom.SetIsAromatic(self.is_aromatic)
        atom.SetBoolProp('is_edited', True)

        return mol

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nodes = nodes.copy()
        for key, val in zip(self.feat_keys, self.feat_vals):
            ind = self.feat_vocab['atom_feat_ind'].get(key, -1)
            if ind != -1:
                ind = self.feat_vocab['atom_feat_ind'][key]
                nodes[self.atom_map1, ind] = self.prop2oh['atom'][key].get(val, 0)
        return adj, nodes

    def __str__(self):
        feat_vals = ', '.join([feat_val_to_str(key, val) for val, key in
                               zip(self.feat_vals, self.feat_keys) if key != 'is_edited'])
        return f'Edit Atom {self.atom_map1}: {feat_vals}'


class AddAtomAction(ReactionAction):   # 生成 Atom 时是连 bond 一起生成的
    def __init__(self, atom_map1: int, atom_map2: int,
                 bond_type: int, bond_stereo: int,
                 atomic_num: int, formal_charge: int, chiral_tag: int,
                 num_explicit_hs: int, is_aromatic: int, feat_vocab: dict,
                 is_hard: bool = False, detach: bool = False):
        super(AddAtomAction, self).__init__(atom_map1, atom_map2, feat_vocab, is_hard)
        self.bond_type = bond_type
        self.bond_stereo = bond_stereo
        self.atomic_num = atomic_num
        self.formal_charge = formal_charge
        self.chiral_tag = chiral_tag
        self.num_explicit_hs = num_explicit_hs
        self.is_aromatic = is_aromatic
        self.detach = detach

        self.new_a = self._gen_new_atom()
        self.new_atom_features = get_atom_features(self.new_a, list(self.prop2oh['atom'].keys()), atom_prop2oh=self.prop2oh['atom'])

        # new atom has 1 neighbour when its created
        self.degree_ind = self.feat_vocab['atom_feat_ind'].get('degree', -1)
        if self.degree_ind != -1:
            self.new_atom_features[self.degree_ind] = self.prop2oh['atom']['degree'][1]

        self.mol_id_ind = self.feat_vocab['atom_feat_ind'].get('mol_id', -1)
        self.is_reactant_ind = self.feat_vocab['atom_feat_ind'].get('is_reactant', -1)

    @property
    def atom_feat_vals(self):
        return self.atomic_num, self.formal_charge, self.chiral_tag, self.num_explicit_hs, self.is_aromatic

    def get_tuple(self) -> Tuple[str, Tuple[Tuple[int, int], Tuple[int, int, int, int, int]]]:
        return 'add_atom', ((self.bond_type, self.bond_stereo), self.atom_feat_vals)

    def _gen_new_atom(self):
        new_a = Chem.Atom(self.atomic_num)
        new_a.SetFormalCharge(self.formal_charge)
        a_chiral = rdchem.ChiralType.values[self.chiral_tag]
        new_a.SetChiralTag(a_chiral)
        new_a.SetNumExplicitHs(self.num_explicit_hs)
        new_a.SetIsAromatic(self.is_aromatic)
        new_a.SetAtomMapNum(self.atom_map2)
        new_a.SetBoolProp('is_edited', True)
        return new_a

    def _get_bond_features(self):
        return [self.bond_type, self.bond_stereo, 1]

    def apply(self, mol: RWMol) -> RWMol:
        num_atoms = mol.GetNumAtoms()
        if self.detach:  # deprecated
            for i, a in enumerate(mol.GetAtoms()):
                m = a.GetAtomMapNum()
                if m == self.atom_map2:
                    for bond in a.GetBonds():
                        mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    mol.RemoveAtom(i)
                    num_atoms -= 1
                    break

        atom_ind = get_atom_ind(mol, self.atom_map1)
        b_type = rdchem.BondType.values[self.bond_type]
        b_stereo = rdchem.BondStereo.values[self.bond_stereo]

        old_atom = mol.GetAtomWithIdx(atom_ind)
        if old_atom.HasProp('in_reactant'):
            self.new_a.SetBoolProp('in_reactant', old_atom.GetBoolProp('in_reactant'))  # TODO:  What is this for? Note that in_reactant != is_reactant
        if old_atom.HasProp('mol_id'):
            self.new_a.SetIntProp('mol_id', old_atom.GetIntProp('mol_id'))  # TODO: What is this for?

        mol.AddAtom(self.new_a)
        new_atom_ind = num_atoms

        bond_ind = mol.AddBond(atom_ind, new_atom_ind, order=b_type) - 1
        new_bond = mol.GetBondWithIdx(bond_ind)
        new_bond.SetStereo(b_stereo)
        new_bond.SetBoolProp('is_edited', True)

        return mol

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        max_num = max(len(adj), self.atom_map2 + 1)
        new_adj = np.full((max_num, max_num, adj.shape[2]), fill_value=0, dtype=int)
        new_adj[:adj.shape[0], :adj.shape[1]] = adj
        new_nodes = np.full((max_num, nodes.shape[1]), fill_value=0, dtype=int)
        new_nodes[:nodes.shape[0]] = nodes
        new_nodes[self.atom_map2] = self.new_atom_features

        if self.is_aromatic:
            new_nodes[self.atom_map1, self.feat_vocab['atom_feature_keys'].index("is_aromatic")] = self.feat_vocab['prop2oh']['atom']['is_aromatic'][1]

        if self.detach:
            for i in range(1, len(new_nodes)):
                if self.degree_ind != -1 and i != self.atom_map2 and new_adj[self.atom_map2, i, 0] != 0:
                    new_nodes[i, self.degree_ind] -= 1
            new_adj[self.atom_map2, :] = 0
            new_adj[:, self.atom_map2] = 0

        # copy "mol_id" and "is_reactant" to new atom from the old neighboring atom
        if self.mol_id_ind != -1:
            new_nodes[self.atom_map2, self.mol_id_ind] = new_nodes[self.atom_map1, self.mol_id_ind]
        if self.is_reactant_ind != -1:
            new_nodes[self.atom_map2, self.is_reactant_ind] = new_nodes[self.atom_map1, self.is_reactant_ind]

        # update 'degree' feature of old atom (+= 1)
        if self.degree_ind != -1:
            new_nodes[self.atom_map1, self.degree_ind] += 1

        bond_features = [self.prop2oh['bond'][key].get(val, 0) for key, val in
                         zip(['bond_type', 'bond_stereo', 'is_edited'], self._get_bond_features())]

        new_adj[self.atom_map1, self.atom_map2] = new_adj[self.atom_map2, self.atom_map1] = bond_features

        new_adj[0, self.atom_map2, 0] = new_adj[self.atom_map2, 0, 0] = self.prop2oh['bond']['bond_type']['supernode']
        new_adj[self.atom_map2, self.atom_map2, 0] = self.prop2oh['bond']['bond_type']['self']

        return new_adj, new_nodes

    def __str__(self):
        new_atom_str = atom_to_str(self.atomic_num, self.formal_charge, self.num_explicit_hs)
        key = 'Detach' if self.detach else 'Add'
        return f'{key} {new_atom_str}:{self.atom_map2} to atom {self.atom_map1} ' \
               f'({feat_val_to_str("bond_type", self.bond_type)})'



class AddMotifAction(ReactionAction):   # TODO: Motif is a combination of a set of atoms and bonds
    def __init__(self, atom_map1: int, atom_map2: int, bond_type: int, bond_stereo:int, new_atoms_map_nums: List[int], motif_info: Dict[str, Any],
                 feat_vocab: dict, is_hard: bool = False):
        super(AddMotifAction, self).__init__(atom_map1, atom_map2, feat_vocab, is_hard)
        self.motif_info = motif_info
        self.bond_type = bond_type
        self.bond_stereo = bond_stereo
        self.prop2oh = feat_vocab['prop2oh']

        
        mol = Chem.MolFromSmiles(motif_info['smiles_with_mapping'], sanitize=False)

            
        # add dummy atom  
        mol = Chem.rdchem.RWMol(mol)
        for source_ind, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomMapNum() == atom_map2:
                new_a = Chem.Atom(0)
                new_a.SetAtomMapNum(100)
                attach_ind = source_ind
                break
            
        new_atom_ind = mol.AddAtom(new_a)
        mol.AddBond(attach_ind, new_atom_ind)
        b_type = rdchem.BondType.values[self.bond_type]
        b_stereo = rdchem.BondStereo.values[self.bond_stereo]

        bond = mol.GetBondBetweenAtoms(attach_ind, new_atom_ind)
        bond.SetBondType(b_type)
        bond.SetStereo(b_stereo)
        
        
            
        self.smiles_atom_mapping = Chem.MolToSmiles(mol)
        
        for source_ind, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(0)
                
        self.smiles = Chem.MolToSmiles(mol)


        # order new atom map nums so map num of the existing atom is first
        self.new_atoms_map_nums = new_atoms_map_nums
        self.atom_ind2 = list(self.new_atoms_map_nums).index(self.atom_map2)

        self.new_atoms = list()
        self.new_bonds = list()

        self.degree_indices = list()
        self.mol_id_indices = list()
        self.is_reactant_indices = list()
        
        self.motifmol = Chem.MolFromSmiles(self.smiles_atom_mapping, sanitize=False)
        
        try:   
            self.motifmol = fix_explicit_hs(self.motifmol)
        except:
            pass
            
        for atom in self.motifmol.GetAtoms():
            atom.SetBoolProp('is_edited', True)
        
        for bond in self.motifmol.GetBonds():
            bond.SetBoolProp('is_edited', True)
        
        self.new_atom_features = {}
        for i, new_atom in enumerate(self.motifmol.GetAtoms()):
            if new_atom.GetAtomicNum()!=0:
                atom_feat = get_atom_features(new_atom, list(self.prop2oh['atom'].keys()), atom_prop2oh=self.prop2oh['atom'])
                
                degree_ind = feat_vocab['atom_feat_ind'].get('degree', -1)
                if degree_ind != -1:
                    atom_feat[self.degree_ind] = self.prop2oh['atom']['degree'][1]
                self.new_atom_features[new_atom.GetAtomMapNum()] = atom_feat
        
        self.new_bond_features = {}
        for bond in self.motifmol.GetBonds():
            atom1 = bond.GetBeginAtom().GetAtomMapNum()
            atom2 = bond.GetEndAtom().GetAtomMapNum()
            bond_feat = get_bond_tuple(bond)
            self.new_bond_features[(atom1, atom2)] = {"bond_type": bond_feat[2], 
                                                      "bond_stereo":bond_feat[3]}
        
        

    @property
    def connected_atom_feat_vals(self):
        data = self.motif_info['atoms_info'][self.atom_ind2]
        return data['atomic_num'], data['formal_charge'], data['chiral_tag'], data['num_explicit_hs'], data['is_aromatic']

    def get_tuple(self) -> Tuple[str, Tuple[Tuple[int, int], Tuple[int, int, int, int, int], str]]:
        return 'add_motif', self.smiles

    def apply(self, mol: RWMol) -> RWMol:   # 修改分子
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == self.atom_map1:
                # atom.SetBoolProp('attach_source', True)
                attach_am = atom.GetAtomMapNum()
                atom.SetAtomMapNum(1000)
                break
        
        for bond in self.motifmol.GetBonds():
            start_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if start_atom.GetAtomicNum() == 0:
                new_bond = bond
                link_am = end_atom.GetAtomMapNum()
                end_atom.SetAtomMapNum(1001)
                break
            elif end_atom.GetAtomicNum() == 0:
                new_bond = bond
                link_am = start_atom.GetAtomMapNum()
                start_atom.SetAtomMapNum(1001)
                break
                
        
        # 合并source mol和motif mol
        mol = Chem.RWMol(CombineMols(mol, self.motifmol))
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomMapNum()==1000:
                source_attach_ind = i
                source_IsAromatic = atom.GetIsAromatic()
            if atom.GetAtomMapNum()==1001:
                motif_attach_ind = i
                motif_IsAromatic = atom.GetIsAromatic()
        
        
        # 连上化学键, 恢复atom mapping
        b_type = rdchem.BondType.values[self.bond_type]
        b_stereo = rdchem.BondStereo.values[self.bond_stereo]
        
        # 和Aromatic原子相连时，被连的原子会自动变成Aromatic类型, 需要使用source_IsAromatic和motif_IsAromatic保存他们原本的类型，并在后面的代码中恢复其本来的原子类型
        bond_ind = mol.AddBond(source_attach_ind, motif_attach_ind, b_type) - 1
        new_bond = mol.GetBondWithIdx(bond_ind)
        new_bond.SetStereo(b_stereo)
        new_bond.SetBoolProp('is_edited', True)
        
        atom1 = mol.GetAtomWithIdx(source_attach_ind)
        atom1.SetAtomMapNum(attach_am)
        atom1.SetIsAromatic(source_IsAromatic)
        atom2 = mol.GetAtomWithIdx(motif_attach_ind)
        atom2.SetAtomMapNum(link_am)
        atom2.SetIsAromatic(motif_IsAromatic) 
        
        # 删除虚拟原子
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum()==0:
                remove_atom_ind = i
                break
        mol.RemoveAtom(remove_atom_ind)

        return mol

    def __str__(self):
        cluster = self.motif_info['cluster_atom_mapping']
        return f'Add Motif {cluster} to atom {self.atom_map1}'

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  
        '''
        TODO: 需要更新
        '''
        # 修改 featurized 的数据        
        max_map_num = max(max(self.new_atoms_map_nums) + 1, adj.shape[0])

        new_adj = np.full((max_map_num, max_map_num, adj.shape[2]), fill_value=0, dtype=int)  # 新 adj 矩阵
        new_adj[:adj.shape[0], :adj.shape[1]] = adj
        new_nodes = np.full((max_map_num, nodes.shape[1]), fill_value=0, dtype=int)  # 新 nodes 列表
        new_nodes[:nodes.shape[0]] = nodes


        for i, map_num in enumerate(self.new_atoms_map_nums):
            new_nodes[map_num] = self.new_atom_features[map_num]
            new_adj[map_num, map_num, 0] = self.prop2oh['bond']['bond_type']['self']
            new_adj[0, map_num, 0] = new_adj[map_num, 0, 0] = self.prop2oh['bond']['bond_type']['supernode']

        # for bond in self.motif_info['bonds_info']:
        #     new_adj[bond['start_atom_map'], bond['end_atom_map']] = new_adj[bond['end_atom_map'], bond['start_atom_map']] = [
        #         self.prop2oh['bond']['bond_type'][bond['bond_type']], 
        #         self.prop2oh['bond']['bond_stereo'][bond['bond_stereo']], 
        #         self.prop2oh['bond']['is_edited'][1]
        #     ]
        
        for (atom1, atom2), values in self.new_bond_features.items():
            if atom1==100 or atom2 == 100:
                continue
            new_adj[atom1, atom2] = new_adj[atom2, atom1] = [
                self.prop2oh['bond']['bond_type'][values['bond_type']], 
                self.prop2oh['bond']['bond_stereo'][values['bond_stereo']], 
                self.prop2oh['bond']['is_edited'][1]
            ]

        new_adj[self.atom_map1, self.atom_map2] = new_adj[self.atom_map2, self.atom_map1] = [
            self.prop2oh['bond']['bond_type'][self.bond_type], 
            self.prop2oh['bond']['bond_stereo'][self.bond_stereo], 
            self.prop2oh['bond']['is_edited'][1]
        ]
        return new_adj, new_nodes




class DelMotifAction:   
    def __init__(self, source_a, del_a, atom_maps: List[int], feat_vocab: dict):
        super(DelMotifAction, self).__init__()
        self.source_a = source_a
        self.atom_map1 = source_a
        self.atom_map2 = del_a
        self.del_atom_maps = atom_maps
        self.prop2oh = feat_vocab['prop2oh']

    @property
    def connected_atom_feat_vals(self):
        data = self.motif_info['atoms_info'][self.atom_ind2]
        return data['atomic_num'], data['formal_charge'], data['chiral_tag'], data['num_explicit_hs'], data['is_aromatic']

    def get_tuple(self) -> Tuple[str, Tuple[Tuple[int, int], Tuple[int, int, int, int, int], str]]:
        return 'delete_motif', None

    def apply(self, mol: RWMol) -> RWMol:  
        mol = Chem.RWMol(mol)
        del_bonds_ind = []
        del_atoms_ind = []
        for atom1 in mol.GetAtoms():
            atom1_ind = atom1.GetIdx()
            if atom1.GetAtomMapNum() in self.del_atom_maps:
                for atom2 in atom1.GetNeighbors():
                    atom2_ind = atom2.GetIdx()
                    del_bonds_ind.append((atom1_ind, atom2_ind))
                del_atoms_ind.append(atom1_ind)
        
        for a1, a2 in del_bonds_ind:
            mol.RemoveBond(a1, a2)
        
        for atom_ind in sorted(del_atoms_ind, reverse=True):
            mol.RemoveAtom(atom_ind)
                    
        return mol

    def __str__(self):
        return f'Delete Motif {self.del_atom_maps}'

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  
        '''
        TODO: 需要更新
        '''
        # 修改 featurized 的数据        
        max_map_num = adj.shape[0]

        new_adj = np.full((max_map_num, max_map_num, adj.shape[2]), fill_value=0, dtype=int)  # 新 adj 矩阵
        new_adj[:adj.shape[0], :adj.shape[1]] = adj
        new_nodes = np.full((max_map_num, nodes.shape[1]), fill_value=0, dtype=int)  # 新 nodes 列表
        new_nodes[:nodes.shape[0]] = nodes

        for i, map_num in enumerate(self.del_atom_maps):
            new_nodes[map_num] = 0
            new_adj[map_num, :,:] = 0
            new_adj[:, map_num, :] = 0
        return new_adj, new_nodes
    
    
class ReplaceMotifAction:   
    def __init__(self, atom_map1: int, atom_map2: int, new_atoms_map_nums: List[int], motif_info: Dict[str, Any], feat_vocab: dict, is_hard: bool = False):
        super(ReplaceMotifAction, self).__init__()
        self.atom_map1 = atom_map1
        self.atom_map2 = atom_map2
        self.new_atoms_map_nums = new_atoms_map_nums
        self.motif_info = motif_info
        self.prop2oh = feat_vocab['prop2oh']
        
        mol = Chem.MolFromSmiles(motif_info['smiles_with_mapping'], sanitize=False)
        
        self.smiles_atom_mapping = Chem.MolToSmiles(mol)
        
        for source_ind, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomMapNum() == self.atom_map1:
                atom.SetAtomMapNum(500)
            elif atom.GetAtomMapNum() == self.atom_map2:
                atom.SetAtomMapNum(501)
            else:
                atom.SetAtomMapNum(0)
        
        self.smiles = Chem.MolToSmiles(mol)
        
        self.motifmol = Chem.MolFromSmiles(self.smiles_atom_mapping, sanitize=False)
        try:
            self.motifmol = fix_explicit_hs(self.motifmol)
        except:
            pass
        
        for atom in self.motifmol.GetAtoms():
            atom.SetBoolProp('is_edited', True)
        
        for bond in self.motifmol.GetBonds():
            bond.SetBoolProp('is_edited', True)
            
        self.new_atom_features = {}
        for i, new_atom in enumerate(self.motifmol.GetAtoms()):
            
            if new_atom.GetAtomicNum()!=0:
                atom_feat = get_atom_features(new_atom, list(self.prop2oh['atom'].keys()), atom_prop2oh=self.prop2oh['atom'])
                
                degree_ind = feat_vocab['atom_feat_ind'].get('degree', -1)
                if degree_ind != -1:
                    atom_feat[self.degree_ind] = self.prop2oh['atom']['degree'][1]
                self.new_atom_features[new_atom.GetAtomMapNum()] = atom_feat
        
        self.new_bond_features = {}
        for bond in self.motifmol.GetBonds():
            atom1 = bond.GetBeginAtom().GetAtomMapNum()
            atom2 = bond.GetEndAtom().GetAtomMapNum()
            bond_feat = get_bond_tuple(bond)
            self.new_bond_features[(atom1, atom2)] = {"bond_type": bond_feat[2], 
                                                      "bond_stereo":bond_feat[3]}
        
        self.neighbor_bonds = []
  


    @property
    def connected_atom_feat_vals(self):
        data = self.motif_info['atoms_info'][self.atom_ind2]
        return data['atomic_num'], data['formal_charge'], data['chiral_tag'], data['num_explicit_hs'], data['is_aromatic']

    def get_tuple(self) -> Tuple[str, Tuple[Tuple[int, int], Tuple[int, int, int, int, int], str]]:
        return 'replace_motif', self.smiles

    def apply(self, mol: RWMol) -> RWMol:  
        if self.atom_map2!=-1: #替换边
            mol = Chem.RWMol(mol)
            
            # 标记mol与motif的common原子
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum()==self.atom_map1:
                    atom.SetAtomMapNum(self.atom_map1+1000)
                
                if atom.GetAtomMapNum()==self.atom_map2:
                    atom.SetAtomMapNum(self.atom_map2+1000)
            
            for atom in self.motifmol.GetAtoms():
                if atom.GetAtomMapNum()==500:
                    atom.SetAtomMapNum(self.atom_map1)
                
                if atom.GetAtomMapNum()==501:
                    atom.SetAtomMapNum(self.atom_map2)
            
            
            
            
            # 将与mol上面的common原子相关的化学键连到motif上
            mol = Chem.RWMol(CombineMols(mol, self.motifmol))
            am2i = {atom.GetAtomMapNum():i for i, atom in enumerate(mol.GetAtoms())}
            
            added_bond = []
            for left_atom_am in [self.atom_map1, self.atom_map2]:
                left_atom_idx = am2i[left_atom_am+1000]
                left_atom = mol.GetAtomWithIdx(left_atom_idx)
                for neighbor_atom in left_atom.GetNeighbors():
                    
                    neighbor_atom_idx = neighbor_atom.GetIdx()
                    bond = mol.GetBondBetweenAtoms(left_atom_idx, neighbor_atom_idx)
                    bond_tuple = get_bond_tuple(bond)
                    a1, a2, b_type, b_stereo = bond_tuple[0], bond_tuple[1], bond_tuple[2], bond_tuple[3]
                    
                    motif_atom_idx = am2i[left_atom_am]
                    
                    b_type = rdchem.BondType.values[b_type]
                    b_stereo = rdchem.BondStereo.values[b_stereo]
                    if motif_atom_idx > neighbor_atom_idx:
                        motif_atom_idx, neighbor_atom_idx = neighbor_atom_idx, motif_atom_idx
                    
                    if (motif_atom_idx, neighbor_atom_idx) in added_bond:
                        continue
                        
                    
                    if mol.GetBondBetweenAtoms(motif_atom_idx, neighbor_atom_idx) is None:
                        bond_ind = mol.AddBond(motif_atom_idx, neighbor_atom_idx, b_type) - 1
                        new_bond = mol.GetBondWithIdx(bond_ind)
                        new_bond.SetStereo(b_stereo)
                        new_bond.SetBoolProp("is_edited", True)
                        self.neighbor_bonds.append((new_bond.GetBeginAtom().GetAtomMapNum(), new_bond.GetEndAtom().GetAtomMapNum()))
                        added_bond.append((motif_atom_idx, neighbor_atom_idx))
            
            
            # 删除mol上面的common原子
            del_atom_ind = [am2i[self.atom_map1+1000], am2i[self.atom_map2+1000]]
            
            for atom_ind in sorted(del_atom_ind, reverse=True):
                mol.RemoveAtom(atom_ind)
        else:
            mol = Chem.RWMol(mol)
            
            # 标记mol与motif的common原子
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum()==self.atom_map1:
                    atom.SetAtomMapNum(self.atom_map1+1000)
            
            for atom in self.motifmol.GetAtoms():
                if atom.GetAtomMapNum()==self.atom_map1:
                    atom.SetAtomMapNum(self.atom_map1)
            
            
            
            # 将与mol上面的common原子相关的化学键连到motif上
            mol = Chem.RWMol(CombineMols(mol, self.motifmol))
            am2i = {atom.GetAtomMapNum():i for i, atom in enumerate(mol.GetAtoms())}
            
            added_bond = []
            for left_atom_am in [self.atom_map1]:
                left_atom_idx = am2i[left_atom_am+1000]
                left_atom = mol.GetAtomWithIdx(left_atom_idx)
                for neighbor_atom in left_atom.GetNeighbors():
                    
                    neighbor_atom_idx = neighbor_atom.GetIdx()
                    bond = mol.GetBondBetweenAtoms(left_atom_idx, neighbor_atom_idx)
                    bond_tuple = get_bond_tuple(bond)
                    a1, a2, b_type, b_stereo = bond_tuple[0], bond_tuple[1], bond_tuple[2], bond_tuple[3]
                    
                    motif_atom_idx = am2i[left_atom_am]
                    
                    b_type = rdchem.BondType.values[b_type]
                    b_stereo = rdchem.BondStereo.values[b_stereo]
                    if motif_atom_idx > neighbor_atom_idx:
                        motif_atom_idx, neighbor_atom_idx = neighbor_atom_idx, motif_atom_idx
                    
                    if (motif_atom_idx, neighbor_atom_idx) in added_bond:
                        continue
                        
                    
                    if mol.GetBondBetweenAtoms(motif_atom_idx, neighbor_atom_idx) is None:
                        bond_ind = mol.AddBond(motif_atom_idx, neighbor_atom_idx, b_type) - 1
                        new_bond = mol.GetBondWithIdx(bond_ind)
                        new_bond.SetStereo(b_stereo)
                        new_bond.SetBoolProp("is_edited", True)
                        self.neighbor_bonds.append((new_bond.GetBeginAtom().GetAtomMapNum(), new_bond.GetEndAtom().GetAtomMapNum()))
                        added_bond.append((motif_atom_idx, neighbor_atom_idx))
            
            
            # 删除mol上面的common原子
            del_atom_ind = [am2i[self.atom_map1+1000]]
            
            for atom_ind in sorted(del_atom_ind, reverse=True):
                mol.RemoveAtom(atom_ind)
                    
        return mol

    def __str__(self):
        return f'Replace Motif {(self.atom_map1, self.atom_map2)} as {self.new_atoms_map_nums}'

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  

        # 修改 featurized 的数据        
        max_map_num = max(max(self.new_atoms_map_nums) + 1, adj.shape[0])

        new_adj = np.full((max_map_num, max_map_num, adj.shape[2]), fill_value=0, dtype=int)  # 新 adj 矩阵
        new_adj[:adj.shape[0], :adj.shape[1]] = adj
        new_nodes = np.full((max_map_num, nodes.shape[1]), fill_value=0, dtype=int)  # 新 nodes 列表
        new_nodes[:nodes.shape[0]] = nodes


        # ------------ Motif的原子特征
        for i, map_num in enumerate(self.new_atoms_map_nums):
            new_nodes[map_num] = self.new_atom_features[map_num]

            new_adj[map_num, map_num, 0] = self.prop2oh['bond']['bond_type']['self']
            new_adj[0, map_num, 0] = new_adj[map_num, 0, 0] = self.prop2oh['bond']['bond_type']['supernode']

        # ------------- Motif的bond特征
        # for bond in self.motif_info['bonds_info']:
        #     new_adj[bond['start_atom_map'], bond['end_atom_map']] = new_adj[bond['end_atom_map'], bond['start_atom_map']] = [
        #         self.prop2oh['bond']['bond_type'][bond['bond_type']], 
        #         self.prop2oh['bond']['bond_stereo'][bond['bond_stereo']], 
        #         self.prop2oh['bond']['is_edited'][1]
        #     ]
        
        for (atom1, atom2), values in self.new_bond_features.items():
            new_adj[atom1, atom2] = new_adj[atom2, atom1] = [
                self.prop2oh['bond']['bond_type'][values['bond_type']], 
                self.prop2oh['bond']['bond_stereo'][values['bond_stereo']], 
                self.prop2oh['bond']['is_edited'][1]
            ]
        
        #给邻接的bond的is_edit属性置1
        for (atom1, atom2) in self.neighbor_bonds:
            if atom1>1000:
                atom1 = atom1-1000
            if atom2>1000:
                atom2 = atom2 - 1000
            new_adj[atom1, atom2, 2] = new_adj[atom2, atom1, 2] = self.prop2oh['bond']['is_edited'][1]

        return new_adj, new_nodes


class AddRingAction(ReactionAction):
    def __init__(self, atom_map1: int, new_atoms_map_nums: List[int], ring_key: str,
                 feat_vocab: dict, is_hard: bool = False):
        super(AddRingAction, self).__init__(atom_map1, -1, feat_vocab, is_hard)
        self.ring_key = ring_key

        # order new atom map nums so map num of the existing atom is first
        map_ind = new_atoms_map_nums.index(self.atom_map1)  # 环中 atom_map1 的 atom map number
        self.new_atoms_map_nums = [self.atom_map1] + new_atoms_map_nums[map_ind + 1:] + new_atoms_map_nums[:map_ind]  # 更改顺序，把在source里的放到第一个

        new_a = Chem.Atom(6)
        new_a.SetIsAromatic(True)
        new_a.SetBoolProp('is_edited', True)
        self.new_atom_features = get_atom_features(new_a, list(self.prop2oh['atom'].keys()), atom_prop2oh=self.prop2oh['atom'])

        b_type = Chem.rdchem.BondType.AROMATIC
        self.new_bond_features = [self.prop2oh['bond'][key][val] for key, val in
                                  (('bond_type', b_type), ('bond_stereo', 0), ('is_edited', 1))]

    def get_tuple(self) -> Tuple[str, str]:
        return 'add_ring', self.ring_key

    def apply(self, mol: RWMol) -> RWMol:   # 修改分子
        atom_ind = get_atom_ind(mol, self.atom_map1)  # 返回 atom_map1 在 mol 中的 index
        if self.ring_key == 'benzene':
            mol = add_benzene_ring(mol, start_atom_ind=atom_ind, ring_atom_maps=self.new_atoms_map_nums)
        else:
            raise ValueError(f'No such ring type: {self.ring_key}')
        return mol

    def __str__(self):
        return f'Add {self.ring_key} ring to atom {self.atom_map1}'

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # 修改 featurized 的数据
        if self.ring_key != 'benzene':
            raise ValueError(f'No such ring type: {self.ring_key}')
        max_map_num = max(max(self.new_atoms_map_nums) + 1, adj.shape[0])

        new_adj = np.full((max_map_num, max_map_num, adj.shape[2]), fill_value=0, dtype=int)  # 新 adj 矩阵
        new_adj[:adj.shape[0], :adj.shape[1]] = adj
        new_nodes = np.full((max_map_num, nodes.shape[1]), fill_value=0, dtype=int)  # 新 nodes 列表
        new_nodes[:nodes.shape[0]] = nodes

        degree_ind = self.feat_vocab['atom_feat_ind'].get('degree', -1)  # 如果不存在, 返回 -1
        mol_id_ind = self.feat_vocab['atom_feat_ind'].get('mol_id', -1)
        is_reactant_ind = self.feat_vocab['atom_feat_ind'].get('is_reactant', -1)

        if mol_id_ind != -1:
            self.new_atom_features[mol_id_ind] = nodes[self.atom_map1, mol_id_ind]
        if is_reactant_ind != -1:
            self.new_atom_features[is_reactant_ind] = nodes[self.atom_map1, is_reactant_ind]

        for i, map_num in enumerate(self.new_atoms_map_nums):
            if degree_ind != -1:
                old_degree = new_nodes[map_num, degree_ind]
            else:
                old_degree = -1
            new_nodes[map_num] = self.new_atom_features

            # starting node has degree increased by 2 (it "closes" the ring)
            if degree_ind != -1:
                if map_num == self.atom_map1:
                    new_nodes[map_num, degree_ind] = old_degree + 2
                # all other nodes have degree 2
                else:
                    new_nodes[map_num, degree_ind] = self.prop2oh['atom']['degree'][2]

            new_adj[map_num, map_num, 0] = self.prop2oh['bond']['bond_type']['self']
            new_adj[0, map_num, 0] = new_adj[map_num, 0, 0] = self.prop2oh['bond']['bond_type']['supernode']

            if i > 0:
                prev_map_num = self.new_atoms_map_nums[i - 1]
                new_adj[prev_map_num, map_num] = new_adj[map_num, prev_map_num] = self.new_bond_features

        # close the ring (connected first atom to the last)
        map_num2 = self.new_atoms_map_nums[-1]
        new_adj[self.atom_map1, map_num2] = new_adj[map_num2, self.atom_map1] = self.new_bond_features

        return new_adj, new_nodes


class BondEditAction(ReactionAction):
    def __init__(self, atom_map1: int, atom_map2: int,
                 bond_type: Optional[int], bond_stereo: Optional[int],
                 feat_vocab: dict, is_hard: bool = False):
        super(BondEditAction, self).__init__(atom_map1, atom_map2, feat_vocab, is_hard)
        self.bond_type = bond_type
        self.bond_stereo = bond_stereo
        self.bond_feat_keys = ['bond_type', 'bond_stereo', 'is_edited']
        self.is_aromatic_val = self.prop2oh['atom']['is_aromatic'][1]

    def get_tuple(self) -> Tuple[str, Tuple[Optional[int], Optional[int]]]:
        return 'change_bond', (self.bond_type, self.bond_stereo)

    def _get_bond_features(self):
        return [self.bond_type, self.bond_stereo, 1]

    def apply(self, mol: RWMol) -> RWMol:
        atom1 = get_atom_ind(mol, self.atom_map1)
        atom2 = get_atom_ind(mol, self.atom_map2)

        if self.bond_type is None:  # delete bond
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if bond is not None:
                mol.RemoveBond(atom1, atom2)
        else:
            b_type = rdchem.BondType.values[self.bond_type]
            b_stereo = rdchem.BondStereo.values[self.bond_stereo]

            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if bond is None:  # add new bond
                bond_ind = mol.AddBond(atom1, atom2, order=b_type) - 1
                bond = mol.GetBondWithIdx(bond_ind)
            else:  # change an existing bond
                bond.SetBondType(b_type)
            bond.SetStereo(b_stereo)
            bond.SetBoolProp('is_edited', True)

            if b_type == BondType.AROMATIC:
                bond.SetIsAromatic(True)
                mol.GetAtomWithIdx(atom1).SetIsAromatic(True)
                mol.GetAtomWithIdx(atom2).SetIsAromatic(True)

        return mol

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        adj = adj.copy()
        nodes = nodes.copy()

        degree_ind = self.feat_vocab['atom_feat_ind'].get('degree', -1)
        is_aromatic_ind = self.feat_vocab['atom_feat_ind'].get('is_aromatic', -1)

        if self.bond_type is None:
            bond_features = [0, 0, 0]

            # for a deleted bond, decrease degree of nodes by 1
            if degree_ind != -1 and adj[self.atom_map1, self.atom_map2, 0] != 0:
                nodes[self.atom_map1, degree_ind] -= 1
                nodes[self.atom_map2, degree_ind] -= 1

        else:
            bond_features = [self.prop2oh['bond'][key].get(val, 0) for key, val in zip(self.bond_feat_keys, self._get_bond_features())]
            # for a new bond, increase degree of nodes by 1
            if degree_ind != -1 and adj[self.atom_map1, self.atom_map2, 0] == 0:
                nodes[self.atom_map1, degree_ind] += 1
                nodes[self.atom_map2, degree_ind] += 1

        if is_aromatic_ind != -1 and self.bond_type == 12:  # aromatic bond
            nodes[self.atom_map1, is_aromatic_ind] = self.is_aromatic_val
            nodes[self.atom_map2, is_aromatic_ind] = self.is_aromatic_val

        adj[self.atom_map1, self.atom_map2] = adj[self.atom_map2, self.atom_map1] = bond_features

        return adj, nodes

    def __str__(self):
        if self.bond_type is None:
            return f'Delete bond {self.atom_map1, self.atom_map2}'
        bond_type = f'{feat_val_to_str("bond_type", self.bond_type)}'
        bond_stereo = f'{feat_val_to_str("bond_stereo", self.bond_stereo)}'
        return f'Edit bond {self.atom_map1, self.atom_map2}: {bond_type}, {bond_stereo}'


class StopAction(ReactionAction):
    def __init__(self, feat_vocab: dict):
        super(StopAction, self).__init__(0, -1, feat_vocab=feat_vocab, is_hard=False)

    def get_tuple(self) -> Tuple[str]:
        return 'stop', None

    def apply(self, mol: RWMol) -> RWMol:
        return mol  # do nothing (stop generation)

    def graph_apply(self, adj: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return adj, nodes  # do nothing (stop generation)

    def __str__(self):
        return 'Stop'


class AttachAtomAction:
    def __init__(self, base_am, attach_am):
        super(AttachAtomAction, self).__init__()
        self.base_am = base_am
        self.attach_am = attach_am
        self.atom_map1 = base_am
        self.atom_map2 = attach_am
    
    def get_tuple(self) -> Tuple[str, Tuple[Tuple[int, int], Tuple[int, int, int, int, int], str]]:
        return 'attach_atom', (self.base_am, self.attach_am)
    
    def apply(self, mol):
        infos = []
        
        for rm_idx, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol()=="*" and atom.GetAtomMapNum()==self.attach_am:
                bond = atom.GetBonds()[0]
                infos.append([rm_idx, self.base_am, self.attach_am, bond])
                
        for rm_idx, _,_,_ in infos:
            mol.RemoveAtom(rm_idx)
        
        am2idx = get_am2idx(mol)
        for _, base_am, attach_am, bond in infos:
            atom1 = am2idx[base_am]
            atom2 = am2idx[attach_am]
            mol.AddBond(atom1, atom2, order=bond.GetBondType())
        return mol
    
    def __str__(self):
        return f'Attach Atom {self.base_am}-->{self.attach_am}'

class AddMotifAction_with_dummySMI:
    '''------------------Explicit valence vs Implicit valence---------------
    Explicit valence in chemistry refers to the number of bonds that an atom can form in a molecule. The valence electrons of an atom are the outermost electrons that participate in chemical reactions and bonding. The explicit valence of an atom is determined by the number of valence electrons that it has.

    Implicit valence in chemistry refers to the potential number of bonds that an atom could form, even if it does not actually participate in bonding in a particular molecule. This can be determined by the electronic structure and configuration of an atom.

    In summary, explicit valence in chemistry refers to the actual number of bonds that an atom forms, while implicit valence refers to the potential number of bonds that an atom could form.
    
    显示化合价是原子通过共价键得失电子的数量, 也就是成键电子被有倾向性得分给化学键一边而造成的原子带电荷量.(和形式电荷下成键电子均匀分配假设刚好相反)
    
    ---------------Explicit hs vs Implicit hs-------------------
    
    In RDKit, explicit hydrogens are atoms that are explicitly represented in a molecule's graph, while implicit hydrogens are not explicitly represented, but rather inferred from the number of valence electrons on each atom.
    explicit hydrogens 在分子图中当做独立的节点呈现;而implicit hydrogens不是独立的节点,可以根据原子的化合价推出来。
    
    
    ------------------------ Formal charge----------------------
    形式电荷是化合物中成键电子均匀分配下的原子带电荷数, 形式电荷=标准最外层电荷数-(孤对电子数+化学键数) = 标准最外层电荷数-(孤对电子数+共价键数+#Hs)
    
    #Hs = 标准最外层电荷数-孤对电子数-共价键数-形式电荷
    
    其中,(标准最外层电荷数-孤对电子数)=可成键数
    
    If (#Hs_old > 最大可成键数-(共价键数+形式电荷)):
        #Hs_new = 最大可成键数-(共价键数+形式电荷)
    '''   
    MAX_BONDS = {"Na":[1],
                 "Zn":[2],
                 "Li":[1],
                 "K":[1],
                 "Mg":[2],
                 "B":[3],
                 "C":[4],
                 "N":[3],
                 "O":[2],
                 "Br":[1],
                 "I":[1],
                 "Cl":[1],
                 "F":[1],
                 "S":[4, 6],
                 "Se":[2,4,6],
                 "Sn":[4],
                 "P":[5],
                 "Si":[4]
                 }
    def __init__(self, motif_smi):
        super(AddMotifAction_with_dummySMI, self).__init__()
        self.smiles_atom_mapping = motif_smi
        
        # MolFromSmiles有时候会报错, 不得已使用MolFromSmiles
        mol = Chem.MolFromSmiles(motif_smi, sanitize=False)
        for source_ind, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol()=="*":
                self.atom_map1 = atom.GetAtomMapNum()
                self.atom_map2 = -1
            atom.SetAtomMapNum(0)
                
        self.smiles = Chem.MolToSmiles(mol)

        # self.motifmol = Chem.MolFromSmiles(self.smiles_atom_mapping, sanitize=False)
        self.motifmol = Chem.MolFromSmiles(self.smiles_atom_mapping, sanitize=False)
        self.motifmol = fix_explicit_hs(self.motifmol)
        


    def get_tuple(self) -> Tuple[str, Tuple[Tuple[int, int], Tuple[int, int, int, int, int], str]]:
        return 'add_motif', self.smiles

    def apply(self, mol: RWMol) -> RWMol:   # 修改分子
        # 统计motif里面和dummy原子相连的边, connect_bonds
        existing_ams = set(get_am2idx(mol).keys())
        connect_bonds = []
        for atom in self.motifmol.GetAtoms():
            if atom.GetSymbol()=="*" and atom.GetAtomMapNum() in existing_ams:
                attach_am = atom.GetAtomMapNum()
                dummy_motif_idx = atom.GetIdx()
                
        for bond in self.motifmol.GetBonds():
            start_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if start_atom.GetAtomMapNum() == attach_am:
                link_am = end_atom.GetAtomMapNum()
                connect_bonds.append([link_am, bond])
            elif end_atom.GetAtomMapNum() == attach_am:
                link_am = start_atom.GetAtomMapNum()
                connect_bonds.append([link_am, bond])
        
        # 去除motif的dummy原子,
        motifmol = Chem.RWMol(copy.deepcopy(self.motifmol))
        motifmol.RemoveAtom(dummy_motif_idx)
        
        #  合并source mol和motif mol, 将connection_set里面的边连接到合并source mol上面   
        mol_merge = Chem.RWMol(CombineMols(mol, motifmol))
        am2idx = get_am2idx(mol_merge)
        source_attach_ind = am2idx[attach_am]
        for link_am, bond in connect_bonds:
            motif_attach_ind = am2idx[link_am]
            atom_source_attach = mol_merge.GetAtomWithIdx(source_attach_ind)
            atom_motif_attach = mol_merge.GetAtomWithIdx(motif_attach_ind)
            source_IsAromatic = atom_source_attach.GetIsAromatic()
            motif_IsAromatic = atom_motif_attach.GetIsAromatic()
            

            b_type = rdchem.BondType.values[int(bond.GetBondType())]
            b_stereo = rdchem.BondStereo.values[int(bond.GetStereo())]
            bond_ind = mol_merge.AddBond(source_attach_ind, motif_attach_ind, b_type ) - 1
            new_bond = mol_merge.GetBondWithIdx(bond_ind)
            new_bond.SetStereo(b_stereo)
            new_bond.SetBoolProp('is_edited', True)
            
            # 和Aromatic原子相连时，被连的原子会自动变成Aromatic类型, 需要使用source_IsAromatic和motif_IsAromatic保存他们原本的类型，并在后面的代码中恢复其本来的原子类型
            atom1 = mol_merge.GetAtomWithIdx(source_attach_ind)
            atom1.SetAtomMapNum(attach_am)
            atom1.SetIsAromatic(source_IsAromatic)
            atom2 = mol_merge.GetAtomWithIdx(motif_attach_ind)
            atom2.SetAtomMapNum(link_am)
            atom2.SetIsAromatic(motif_IsAromatic) 

        # 矫正source mol上面被attach部分的氢原子数量
        source_attach_atom = mol_merge.GetAtomWithIdx(source_attach_ind)
        
        symbol = source_attach_atom.GetSymbol()
        if symbol!="*":
            explicit_bonds = int(sum([b.GetBondTypeAsDouble() for b in source_attach_atom.GetBonds()]))
            formal_charge = source_attach_atom.GetFormalCharge()
            Hs = source_attach_atom.GetNumExplicitHs()

            
            new_Hs_max_list = [max_bonds - (explicit_bonds+formal_charge) for max_bonds in self.MAX_BONDS[symbol]]
            new_Hs_max_list = [one for one in new_Hs_max_list if one>=0]
            if len(new_Hs_max_list)>0:# 可以通过修改Hs来平衡化合价
                new_Hs_max = min(new_Hs_max_list)
                if Hs>new_Hs_max:
                    source_attach_atom.SetNumExplicitHs(new_Hs_max)
            else:# 无法通过修改Hs来平衡化合价
                pass
                
        return mol_merge

    def __str__(self):
        return f'Add Motif {self.smiles_atom_mapping}'