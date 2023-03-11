from typing import Tuple, List

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import RenumberAtoms
from tqdm import tqdm
from src.feat.graph_features import try_get_atom_feature
import torch


def get_bond_tuple(bond) -> Tuple[int, int, int, int]:
    a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
    bt = int(bond.GetBondType())
    st = int(bond.GetStereo())
    if a1 > a2:
        a1, a2 = a2, a1
    return a1, a2, bt, st


def atom_to_edit_tuple(atom) -> Tuple:  # 考察原子的四个 property： ['formal_charge', 'chiral_tag', 'num_explicit_hs', 'is_aromatic']
    atom_feat_keys = ['formal_charge', 'chiral_tag', 'num_explicit_hs', 'is_aromatic']
    feat = [try_get_atom_feature(atom, key) for key in atom_feat_keys]
    return tuple(feat)

def fix_incomplete_mappings(sub_mol: Mol, prod_mol: Mol) -> Tuple[Mol, Mol]:
    max_map = max(a.GetAtomMapNum() for a in sub_mol.GetAtoms())
    max_map = max(max(a.GetAtomMapNum() for a in prod_mol.GetAtoms()), max_map)

    for mol in (sub_mol, prod_mol):
        for a in mol.GetAtoms():
            map_num = a.GetAtomMapNum()
            if map_num is None or map_num < 1:
                max_map += 1
                a.SetAtomMapNum(max_map)
    return sub_mol, prod_mol


def reac_to_canonical(sub_mol, prod_mol): # converting to smiles to mol and again to smiles makes atom order canonical
    sub_mol = Chem.MolFromSmiles(Chem.MolToSmiles(sub_mol))
    prod_mol = Chem.MolFromSmiles(Chem.MolToSmiles(prod_mol))

    # in RdKit chirality can be marked different depending on order of atoms in molecule list
    # here we remap atoms so the map order is consistent with atom list order

    map2map = {}
    for i, a in enumerate(prod_mol.GetAtoms()):
        map2map[a.GetAtomMapNum()] = i + 1
        a.SetAtomMapNum(i + 1)

    max_map = max(map2map.values())
    for i, a in enumerate(sub_mol.GetAtoms()):
        m = a.GetAtomMapNum()
        if m in map2map:
            a.SetAtomMapNum(map2map[m])
        else:
            max_map += 1
            a.SetAtomMapNum(max_map)

    return sub_mol, prod_mol

# def reac_to_canonical(sub_mol, prod_mol): # converting to smiles to mol and again to smiles makes atom order canonical
#     sub_mol = Chem.MolFromSmiles(Chem.MolToSmiles(sub_mol))
#     prod_mol = Chem.MolFromSmiles(Chem.MolToSmiles(prod_mol))

#     # in RdKit chirality can be marked different depending on order of atoms in molecule list
#     # here we remap atoms so the map order is consistent with atom list order
    
#     map2idx = {}
#     for i, a in enumerate(prod_mol.GetAtoms()):
#         map2idx[a.GetAtomMapNum()] = i
    
#     idx2rand = np.random.permutation(list(map2idx.values())).tolist()
    
#     map2map = {key: idx2rand[val] for key,val in map2idx.items()}
    
    
#     for i, a in enumerate(prod_mol.GetAtoms()):
#         a.SetAtomMapNum(map2map[a.GetAtomMapNum()])

#     max_map = max(map2map.values())
#     for i, a in enumerate(sub_mol.GetAtoms()):
#         m = a.GetAtomMapNum()
#         if m in map2map:
#             a.SetAtomMapNum(map2map[m])
#         else:
#             max_map += 1
#             a.SetAtomMapNum(max_map)

#     return sub_mol, prod_mol

# rdkit has a problem with implicit hs. By default there are only explicit hs.
# This is a hack to fix this error
def fix_explicit_hs(mol: Mol) -> Mol:
    for a in mol.GetAtoms():
        a.SetNoImplicit(False)

    mol = Chem.AddHs(mol, explicitOnly=True)
    mol = Chem.RemoveHs(mol)

    Chem.SanitizeMol(mol)
    return mol

def renumber_atoms_for_mapping(mol: Mol) -> Mol:
    new_order = []
    for a in mol.GetAtoms():
        new_order.append(a.GetAtomMapNum())
    new_order = [int(a) for a in np.argsort(new_order)]
    return RenumberAtoms(mol, new_order)


def mark_reactants(source_mol: Mol, target_mol: Mol):
    target_atoms = set(a.GetAtomMapNum() for a in reversed(target_mol.GetAtoms()))
    for a in source_mol.GetAtoms():
        m = a.GetAtomMapNum()
        if m is not None and m > 0 and m in target_atoms:
            a.SetBoolProp('in_target', True)
            

def to_torch_tensor(arr, long: bool = False) -> torch.Tensor:
    if not isinstance(arr, np.ndarray):
        arr = arr.toarray()
    # noinspection PyUnresolvedReferences
    ten = torch.from_numpy(arr)
    if long:
        ten = ten.long()
    else:
        ten = ten.float()

    # if torch.cuda.is_available():
    #     return ten.cuda()
    return ten


