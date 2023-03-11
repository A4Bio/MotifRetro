import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import numpy as np
from src.utils.feat_utils import atom_to_edit_tuple, get_bond_tuple, fix_incomplete_mappings, reac_to_canonical, fix_explicit_hs, renumber_atoms_for_mapping, mark_reactants
from copy import copy
from src.utils.chem_utils import get_ams, get_am2idx
from src.utils.chem_utils import MultiElement, get_submol

def preprocess_mols(target, source):
    target_mol = Chem.MolFromSmiles(target)
    source_mol = Chem.MolFromSmiles(source)

    target_mol, source_mol = fix_incomplete_mappings(target_mol, source_mol)
    target_mol, source_mol = reac_to_canonical(target_mol, source_mol) 
    source_mol = fix_explicit_hs(source_mol)
    target_mol = fix_explicit_hs(target_mol)
    source_mol = renumber_atoms_for_mapping(source_mol)
    target_mol = renumber_atoms_for_mapping(target_mol)
    return target_mol, source_mol


# get edit states
def get_synthons(sample_generator):
    state = [copy(sample_generator.source_mol)]

    for i in range(100):
        reaction_action = sample_generator.generate_gen_action()
        if ("Delete bond" not in str(reaction_action)) and ("Edit Atom" not in str(reaction_action)) and ("Edit Bond" not in str(reaction_action)): # 看是否需要追加Delete atom操作，目前没有这个操作
            break

        sample_generator.source_mol = reaction_action.apply(sample_generator.source_mol) # 这是关键部分
        latent_mol = copy(sample_generator.source_mol)
        latent_mol.UpdatePropertyCache(strict=False)
        state.append(latent_mol)
        
    return state[-1]


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
        
    def get_submol_with_ams(mol, ams):
        am2idx = get_am2idx(mol)
        idxs = []
        for am in ams:
            idxs.append(am2idx[am])

        sub_mol = get_submol(mol, idxs)
        return sub_mol

    frag_mol = get_submol_with_ams(reactant, frag+attach)
    
    # 构造dummy atom
    frag_mol = Chem.RWMol(frag_mol)
    for atom in frag_mol.GetAtoms():
        if atom.GetAtomMapNum() in attach:
            atom.SetAtomicNum(0)
            atom.SetFormalCharge(0)
    return frag_mol

def filter_reactants(sub_mols, prod_mol):
    mol_maps = set(a.GetAtomMapNum() for a in prod_mol.GetAtoms())
    reactants = []
    for mol in sub_mols:
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() in mol_maps:
                reactants.append(mol)
                break
    return Chem.MolFromSmiles('.'.join([Chem.MolToSmiles(m) for m in reactants]))

def mol_to_unmapped(mol):
    mol_copy = Chem.Mol(mol)
    for a in mol_copy.GetAtoms():
        a.ClearProp("molAtomMapNumber")
    return mol_copy


def mol_to_unmapped_smiles(mol) -> str:
    mol_copy = mol_to_unmapped(mol)
    smi_unmapped = Chem.MolToSmiles(mol_copy, canonical=True)
    return smi_unmapped