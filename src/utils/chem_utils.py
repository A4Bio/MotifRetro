import numpy as np
from rdkit import Chem
import networkx as nx
from typing import List, Tuple, Union
import copy
import ast

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':5, 'O':2, 'P':5, 'S':6}  # 'Se':4, 'Si':4, 'Mg': 2}

BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_DELTAS = {-3: 0, -2: 1, -1.5: 2, -1: 3, -0.5: 4, 0: 5, 0.5: 6, 1: 7, 1.5: 8, 2:9, 3:10}
BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5]


class RxnElement:
    """
    RxnElement is an abstract class for dealing with single molecule. The graph
    and corresponding molecule attributes are built for the molecule. The constructor
    accepts only mol objects, sidestepping the use of SMILES string which may always
    not be achievable, especially for a unkekulizable molecule.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        self.mol = mol
        self.rxn_class = rxn_class
        self._build_mol()
        self._build_graph()

    def _build_mol(self) -> None:
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()
        self.num_bonds = self.mol.GetNumBonds()
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                            for atom in self.mol.GetAtoms()}
        self.idx_to_amap = {value: key for key, value in self.amap_to_idx.items()}

    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index( bond.GetBondType() )
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        self.atom_scope = (0, self.num_atoms)
        self.bond_scope = (0, self.num_bonds)

    #CHECK IF THESE TWO ARE NEEDED
    def update_atom_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the atom indices by the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        if isinstance(self.atom_scope, list):
            return [(st + offset, le) for (st, le) in self.atom_scope]
        st, le = self.atom_scope
        return (st + offset, le)

    def update_bond_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the atom indices by the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        if isinstance(self.bond_scope, list):
            return [(st + offset, le) for (st, le) in self.bond_scope]
        st, le = self.bond_scope
        return (st + offset, le)


class MultiElement(RxnElement):
    """
    MultiElement is an abstract class for dealing with multiple molecules. The graph
    is built with all molecules, but different molecules and their sizes are stored.
    The constructor accepts only mol objects, sidestepping the use of SMILES string
    which may always not be achievable, especially for an invalid intermediates.
    """
    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index( bond.GetBondType() )
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        frag_indices = [c for c in nx.strongly_connected_components(self.G_dir)]
        self.mols = [get_submol(self.mol, sub_atoms) for sub_atoms in frag_indices]

        atom_start = 0
        bond_start = 0
        self.atom_scope = []
        self.bond_scope = []

        for mol in self.mols:
            self.atom_scope.append((atom_start, mol.GetNumAtoms()))
            self.bond_scope.append((bond_start, mol.GetNumBonds()))
            atom_start += mol.GetNumAtoms()
            bond_start += mol.GetNumBonds()

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

def get_ams(mol):
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]

def get_am2idx(mol):
    return {atom.GetAtomMapNum():idx for idx, atom in enumerate(mol.GetAtoms())}

def remove_am(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def am2inchi_order(mol):
    inchi, aux_info = Chem.MolToInchiAndAuxInfo(mol)
    for i in aux_info.split('/'):
        if i[0]=='N':
            print(i)
            pos=i[2:].split(',')
    mm_map = {int(j)-1:i for i,j in enumerate(pos)}
    return mm_map

def get_am2idx_smiles_order(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol()=="*":
            root = atom.GetIdx()
            break
    
    smi = Chem.MolToSmiles(mol, rootedAtAtom=root, canonical=True)
    idx = mol.GetProp('_smilesAtomOutputOrder')
    idx = ast.literal_eval(idx)
    ams2idx = {}
    for t, atom in enumerate(mol.GetAtoms()):
        ams2idx[atom.GetAtomMapNum()] = idx[t]
    return smi, ams2idx

def get_submol(mol: Chem.Mol, sub_atoms: List[int]) -> Chem.Mol:
    """Extract subgraph from molecular graph.
    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object,
    sub_atoms: List[int],
        List of atom indices in the subgraph.
    """
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

# def get_submol(raw_mol, atom_indices, kekulize=False):
#     mol = copy.deepcopy(raw_mol)
#     atom_indices = list(set(atom_indices))
#     if len(atom_indices) == 1:
#         return smi2mol(mol.GetAtomWithIdx(atom_indices[0]).GetSymbol(), kekulize)
#     aid_dict = { i: True for i in atom_indices }
#     edge_indices = []
#     for i in range(mol.GetNumBonds()):
#         bond = mol.GetBondWithIdx(i)
#         begin_aid = bond.GetBeginAtomIdx()
#         end_aid = bond.GetEndAtomIdx()
#         if begin_aid in aid_dict and end_aid in aid_dict:
#             edge_indices.append(i)
#     mol = Chem.PathToSubmol(mol, edge_indices)# 这里子图中每个原子的Hs与原分子是相同的, 但通过mol2 = smi2mol(mol2smi(mol))转化后, rdkit在mol2smi这一步会自动补H，导致mol2原子的Hs与原分子mol不一致
#     return mol

def get_sub_mol_stereo(mol: Chem.Mol, sub_atoms: List[int]) -> Chem.Mol:
    """Extract subgraph from molecular graph, while preserving stereochemistry.
    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object,
    sub_atoms: List[int],
        List of atom indices in the subgraph.
    """
    # This version retains stereochemistry, as opposed to the other version
    new_mol = Chem.RWMol(Chem.Mol(mol))
    atoms_to_remove = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in sub_atoms:
            atoms_to_remove.append(atom.GetIdx())

    for atom_idx in sorted(atoms_to_remove, reverse=True):
        new_mol.RemoveAtom(atom_idx)

    return new_mol.GetMol()


def cnt_atom(smi, return_dict=False):
    atom_dict = { atom: 0 for atom in MAX_VALENCE }
    for i in range(len(smi)):
        symbol = smi[i].upper()
        next_char = smi[i+1] if i+1 < len(smi) else None
        if symbol == 'B' and next_char == 'r':
            symbol += next_char
        elif symbol == 'C' and next_char == 'l':
            symbol += next_char
        if symbol in atom_dict:
            atom_dict[symbol] += 1
    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())
    

def compare_mol(mol_pred, mol_true):
    '''
    
    '''
    # mol_pred = smi2mol(smi_pred)
    # mol_true = smi2mol(smi_true)
    
    am2idx1 = get_am2idx(mol_pred)
    am2idx2 = get_am2idx(mol_true)
    if not (set(am2idx1.keys())==set(am2idx2.keys())):
        return False
    
    for am in am2idx1.keys():
        atom_pred = mol_pred.GetAtomWithIdx(am2idx1[am])
        atom_true = mol_true.GetAtomWithIdx(am2idx2[am])
        sym_pred = atom_pred.GetSymbol()
        sym_true = atom_true.GetSymbol()
        if sym_pred=="c":
            print()
        if sym_pred!=sym_true:
            if (sym_pred, sym_true) == ("c", "C"):
                atom_pred.SetIsAromatic(False)
            elif (sym_pred, sym_true) == ("C", "c"):
                atom_pred.SetIsAromatic(True)
            else:
                return False
            
    smi_pred = mol2smi(mol_pred)
    smi_true = mol2smi(mol_true)
    smi_pred = Chem.CanonSmiles(smi_pred.replace("@@", "@").replace("@", ""))
    smi_true = Chem.CanonSmiles(smi_true.replace("@@", "@").replace("@", ""))
    return smi_pred == smi_true



def compare_smi(smi_pred, smi_true):
    '''
    
    '''
    smi_pred = smi_pred.replace("@@", "@").replace("@", "").replace("/","").replace("\\","")
    smi_true = smi_true.replace("@@", "@").replace("@", "").replace("/","").replace("\\","")
    
    smi_pred = Chem.CanonSmiles(smi_pred)
    smi_true = Chem.CanonSmiles(smi_true)
    return smi_pred == smi_true


def revise_mol_with_refsmi(rdmol, refsmi):
    '''
    防止rdkit读取smiles时自动补H和改变原子类型(c-->C)和化学键类型(C-C单键-->芳香键)
    '''
    ref_mol = Chem.MolFromSmiles(refsmi, sanitize=False)
    for ref_atom, now_atom in zip(ref_mol.GetAtoms(), rdmol.GetAtoms()):
        assert ref_atom.GetSymbol() == now_atom.GetSymbol()
        now_atom.SetIsAromatic(ref_atom.GetIsAromatic())
        now_atom.SetNumExplicitHs(ref_atom.GetNumExplicitHs())
        
    for ref_bond, now_bond in zip(ref_mol.GetBonds(), rdmol.GetBonds()):
        now_bond.SetIsAromatic(ref_bond.GetIsAromatic())
        now_bond.SetBondType(ref_bond.GetBondType())
    return rdmol