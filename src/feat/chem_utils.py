import numpy as np
from rdkit import Chem
import networkx as nx
from typing import List, Tuple, Union

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
        self.mols = [get_sub_mol(self.mol, sub_atoms) for sub_atoms in frag_indices]

        atom_start = 0
        bond_start = 0
        self.atom_scope = []
        self.bond_scope = []

        for mol in self.mols:
            self.atom_scope.append((atom_start, mol.GetNumAtoms()))
            self.bond_scope.append((bond_start, mol.GetNumBonds()))
            atom_start += mol.GetNumAtoms()
            bond_start += mol.GetNumBonds()


def get_sub_mol(mol: Chem.Mol, sub_atoms: List[int]) -> Chem.Mol:
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