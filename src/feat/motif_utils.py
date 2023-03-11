import json
from src.feat.ps.utils.chem_utils import smi2mol, mol2smi, get_submol
from rdkit import Chem
from copy import copy
import numpy as np

class MolInSubgraph:
    def __init__(self, mol, kekulize=False):
        self.mol = mol
        self.smi = mol2smi(mol)
        self.kekulize = kekulize
        self.subgraphs, self.subgraphs_smis = {}, {}  # pid is the key (init by all atom idx)
        for atom in mol.GetAtoms():
            idx, symbol = atom.GetIdx(), atom.GetSymbol()
            self.subgraphs[idx] = { idx: symbol }
            self.subgraphs_smis[idx] = symbol
        self.inversed_index = {} # assign atom idx to pid
        self.upid_cnt = len(self.subgraphs)
        for aid in range(mol.GetNumAtoms()):
            for key in self.subgraphs:
                subgraph = self.subgraphs[key]
                if aid in subgraph:
                    self.inversed_index[aid] = key
        self.dirty = True
        self.smi2pids = {} # private variable, record neighboring graphs and their pids

    def get_nei_subgraphs(self):
        nei_subgraphs, merge_pids = [], []
        for key in self.subgraphs:
            subgraph = self.subgraphs[key]
            local_nei_pid = []
            for aid in subgraph:
                atom = self.mol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if nei_idx in subgraph or nei_idx > aid:   # only consider connecting to former atoms
                        continue
                    local_nei_pid.append(self.inversed_index[nei_idx])
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                new_subgraph = copy(subgraph)
                new_subgraph.update(self.subgraphs[nei_pid])
                nei_subgraphs.append(new_subgraph)
                merge_pids.append((key, nei_pid))
        return nei_subgraphs, merge_pids
    
    def get_nei_smis(self):
        if self.dirty:
            nei_subgraphs, merge_pids = self.get_nei_subgraphs()
            nei_smis, self.smi2pids = [], {}
            for i, subgraph in enumerate(nei_subgraphs):
                submol = get_submol(self.mol, list(subgraph.keys()), kekulize=self.kekulize)
                smi = mol2smi(submol)
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, [])
                self.smi2pids[smi].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis

    def merge(self, smi):
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            merge_pids = self.smi2pids[smi]
            for pid1, pid2 in merge_pids:
                if pid1 in self.subgraphs and pid2 in self.subgraphs: # possibly del by former
                    self.subgraphs[pid1].update(self.subgraphs[pid2])
                    self.subgraphs[self.upid_cnt] = self.subgraphs[pid1]
                    self.subgraphs_smis[self.upid_cnt] = smi
                    # self.subgraphs_smis[pid1] = smi
                    for aid in self.subgraphs[pid2]:
                        self.inversed_index[aid] = pid1
                    for aid in self.subgraphs[pid1]:
                        self.inversed_index[aid] = self.upid_cnt
                    del self.subgraphs[pid1]
                    del self.subgraphs[pid2]
                    del self.subgraphs_smis[pid1]
                    del self.subgraphs_smis[pid2]
                    self.upid_cnt += 1
        self.dirty = True   # mark the graph as revised

    def get_smis_subgraphs(self):
        # return list of tuple(smi, idxs)
        res = []
        for pid in self.subgraphs_smis:
            smi = self.subgraphs_smis[pid]
            group_dict = self.subgraphs[pid]
            idxs = list(group_dict.keys())
            res.append((smi, idxs))
        return res
    

class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        # load kekulize config
        config = json.loads(lines[0])
        self.kekulize = config['kekulize']
        lines = lines[1:]
        
        self.vocab_dict = {}
        self.idx2subgraph, self.subgraph2idx = [], {}
        self.max_num_nodes = 0
        for line in lines:
            smi, atom_num, freq = line.strip().split('\t')
            self.vocab_dict[smi] = (int(atom_num), int(freq))
            self.max_num_nodes = max(self.max_num_nodes, int(atom_num))
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)
        self.pad, self.end = '<pad>', '<s>'
        for smi in [self.pad, self.end]:
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)
        # for fine-grained level (atom level)
        self.bond_start = '<bstart>'
        self.max_num_nodes += 2 # start, padding
        self.vocab_array = np.array(list(self.vocab_dict.keys()))
    
    def tokenize(self, mol):
        smiles = mol
        if isinstance(mol, str):
            mol = smi2mol(mol, self.kekulize)
        else:
            smiles = mol2smi(mol)
        rdkit_mol = mol
        for a in mol.GetAtoms():
            a.SetNoImplicit(True)
        mol = MolInSubgraph(mol, kekulize=self.kekulize)
        while True:
            nei_smis = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''
            for smi in nei_smis:
                if smi not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi
            if max_freq == -1:
                break
            mol.merge(merge_smi)
        res = mol.get_smis_subgraphs()
        # construct reversed index
        aid2pid = {}
        for pid, subgraph in enumerate(res):
            _, aids = subgraph
            for aid in aids:
                aid2pid[aid] = pid
        # construct adjacent matrix
        ad_mat = [[0 for _ in res] for _ in res]
        for aid in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(aid)
            for nei in atom.GetNeighbors():
                nei_id = nei.GetIdx()
                i, j = aid2pid[aid], aid2pid[nei_id]
                if i != j:
                    ad_mat[i][j] = ad_mat[j][i] = 1
        group_idxs = [x[1] for x in res]
        smiles = [x[0] for x in res]

        return group_idxs, smiles, ad_mat
        # print(group_idxs)
        # return Molecule(smiles, group_idxs, self.kekulize)

    def idx_to_subgraph(self, idx):
        return self.idx2subgraph[idx]
    
    def subgraph_to_idx(self, subgraph):
        return self.subgraph2idx[subgraph]
    
    def pad_idx(self):
        return self.subgraph2idx[self.pad]
    
    def end_idx(self):
        return self.subgraph2idx[self.end]
    
    def atom_vocab(self):
        return copy(self.atom_level_vocab)

    def num_subgraph_type(self):
        return len(self.idx2subgraph)
    
    def atom_pos_pad_idx(self):
        return self.max_num_nodes - 1
    
    def atom_pos_start_idx(self):
        return self.max_num_nodes - 2

    def __call__(self, mol):
        return self.tokenize(mol)
    
    def __len__(self):
        return len(self.idx2subgraph)


class MotifGenerator(object):
    def __init__(self, vocab_path):
        self.tokenizer = Tokenizer(vocab_path)  # TODO: 
    
    def generate_motif(self, mol):

        idx2am = dict()
        am2idx = dict()

        for atom in mol.GetAtoms():
            idx2am[atom.GetIdx()] = atom.GetAtomMapNum()
            am2idx[atom.GetAtomMapNum()] = atom.GetIdx()
            atom.SetAtomMapNum(0)
            atom.SetNumExplicitHs(0)
        
        group_idxs, group_smis, ad_mat = self.tokenizer(mol)  # TODO: 这里 Idx 会改变

        all_motifs_info = {'motifs': [], 'smiles': [], 'smiles_with_mapping': [], 'motif_id':[]}  # , 'motif_bonds': [], 'mol': mol}
        for group, _smiles in zip(group_idxs, group_smis):
            # motif_info = []

            # for idx in group:
            #     motif_info.append(idx2am[idx])

            # 这段代码是用来找这个motif与其他的motif的连接点的 =>
            editable_mol = Chem.EditableMol(mol)
            removed_atoms_idx = list(set(idx2am.keys()) - set(group))
            removed_atoms_idx.reverse()
            removed_atoms_idx = sorted(removed_atoms_idx, reverse=True)
            for _idx in removed_atoms_idx:
                editable_mol.RemoveAtom(_idx)
            smiles_with_mapping = Chem.MolToSmiles(editable_mol.GetMol())

            all_motifs_info['motifs'].append(tuple(group))
            all_motifs_info['smiles'].append(_smiles)
            mask = self.tokenizer.vocab_array == _smiles
            if mask.sum()>0:
                motif_id = mask.nonzero()[0][0] + 1
            else:
                motif_id = 0
                
            all_motifs_info['motif_id'].append(motif_id)
            all_motifs_info['smiles_with_mapping'].append(smiles_with_mapping)
            # <= 这段代码是用来找这个motif与其他的motif的连接点的

        return all_motifs_info, am2idx, idx2am