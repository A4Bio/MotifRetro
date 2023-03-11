

#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys; sys.path.append("/gaozhangyang/experiments/MotifRetro")
import json
# from copy import copy
import copy
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from src.utils.chem_utils import smi2mol, mol2smi, get_submol, cnt_atom, revise_mol_with_refsmi
import networkx as nx
from networkx.readwrite import json_graph
# from molvs import standardize_smiles

# 还可以参考这个寻找fragment: https://www.rdkit.org/docs/source/rdkit.Chem.Fraggle.FraggleSim.html


get_ams = lambda mol: [atom.GetAtomMapNum() for atom in mol.GetAtoms()]

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def fix_incomplete_mappings(sub_mol) :
    max_map = max(a.GetAtomMapNum() for a in sub_mol.GetAtoms())

    for a in sub_mol.GetAtoms():
        map_num = a.GetAtomMapNum()
        if map_num is None or map_num < 1:
            max_map += 1
            a.SetAtomMapNum(max_map)
    return sub_mol

def fix_incomplete_mappings_with_idx(sub_mol) :

    for idx, a in enumerate(sub_mol.GetAtoms()):
        a.SetAtomMapNum(idx)
    return sub_mol

'''classes below are used for principal subgraph extraction'''

class MolInSubgraph:
    def __init__(self, mol, raw_smi=None, source_mol=None, weight = 1, kekulize=False, keep_ring=False, is_training=True):
        mol = mol_with_atom_index(mol)
            
        self.mol = mol
        self.source_mol = source_mol
        # self.smi = mol2smi(mol)
        self.smi = raw_smi
        self.weight = weight
        self.kekulize = kekulize
        # 需要给subgraph加上attach属性, 在统计频率的时候考虑attach属性
        ## -------------------subgraphs
        # element: {0: {0: '*'}, 1: {1: 'O'}, 2: {2: 'C'}, 3: {3: 'O'}, 4: {4: 'C'}, 5: {5: 'F'}, 6: {6: 'F'}, 7: {7: 'F'}}
        # attach: {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4}
        # smi: {0: '*', 1: 'O', 2: 'C', 3: 'O', 4: 'C', 5: 'F', 6: 'F', 7: 'F'}
        self.subgraphs, self.subgraphs_smis = {}, {}  
        # self.subgraphs_attach = {}
        self.subgraphs_isring = {}
        self.is_training = is_training
        
        
        for atom in self.mol.GetAtoms():
            if atom.GetSymbol() == "*":
                self.dummy_idx = atom.GetIdx() # 只有一个dummy atom
            
        # self.Dist = GetDistanceMatrix(mol)

        if not keep_ring:
            for atom in mol.GetAtoms():
                idx, symbol = atom.GetIdx(), atom.GetSymbol()
                self.subgraphs[idx] = { idx: symbol }
                self.subgraphs_smis[idx] = symbol
        else: 
            #---------------------我们加入的代码, 以环为基本单元，不可能开环
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()
            
            # 合并具有公共边的环
            seen = np.zeros(len(rings))
            assign_matrix = np.eye(len(rings))
            for idx1, ring1 in enumerate(rings):
                seen[idx1] = 1
                for idx2, ring2 in enumerate(rings):
                    if (idx2>idx1) and len(set(ring1)&set(ring2))>0 and (not seen[idx2]):
                        assign_matrix[idx1, idx2] = 1
                        assign_matrix[idx2, idx2] = 0
                        seen[idx2]=1
                        
            if assign_matrix.sum()>0:
                ring_merge_edge = []
                for idx in range(assign_matrix.shape[0]):
                    if assign_matrix[idx].sum()>0:
                        new_ring = []
                        merge_idx = assign_matrix[idx].nonzero()[0].tolist()
                        for idx2 in merge_idx:
                            new_ring.extend(rings[idx2])
                        ring_merge_edge.append(list(set(new_ring)))
                rings = ring_merge_edge
            
            covered_atoms = []
            motif_ind = mol.GetNumAtoms() + 1 + 1
            for ring in rings:
                subgraph = {}
                for idx in ring:
                    atom = mol.GetAtomWithIdx(idx)
                    symbol = atom.GetSymbol()
                    subgraph[idx] = symbol
                self.subgraphs[motif_ind] = subgraph
                
                submol = get_submol(mol, ring)
                for atom in submol.GetAtoms():
                    atom.SetAtomMapNum(0)
                
                covered_atoms.extend(list(ring))
                self.subgraphs_smis[motif_ind] = mol2smi(submol)
                
                self.subgraphs_isring[motif_ind] = True

                
                
                motif_ind += 1
            
            for atom in mol.GetAtoms():
                idx, symbol = atom.GetIdx(), atom.GetSymbol()
                if idx in covered_atoms:
                    continue
                self.subgraphs[idx] = { idx: symbol }
                self.subgraphs_smis[idx] = symbol
                
                self.subgraphs_isring[motif_ind] = False
            #--------------------------------
            
        
        
        self.inversed_index = {} # assign atom idx to pid
        self.idx2am = {}
        # self.upid_cnt = len(self.subgraphs)
        self.upid_cnt = max(self.subgraphs.keys())+1 # 用于当做最新motif的索引
        for aid, atom in enumerate(mol.GetAtoms()):
            self.idx2am[aid] = atom.GetAtomMapNum()
            for key in self.subgraphs:
                subgraph = self.subgraphs[key]
                if aid in subgraph:
                    self.inversed_index[aid] = key
        self.dirty = True
        self.smi2pids = {} # private variable, record neighboring graphs and their pids

    def rm_dummy_charge(self, mol):
        for atom in mol.GetAtoms():
            if atom.GetSymbol()=="*":
                # print(atom.GetFormalCharge())
                atom.SetFormalCharge(0)
        return mol
    
    def get_submol_with_dummy(self, raw_mol, atom_indices, kekulize=False, dummy=True):
        mol = copy.deepcopy(raw_mol)
        src, nei = self.get_neighbor(mol, atom_indices)
        
        if dummy:
            for idx in nei:
                dummy_atom = mol.GetAtomWithIdx(idx)
                dummy_atom.SetAtomicNum(0)
                dummy_atom.SetFormalCharge(0)
                atom_indices = atom_indices+[idx]
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
        mol = Chem.PathToSubmol(mol, edge_indices)# 这里子图中每个原子的Hs与原分子是相同的, 但通过mol2 = smi2mol(mol2smi(mol))转化后, rdkit在mol2smi这一步会自动补H，导致mol2原子的Hs与原分子mol不一致
        
        for atom in mol.GetAtoms():# 清楚dummy原子上的电荷
            if atom.GetSymbol()=="*":
                atom.SetFormalCharge(0)
        return mol
        
    
            
        

    def get_nei_subgraphs(self):
        '''
        根据化学键找到相邻的motif pair
        nei_subgraphs: [{aid1:"C", aid2:"O"}, {aid4:"C", aid9:"N"}], 记录按照neighborhood合并后的subgraphs
        merge_pids: [(1,2), (4,9)], 记录合并的motif pair的pid (principle graph id)
        
        在此实现我们的idea:基于source提取target motif
        '''
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
                    # if (nei_idx in subgraph) or self.subgraphs_isring.get(self.inversed_index[nei_idx]):
                    #     continue
                    if (nei_idx in subgraph):
                        continue
                    
                    local_nei_pid.append(self.inversed_index[nei_idx])
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                new_subgraph = copy.deepcopy(subgraph)
                new_subgraph.update(self.subgraphs[nei_pid])
                nei_subgraphs.append(new_subgraph)
                merge_pids.append((key, nei_pid))
        
        return nei_subgraphs, merge_pids
    
    def get_nei_smis(self):
        '''
        self.smi2pids: {"C=C": [(1,0)], "CC": [(2,0), (3,2)]}, 更新neighborhood smiles对应的motif的pid集合
        nei_smis: ["C=C", "CC", "CC", "CO"], 记录当前阶段的neighborhood smiles
        '''
        if self.dirty:
            '''初始化'''
            nei_subgraphs, merge_pids = self.get_nei_subgraphs()
            nei_smis, self.smi2pids = [], {}
            for i, subgraph in enumerate(nei_subgraphs):
                
                # # ---------环不与任何邻居合并--------
                # if self.subgraphs_isring.get(merge_pids[i][0]):
                #     continue
                # if self.subgraphs_isring.get(merge_pids[i][1]):
                #     continue
                # # ----------------------------------
                
                submol_with_dummy = self.get_submol_with_dummy(copy.deepcopy(self.mol), list(subgraph.keys()), kekulize=self.kekulize)
                # for atom in submol_with_dummy.GetAtoms():
                #     atom.SetAtomMapNum(0)
                smi = mol2smi(submol_with_dummy, rm_am=True) 
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, [])
                self.smi2pids[smi].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis


    def merge(self, smi):
        '''
        合并smi所对应的neighorhood smiles
        
        self.subgraphs: {pid1: {aid1: "C", aid2:"O}}, 记录principle motif对应的atom index和原子类型
        self.subgraphs_smis: {pid1: smi1}, 记录principle motif对应的smiles
        '''
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            merge_pids = self.smi2pids[smi]
            for pid1, pid2 in merge_pids: # 将具备smi的局部结构合并, merge_pids: [(left1, right1), (left2, right2)]
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

    def get_neighbor(self, rdmol, element):
        neighbor = []
        for bond in rdmol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if (a in element) and (b not in element):
                neighbor.append(b)
            if (b in element) and (a not in element):
                neighbor.append(a)
        return list(set(element)), list(set(neighbor))
    
    def get_motif_tree(self, len_freq=None, raw_vocab=None):
        rdmol = Chem.MolFromSmiles(self.smi)
        rdmol = fix_incomplete_mappings_with_idx(rdmol)
        
        # subgraphs = {min(subgraph.keys()) :subgraph for pid, subgraph in self.subgraphs.items()}
        subgraphs = copy.copy(self.subgraphs)
        # 构造graph
        g = nx.Graph()
            
        # ----------------防止rdkit读取smiles时自动补H和改变原子类型(c-->C)和化学键类型(C-C单键-->芳香键)
        rdmol = revise_mol_with_refsmi(rdmol, self.smi)
        # ----------------------------------------------
        
        
        g.graph['smi'] = self.smi
        g.graph['am_smi'] =  Chem.MolToSmiles(rdmol)
        g.graph['freq'] = raw_vocab[self.smi][1]
        
        merge_dummy = []
        for pid, subgraph in subgraphs.items(): # 将dummy合并到最近的motif中
            if "*" in subgraph.values() and len(subgraph)==1:
                dummy_idx = list(subgraph.keys())[0]
                src, nei = self.get_neighbor(rdmol, [dummy_idx])
                merge_pid = self.inversed_index[nei[0]]
                merge_dummy.append((pid, merge_pid, dummy_idx))
        
        for (pid, merge_pid, dummy_idx) in merge_dummy:
            subgraphs[merge_pid].update({dummy_idx:"*"})
            del subgraphs[pid]
            
        for pid, subgraph in subgraphs.items():
            if "*" in subgraph.values():
                start = pid
                g.graph['root'] = start
                break
        
        # 添加motif节点
        for pid, subgraph in subgraphs.items():
            src, nei = self.get_neighbor(rdmol, list(subgraph.keys()))
            if len(src)==1 and rdmol.GetAtomWithIdx(src[0]).GetSymbol()=="*":
                g.add_node(pid, element=src, neighbor = nei, am_smi="", smi="")
            else:
                g.add_node(pid, element=src, neighbor = nei)
            
        # 添加motif之间的edge
        for node1 in g.nodes():
            for node2 in g.nodes():
                if (node1<node2) and len(set(g.nodes[node1]['element'])&set(g.nodes[node2]['neighbor']))>0:
                    attach1 = list(set(g.nodes[node1]['element'])&set(g.nodes[node2]['neighbor']))
                    attach2 = list(set(g.nodes[node1]['neighbor'])&set(g.nodes[node2]['element']))
                    g.add_edge(node1, node2, attach={node1:attach1, node2:attach2})
        
        
        # 按照树的顺序遍历graph, 为每个节点生成adding_motif
        adding_motif_dict = {}
        T = nx.dfs_tree(g, source=start)
        if len(T.edges())>0:
            for t, edge in enumerate(T.edges()):
                prev, now = edge
                if t==0: #第一个节点自带dummy原子,因此不需要添加dummy
                    node_ids = g.nodes[prev]['element']
                    adding_motif = self.get_submol_with_dummy(rdmol, node_ids, dummy=False)
                    adding_motif = self.rm_dummy_charge(adding_motif)
                    am_smi = Chem.MolToSmiles(adding_motif)
                    smi = mol2smi(smi2mol(am_smi), rm_am=True)
                    g.nodes[prev]['am_smi'] = am_smi
                    g.nodes[prev]['smi'] = smi
                    adding_motif_dict[smi] = adding_motif_dict
                    
                    
                # 为now节点添加dummy原子
                
                attach = g.edges[(edge)]['attach'][prev]
                node_ids = list(set(g.nodes[now]['element'])|set(attach))
                
                adding_motif = self.get_submol_with_dummy(rdmol, node_ids, dummy=False)
                for atom in adding_motif.GetAtoms():
                    if atom.GetAtomMapNum()==list(attach)[0]:
                        atom.SetAtomicNum(0)
                adding_motif = self.rm_dummy_charge(adding_motif)
                am_smi = Chem.MolToSmiles(adding_motif)
                g.nodes[now]['am_smi'] = am_smi
                smi = mol2smi(smi2mol(am_smi), rm_am=True)
                g.nodes[now]['smi'] = smi
                adding_motif_dict[smi] = adding_motif_dict

        else:
            node_ids = list(g.nodes[start]['element'])
            smi = self.smi
            adding_motif = self.get_submol_with_dummy(rdmol, node_ids)
            adding_motif = self.rm_dummy_charge(adding_motif)
            am_smi = Chem.MolToSmiles(adding_motif)
            # smi = mol2smi(smi2mol(am_smi, sanitize=False), rm_am=True)
            g.nodes[start]['am_smi'] = am_smi
            g.nodes[start]['smi'] = smi
            adding_motif_dict[smi] = adding_motif_dict
            

        
        g = g.to_directed()
        
        g.remove_edges_from(list(set(g.edges)-set(T.edges)))
        
        # json.dump(data, open("./motif_tree.json", "w"), cls=NpEncoder, indent=4)
        return g, adding_motif_dict


def freq_cnt(mol):
    freqs = {}
    nei_smis = mol.get_nei_smis()
    for smi in nei_smis:
        freqs.setdefault(smi, 0)
        freqs[smi] += mol.weight
    return freqs, mol

def add_am_to_dummy(mol, attach):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            atom.SetAtomMapNum(attach)
    return mol

def get_dummy_am(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol()=="*":
            return atom.GetAtomMapNum()
    return None


def readout_adding_motif_trees(mols, len_freq, raw_vocab, meta_set):
    edit_paths = {}
    adding_motif_trees = {}
    adding_motif_dict = {}
    for idx, mol in enumerate(mols):

        adding_motif_tree, adding_motif_dict_ = mol.get_motif_tree(len_freq, raw_vocab)

        adding_motif_trees[mol.smi] = adding_motif_tree
        adding_motif_dict.update(adding_motif_dict_)
            
    
    # 合并meta 集合
    for raw_smi in meta_set:
        if raw_smi not in adding_motif_trees:
            motif_mol = Chem.MolFromSmiles(raw_smi)
            motif_mol = fix_incomplete_mappings_with_idx(motif_mol)
            raw_am_smi = mol2smi(motif_mol)
            
            g = nx.DiGraph()
            g.graph['root'] = 0
            g.graph['smi'] = raw_smi
            g.graph['am_smi'] = raw_am_smi
            g.graph['freq'] = raw_vocab[raw_smi][1]
            g.add_node(0, element = list(get_ams(motif_mol)), neighbor=[], am_smi=raw_am_smi, smi=raw_smi)

            adding_motif_trees[raw_smi] = g
            # adding_motif_dict[smi] = motif_mol
    
    # 更新最终的freq信息
    final_freq = {}
    for smi, tree in adding_motif_trees.items():
        freq = tree.graph['freq']
        for idx in tree.nodes():
            node = tree.nodes[idx]
  
            motif_smi = node['smi']

            if final_freq.get(motif_smi) is not None:
                final_freq[motif_smi] += freq
            else:
                final_freq[motif_smi] = freq
    
    # 向adding_motif_tree写入freq信息
    for smi, tree in adding_motif_trees.items():
        for idx in tree.nodes():
            node = tree.nodes[idx]
            node["freq"] = final_freq[node["smi"]]
    return adding_motif_trees


def retro_bpe(fname, vocab_len, vocab_path, cpus, kekulize, is_training=False):
    # load molecules
    print(f'Loading mols from {fname} ...')
    data = json.load(open(fname, "r"))
    if "" in data:
        del data[""]
    if "N_all" in data:
        del data["N_all"]
    if "N_invalid" in data:
        del data["N_invalid"]
        
    smis = list(data.keys())
    weight = list(data.values())
    
    meta_set = {}
    selected_smis = []
    details = {}
    raw_vocab = {}
    for smi, freq in zip(smis, weight):
        if Chem.MolFromSmiles(smi).GetNumAtoms()<5:
            selected_smis.append(smi)
            details[smi] = [cnt_atom(smi), freq]
            meta_set[smi] = [cnt_atom(smi), freq]
        
        raw_vocab[smi] = [cnt_atom(smi), freq]
    
    mols = [MolInSubgraph(smi2mol(smi, kekulize), smi, kekulize, weight = weight[i], keep_ring=True, is_training=is_training) for i, smi in enumerate(smis) if smi not in selected_smis]
    
    
    # bpe process
    add_len = vocab_len - len(selected_smis)
    pool = mp.Pool(cpus)
    for _ in tqdm(range(add_len)):
        # 得到单步合并后的所有smi并统计频率
        # res_list = pool.map(freq_cnt, mols)  # each element is (freq, mol) (because mol will not be synced...)
        res_list = []
        for mol in mols:
            res_list.append(freq_cnt(mol))
        
        freqs, mols = {}, []
        for freq, mol in res_list:
            mols.append(mol)
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]
        # 寻找最频繁结构, find the subgraph to merge
        max_cnt, merge_smi = 0, ''
        for smi in freqs:
            cnt = freqs[smi]
            # if (cnt > max_cnt) and (cnt_atom(smi)>3):
            if (cnt > max_cnt):
                max_cnt = cnt
                merge_smi = smi
        
        if max_cnt<99:
            adding_motif_trees = readout_adding_motif_trees(mols, details, raw_vocab, meta_set)
            break
        
        # 合并最频繁结构, merge
        for idx, mol in enumerate(mols):
            # if idx==100:
            #     print()
            mol.merge(merge_smi)
        
        if merge_smi in selected_smis:
            details[merge_smi][1] += max_cnt
        else:
            selected_smis.append(merge_smi)
            details[merge_smi] = [cnt_atom(merge_smi), max_cnt]
    print('sorting vocab by atom num')
    pool.close()
    selected_smis.sort(key=lambda x: details[x][1], reverse=True)
    

    sv_trees = {}
    for smi, tree in adding_motif_trees.items():
        data = json_graph.tree_data(tree, root=tree.graph['root'])
        # data = json_graph.node_link_data(tree)
        # json_graph.node_link_data(dictionary)
        sv_trees[smi] = {"graph": tree.graph, "tree":data}
    
    with open("/".join(vocab_path.split("/")[:-1] + ["adding_motif_trees.json"]), "w") as f:
        json.dump(sv_trees, f, indent=5)
    
    
        
    new_vocab = {}
    for smi, tree in adding_motif_trees.items():
        for idx in tree.nodes():
            node = tree.nodes[idx]
            new_vocab[node["smi"]] = node["freq"]
                    
    
    new_vocab = sorted(new_vocab.items(),key = lambda x:x[1],reverse = True)
    new_vocab = {key:val for key, val in new_vocab}
    with open("/".join(vocab_path.split("/")[:-1] + ["frag_decompsed.json"]), "w") as f:
        json.dump(new_vocab, f, indent=4)
    return selected_smis, details




def parse():
    parser = argparse.ArgumentParser(description='Principal subgraph extraction motivated by bpe')
    parser.add_argument('--data', type=str, default="/gaozhangyang/experiments/MotifRetro/data/uspto_50k/frag_smiles_count.json", help='Path to molecule corpus')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Length of vocab')
    parser.add_argument('--output', type=str, default="/gaozhangyang/experiments/MotifRetro/data/uspto_50k/frag_decompsed.json", help='Path to save vocab')
    parser.add_argument('--workers', type=int, default=16, help='Number of cpus to use')
    parser.add_argument('--kekulize', action='store_true', help='Whether to kekulize the molecules (i.e. replace aromatic bonds with alternating single and double bonds)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    retro_bpe(args.data, vocab_len=args.vocab_size, vocab_path=args.output,
              cpus=args.workers, kekulize=args.kekulize)
    print("Finished!")
