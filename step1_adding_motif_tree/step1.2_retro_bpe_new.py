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
from src.utils.chem_utils import smi2mol, mol2smi, cnt_atom, revise_mol_with_refsmi
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
    def __init__(self, rdmol, raw_smi=None, source_mol=None, weight = 1, kekulize=False, keep_ring=False, is_training=True):
        rdmol = mol_with_atom_index(rdmol)
        if raw_smi=='*Cc1ccccc1':
            print()
        self.rdmol = rdmol
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
        self.subgraphs = {}  
        for atom in self.rdmol.GetAtoms():
            if atom.GetSymbol() == "*":
                self.dummy_idx = atom.GetIdx() # 只有一个dummy atom
            
        # self.Dist = GetDistanceMatrix(mol)

        if not keep_ring:
            for atom in self.rdmol.GetAtoms():
                idx, symbol = atom.GetIdx(), atom.GetSymbol()
                self.subgraphs[idx] = {"element":{ idx: symbol },
                                       "smi":symbol,
                                       "finished":False,
                                       "isring":False}
        else: 
            #---------------------我们加入的代码, 以环为基本单元，不可能开环
            ring_info = self.rdmol.GetRingInfo()
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
            motif_ind = self.rdmol.GetNumAtoms() + 1 + 1
            for ring in rings:
                element = {}
                for idx in ring:
                    atom = self.rdmol.GetAtomWithIdx(idx)
                    symbol = atom.GetSymbol()
                    element[idx] = symbol
                    covered_atoms.append(idx)
                
                # submol = get_submol(self.rdmol, ring)
                submol = self.get_submol_with_dummy(self.rdmol, ring, dummy=False)
                    
                self.subgraphs[motif_ind] = {"element":element,
                                       "smi":mol2smi(submol, rm_am=True),
                                       "finished":False,
                                       "isring":True}
                motif_ind += 1
            
            for atom in self.rdmol.GetAtoms():
                idx, symbol = atom.GetIdx(), atom.GetSymbol()
                if idx in covered_atoms:
                    continue
                self.subgraphs[idx] = {"element":{ idx: symbol },
                                       "smi":symbol,
                                       "finished":False,
                                       "isring":False}
            #--------------------------------
            
        
        
        self.inversed_index = {} # assign atom idx to pid
        self.upid_cnt = max(self.subgraphs.keys())+1 # 用于当做最新motif的索引
        for aid, atom in enumerate(self.rdmol.GetAtoms()):
            for key in self.subgraphs:
                record = self.subgraphs[key]
                if aid in record['element']:
                    self.inversed_index[aid] = key


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

    
    def update_nei_contextual_smis(self):
        '''
        nei_context_graphs记录所有可能的neighbring motif的合并情况
        '''
        nei_context_graphs, merge_pids = [], []
        for pid, record in self.subgraphs.items():
            subgraph = record['element']
            local_nei_pid = []
            for aid in subgraph.keys():
                atom = self.rdmol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if nei_idx in subgraph or nei_idx > aid: 
                        continue
                    
                    local_nei_pid.append(self.inversed_index[nei_idx])
            local_nei_pid = set(local_nei_pid)
            
            # 遍历所有邻居, 分别合并, 每一次合并产生一条new_record, 
            for nei_pid in local_nei_pid:
                new_record = copy.deepcopy(record)
                new_record['element'].update(self.subgraphs[nei_pid]['element'])
                submol = self.get_submol_with_dummy(self.rdmol, list(new_record['element'].keys()))
                new_record['smi'] = mol2smi(submol, rm_am=True)
                new_record['merge_pid'] = (pid, nei_pid)
                nei_context_graphs.append(new_record)
        
        self.nei_context_graphs = nei_context_graphs
        return nei_context_graphs
    
    def update_self_contextual_smis(self):
        '''
        self_context_graphs记录atom element及其一阶邻居边构成的子图
        
        # element: 原子集合
        # smi: 包含一阶邻居边信息的smiles
        # finished: 当前motif是否已经finish
        # isring: 是否是环
        # merge_pid: 当前motif包括的pid集合
        '''
        self_context_graphs = []
        for pid, record in self.subgraphs.items():  
            subgraph = record['element']
            submol_with_dummy = self.get_submol_with_dummy(copy.deepcopy(self.rdmol), list(subgraph.keys()), kekulize=self.kekulize)
            nn_smi = mol2smi(submol_with_dummy, rm_am=True)
            if nn_smi=="**" or nn_smi=="":
                continue
            new_record = copy.deepcopy(record)
            new_record['smi'] = nn_smi
            new_record['merge_pid'] = (pid,)
            self_context_graphs.append(new_record)
        self.self_context_graphs = self_context_graphs
        return self_context_graphs

    def merge_self_subgraph(self, smi, finished=True):
        '''
        合并smi所对应的neighorhood smiles
        
        self.subgraphs: {pid1: {aid1: "C", aid2:"O}}, 记录principle motif对应的atom index和原子类型
        '''
        
        for record in self.self_context_graphs:
            if record['smi'] == smi:
                pid = record['merge_pid'][0]
                self.subgraphs[pid]['finished'] = finished
                
        
        

    def merge_nei_subgraph(self, merge_smi, finished=False):
        '''
        合并smi所对应的neighorhood smiles
        
        self.subgraphs: {pid1: {aid1: "C", aid2:"O}}, 记录principle motif对应的atom index和原子类型
        '''
        for record in self.nei_context_graphs:
            (pid1, pid2) = record['merge_pid']
            if pid1 in self.subgraphs and pid2 in self.subgraphs: # possibly del by former
                if record['smi'] == merge_smi:
                    self.subgraphs[pid1]['element'].update(self.subgraphs[pid2]['element'])
                    element = self.subgraphs[pid1]['element'] 
                    submol = self.get_submol_with_dummy(self.rdmol, list(element.keys()), dummy=False)
                    self.subgraphs[self.upid_cnt] = {"element":element,
                                                    "smi":mol2smi(submol, rm_am=True),
                                                    "finished":finished,
                                                    "isring":False}
                    
                    for aid in self.subgraphs[pid2]['element']:
                        self.inversed_index[aid] = pid1
                    for aid in self.subgraphs[pid1]['element']:
                        self.inversed_index[aid] = self.upid_cnt
                    del self.subgraphs[pid1]
                    del self.subgraphs[pid2]
                    self.upid_cnt += 1
        


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
    
    def get_motif_tree(self, bpe_results, raw_vocab):
        rdmol = Chem.MolFromSmiles(self.smi)
        rdmol = fix_incomplete_mappings_with_idx(rdmol)
        
        subgraphs = copy.copy(self.subgraphs)
        # 构造graph
        g = nx.Graph()
            
        # ----------------防止rdkit读取smiles时自动补H和改变原子类型(c-->C)和化学键类型(C-C单键-->芳香键)
        rdmol = revise_mol_with_refsmi(rdmol, self.smi)
        # ----------------------------------------------
        
        
        g.graph['smi'] = self.smi
        g.graph['am_smi'] =  Chem.MolToSmiles(rdmol)
        g.graph['freq'] = raw_vocab[self.smi]['raw_freq']
        
        merge_dummy = []
        for pid, record in subgraphs.items(): # 将dummy合并到最近的motif中
            subgraph = record['element']
            if "*" in subgraph.values() and len(subgraph)==1:
                dummy_idx = list(subgraph.keys())[0]
                src, nei = self.get_neighbor(rdmol, [dummy_idx])
                merge_pid = self.inversed_index[nei[0]]
                merge_dummy.append((pid, merge_pid, dummy_idx))
        
        for (pid, merge_pid, dummy_idx) in merge_dummy:
            subgraphs[merge_pid]['element'].update({dummy_idx:"*"})
            element = subgraphs[merge_pid]['element']
            submol = self.get_submol_with_dummy(self.rdmol, list(element.keys()), dummy=False)
            subgraphs[merge_pid]['smi'] = mol2smi(submol, rm_am=True)
            del subgraphs[pid]
            
        for pid, record in subgraphs.items():
            subgraph = record['element']
            if "*" in subgraph.values():
                start = pid
                g.graph['root'] = start
                break
        
        # 添加motif节点
        for pid, record in subgraphs.items():
            subgraph = record['element']
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
                try:
                    smi = mol2smi(smi2mol(am_smi), rm_am=True)
                except:
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


def freq_cnt_nei_smi(mol):
    freqs = {}
    # nei_smis = mol.get_nei_smis()
    for record in mol.nei_context_graphs:
        smi = record['smi']
        freqs.setdefault(smi, 0)
        freqs[smi] += mol.weight
    return freqs, mol

def freq_cnt_self_contextual_smi(mol):
    freqs = {}
    # self_smis = mol.get_self_smis()
    for record in mol.self_context_graphs:
        smi = record['smi']
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


def readout_adding_motif_trees(mols, bpe_results, raw_vocab):
    mols = copy.deepcopy(mols)
    bpe_results = copy.deepcopy(bpe_results)
    raw_vocab = copy.deepcopy(raw_vocab)
    adding_motif_trees = {}
    adding_motif_dict = {}
    
    # 将meta data转化为motif tree
    
    # 将剩余molecule转化为motif tree
    for idx, mol in enumerate(mols):

        adding_motif_tree, adding_motif_dict_ = mol.get_motif_tree(bpe_results, raw_vocab)

        adding_motif_trees[mol.smi] = adding_motif_tree
        adding_motif_dict.update(adding_motif_dict_)
    
    for raw_smi in raw_vocab.keys():
        if raw_smi not in adding_motif_trees:
            motif_mol = Chem.MolFromSmiles(raw_smi)
            motif_mol = fix_incomplete_mappings_with_idx(motif_mol)
            raw_am_smi = mol2smi(motif_mol)
            g = nx.DiGraph()
            g.graph['root'] = 0
            g.graph['smi'] = raw_smi
            g.graph['am_smi'] = raw_am_smi
            g.graph['freq'] = raw_vocab[raw_smi]['freq']
            g.add_node(0, element = list(get_ams(motif_mol)), neighbor=[], am_smi=raw_am_smi, smi=raw_smi)
            adding_motif_trees[raw_smi] = g
    
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


class RetroBPE:
    def __init__(self, fname):
        # load molecules
        print(f'Loading mols from {fname} ...')
        data = json.load(open(fname, "r"))
        if "" in data:
            del data[""]
        if "N_all" in data:
            del data["N_all"]
        if "N_invalid" in data:
            del data["N_invalid"]
            
        self.smis = list(data.keys())
        self.weight = list(data.values())
    
    def get_meta_motifs(self, max_num_atoms=3):
        selected_smis = []
        bpe_results = {}
        raw_vocab = {}
        for smi, freq in zip(self.smis, self.weight):
            if Chem.MolFromSmiles(smi).GetNumAtoms()<max_num_atoms:
                selected_smis.append(smi)
                bpe_results[smi] = {"num_atoms": cnt_atom(smi),
                                  "raw_freq": freq,
                                  "freq": freq}
            raw_vocab[smi] = {"num_atoms": cnt_atom(smi),
                                  "raw_freq": freq,
                                  "freq": freq}
        return bpe_results, raw_vocab
    
    def statistic_self_smi(self, mols, pool):
        res_list = pool.map(freq_cnt_self_contextual_smi, mols)
        
        # res_list = []
        # for mol in mols:
        #     res_list.append(freq_cnt_self_contextual_smi(mol))
        
        freqs = {}
        for freq, mol in res_list:
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]
        return freqs
    
    
    
    def statistic_neighbor_smi(self, mols, pool):
        # res_list = pool.map(freq_cnt_nei_smi, mols)
        
        res_list = []
        for mol in mols:
            res_list.append(freq_cnt_nei_smi(mol))
        
        smi_freqs = {}
        for smi_freq, mol in res_list:
            for key in smi_freq:
                smi_freqs.setdefault(key, 0)
                smi_freqs[key] += smi_freq[key]
        return smi_freqs
    
    def find_merge_smi(self, freqs_table, bpe_results):
        smi_freqs = sorted(freqs_table.items(), key=lambda x: x[1], reverse=True)
        
        FINISHED = False
        for smi, freq in smi_freqs:
            if smi in bpe_results:
                FINISHED = True
                return smi, freq, FINISHED
        
        return smi_freqs[0][0], smi_freqs[0][1], FINISHED
    
    
    def merge_meta_smis(self, mols, pool, bpe_results):
        self_freqs_table = self.statistic_self_smi(mols,pool)
        merge_smi_freq_list = []
        for smi, freq in self_freqs_table.items():
            if smi in bpe_results: # TODO: add constraints
                merge_smi_freq_list.append((smi, freq))
                bpe_results[smi]['freq'] += freq
        
        for merge_smi, freq in merge_smi_freq_list:
            for mol in mols:
                mol.merge_self_subgraph(merge_smi)
        return bpe_results

    def scoring(self, motif_trees):
        N = 0
        freq = 0
        atom_num = 0
        for smi, tree in motif_trees.items():
            T = nx.bfs_tree(tree, source=tree.graph['root'])
            for node_idx in T.nodes:
                node = tree.nodes[node_idx]
                freq += node['freq']
                mol = smi2mol(node['smi'])
                atom_num += len([atom for atom in mol.GetAtoms() if atom.GetSymbol()!="*"])
                N+=1
        
        simplicity_score = freq/atom_num
        generality_score = freq/N
        return simplicity_score, generality_score

    def save(self, adding_motif_trees, vocab_path, name):
        sv_trees = {}
        for smi, tree in adding_motif_trees.items():
            data = json_graph.tree_data(tree, root=tree.graph['root'])
            sv_trees[smi] = {"graph": tree.graph, "tree":data}
        
        with open("/".join(vocab_path.split("/")[:-1] + [f"motif_trees_{name}.json"]), "w") as f:
            json.dump(sv_trees, f, indent=5)
        
        
        new_vocab = {}
        for smi, tree in adding_motif_trees.items():
            for idx in tree.nodes():
                node = tree.nodes[idx]
                new_vocab[node["smi"]] = node["freq"]
                        
        
        new_vocab = sorted(new_vocab.items(),key = lambda x:x[1],reverse = True)
        new_vocab = {key:val for key, val in new_vocab}
        with open("/".join(vocab_path.split("/")[:-1] + [f"frag_decompsed_{name}.json"]), "w") as f:
            json.dump(new_vocab, f, indent=5)
        
    
    def retro_bpe(self, vocab_len, vocab_path, cpus, kekulize, is_training=False):
        motif_forest = []
        ##------------------- step1.1: recored meta motifs
        bpe_results, raw_vocab = self.get_meta_motifs(max_num_atoms=3)
        mols = []
        
        ##------------------- step1.2: merge meta motifs
        mols = [MolInSubgraph(smi2mol(smi, kekulize), smi, kekulize, weight = self.weight[i], keep_ring=True, is_training=is_training) for i, smi in enumerate(self.smis) if smi not in bpe_results]
        
        # bpe process
        add_len = vocab_len - len(bpe_results)
        pool = mp.Pool(cpus)
        
        for mol in mols:
            mol.update_self_contextual_smis()
        bpe_results = self.merge_meta_smis(mols, pool, bpe_results)
        base_motif_trees = readout_adding_motif_trees(mols, bpe_results, raw_vocab)
        simplicity_score0, generality_score0 = self.scoring(base_motif_trees)
        motif_forest.append((simplicity_score0, generality_score0,base_motif_trees))
            
        simplicity_score_list = [simplicity_score0]
        generality_score_list = [generality_score0]
        for step in tqdm(range(add_len)):
            # 得到单步合并后的所有可能的neighboring smi并统计频率
            for mol in mols:
                mol.update_nei_contextual_smis()
                
            freqs_table = self.statistic_neighbor_smi(mols,pool)
            
            # 寻找最频繁结构, find the subgraph to merge
            merge_smi, max_cnt, FINISHED = self.find_merge_smi(freqs_table, bpe_results)
            
            # print(len(freqs_table), max_cnt)
            if max_cnt<5:
                break
            
            # 合并最频繁结构, merge
            for idx, mol in enumerate(mols):
                mol.merge_nei_subgraph(merge_smi, FINISHED)
            
            if merge_smi in bpe_results:
                bpe_results[merge_smi]["freq"] += max_cnt
            else:
                bpe_results[merge_smi] = {"num_atoms": cnt_atom(merge_smi),
                                  "raw_freq": 0,
                                  "freq": max_cnt}
            
            motif_trees = readout_adding_motif_trees(mols, bpe_results, raw_vocab)
            simplicity_score, generality_score = self.scoring(motif_trees)
            motif_forest.append((simplicity_score, generality_score,motif_trees))
            simplicity_score_list.append(simplicity_score)
            generality_score_list.append(generality_score)
            
                
        print('sorting vocab by atom num')
        pool.close()
        
        simplicity_scores = np.array(simplicity_score_list)
        generality_scores = np.array(generality_score_list)
        
        
        simplicity_scores = simplicity_scores[0]/simplicity_scores
        simplicity_scores = simplicity_scores/simplicity_scores.max()
        generality_scores = generality_scores/generality_scores[0]
        
        for i, (_, _, motif_trees) in enumerate(motif_forest):
            simplicity_score = simplicity_scores[i]
            generality_score = generality_scores[i]
            self.save(motif_trees, vocab_path, f"{simplicity_score:.4f}_{generality_score:.4f}")
        
        

        




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
    BPE = RetroBPE(args.data)
    BPE.retro_bpe(vocab_len=args.vocab_size, vocab_path=args.output,
              cpus=args.workers, kekulize=args.kekulize)
    print("Finished!")

