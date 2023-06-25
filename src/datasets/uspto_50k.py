# -*- coding: utf-8 -*-
"""
A dataset containing 50k reactions of 10 types from USPTO data. It is commonly used in papers.
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.datasets import Dataset
from src.utils.feat_utils import fix_incomplete_mappings
from src.utils.chem_utils import smi2mol, mol2smi
from torch.utils.data import Dataset as Dataset_torch
import torch
from src.feat.megan_graph import MeganTrainingSamplesFeaturizer
from src.split.basic_splits import DefaultSplit
import json
import ast
from networkx import json_graph
N_JOBS = 128
DEFAULT_SPLIT = DefaultSplit()

logger = logging.getLogger(__name__)

def lists_to_tuple(x):
    if type(x) != list:
        return x
    elif len(x) == 0:
        return ()
    elif len(x) == 1:
        return (lists_to_tuple(x[0]), )

    a, b = x[0], x[1:]
    return (lists_to_tuple(a),) + lists_to_tuple(b)

REACTION_TYPES = {
    1: 'heteroatom alkylation and arylation',
    2: 'acylation and related processes',
    3: 'C-C bond formation',
    4: 'heterocycle formation',
    5: 'protections',
    6: 'deprotections',
    7: 'reductions',
    8: 'oxidations',
    9: 'functional group interconversion (FGI)',
    10: 'functional group addition (FGA)'
}


class Uspto50k(Dataset):
    def __init__(self, DATA_DIR):
        super().__init__(DATA_DIR)
        self.raw_data_path = os.path.join(self.feat_dir, 'data_processed.csv')

    @property
    def meta_info(self) -> dict:
        return {'reaction_types': REACTION_TYPES, 'max_n_nodes': 100}

    @property
    def key(self) -> str:
        return 'uspto_50k'

    def acquire(self):
        x = {
            'product': [],
            'substrates': [],
            'reaction_type_id':[],
            'id': [],
            'train': [],
            'valid': [],
            'test': []
        }
        # meta = {
        #     'reaction_type_id': [],
        #     'id': []
        # }

        for split_key, filename in (('train', 'raw_train.csv'), ('valid', 'raw_val.csv'), ('test', 'raw_test.csv')):
            data_path = os.path.join(self.DATA_DIR, f'uspto_50k/{filename}')
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f'File not found at: {data_path}. Please download data manually from '
                    'https://www.dropbox.com/sh/6ideflxcakrak10/AAAESdZq7Y0aNGWQmqCEMlcza/typed_schneider50k '
                    'and extract to the required location.')
            data_df = pd.read_csv(data_path)

            for reaction_smiles, reaction_type_id, id in tqdm(zip(data_df['reactants>reagents>production'], data_df['class'], data_df['id']), total=len(data_df),desc="generating product/substrates"):
                subs, prod = tuple(reaction_smiles.split('>>'))
                subs, prod = smi2mol(subs), smi2mol(prod)
                if subs is None or prod is None:
                    continue
                subs, prod = fix_incomplete_mappings(subs, prod)
                subs, prod = mol2smi(subs), mol2smi(prod)
                x['substrates'].append(subs)
                x['product'].append(prod)
                x['reaction_type_id'].append(reaction_type_id)
                x['id'].append(id)
                
                for split_key2 in ['train', 'valid', 'test']:
                    if split_key == split_key2:
                        x[split_key2].append(1)
                    else:
                        x[split_key2].append(0)

        logger.info(f"Saving 'x' to {self.x_path}")
        pd.DataFrame(x).to_csv(self.x_path, sep='\t')

class Uspto50k_torch(Dataset_torch):
    def __init__(
        self,
        args,
        data_dir, 
        featurizer_key, 
        keep_action,
        use_reaction_type,
        vocab_path,
        mode ="train"
    ) -> None:
        super().__init__()
        self.args = args
        self.data_dir = data_dir
        self.featurizer_key = featurizer_key
        self.vocab_path = vocab_path
        
        
        self.use_reaction_type = use_reaction_type
        self.dataset, self.featurizer, self.feat_vocab, self.action_vocab = self._preprocess()
        self.data = self.featurizer.load(self.dataset.feat_dir)
        self.sampleind2action = {ind:(am1, am2, n_node, action) for ind, am1, am2, n_node, action in self.data['action_tuples']}
        
        self.metadata = self.data['reaction_metadata']

        if mode == 'train':
            select_mask= self.metadata['is_train'] == 1
        if mode == 'valid':
            select_mask = self.metadata['is_train'] == 0
            
        self.select_reaction_ind = self.metadata.index[select_mask]
       

        self.cache = {}
    
    def read_action_vocab(self, path):
        action_vocab = json.load(open(path,"r"))
        action_vocab["action_freq"] = {ast.literal_eval(key):val for key, val in action_vocab["action_freq"].items()}
        
        action_vocab["atom_action2ind"] = {ast.literal_eval(key):val for key, val in action_vocab["atom_action2ind"].items()}
        action_vocab["atom_action2ind"][("stop", None)] = len(action_vocab["atom_action2ind"])
        
        action_vocab["bond_action2ind"] = {ast.literal_eval(key):val for key, val in action_vocab["bond_action2ind"].items()}
        
        action_vocab['atom_ind2action'] = {val:key for key, val in action_vocab['atom_action2ind'].items()}
        action_vocab['bond_ind2action'] = {val:key for key, val in action_vocab['bond_action2ind'].items()}
        
        # action_vocab["graph_action2ind"] = {("stop", None):1}


        action_vocab["n_atom_actions"] = len(action_vocab['atom_action2ind'])
        action_vocab["n_bond_actions"] = len(action_vocab['bond_action2ind'])
        # action_vocab["n_graph_actions"] = len(action_vocab['graph_action2ind'])
        
        
        
        action_vocab['n_target_actions'] = max([action_vocab["n_atom_actions"], action_vocab["n_bond_actions"]])
        
        
        action_vocab["action2ind"] = {key:idx for idx, key in enumerate(action_vocab["action_freq"].keys())}
        action_vocab["ind2action"] = {val:key for key,val in action_vocab["action2ind"].items()}
        return action_vocab
    
    def _preprocess(self):
        data_dir = self.data_dir
        featurizer_key = self.featurizer_key 

        dataset = Uspto50k(data_dir)
        if not os.path.exists(dataset.x_path):
            dataset.acquire()

        DEFAULT_SPLIT = DefaultSplit()
        
        featurizer = MeganTrainingSamplesFeaturizer(n_jobs=120, max_n_steps=16,
                                                    split=DEFAULT_SPLIT, key=featurizer_key,
                                                    action_order='bfs_randat',
                                                    # action_order='dfs_cano',
                                                    use_motif_action=1,
                                                    use_decomposed_fragment = 1,
                                                    vocab_path=self.vocab_path)

        if not os.path.exists(os.path.join(dataset.feat_dir, featurizer_key)):
            featurizer.featurize_dataset(dataset)

        feat_vocab = json.load(open(os.path.join(featurizer.dir(dataset.feat_dir), "feat_vocab.json"),"r"))
        
        for prop_name in feat_vocab['prop2oh']['atom']:
            new_dict = {}
            for key, val in feat_vocab['prop2oh']['atom'][prop_name].items():
                new_dict[int(key)] = val
            feat_vocab['prop2oh']['atom'][prop_name] = new_dict
        
        for prop_name in feat_vocab['prop2oh']['bond']:
            new_dict = {}
            for key, val in feat_vocab['prop2oh']['bond'][prop_name].items():
                if key in ["self", "supernode"]:
                    new_dict[key] = val
                elif prop_name == "bond_length":
                    new_dict[key] = float(key)
                else:
                    new_dict[int(key)] = val
                    
            feat_vocab['prop2oh']['bond'][prop_name] = new_dict
        
        action_vocab = self.read_action_vocab(os.path.join(featurizer.dir(dataset.feat_dir), "all_actions.json"))
        
        return dataset, featurizer, feat_vocab, action_vocab

            
    def __len__(self) -> int:
        return len(self.select_reaction_ind )

    def __getitem__(
        self, ind
    ):
        reaction_ind = self.select_reaction_ind[ind]
        # reaction_ind = 4652

        start_ind = self.metadata['start_ind'][reaction_ind] 
        n_steps = self.metadata['n_samples'][reaction_ind]
        source_smi = self.metadata['source_smi'][reaction_ind]
        target_smi = self.metadata['target_smi'][reaction_ind]

        reaction_type = self.metadata.loc[reaction_ind, "reaction_type"]
        sample_ind = np.arange(start_ind, start_ind + n_steps)
        sample_data = [self.sampleind2action[i] for i in sample_ind]
        
        # # # ---------------- filter attach_atom
        # sample_ind_old = np.arange(start_ind, start_ind + n_steps)

        # sample_data_old = [self.sampleind2action[i] for i in sample_ind_old]
        # sample_data = []
        # sample_ind = []
        # for i, one in enumerate(sample_data_old):
        #     if one[-1][0]!="attach_atom":
        #         sample_data.append(sample_data_old[i])
        #         sample_ind.append(sample_ind_old[i])
        # sample_ind = np.array(sample_ind)
        # n_steps = len(sample_ind)
        # # #---------------------------------
        
        if hasattr(sample_data, 'toarray'):
            sample_data = sample_data.toarray().astype(int)
        
        n_nodes = [one[2] for one in sample_data]
        n_max_steps = sample_ind.shape[0]
        n_max_nodes = int(max(n_nodes))
        
        
        
        tensor_data = {
            'atom': self.data['atom'][sample_ind],
            'bond': self.data['bond'][sample_ind],
            'max_n_nodes': n_max_nodes
        }

        
        tensor_data = self.featurizer.to_tensor_batch(tensor_data, self.feat_vocab)
        
        
        node_feats = tensor_data['atom']
        adj = tensor_data['bond']
        # is_hard_matrix = torch.from_numpy(is_hard)
        use_reaction_type = self.use_reaction_type
        if use_reaction_type:
            reaction_type = reaction_type
        else:
            reaction_type = 0
        
        node_mask = torch.sign(torch.max(node_feats, dim=-1)[0]) 
        adj_mask = torch.sign(torch.max(adj, dim=-1)[0]).unsqueeze(-1)
        atom_action_mask = torch.ones((*node_mask.shape, self.action_vocab['n_atom_actions']),  dtype=torch.float)
        bond_action_mask = torch.ones((*node_mask.shape, n_max_nodes, self.action_vocab['n_bond_actions']), dtype=torch.float)
        
        
        # supernode (first node) can only predict "stop" action
        # (and "stop" action can be predicted only by supernode)
        # (this masking is always applied)
        bond_action_mask[:, 0, :] = 0
        bond_action_mask[:, :, 0] = 0
        atom_action_mask[:, 0] = 0
        atom_action_mask[:, 0, self.action_vocab['atom_action2ind'][("stop", None)]] = 1
        atom_action_mask[:, 1:, self.action_vocab['atom_action2ind'][("stop", None)]] = 0
        
        # mask out bond actions for diagonal ('self' node)
        # mask out bond actions for upper half (matrix is symmetric)
        triu = torch.triu(torch.ones((n_max_nodes, n_max_nodes), dtype=torch.int), diagonal=1)
        bond_action_mask *= triu.reshape(1,n_max_nodes, n_max_nodes,1)
        
        # only bonds between existing atoms can be edited
        atom_exists = node_mask.unsqueeze(-1).unsqueeze(-1)
        bond_action_mask *= atom_exists
        
        unflatten_target = torch.zeros((n_max_steps, n_max_nodes + 1, n_max_nodes,  self.action_vocab['n_target_actions']), dtype=torch.float) 
        node_mask = node_mask.unsqueeze(-1)
        
        
        batch_action_ind = torch.zeros(n_max_steps, dtype=torch.long)
        
        action_tuple_list = []
        for j in range(n_steps):
            a1, a2, n_node, action_tuple = sample_data[j]
            # this_action_ind = self.action_vocab['action2ind'][action_tuple]
            action_tuple_list.append([a1, a2, action_tuple])
            
            if action_tuple[0] in ['change_atom', 'add_motif']:
                assert a2 == -1
                action_num = self.action_vocab['atom_action2ind'][action_tuple]
                unflatten_target[j, -1, a1, action_num] = 1
                batch_action_ind[j] = self.action_vocab['action2ind'][action_tuple]
            elif action_tuple[0] in ['change_bond']:
                if a2 < a1:
                    a2, a1 = a1, a2
                action_num = self.action_vocab['bond_action2ind'][action_tuple]
                unflatten_target[j, a1, a2, action_num] = 1
                
                batch_action_ind[j] = self.action_vocab['action2ind'][action_tuple]
            elif action_tuple[0] in ['attach_atom']:
                assert a1>a2
                if a2 < a1:
                    a2, a1 = a1, a2
                action_num =  self.action_vocab['bond_action2ind'][('attach_atom',None)]
                unflatten_target[j, a1, a2, action_num] = 1
                batch_action_ind[j] = self.action_vocab['action2ind'][('attach_atom',None)]
            elif action_tuple[0] in ['stop']:
                action_num =  self.action_vocab['atom_action2ind'][('stop',None)]
                unflatten_target[j, -1, 0, action_num] = 1
                batch_action_ind[j] = self.action_vocab['action2ind'][('stop',None)]
        

        result = {  
            'source_smi': source_smi,
            'target_smi': target_smi,
            'sample_data': sample_data,
            'node_features': node_feats, # [4,47,8]
            'node_mask': node_mask, # [4,47,1]
            'adj': adj, # [4,47,47,3]
            'adj_mask': adj_mask, # [4,47,47,1]
            'unflatten_target': unflatten_target, # [4,48,47,149]
            'atom_action_mask': atom_action_mask, # [4,47,149]
            'bond_action_mask': bond_action_mask, # [4,47,47,7]
            'n_bond_actions': self.action_vocab["n_bond_actions"],
            'n_steps': torch.tensor([n_steps]), # [1]
            "action_ind":batch_action_ind, # [4]
            "reaction_type": reaction_type,
            "action_tuples": action_tuple_list
        }
        
        return result
    
    

class USPTO50K_torch_test(Dataset_torch):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

        self.data = pd.read_csv(os.path.join(data_dir, "uspto_50k", "x.tsv"), index_col=0, sep="\t")
        
        split_ind = self.data['test'] == 1
        
        self.product = list(self.data['product'][split_ind])
        self.substrates = list(self.data['substrates'][split_ind])
        self.reaction_type = list(self.data['reaction_type_id'][split_ind])
        with open(os.path.join(data_dir, "uspto_50k", "adding_motif_trees.json"), "r") as f:
            all_trees = json.load(f)
        
        self.motiftrees = {}
        for key, val in all_trees.items():
            tree = json_graph.tree_graph(val['tree'])
            self.motiftrees[key] = tree
    
    def __len__(self):
        return len(self.product)
    
    def __getitem__(self, idx):
        # idx = 0
        target_smi = self.substrates[idx]
        source_smi = self.product[idx]
        reaction_type_id = self.reaction_type[idx]
        # source_smi = '[OH:1][CH2:2][c:3]1[n:4][c:5]2[c:6]([Cl:7])[n:8][cH:9][n:10][c:11]2[n:12]1[CH2:13][C:14]([CH3:15])([CH3:16])[CH3:17]'
        # target_smi = '[O:1]([CH2:2][c:3]1[n:4][c:5]2[c:6]([Cl:7])[n:8][cH:9][n:10][c:11]2[n:12]1[CH2:13][C:14]([CH3:15])([CH3:16])[CH3:17])[C:19]([CH3:18])=[O:20]'
        return {
            'source_smi': source_smi,
            'target_smi': target_smi,
            "reaction_type_id": reaction_type_id
        }
