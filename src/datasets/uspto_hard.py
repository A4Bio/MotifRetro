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
from torch.utils.data import Dataset as Dataset_torch
import torch
from src.feat.megan_graph import MeganTrainingSamplesFeaturizer, get_sample_data_path
from src.split.basic_splits import DefaultSplit

from scipy import sparse

logger = logging.getLogger(__name__)


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


class UsptoHard:
    def __init__(self, DATA_DIR):
        super().__init__()
        self.DATA_DIR = DATA_DIR
        self.raw_data_path = os.path.join(self.feat_dir, 'data_processed.csv')

    @property
    def meta_info(self) -> dict:
        return {'reaction_types': REACTION_TYPES, 'max_n_nodes': 1024}

    @property
    def key(self) -> str:
        return 'uspto_hard'

    def acquire(self):
        x = {
            'product': [],
            'substrates': []
        }
        split = {
            'train': [],
            'valid': [],
            'test': []
        }
        meta = {
            'reaction_type_id': [],
            'id': []
        }

        for split_key, filename in (('train', 'raw_train.csv'), ('valid', 'raw_val.csv'), ('test', 'raw_test.csv')):
            data_path = os.path.join(self.DATA_DIR, f'uspto_hard/{filename}')
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f'File not found at: {data_path}. Please download data manually from '
                    'https://www.dropbox.com/sh/6ideflxcakrak10/AAAESdZq7Y0aNGWQmqCEMlcza/typed_schneider50k '
                    'and extract to the required location.')
            data_df = pd.read_csv(data_path)

            for reaction_smiles in tqdm(data_df['reactants>reagents>production'], total=len(data_df),
                                        desc="generating product/substrates SMILES'"):
                subs, prod = tuple(reaction_smiles.split('>>'))
                subs, prod = fix_incomplete_mappings(subs, prod)  # preprocessing subs and prod from the raw data files.
                x['substrates'].append(subs)
                x['product'].append(prod)

            for split_key2 in ['train', 'valid', 'test']:
                if split_key == split_key2:
                    split[split_key2] += [1 for _ in range(len(data_df))]
                else:
                    split[split_key2] += [0 for _ in range(len(data_df))]

            meta['reaction_type_id'] += data_df['class'].tolist()
            meta['id'] += data_df['id'].tolist()

        logger.info(f"Saving 'x' to {self.x_path}")
        pd.DataFrame(x).to_csv(self.x_path, sep='\t')

        logger.info(f"Saving {self.metadata_path}")
        pd.DataFrame(meta).to_csv(self.metadata_path, sep='\t')

        split_path = os.path.join(self.dir, 'default_split.csv')
        logger.info(f"Saving default split to {split_path}")
        pd.DataFrame(split).to_csv(split_path)

    def load_x(self) -> pd.DataFrame:
        """
        Load X of an acquired dataset. Yields exception if dataset is not acquired.
        :return: dataset as a pandas DataFrame
        """
        return pd.read_csv(self.x_path)
    
    @property
    def feat_dir(self) -> str:
        """
        :return: directory in which featurized versions of the dataset will be stored.
        """
        return os.path.join(self.dir, 'feat')
    
    @property
    def dir(self) -> str:
        """
        :return: directory of the dataset data, which depends on DATA_DIR env variable and the key of the dataset.
        """
        return os.path.join(self.DATA_DIR, self.key)

    @property
    def x_path(self) -> str:
        """
        :return: path to a .tsv containing SMILES and/or other 'X' (e. g. changed_atoms) of the processed dataset.
        """
        return os.path.join(self.dir, 'USPTOHard.csv')

class UsptoHard_torch(Dataset_torch):
    def __init__(
        self,
        args,
        data_dir, 
        featurizer_key, 
        use_reaction_type,
        use_motif_action, 
        use_motif_feature,
        use_hierachical_action,
        mode ="train"
    ) -> None:
        super().__init__()
        self.args = args
        self.data_dir = data_dir
        self.featurizer_key = featurizer_key
        self.use_motif_action = use_motif_action
        self.use_motif_feature = use_motif_feature
        
        self.use_reaction_type = use_reaction_type
        self.use_hierachical_action = use_hierachical_action
        
        
        self.dataset, self.featurizer, self.action_vocab = self._preprocess()
        self.data = self.featurizer.load(self.dataset.feat_dir) # self.data有问题
        self.metadata = self.data['reaction_metadata']
        if mode=='train':
            select_ind = self.metadata['is_train'] == 1
        if mode=='valid':
            select_ind = self.metadata['is_train'] == 0
        self.select_ind = np.argwhere(np.array(select_ind)).flatten()
        
    
    def _preprocess(self):
        data_dir = self.data_dir
        featurizer_key = self.featurizer_key 

        dataset = UsptoHard(data_dir)
        DEFAULT_SPLIT = DefaultSplit()
        featurizer = MeganTrainingSamplesFeaturizer(n_jobs=200, max_n_steps=50,
                                                    split=DEFAULT_SPLIT, key=featurizer_key,
                                                    action_order='bfs_randat',
                                                    use_motif_action=self.use_motif_action,
                                                    use_motif_feature=self.use_motif_feature,
                                                    vocab_path="/gaozhangyang/chenxran/MotifRetro/data/uspto_hard/uspto_bpe_300.txt")
        
        if not os.path.exists(os.path.join(dataset.feat_dir, featurizer_key)):
            featurizer.featurize_dataset(dataset)
        
        # for topk in [50, 100, 150, 200, 300, 500]:
        # for topk in [50]:
        # keep_actions_list = self.truncate_action_vocab(dataset, featurizer, topk=self.args.topk)

        #     # if not os.path.exists(os.path.join(dataset.feat_dir, featurizer_key)):
        # featurizer.featurize_dataset(dataset, keep_actions_list, self.args.topk, new_key=f'-truncate-top-{self.args.topk}-solve-unparsed')
        # featurizer.featurize_dataset(dataset, keep_actions_list, self.args.topk)
        # sys.exit()
        action_vocab = featurizer.get_actions_vocabulary(dataset.feat_dir)
        
        # action_vocab['props']['atom'] = {k:v for k,v in action_vocab['props']['atom'].items() if k in self.args.atom_feature_keys}
        # action_vocab['props']['bond'] = {k:v for k,v in action_vocab['props']['atom'].items() if k in self.args.atom_feature_keys}
        
        # action_vocab['prop2oh']['atom'] = {k:v for k,v in action_vocab['prop2oh']['atom'].items() if k in self.args.atom_feature_keys}
        # action_vocab['prop2oh']['bond'] = {k:v for k,v in action_vocab['prop2oh']['bond'].items() if k in self.args.atom_feature_keys}
        
        # action_vocab['atom_feature_keys'] = self.args.atom_feature_keys
        # action_vocab['bond_feature_keys'] = self.args.bond_feature_keys
        
        # action_vocab['atom_feat_ind'] = dict((k, i) for i, k in enumerate(self.args.atom_feature_keys))
        # action_vocab['bond_feat_ind'] = dict((k, i) for i, k in enumerate(self.args.bond_feature_keys))
        
        
        return dataset, featurizer, action_vocab

    def truncate_action_vocab(self, dataset, featurizer, topk=50):
        action_vocab = featurizer.get_actions_vocabulary(dataset.feat_dir)
        sample_data = sparse.load_npz(get_sample_data_path(featurizer.dir(dataset.feat_dir))).toarray()
        # change datatype to int
        sample_data = sample_data.astype(np.int32)
        
        action_frequency = {}
        for i in tqdm(range(sample_data.shape[0])):
            action_ind, atom_map1, atom_map2, n_nodes, is_atom_action = \
            sample_data[i, 0], sample_data[i, 1], sample_data[i, 2], sample_data[i, 3], sample_data[i, 4]
            this_action_ind, a1, a2 = action_ind, atom_map1, atom_map2


            action = str(action_vocab['action_tuples'][this_action_ind])
            
            if action not in action_frequency:
                action_frequency[action] = 1
            action_frequency[action] += 1
        
        action_frequency = sorted(action_frequency.items(), key=lambda x: x[1], reverse=True)
        keep_actions_list = action_frequency[:topk]

        return keep_actions_list
         
    def __len__(self) -> int:
        return len(self.select_ind )

    def __getitem__(
        self, ind
    ):
        # graph_ind --> samples 
        ind = self.select_ind[ind]
        # ind = 13107
        start_ind = self.metadata['start_ind'][ind] 
        n_steps = self.metadata['n_samples'][ind]
        source_smi = self.metadata['source_smi'][ind]
        target_smi = self.metadata['target_smi'][ind]
        sample_ind = np.arange(start_ind, start_ind + n_steps)

        sample_data = self.data['sample_data'][sample_ind]
        if hasattr(sample_data, 'toarray'):
            sample_data = sample_data.toarray().astype(int)
        
        action_ind, atom_map1, atom_map2, n_nodes, is_atom_action = \
        sample_data[:, 0], sample_data[:, 1], sample_data[:, 2], sample_data[:, 3], sample_data[:, 4]
        n_max_steps = sample_ind.shape[0]
        n_max_nodes = int(max(n_nodes))
        
        tensor_data = {
            'atom': self.data['atom'][sample_ind],
            'bond': self.data['bond'][sample_ind],
            'max_n_nodes': n_max_nodes
        }

        
        tensor_data = self.featurizer.to_tensor_batch(tensor_data, self.action_vocab, self.use_motif_feature)
        
        
        node_feats = tensor_data['atom']
        adj = tensor_data['bond']

        
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
        atom_action_mask[:, 0, self.action_vocab['stop_action_num']] = 1
        atom_action_mask[:, 1:, self.action_vocab['stop_action_num']] = 0
        
        # mask out bond actions for diagonal ('self' node)
        # mask out bond actions for upper half (matrix is symmetric)
        triu = torch.triu(torch.ones((n_max_nodes, n_max_nodes), dtype=torch.int), diagonal=1)
        bond_action_mask *= triu.reshape(1,n_max_nodes, n_max_nodes,1)
        
        # only bonds between existing atoms can be edited
        atom_exists = node_mask.unsqueeze(-1).unsqueeze(-1)
        bond_action_mask *= atom_exists
        
        
        target = torch.zeros((n_max_steps, n_max_nodes + 1, n_max_nodes,  self.action_vocab['n_target_actions']), dtype=torch.float) 
        node_mask = node_mask.unsqueeze(-1)
        
        
        batch_action_ind = torch.zeros(n_max_steps, dtype=torch.long)
        
        
        k=0
        for j in range(n_steps):
            this_action_ind, a1, a2 = action_ind[k], atom_map1[k], atom_map2[k]
            if self.featurizer_key == 'megan-baseline':
                if a1 < a2:
                    a2, a1 = a1, a2
                if a2 == -1:
                    action_num = self.action_vocab['atom_action_num'][this_action_ind]
                else:
                    action_num = self.action_vocab['bond_action_num'][this_action_ind]
            else:
                if is_atom_action[j]:
                    if self.action_vocab['action_tuples'][this_action_ind][0] in ['add_atom', 'change_atom', 'add_motif']:
                        a2 = -1
                        action_num = self.action_vocab['atom_action_num'][this_action_ind]
                    else:
                        if a1 < a2:
                            a2, a1 = a1, a2
                        # try:
                        #     assert a2 == -1
                        # except:
                        #     print(self.action_vocab['action_tuples'][this_action_ind])

                        action_num = self.action_vocab['atom_action_num'][this_action_ind]
                        a2 = -1
                else:
                    if a1 < a2:
                        a2, a1 = a1, a2
                    action_num = self.action_vocab['bond_action_num'][this_action_ind]

            batch_action_ind[j] = this_action_ind
            target[j, a2, a1, action_num] = 1 # 在a2这个维度，最后一行表示atom_action, 前面所有行表示bond_action
            k+=1
        
        
        result = {
            'add_mask': self.args.add_mask,
            'source_smi': source_smi,
            'target_smi': target_smi,
            'node_features': node_feats, # [4,47,8]
            'node_mask': node_mask, # [4,47,1]
            'adj': adj, # [4,47,47,3]
            'adj_mask': adj_mask, # [4,47,47,1]
            'target': target, # [4,48,47,149]
            'atom_action_mask': atom_action_mask, # [4,47,149]
            'bond_action_mask': bond_action_mask, # [4,47,47,7]
            'n_steps': torch.tensor([n_steps]), # [1]
            'n_paths': torch.tensor([1]), # [1]
            "action_ind":batch_action_ind, # [4]
            "unflatten_target":target, # [4,51,50,149]
            "target": target.reshape(n_max_steps, -1), # [4,51*50*149]
            "atom_action_num": len(self.action_vocab['atom_action_num']),
            "bond_action_num": len(self.action_vocab['bond_action_num']),
        }
    
        
        return result