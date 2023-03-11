# -*- coding: utf-8 -*-
"""
Generates featurized training/validation samples for training MEGAN reaction generation model.
Each sample represents a single edit of the target molecule (for training for retrosynthesis).
Training can be stateful or stateless.
"""
import json
import logging
import os
import shutil
from multiprocessing.pool import Pool
from collections import Counter
import numpy as np
import pandas as pd
from src.feat import ReactionFeaturizer
import pickle

# from src.feat import ORDERED_ATOM_OH_KEYS, ORDERED_BOND_OH_KEYS, ORDERED_MOTIF_OH_KEYS

# from src.feat.graph_features import ATOM_PROPS, BOND_PROPS, ATOM_PROP2OH, BOND_PROP2OH
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm

from src import USE_MOTIF_ACTION_KEYS, USE_MOTIF_FEATURES_KEYS
from src.datasets import Dataset

from src.feat.find_properties import find_properties_parallel
from src.feat.mol_graph import unravel_atom_features, unravel_bond_features
from src.split import DatasetSplit
from src.utils.feat_utils import to_torch_tensor

logger = logging.getLogger(__name__)

# there is a warning about hydrogen atoms that do not have neighbors that could not be deleted (that is OK)
RDLogger.DisableLog('rdApp.*')

ATOM_PROPS = {
    'atomic_num': set(),
    'formal_charge': set(),
    'chiral_tag': set(),
    'num_explicit_hs': set(),
    'is_aromatic': set(),
    'is_supernode': {0, 1},  
    'is_edited': {0, 1},  # we mark atoms that have been added/edited by the model
    # 'is_reactant': {0, 1}  # this feature is used to mark reactants in "SEPARATED" variant of forward prediction
}

BOND_PROPS = {
    'bond_type': {'supernode', 'self'},
    'bond_stereo': set(),
    # 'is_aromatic': set(),
    'is_edited': {0, 1},
}

# ATOM_PROP2OH = dict((k, (dict((ap, i + 1) for i, ap in enumerate(vals)))) for k, vals in ATOM_PROPS.items())
# BOND_PROP2OH = dict((k, (dict((ap, i + 1) for i, ap in enumerate(vals)))) for k, vals in BOND_PROPS.items())


def get_adj_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'adj.npz')


def get_nodes_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'nodes.npz')

def get_motif_adj_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'motif_adj.npz')

def get_motif_node_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'motif_nodes.npz')

def get_motif2atom_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'motif2atom.npz')

def get_actionlabel_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'all_action_labels.pkl')

def get_sample_data_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'sample_data.npz')


def get_metadata_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'metadata.csv')


def get_actions_vocab_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'all_actions.json')

def get_nodes_motif_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'nodes_motif_mat.npz')

def get_atom2motif_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'nodes_atom2motif_mat.npz')




def get_prop2oh_vocab_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'prop2oh.json')


class MeganTrainingSamplesFeaturizer(ReactionFeaturizer):
    """
    Converts a mapped reaction SMILES to series of training/validation samples.
    """

    def __init__(self, split: DatasetSplit, n_jobs: int = 1,
                 key: str = 'megan', max_n_steps: int = 32,
                 forward: bool = False, action_order: str = 'dfs', 
                 use_motif_action: bool = False, use_motif_feature: bool = False, use_decomposed_fragment=False, vocab_path: str = None):
        """
        :param n_jobs: number of threads
        :param key: key of the featurizer
        :param max_n_steps: maximum number of generation steps
        :param split: split to use to filter out 'test' samples
        :param forward: generate samples for learning forward synthesis instead of backward
        :param action_order: type of action ordering ('dfs'/'bfs'/'dfs_random'/'bfs_random','random')
        """
        super(MeganTrainingSamplesFeaturizer, self).__init__()
        assert max_n_steps > 0
        assert n_jobs != 0

        self.n_jobs = n_jobs
        self.key = key
        self.max_n_steps = max_n_steps
        self.split = split
        self._vocabulary = None
        self.forward = forward
        self.action_order = action_order
        self.use_motif_action = use_motif_action
        self.use_motif_feature = use_motif_feature
        self.use_decomposed_fragment = use_decomposed_fragment
        self.vocab_path = vocab_path

    def dir(self, feat_dir: str) -> str:
        return os.path.join(feat_dir, self.key)


    def update_key(self, string):
        self.key += string
        
    def get_feat_vocab(self, dataset, split, data):
        
        if 'max_n_nodes' in dataset.meta_info:
            max_n_nodes = dataset.meta_info['max_n_nodes']
        else:
            max_n_nodes = 1024
        logger.info("Max. number of nodes: {}".format(max_n_nodes))

        # we do not featurize test set for training
        all_inds = np.argwhere(np.array(split['test'] == 0)).flatten()
        # shuffle indices for featurization in multiple threads
        np.random.shuffle(all_inds)

        

        chunk_size = int(len(all_inds) / self.n_jobs)
        chunk_ends = [chunk_size * i for i in range(self.n_jobs + 1)]
        chunk_ends[-1] = len(all_inds)
        chunk_inds = [all_inds[chunk_ends[i]: chunk_ends[i + 1]] for i in range(len(chunk_ends) - 1)]  # assign different data to different thread by splitting ID.
        
        prop_dict = {'atom': ATOM_PROPS, 'bond': BOND_PROPS}   # result of find_properties_parallel
        # add some 'special' atom/bond feature values

        logger.info(f'Finding all possible values of atom and bond properties '
                    f'on {len(all_inds)} reactions using {self.n_jobs} chunks')
        parallel_args = []
        # for i, ch_inds in enumerate(chunk_inds):
        for i, ch_inds in enumerate(chunk_inds):
            new_x = dict((k, x.values[ch_inds]) for k, x in data.items())   # type(data) = pandas.DataFrame; data.columns = Index(['Unnamed: 0', 'product', 'substrates', 'reaction_type'], dtype='object')
            parallel_args.append((i, new_x, tqdm, prop_dict['atom'], prop_dict['bond']))

        
        if self.n_jobs == 1:
            chunk_results = [find_properties_parallel(parallel_args[0])]
        else:
            pool = Pool(self.n_jobs)
            chunk_results = pool.imap(find_properties_parallel, parallel_args)

        for chunk_prop_dict in chunk_results:  # merge prop dict of each chunk.
            for type_key in prop_dict.keys():  # ['atom', 'bond']
                for key, values in chunk_prop_dict[type_key].items():
                    if key not in prop_dict[type_key]:
                        prop_dict[type_key][key] = set()
                    prop_dict[type_key][key].update(values)  



        # counting the number of types of each feature
        atom_feat_counts = ', '.join(['{:s}: {:d}'.format(key, len(values))  # 'atomic_num: 16, formal_charge: 3, chiral_tag: 3, num_explicit_hs: 4, is_aromatic: 2, is_supernode: 2, is_edited: 2, is_reactant: 2'
                                      for key, values in prop_dict['atom'].items()])
        logger.info(f'Found atom features: {atom_feat_counts}')

        # similar to atom feature
        bond_feat_counts = ', '.join(['{:s}: {:d}'.format(key, len(values))
                                      for key, values in prop_dict['bond'].items()])
        logger.info(f'Found bond features: {bond_feat_counts}')

        # make a dictionary for conversion of atom/bond features to OH numbers
        # difference between "prop_dict" and "props": “props" convert set of prop_dict to list.
        # difference between "prop_dict" and "prop2oh": "prop2oh" convert set of prop_dict to dict, where key is the type of feature and value is the OH numbers/index. (e.g. prop2oh['atomic_num"] = {5: 1})
        prop2oh = {'atom': {}, 'bond': {}}
        props = {'atom': {}, 'bond': {}}
        for type_key, prop_values in prop_dict.items():
            for prop_key, values in prop_values.items():
                sorted_vals = list(sorted(values, key=lambda x: x if isinstance(x, int) else 0))
                if "self" == sorted_vals[1]:
                    sorted_vals = ['self', 'supernode'] + sorted_vals[2:]
                props[type_key][prop_key] = sorted_vals
                oh = dict((k, i + 1) for i, k in enumerate(sorted_vals))  # dictionary of {sorted_value: rank}
                prop2oh[type_key][prop_key] = oh

        
        

        atom_feature_keys = [k for k in prop2oh['atom']]
        bond_feature_keys = [k for k in prop2oh['bond']]  # 'is_aromatic' feature is revmoed.
        feat_vocab = {
            'prop2oh': prop2oh,
            'atom_feature_keys': atom_feature_keys,
            'bond_feature_keys': bond_feature_keys,
            'atom_feat_ind': dict((k, i) for i, k in enumerate(atom_feature_keys)),
            'bond_feat_ind': dict((k, i) for i, k in enumerate(bond_feature_keys))
        }
        return feat_vocab, all_inds, chunk_inds, max_n_nodes
    
    
    def get_sample_data(self,data, all_inds, chunk_inds, split, feat_dir, max_n_nodes, reaction_type_given, feat_vocab, featurize_parallel):
        data_len = len(data)
        samples_len = data_len * self.max_n_steps
        parallel_args = []
        chunk_save_paths = []
        for i, ch_inds in enumerate(chunk_inds):
            new_x = dict((k, x.values[ch_inds]) for k, x in data.items())
            is_train = split['train'][ch_inds].values
            chunk_save_path = os.path.join(feat_dir, f'chunk_result_{i}')
            chunk_save_paths.append(chunk_save_path)
            parallel_args.append((i, samples_len, ch_inds, new_x, max_n_nodes, tqdm,
                                  self.max_n_steps, is_train, reaction_type_given, self.use_motif_action, self.use_motif_feature, self.vocab_path,
                                  self.forward, self.action_order,
                                  feat_vocab, chunk_save_path))

        logger.info(f'Featurizing {len(all_inds)} reactions with {self.n_jobs} threads')
        logger.info(f"Number of generated paths (train+valid): {data_len}")
        logger.info(f"Upper bound for number of generated samples: {samples_len} ({data_len} * {self.max_n_steps})")
        
        # self.n_jobs = 1 # for debug
        
        if self.n_jobs == 1:
            chunk_results = [featurize_parallel(parallel_args[0])]
        else:
            # leave one job for merging results
            pool = Pool(max(self.n_jobs - 1, 1))
            chunk_results = pool.imap(featurize_parallel, parallel_args)

        logger.info(f"Merging featurized data from {self.n_jobs} chunks")
        return chunk_results, chunk_save_paths, samples_len
        

    def featurize_dataset(self, dataset: Dataset, topk: int = None, new_key: str = None):
        if self.use_motif_action and self.use_decomposed_fragment:
            from src.feat.featurize_gzy_psvae import featurize_parallel
        else:
            from src.feat.featurize import featurize_parallel

        if new_key is not None:
            self.update_key(new_key)
            
        logger.info(f"Loading dataset {dataset.key} and {self.split.key} split")
        data = dataset.load_x()
        # split = self.split.load(dataset.dir)
        split = data[["train", "valid", "test"]]
        feat_dir = self.dir(dataset.feat_dir)

        reaction_type_given = False

        # debug
        # =======================
        # self.n_jobs = 1
        # data = data[:5000]
        # split = split[:5000]

        # split[800:1000]['train'] = 0
        # split[800:1000]['valid'] = 1
        # metadata = metadata[:1000]
        # =======================
        
        ## ---------------------statistic graph features------------------------
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        feat_vocab, all_inds, chunk_inds, max_n_nodes = self.get_feat_vocab(dataset, split, data)
        with open(os.path.join(feat_dir, 'feat_vocab.json'), 'w') as fp:
            json.dump(feat_vocab, fp, indent=10)
        ## ---------------------------------------------------------------------

        # self.n_jobs = 1
        ## -----------------data sample featurize_parallels--------------------
        chunk_results, chunk_save_paths, samples_len = self.get_sample_data(data,  all_inds, chunk_inds, split, feat_dir, max_n_nodes, reaction_type_given, feat_vocab, featurize_parallel)
        ## ---------------------------------------------------------------------

        nodes_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes))   # [dataset_size, max_n_nodes]
        adj_mat = sparse.csr_matrix(([], ([], [])), shape=(samples_len, max_n_nodes ** 2))  # [dataset_size, max_n_nodes, max_n_nodes]
        all_actions_tuples = []
        meta = []

        # vocabulary of actions
        actions_vocab = []
        sample_inds = []
        all_unsparsed = 0
        for ch_inds, result_code, chunk_save_path in tqdm(zip(chunk_inds, chunk_results, chunk_save_paths),desc='merging reactions from chunks', total=self.n_jobs):
            # sample_data_path = os.path.join(chunk_save_path, 'sample_data.npz')
            # sample_data_mat += sparse.load_npz(sample_data_path)

            nodes_mat_path = os.path.join(chunk_save_path, 'nodes_mat.npz')
            nodes_mat += sparse.load_npz(nodes_mat_path)

            adj_mat_path = os.path.join(chunk_save_path, 'adj_mat.npz')
            adj_mat += sparse.load_npz(adj_mat_path)
            


            meta_save_path = os.path.join(chunk_save_path, 'metadata.csv')
            chunk_meta = pd.read_csv(meta_save_path, index_col=0)
            meta.append(chunk_meta)

            actions_save_path = os.path.join(chunk_save_path, 'actions.txt')
            for line in open(actions_save_path, 'r'):
                action = eval(line.strip())
                all_actions_tuples.append(action)
        
            # remove temporary chunk files
            shutil.rmtree(chunk_save_path)
            logger.info(f"Merged chunk {len(meta)} (unparsed samples: {result_code}/{len(ch_inds)})")
            all_unsparsed += result_code
            
        
            

        logger.info("Unsparsed sample vs Sparsed sample: {}/{}".format(all_unsparsed, len(sample_inds)))
        logger.info("Concatenating metadata")
        meta = pd.concat(meta)

        logger.info("Saving found actions")
        # sample_data_mat[sample_inds, 0] = action_inds # 记录标签
        
        
        #--------------------- action
        all_actions2 = []
        for ind, atom_map1, atom_map2, n_node, action in all_actions_tuples:
            if action[0]=="attach_atom":
                all_actions2.append(("attach_atom",None))
            else:
                all_actions2.append(action)
        
        atom_action2ind = {}
        bond_action2ind = {}
        atom_prednum = 0
        bond_prednum=0
        for action in all_actions2:
            if action[0] in ['change_atom', 'add_motif']:
                if action not in atom_action2ind:
                    atom_action2ind[action] = atom_prednum
                    atom_prednum+=1
            elif action[0] in ['change_bond']:
                if action not in bond_action2ind:
                    bond_action2ind[action] = bond_prednum
                    bond_prednum+=1
        bond_action2ind[("attach_atom",None)] = bond_prednum
        
                
                
        
        action_vocab = {"action_freq": 
                                    {str(key):val for key, val in dict       (Counter(all_actions2)).items()},
                        "atom_action2ind":
                            {str(key):val for key, val in atom_action2ind.items()},
                        "bond_action2ind":{str(key):val for key, val in bond_action2ind.items()}}
        with open(get_actions_vocab_path(feat_dir), 'w') as fp:
            json.dump(action_vocab, fp)
        
        logger.info(f"Found {len(actions_vocab)} reaction actions")
        #---------------------------
            
        
        
        #----------------------graph & meta
        n_samples = meta['n_samples']
        logger.info(f"Number of steps: max: {np.max(n_samples)}, avg: {np.mean(n_samples)}")

        logger.info("Saving featurized data")
        meta.to_csv(get_metadata_path(feat_dir))
        pickle.dump(all_actions_tuples, open(get_actionlabel_path(feat_dir), 'wb'))
        sparse.save_npz(get_nodes_path(feat_dir), nodes_mat)
        sparse.save_npz(get_adj_path(feat_dir), adj_mat)


        n_saved_reacs = len(np.unique(meta['reaction_ind']))

        logger.info(f"Saved {n_saved_reacs}/{len(all_inds)} reactions ({n_saved_reacs / len(all_inds) * 100}%)")
        logger.info(f"Saved {len(meta)} paths (avg. {len(meta) / n_saved_reacs} paths per reaction)")

        logger.info("Saving featurization metadata")
        meta_info = {
            'description': 'Graph representation of molecules with discrete node and edge features for MEGAN',
            'features': ['atom', 'bond'],
            'features_type': ['atom', 'bond'],
            'max_n_nodes': max_n_nodes,
            'format': 'sparse',
            'all_unparsed': all_unsparsed,
            'max_steps': int(np.max(n_samples)),
            'avg_steps': float(np.mean(n_samples))
        }
        meta_path = self.meta_info_path(dataset.feat_dir)
        with open(meta_path, 'w') as fp:
            json.dump(meta_info, fp, indent=2)

    def featurize_batch(self, metadata_dir: str, batch: dict) -> dict:
        raise NotImplementedError("TODO")



    def load(self, feat_dir: str) -> dict:
        this_feat_dir = self.dir(feat_dir)
        result = {
            'reaction_metadata': pd.read_csv(get_metadata_path(this_feat_dir), index_col='reaction_ind'),
            'atom': sparse.load_npz(get_nodes_path(this_feat_dir)),
            'bond': sparse.load_npz(get_adj_path(this_feat_dir)),
            'action_tuples': pickle.load(open(get_actionlabel_path(this_feat_dir),"rb"))
        }
        return result

    # noinspection PyMethodOverriding
    def to_tensor_batch(self, data: dict, feat_vocab: dict, use_motif_feature=False) -> dict:
        batch_max_nodes = data['max_n_nodes']  # number of nodes in each graph in batch
        props = feat_vocab['prop2oh']

        nodes = data['atom'][:, :batch_max_nodes]
        if hasattr(nodes, 'toarray'):
            nodes = nodes.toarray()
        nodes = nodes.astype(int)
        
        if use_motif_feature:
            nodes_motif = data['nodes_motif'][:, :batch_max_nodes].toarray().astype(int)
            atom2motif = data['atom2motif'][:, :batch_max_nodes].toarray().astype(int)

        edges = data['bond']
        if hasattr(edges, 'toarray'):
            edges = edges.toarray()
        max_n = int(np.sqrt(edges.shape[-1]))
        edges = edges.reshape(edges.shape[0], max_n, max_n)
        edges = edges[:, :batch_max_nodes, :batch_max_nodes].astype(int)

        # unravel discrete features
        node_oh_dim = [len(props['atom'][feat_key]) + 1 for feat_key in feat_vocab['atom_feature_keys']]
        unraveled_nodes = unravel_atom_features(nodes, node_oh_dim=node_oh_dim)  # 返回每一个元素在形状为 node_oh_dim 的张量中的坐标 [len(node_oh_dim), sample_size, max_nodes] 
        unraveled_nodes = unraveled_nodes.transpose(1, 2, 0)  # [sample_size, max_nodes, len(node_oh_dim)]
        data['atom'] = to_torch_tensor(unraveled_nodes, long=True)
        if use_motif_feature:
            data['nodes_motif'] = to_torch_tensor(nodes_motif, long=True)
            data['atom2motif'] = to_torch_tensor(atom2motif, long=True)

        edge_oh_dim = [len(props['bond'][feat_key]) + 1 for feat_key in feat_vocab['bond_feature_keys']]
        try:
            unraveled_edges = unravel_bond_features(edges, edge_oh_dim=edge_oh_dim)
            unraveled_edges = unraveled_edges.transpose(1, 2, 3, 0)
        except:
            unraveled_edges = unravel_bond_features(edges, edge_oh_dim=edge_oh_dim) #TODO: for debug
            unraveled_edges = unraveled_edges.transpose(1, 2, 3, 0)
        data['bond'] = to_torch_tensor(unraveled_edges, long=True)  # 类似 atom

        

        if 'reaction_type' in data:
            data['reaction_type'] = to_torch_tensor(data['reaction_type'], long=True)

        return data
