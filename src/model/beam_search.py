"""
Scripts for performing beam search on MEGAN
"""
import logging
from typing import List, Tuple, Optional

import networkx as nx
from torch.autograd import Variable
import numpy as np
import torch
import copy

# from src.feat import ORDERED_BOND_OH_KEYS, ORDERED_ATOM_OH_KEYS
from src.feat.mol_graph import get_graph  # , mol_to_nx
from src.feat.reaction_actions import ReactionAction, StopAction, AtomEditAction, AddAtomAction, AddRingAction, \
    BondEditAction, AddMotifAction, AddMotifAction_with_dummySMI
from src.feat.utils import atom_to_edit_tuple, get_bond_tuple, fix_incomplete_mappings, reac_to_canonical, \
    fix_explicit_hs
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from src.model.megan_utils import generate_eval_batch, mols_from_graph, RdkitCache

from torch_scatter import scatter_sum, scatter_max, scatter_softmax

logger = logging.getLogger(__name__)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_top_k_paths(paths: List[dict], beam_size: int, n_mols: int, sort: bool = True):
    filtered_paths = []
    for i in range(n_mols):
        mol_paths = [p for p in paths if p['mol_ind'] == i]
        if sort:
            path_argsort = np.argsort([-p['prob'].cpu().numpy() for p in mol_paths])
            mol_paths = [mol_paths[j] for j in path_argsort[:beam_size]]
        else:
            mol_paths = mol_paths[:beam_size]
        filtered_paths += mol_paths
    return filtered_paths


def paths_are_probably_same(path1: dict, path2: dict, eps: float = 1e-6) -> bool:
    if abs(path1['prob'] - path2['prob']) < eps:
        return True
    if path1['action_set'] == path2['action_set']:
        return True
    return False


def filter_duplicate_paths(paths: List[dict], n_mols: int):
    filter_paths = np.full(len(paths), fill_value=True, dtype=bool)

    for path in paths:
        path['action_set'] = set(str(a) for a, p in path['actions'])

    for mol_i in range(n_mols):
        mol_paths_i = [i for i in range(len(paths)) if paths[i]['mol_ind'] == mol_i]
        mol_paths = [paths[i] for i in mol_paths_i]
        
        is_unique = np.full(len(mol_paths), fill_value=True, dtype=bool)
        for path_i in range(len(mol_paths)):
            for prev_path in mol_paths[:path_i]:
                if paths_are_probably_same(prev_path, mol_paths[path_i]):
                    is_unique[path_i] = False
                    prev_path['prob'] += mol_paths[path_i]['prob']
                    mol_paths[path_i]['prob'] = prev_path['prob']
                    break
        filter_paths[mol_paths_i] = is_unique
    paths = [paths[i] for i, is_unique in enumerate(filter_paths) if is_unique]
    return paths


def to_one_hot(x, dims: int):
    one_hot = torch.FloatTensor(*x.shape, dims).zero_().to(device)
    x = torch.unsqueeze(x, -1)
    target = one_hot.scatter_(x.dim() - 1, x.data, 1)

    target = Variable(target)
    return target


def merge_eval_graph(graph_list, feat_vocab):
        if graph_list[0]['reaction_type'] is not None:
            reaction_type = torch.tensor([one["reaction_type"] for one in graph_list if one is not None])
        else:
            reaction_type = None
        
        num_nodes = [one["node_features"].shape[0] if one is not None else 0 for one in graph_list ]
        shift = np.cumsum([0, *num_nodes]).tolist()

        graph_id = torch.arange(len(num_nodes)).repeat_interleave(torch.tensor(num_nodes), dim=0)
        node_features = torch.cat([one["node_features"]  for one in graph_list if one is not None])
        atom_action_mask = torch.cat([one['node_mask'].expand(*one["atom_action_mask"].shape) * one["atom_action_mask"] for one in graph_list if one is not None])


        edge_idx = []
        edge_val = []
        bond_action_mask = []

        for i in range(len(graph_list)):
            if graph_list[i] is not None:
                adj_mat = graph_list[i]['adj_mask'][:,:,0]
                local_edge = torch.nonzero(adj_mat)
                
                val = graph_list[i]["adj"][local_edge[:,0], local_edge[:,1]]
                edge_idx.append(local_edge + shift[i]) # 索引偏移，如[[2+10,5+10,7+10], [1+10,3+10,5+10]]
                edge_val.append(val)

                node_adj_mask = graph_list[i]['node_mask'].unsqueeze(1)
                node_adj_mask = node_adj_mask.expand(*node_adj_mask.shape)
                node_adj_mask = node_adj_mask * node_adj_mask.permute(1, 0, 2).contiguous()

                bond_action_mask.append((node_adj_mask.contiguous() * graph_list[i]['bond_action_mask'])[local_edge[:, 0], local_edge[:, 1]])
                

                
        edge_idx = torch.cat(edge_idx)
        # remove supernode

        supernode_idx = torch.where(node_features[:, -2] == 2)[0]

        edge_idx_remove_supernode_idx = [a not in supernode_idx and b not in supernode_idx for a, b in edge_idx]
        edge_idx_remove_supernode = edge_idx[edge_idx_remove_supernode_idx]

        padding_idx = torch.where(node_features[:, -2] == 0)[0]
        # edge_idx_remove_padding = edge_idx[[a not in padding_idx and b not in padding_idx for a, b in edge_idx]]

        # find connected subgraph based on edge_idx
        G = nx.Graph()
        G.add_edges_from(edge_idx_remove_supernode.tolist())
        node_ids_rm_supernode = list(set(torch.arange(node_features.shape[0]).tolist())-set(supernode_idx.tolist()))
        G.add_nodes_from(node_ids_rm_supernode)
        
        sub_graphs = list(nx.connected_components(G))
        sub_graph_id = torch.zeros(node_features.shape[0])-1
        for idx, sub_id in enumerate(sub_graphs):
            sub_graph_id[list(sub_id)] = idx
            
        degree = []
        for i in range(node_features.shape[0]):
            if i in supernode_idx or i in padding_idx:
                degree.append(0)
                continue
            degree.append(G.degree[i])

        degree = torch.tensor(degree)
        
        edge_val = torch.cat(edge_val)
        bond_action_mask = torch.cat(bond_action_mask)


        oh_atom_feats = []
        for i, key in enumerate(feat_vocab['atom_feature_keys']):
            oh_feat = to_one_hot(node_features[:,i], dims=len(feat_vocab['prop2oh']['atom'][key]) + 1)
            oh_atom_feats.append(oh_feat)
        atom_feats = torch.cat(oh_atom_feats, dim=-1)  # slight different when add_mask = 1 or not
        atom_mask = torch.sign(torch.max(node_features, dim=-1)[0])
        
        oh_bond_feats = []

        for i, key in enumerate(feat_vocab['bond_feature_keys']):
            oh_feat = to_one_hot(edge_val[:,i], dims=len(feat_vocab['prop2oh']['bond'][key]) + 1)
            oh_bond_feats.append(oh_feat)
        bond_feats = torch.cat(oh_bond_feats, dim=-1)
        edge_idx_remove_supernode_idx = torch.tensor(edge_idx_remove_supernode_idx)
        bond_action_mask = bond_action_mask[:,:-1]# remove attach action
        
        # # remove supernode edges
        # bond_feats = bond_feats[edge_idx_remove_supernode_idx]
        # edge_idx = edge_idx[edge_idx_remove_supernode_idx]
        # bond_action_mask = bond_action_mask[edge_idx_remove_supernode_idx]


        if reaction_type is not None:
            return {"graph_id": graph_id,
                    "atom_feats": atom_feats,
                    "bond_feats": bond_feats,
                    "atom_mask": atom_mask,
                    "padding_idx": padding_idx.to(device),
                    "supernode_idx": supernode_idx.to(device),
                    "degree": degree.to(device),
                    "atom_action_mask": atom_action_mask,
                    "bond_action_mask": bond_action_mask,
                    "sub_graph_id":sub_graph_id.to(device),
                    "edge_idx": edge_idx,
                    "edge_idx_remove_supernode": edge_idx_remove_supernode_idx,
                    "reaction_type": reaction_type.to(device),
            }
        else:
            return {"graph_id": graph_id,
                    "atom_feats": atom_feats,
                    "bond_feats": bond_feats,
                    "atom_mask": atom_mask,
                    "padding_idx": padding_idx.to(device),
                    "supernode_idx": supernode_idx.to(device),
                    "degree": degree.to(device),
                    "atom_action_mask": atom_action_mask,
                    "bond_action_mask": bond_action_mask,
                    "sub_graph_id":sub_graph_id.to(device),
                    "edge_idx": edge_idx,
                    "edge_idx_remove_supernode": edge_idx_remove_supernode_idx,
                    "reaction_type": None,
            }


def get_batch(paths: List[dict], base_action_masks: dict,
              reaction_types: Optional[np.ndarray] = None, feat_vocab=None) -> Tuple[dict, List[torch.Tensor]]:
    batch = generate_eval_batch([p['mol_graph'] for p in paths],
                                base_action_masks=base_action_masks, reaction_types=reaction_types)

    node_feature_list = [one['node_features'].squeeze(0) for one in batch]
    node_mask_list = [one['node_mask'].squeeze(0) for one in batch]
    adj_list = [one['adj'].squeeze(0) for one in batch]
    adj_mask_list = [one['adj_mask'].squeeze(0) for one in batch]
    atom_action_mask_list = [one['atom_action_mask'].squeeze(0) for one in batch]
    bond_action_mask_list = [one['bond_action_mask'].squeeze(0) for one in batch]
    max_map_num_list = [p['max_map_num'] for p in paths]

    max_path = max([one.shape[0] for one in node_feature_list])
  
    
    graphs = [None for i in range(len(paths))]
    
    for i in range(len(paths)):
        if reaction_types is None:
            graphs[i] = {"node_features": node_feature_list[i],
                            "node_mask":node_mask_list[i],
                            'adj':adj_list[i],
                            'adj_mask': adj_mask_list[i], 
                            'atom_action_mask': atom_action_mask_list[i], 
                            'bond_action_mask': bond_action_mask_list[i], 
                            'max_map_num': torch.tensor(max_map_num_list),
                            'reaction_type': None
            }
        else:
            graphs[i] = {"node_features": node_feature_list[i],
                            "node_mask":node_mask_list[i],
                            'adj':adj_list[i],
                            'adj_mask': adj_mask_list[i], 
                            'atom_action_mask': atom_action_mask_list[i], 
                            'bond_action_mask': bond_action_mask_list[i], 
                            'max_map_num': torch.tensor(max_map_num_list),
                            'reaction_type': reaction_types[i]
            }
    
    batch = merge_eval_graph(graphs, feat_vocab)
    
    models_step_states = []
    for model_i in range(len(paths[0]['state'])):
        if paths[0]['state'][model_i] is None:
            step_state = None
        else:
            stacked_tensors = []
            graph_ids = []
            for i, p in enumerate(paths):
                stacked_tensors.append(p['state'])
                graph_ids += [i] * p['state'].size(0)
            step_state = {'val': torch.vstack(stacked_tensors).cuda(), 'graph_id': torch.tensor(graph_ids).cuda()}
        models_step_states.append(step_state)

    # attach_source and attach_target
    attach_source = []
    attach_target = []
    for path in paths:
        temp = None
        for atom in path['mol'].GetAtoms():
            if atom.GetSymbol()=="*":
                temp = atom.GetAtomMapNum()
                break
                
        attach_source.append(temp)
        attach_target.append(temp)
    batch['attach_source'] = attach_source
    batch['attach_target'] = attach_target
    return batch, models_step_states


def tuple2action(action: Tuple, atom1: int, atom2: int, max_map_num: int, feat_vocab: dict) -> ReactionAction:
    action_type = action[0]

    if action_type == 'stop':
        return StopAction(feat_vocab=feat_vocab)
    elif action_type == 'change_atom':
        return AtomEditAction(atom1, *action[1], feat_vocab=feat_vocab)
    elif action_type == 'add_atom':
        atom2 = max_map_num + 1
        return AddAtomAction(atom1, atom2, *action[1][0], *action[1][1], feat_vocab=feat_vocab)
    elif action_type == 'add_ring':
        new_atoms_map_nums = [atom1] + [max_map_num + i + 1 for i in range(5)]
        return AddRingAction(atom1, new_atoms_map_nums, action[1], feat_vocab=feat_vocab)
    elif action_type == 'change_bond':
        return BondEditAction(atom1, atom2, *action[1], feat_vocab=feat_vocab)
    elif action_type == 'add_motif':
        motif = Chem.RWMol(Chem.MolFromSmiles(action[1]))
        add_attach_map = False
        new_atoms_map_nums = []
        i = 0
        for atom in motif.GetAtoms():
            if atom.GetSymbol()=="*" and not add_attach_map:
                atom.SetAtomMapNum(atom1)
                add_attach_map=True
            else:
                atom.SetAtomMapNum(max_map_num + i + 1)
                new_atoms_map_nums.append(max_map_num + i + 1)
                i += 1
    
        motif_smi = Chem.MolToSmiles(motif)
        action = AddMotifAction_with_dummySMI(motif_smi)
        action.new_atoms_map_nums = new_atoms_map_nums
        return action
    else:
        raise ValueError(f'Unknown action type: {action_type}')


def get_topk(output, mask, beam_size):
    nonzero_where = torch.nonzero(mask, as_tuple=False)[:, 0]
    n_nonzeros = nonzero_where.shape[0]

    if n_nonzeros == 0:
        return torch.tensor((), dtype=torch.float, device=device), \
               torch.tensor((), dtype=torch.long, device=device)
    else:
        beam_size = min(beam_size, output[mask].shape[0])
        action_probs, action_numbers = torch.topk(output[mask], beam_size, sorted=False)

    action_numbers = nonzero_where[action_numbers]
    return action_probs, action_numbers


def beam_search_constraint(actions: List[ReactionAction], edit_path: dict, feat_vocab: dict):
    past_edit_path = []
    past_edit_atom_maps = []
    for i in range(len(actions)):
        action = actions[-i]
        if isinstance(action[0], AddMotifAction_with_dummySMI):
            past_edit_path.append(action[0].smiles)
            past_edit_atom_maps.extend(action[0].new_atoms_map_nums)
        elif isinstance(action[0], AtomEditAction):
            break
        elif isinstance(action[0], BondEditAction):
            break
        else:
            raise ValueError
    # reverse the past_edit_path list
    past_edit_path = past_edit_path[::-1]
    past_edit_atom_maps = list(set(past_edit_atom_maps))
    past_edit_path_id = [feat_vocab['motif2action_ind'][motif] for motif in past_edit_path]

    future_action = []
    if len(past_edit_path_id) != 0:
        for path in edit_path:
            if past_edit_path_id == path[:len(past_edit_path_id)] and len(path) > len(past_edit_path_id):
                future_action.append(path[len(past_edit_path_id)])
        future_action = list(set(future_action))
    return future_action, past_edit_atom_maps



def get_best_actions(model_batch, prediction_scores, batch_beam_size) \
        -> List[List[Tuple[float, Tuple, int]]]:
    
    graph_id = model_batch['graph_id'].unique()
    node_graph_id = model_batch['graph_id']
    edge_graph_id = node_graph_id[model_batch['edge_idx'][:,0]]

    node_num = scatter_sum(torch.ones_like(node_graph_id), node_graph_id)
    edge_num = scatter_sum(torch.ones_like(edge_graph_id), edge_graph_id)


    result = []
    assert max(graph_id) == len(graph_id) - 1
    for id in graph_id:
        prediction_indices = torch.cat([prediction_scores['atom_idx'], prediction_scores['bond_idx'], prediction_scores['attach_idx']], dim=1)
        
        
        pred_action_type = prediction_scores['pred_action_type'].argmax(dim=1)
        
        # is_atom_action_type = pred_action_type[prediction_scores['atom_idx'][0]]
        # is_atom_action_type = (is_atom_action_type==0)|(is_atom_action_type==3)
        
        # is_bond_action_type = pred_action_type[prediction_scores['bond_idx'][0]]
        # is_bond_action_type = is_bond_action_type==1
        
        # is_attach_action_type = pred_action_type[prediction_scores['attach_idx'][0]]
        # is_attach_action_type = is_attach_action_type==2
        
        
        # prediction_score = torch.cat([
        #     prediction_scores['atom_action']*is_atom_action_type, 
        #     prediction_scores['bond_action']*is_bond_action_type, 
        #     prediction_scores['attach_action']*is_attach_action_type])
        
        prediction_score = torch.cat([
            prediction_scores['atom_action'], 
            prediction_scores['bond_action'], 
            prediction_scores['attach_action']])
        
        
        this_score = prediction_score[prediction_indices[0] == id]
        this_indices = prediction_indices[:, prediction_indices[0] == id]
        this_atom_num = node_num[id]
        this_edge_num = edge_num[id]


        topk_probs, topk_indices = torch.topk(this_score, k=batch_beam_size)
        topk_actions = this_indices[:, topk_indices]


        this_result = []
        for prob, action in zip(topk_probs, topk_actions.t()):
            action = action.tolist()
            if action[1] == max(node_num):
                this_result.append({
                    "prob": prob,
                    "atom1": action[2],
                    "atom2": -1,
                    "action": action[3],
                    "n_node": this_atom_num,
                })
            elif action[1] < max(node_num):
                this_result.append({
                    "prob": prob,
                    "atom1": action[1],
                    "atom2": action[2],
                    "action": action[3],
                    "n_node": this_atom_num,
                })
            else:
                raise ValueError     
        result.append(this_result)
    return result
        



def get_action_object(action_info: dict, max_map_num: int, feat_vocab:dict, action_vocab: dict) -> ReactionAction:
    # action_number = action_inds[-1]
    if action_info['atom2'] == -1:
        action = action_vocab['atom_ind2action'][action_info['action']] # TODO:
    else:
        action = action_vocab['bond_ind2action'][action_info['action']]


    return tuple2action(action, action_info['atom1'], action_info['atom2'], max_map_num, feat_vocab)


def transform_edit_path(edit_path, feat_vocab):
    new_edit_path = []
    for path in edit_path:
        try:
            new_path = []
            for motif in path:
                new_path.append(feat_vocab['motif2action_ind'][motif])
            new_edit_path.append(new_path)
        except:
            continue
    return new_edit_path


class BeamSearch:
    def __init__(self, models, base_action_masks: dict, feat_vocab: dict, action_vocab:dict, max_steps: int = 16, beam_size: int = 1,batch_size: int = 32, max_atoms: int = 200, min_prob: float = 0.0, min_stop_prob: float = 0.0,filter_duplicates: bool = False, filter_incorrect: bool = True,reaction_types: Optional[np.ndarray] = None, softmax_base: float = 1.0,export_samples: bool = False) -> List[List[dict]]:
        super(BeamSearch, self).__init__()
        self.models = models
        self.feat_vocab = feat_vocab
        self.paths = []
        self.export_samples = export_samples
        self.reaction_types = reaction_types
        self.base_action_masks = base_action_masks
        self.beam_size = beam_size
        self.action_vocab = action_vocab
        self.min_prob = min_prob
        self.min_stop_prob = min_stop_prob
        self.max_atoms = max_atoms
        self.max_steps = max_steps
        self.filter_duplicates = filter_duplicates
        self.batch_size = batch_size
        self.filter_incorrect = filter_incorrect
    
    
    def get_start_path(self, mols):
        paths = []
        for i, input_mol in enumerate(mols):
            mol_bond_dirs = {}
            if input_mol is not None and input_mol.GetNumAtoms() > 0:
                mol_graph = get_graph(input_mol, ravel=False, to_array=True,
                                    atom_feature_keys=self.feat_vocab['atom_feature_keys'],
                                    bond_feature_keys=self.feat_vocab['bond_feature_keys'],
                                    atom_prop2oh=self.feat_vocab['prop2oh']['atom'],
                                    bond_prop2oh=self.feat_vocab['prop2oh']['bond'])

                start_path = {
                    'n_steps': 0,
                    'prob': 1.0,
                    'mol_graph': mol_graph,
                    'max_map_num': max(a.GetAtomMapNum() for a in input_mol.GetAtoms()),
                    'state': [None for _ in range(len(self.models))],
                    'actions': [],
                    'action_str':[],
                    'action_prob':[],
                    'finished': False,
                    'changed_atoms': set(),
                    'mol_ind': i,
                    "mol": Chem.RWMol(input_mol)
                }

                paths.append(start_path)
        return paths
    
    def ravel_graph(self, adj, nodes):
        feat_vocab = self.feat_vocab
        edge_oh_dim = [len(feat_vocab['prop2oh']['bond'][feat_key]) + 1 for feat_key in feat_vocab['bond_feature_keys'] if feat_key in feat_vocab['prop2oh']['bond']]
        node_oh_dim = [len(feat_vocab['prop2oh']['atom'][feat_key]) + 1 for feat_key in feat_vocab['atom_feature_keys'] if feat_key in feat_vocab['prop2oh']['atom']]
        
        ravel_nodes = np.ravel_multi_index(nodes.T, node_oh_dim)
        ravel_adj = np.ravel_multi_index(adj.reshape(-1, adj.shape[-1]).T, edge_oh_dim)
        ravel_adj = ravel_adj.reshape((adj.shape[0], adj.shape[1]))

        return ravel_adj, ravel_nodes
        
    def process_paths_batch(self, path_batch) -> List[dict]:
        
        if self.reaction_types is not None:
            batch_reaction_types = np.asarray([self.reaction_types[p['mol_ind']] for p in path_batch])
        else:
            batch_reaction_types = None
        step_batch, step_state = get_batch(path_batch, self.base_action_masks, batch_reaction_types, feat_vocab=self.feat_vocab)
        new_batch_paths = []

        # cancel model ensemble
        model_i = 0
        model = self.models[model_i]
        model.eval()

        state_dict = None if step_state[model_i] is None else {'state': step_state[model_i]}
        model_batch = {}
        for k, v in step_batch.items():
            if type(v) == torch.Tensor:
                model_batch[k] = v.to(device)
            else:
                model_batch[k] = v

        # if tuple(model_batch['edge_idx'].shape) == (5432, 2):
        #     print()
        model_step_results = model.forward_one_step(step_batch=model_batch, state_dict=state_dict)

        batch_actions = get_best_actions(model_batch, model_step_results['pred_scores'], self.beam_size)

        for path_i, path in enumerate(path_batch):
            max_map_num = path_batch[path_i]['max_map_num']
            if max_map_num > self.max_atoms:
                continue
            actions = batch_actions[path_i]
            n_steps = path['n_steps'] + 1
            state = model_step_results['state']['val'][model_step_results['state']['graph_id'] == path_i]

            for action in actions:
                
                last_action = action  # action_prob, action_inds, n_nodes
                new_prob = path['prob'] * action['prob']

                if action['atom2'] == -1 and self.action_vocab['atom_ind2action'][action['action']][0] == 'stop':

                    final_path = {
                        'finished': True,
                        'prob': new_prob,
                        'n_steps': n_steps,
                        'actions': path['actions'] + [(StopAction(feat_vocab=self.feat_vocab),action['prob'])],
                        'changed_atoms': path['changed_atoms'],
                        'mol_graph': path['mol_graph'],
                        'state': state ,
                        'mol_ind': path['mol_ind'],
                        'action_str': path['action_str'],
                        'action_prob': path['action_prob']+[action['prob'].item()],
                        'mol': copy.deepcopy(path['mol']),
                    }

                    if self.export_samples:
                        final_path['mol_graphs'] = path['mol_graphs'] + [self.ravel_graph(*path['mol_graph'])]
                    new_batch_paths.append(final_path)
                elif action['atom2'] == -1:
                    new_path = {
                        'finished': False,
                        'n_steps': n_steps,
                        'prob': new_prob,
                        'mol_graph': path['mol_graph'],
                        'max_map_num': max_map_num,
                        'state': state ,
                        'actions': path['actions'],
                        'last_action': last_action,
                        'changed_atoms': path['changed_atoms'],
                        'mol_ind': path['mol_ind'],
                        'action_str': path['action_str'],
                        'action_prob': path['action_prob']+[action['prob'].item()],
                        'mol': copy.deepcopy(path['mol']),
                    }
                    if self.export_samples:
                        new_path['mol_graphs'] = path['mol_graphs'] + [self.ravel_graph(*path['mol_graph'])]
                    new_batch_paths.append(new_path)               
                else:
                    new_path = {
                        'finished': False,
                        'n_steps': n_steps,
                        'prob': new_prob,
                        'mol_graph': path['mol_graph'],
                        'max_map_num': max_map_num,
                        'state': state ,
                        'actions': path['actions'],
                        'last_action': last_action,
                        'changed_atoms': path['changed_atoms'],
                        'mol_ind': path['mol_ind'],
                        'action_str': path['action_str'],
                        'action_prob': path['action_prob']+[action['prob'].item()],
                        'mol': copy.deepcopy(path['mol']),
                    }
                    if self.export_samples:
                        new_path['mol_graphs'] = path['mol_graphs'] + [self.ravel_graph(*path['mol_graph'])]
                    new_batch_paths.append(new_path)

        return new_batch_paths  

    def apply_actions(self, paths):
        for j, path in enumerate(paths):  # 当 step_i == 0 时，这里都跳过了
            if path['finished'] or 'last_action' not in path:  # first step of generation or finished
                continue

            # action_prob, action_inds, n_nodes = path['last_action']
            max_map_num = path['max_map_num']

            action = get_action_object(path['last_action'], max_map_num, feat_vocab = self.feat_vocab, action_vocab=self.action_vocab)

            changed_atoms = path['changed_atoms']
            changed_atoms = changed_atoms.copy()
            if action.atom_map1 > 0:
                changed_atoms.add(action.atom_map1)
            if action.atom_map2 > 0:
                changed_atoms.add(action.atom_map2)
            if isinstance(action, AddRingAction):
                for map_num in action.new_atoms_map_nums:
                    changed_atoms.add(map_num)
            if isinstance(action, AddMotifAction_with_dummySMI):
                for map_num in action.new_atoms_map_nums:
                    changed_atoms.add(map_num)               

            if len(changed_atoms) > 0:
                path['max_map_num'] = max(max(changed_atoms), max_map_num)
            path['changed_atoms'] = changed_atoms
            path['actions'] = path['actions'] + [(action, path['last_action']['prob'])]

            # try:
            path['mol'] = action.apply(path['mol'])
            # print("Action:", action)
            # print("SMI:", Chem.MolToSmiles(path['mol']))
            # print()

            path['mol_graph'] = get_graph(path['mol'],
                                            ravel=False, to_array=True,
                                atom_feature_keys=self.feat_vocab['atom_feature_keys'],
                                bond_feature_keys=self.feat_vocab['bond_feature_keys'],
                                atom_prop2oh=self.feat_vocab['prop2oh']['atom'],
                                bond_prop2oh=self.feat_vocab['prop2oh']['bond'])
            path['action_str'] = path['action_str']+[str(action)]

    def split_batch(self,analyzed_paths):
        '''
        同一个mol_ind的分子不能在相同batch内,否则graph_ind会出问题
        '''
        for i, path in enumerate(analyzed_paths):
            path['mol_ind']
            pass
    
    def beamsearch(self,mols):
        self.paths+=self.get_start_path(mols)
        
        for step_i in range(self.max_steps):
            # apply last actions for all paths
            self.apply_actions(self.paths)
            
            # find paths with duplicate graphs
            if step_i > 0 and self.filter_duplicates:
                self.paths = filter_duplicate_paths(self.paths, len(mols))

            analyzed_paths = [path for path in self.paths if not path['finished']]  # 判断是否所有的 samples 都完成 generation 了
            if len(analyzed_paths) == 0:
                break
            
            self.paths = [path for path in self.paths if path['finished']]

            
            
            n_batches = int(np.ceil(len(analyzed_paths) / self.batch_size))
            path_batches = np.array_split(analyzed_paths, n_batches)

            for k, p_batch in enumerate(path_batches):
                new_paths = self.process_paths_batch(p_batch)
                self.paths += new_paths
            
            # sort paths by probabilities and limit number of paths
            self.paths = get_top_k_paths(self.paths, self.beam_size, len(mols))
            
            
            if all(p['finished'] for p in self.paths):
                break
        
        finished_paths = [[] for _ in range(len(mols))]


        for path in self.paths:
            ind = path['mol_ind']
            if path['finished']:
                # try:
                adj, nodes = path['mol_graph']
                mol_ind = path['mol_ind']
                output_mols =[path['mol']]  # 
                
                # mols_from_graph(rdkit_cache, mols[mol_ind], bond_dirs[mol_ind], adj, nodes, changed_atoms=path['changed_atoms'], only_edited=False)
                final_smi = '.'.join([Chem.MolToSmiles(mol) for mol in output_mols if mol is not None])

                for mol in output_mols:
                    if mol:
                        for a in mol.GetAtoms():
                            a.ClearProp("molAtomMapNumber")
                final_smi_unm = '.'.join([Chem.MolToSmiles(mol) for mol in output_mols if mol is not None])


                if not self.filter_incorrect or final_smi_unm:
                    result_path = {
                        'final_smi': final_smi,
                        'final_smi_unmapped': final_smi_unm,
                        'prob': path['prob'],
                        'actions': path['actions'],
                        'n_steps': path['n_steps'],
                        'changed_atoms': path['changed_atoms'],
                        'mol_graph': path['mol_graph']
                    }
                    if 'class_output' in path:
                        result_path['class_output'] = path['class_output']
                    if self.export_samples:
                        result_path['mol_graphs'] = path['mol_graphs']

                    finished_paths[ind].append(result_path)

        return finished_paths


