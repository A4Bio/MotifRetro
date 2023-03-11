"""
MEGAN model for reaction generation
"""
from typing import Tuple, Optional, List

import copy
import torch
from torch import nn
from modules.MotifRetro2_GNN_module import MeganDecoder
from modules.MotifRetro2_GNN_module import MeganEncoder
from torch_scatter import scatter_softmax, scatter_mean, scatter_max

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



class MotifRetro_model(nn.Module):
    def __init__(self, 
                 n_atom_actions: int, 
                 n_bond_actions: int, 
                 feat_vocab: dict,
                 bond_emb_dim: int = 8, 
                 hidden_dim: int = 512, 
                 reaction_type_given: bool = False, 
                 n_reaction_types: int = 10,
                 reaction_type_emb_dim: int = 16,
                 useAtomAction: bool = True,
                 useBondAction: bool = True,
                 useAttachAction: bool = True,
                 use_degree_feat: bool = False,
                 predict_atom_num: bool = False,
                 n_encoder_conv = 3,
                 enc_residual = True,
                 enc_dropout = 0.0,
                 n_fc = 2,
                 n_decoder_conv = 2,
                 dec_residual = True,
                 bond_atom_dim = 128,
                 atom_fc_hidden_dim = 128,
                 bond_fc_hidden_dim = 128,
                 dec_dropout = 0.0,
                 att_heads = 8,
                 att_dim = 128,
                 attention_dropout = 0.1,
                 temperature = 1.0,
    ):
        super(MotifRetro_model, self).__init__()
        self.prop2oh = feat_vocab['prop2oh']
        self.reaction_type_given = reaction_type_given
        self.useAtomAction = useAtomAction
        self.useBondAction = useBondAction
        self.useAttachAction = useAttachAction
        self.use_degree_feat = use_degree_feat

        total_atom_oh_len = sum(len(self.prop2oh['atom'][key]) + 1 for key in self.prop2oh['atom'])
        total_bond_oh_len = sum(len(self.prop2oh['bond'][key]) + 1 for key in self.prop2oh['bond'])


        if reaction_type_given:
            assert reaction_type_emb_dim < hidden_dim
            assert reaction_type_emb_dim < bond_emb_dim
            self.atom_reaction_type_embedding = nn.Embedding(n_reaction_types, hidden_dim)
            self.bond_reaction_type_embedding = nn.Embedding(n_reaction_types, bond_emb_dim)
            self.atom_embedding = nn.Linear(total_atom_oh_len, hidden_dim)
            self.bond_embedding = nn.Linear(total_bond_oh_len, hidden_dim)
        else:
            self.reaction_type_embedding = None
            self.atom_embedding = nn.Linear(total_atom_oh_len, hidden_dim)
            self.bond_embedding = nn.Linear(total_bond_oh_len, hidden_dim)
        
        if self.use_degree_feat:
            self.degree_embedding = nn.Embedding(10, hidden_dim)

        self.encoder = MeganEncoder(hidden_dim=hidden_dim, 
                                    bond_emb_dim=bond_emb_dim, 
                                    n_encoder_conv=n_encoder_conv, 
                                    enc_residual=enc_residual, 
                                    enc_dropout=enc_dropout,
                                    att_heads = att_heads,
                                    att_dim = att_dim,
                                    attention_dropout=attention_dropout,
                                    )
        
        self.decoder = MeganDecoder(hidden_dim=hidden_dim, 
                                    bond_emb_dim=bond_emb_dim,
                                    n_atom_actions=n_atom_actions, #
                                    n_bond_actions=n_bond_actions-1,#
                                    feat_vocab=feat_vocab, 
                                    n_fc = n_fc,
                                    n_decoder_conv = n_decoder_conv,
                                    dec_residual = dec_residual,
                                    bond_atom_dim = bond_atom_dim,
                                    atom_fc_hidden_dim = atom_fc_hidden_dim,
                                    bond_fc_hidden_dim = bond_fc_hidden_dim,
                                    dec_dropout = dec_dropout,
                                    predict_atom_num = predict_atom_num,
                                    att_heads = att_heads,
                                    att_dim = att_dim,
                                    attention_dropout = attention_dropout,
                                    temperature = temperature,
                                    )
        
        self.graph_predictor = nn.Sequential(nn.Linear(2*hidden_dim, 128),
                                             nn.ReLU(),
                                             nn.Linear(128,4))

    def _preprocess_sparse(self,x):
        atom_feats = self.atom_embedding(x['atom_feats']) 
        bond_feats = self.bond_embedding(x['bond_feats'])

        if self.reaction_type_given:
            graph_id = x['graph_id']
            new_graph_id = copy.deepcopy(x['graph_id'])
            unique_graph_id = graph_id.unique().tolist()
            for id in unique_graph_id:
                new_graph_id[graph_id == id] = unique_graph_id.index(id)
            atom_reaction_type_emb = self.atom_reaction_type_embedding(x['reaction_type'][new_graph_id] - 1)
            bond_reaction_type_emb = self.bond_reaction_type_embedding(x['reaction_type'][new_graph_id[x['edge_idx'][:,0]]] - 1)
            atom_feats += atom_reaction_type_emb
            bond_feats += bond_reaction_type_emb

        if self.use_degree_feat:
            atom_feats += self.degree_embedding(x['degree'])

        
        results = {"atom_mask": x['atom_mask'],
                "atom_feats": atom_feats,
                "bond_feats": bond_feats,
                "atom_action_mask": x['atom_action_mask'],
                "bond_action_mask": x['bond_action_mask'],
                "graph_id": x['graph_id'],
                "edge_idx": x["edge_idx"],
                "supernode_idx": x['supernode_idx'],
                "padding_idx": x['padding_idx'],
                "edge_idx_remove_supernode": x["edge_idx_remove_supernode"]
        }

        x.update(results)
        return x
    
    def forward(self, batch):
        n_steps = len(batch)
        prediction_scores = []

        state_dict = None
        for step_i in range(n_steps):
            step_batch = batch[step_i]
            step_batch = self._preprocess_sparse(step_batch)
            if step_i==0:
                step_batch['x'] = step_batch['atom_feats']
                step_batch['x_e'] = step_batch['bond_feats']
                step_batch['m'] = None
                step_batch['m_e'] = None
                step_batch['hidden_states'] = None
            else:
                # update step_batch
                step_batch['x'] = step_batch['atom_feats']
                step_batch['x_e'] = step_batch['bond_feats']
                
                # maked node_id
                state_node_id = step_results['state']['graph_id']*1000+step_results['state']['local_node_idx']
                new_node_id = step_batch['graph_id']*1000+step_batch['local_node_idx']
                
                # make edge_id
                graph_id = step_results['state']['graph_id'][step_results['state']['edge_idx'][:,0]]
                local_id = step_results['state']['local_edge_idx'][:,0]*300 + step_results['state']['local_edge_idx'][:,1]
                state_edge_id = graph_id*100000+local_id
                new_edge_id = step_batch['graph_id'][step_batch['edge_idx'][:,0]]*100000+step_batch['local_edge_idx'][:,0]*300+step_batch['local_edge_idx'][:,1]
                
                step_batch['m'], new_node_idx, old_node_idx = self.make_state(
                    step_results['state']['m'], 
                    state_node_id,
                    new_node_id)
                
                step_batch['m_e'], new_edge_idx, old_edge_idx = self.make_state(
                    step_results['state']['m_e'], 
                    state_edge_id, 
                    new_edge_id)
                
                step_batch['hidden_states'] = {}
                for layer_id, (h,e) in step_results['state']['hidden_states'].items():
                    step_batch['hidden_states'][layer_id] = (self.make_state(
                        h, 
                        state_node_id,
                        new_node_id,
                        new_node_idx, old_node_idx)[0],
                        self.make_state(
                        e, 
                        state_edge_id,
                        new_edge_id,
                        new_edge_idx, old_edge_idx)[0])
                
                # step_batch['h'], _, _ = self.make_state(
                #     step_results['state']['h'], 
                #     state_node_id,
                #     new_node_id,
                #     new_node_idx, old_node_idx)
                
                # step_batch['h_e'], _, _ = self.make_state(
                #     step_results['state']['h_e'], 
                #     state_edge_id,
                #     new_edge_id,
                #     new_edge_idx, old_edge_idx)
            
            step_results = self.forward_step_sparse(step_batch, state_dict=state_dict)
            state_dict = {
                'state': step_results['state'],
            }
            prediction_scores.append(step_results['pred_scores'])
            
        
        return prediction_scores

    def make_state(self, state, state_id, new_id, new_idx=None, old_idx=None):
        """Create a new state tensor with the same size as graph_id"""
        new_state = torch.zeros(new_id.shape[0], state.shape[1], device=state.device)
        if new_idx is None:
            idx = (new_id.reshape(-1,1) == state_id.reshape(1,-1)).nonzero()
            new_idx, old_idx = idx[:,0], idx[:,1]
        new_state[new_idx] = state[old_idx]
        
        return new_state, new_idx, old_idx
    
    def patch_softmax(self, pred_scores, temperature=1.0, useAtomAction=True, useBondAction=True, useAttachAction=False):

        # Get all the scores and the batch ids
        scores = torch.cat([pred_scores['atom_action'], 
                            pred_scores['bond_action'],
                            pred_scores['attach_action']])
        batch_id = torch.cat([pred_scores['atom_idx'][0], 
                              pred_scores['bond_idx'][0],
                              pred_scores['attach_idx'][0]]).long()
        
        # Get the mask for all the actions
        all_actions_mask = torch.cat([pred_scores['atom_mask']*useAtomAction,
                                      pred_scores['bond_mask']*useBondAction,
                                      pred_scores['attach_mask']*useAttachAction])
        soft_mask = (1.0 - all_actions_mask) * -1e9
        
        # Apply softmax
        scores = scatter_softmax(scores / temperature + soft_mask, batch_id) * all_actions_mask
        
        # Get the length of each type of action
        L_atom = len(pred_scores['atom_action'])
        L_bond = len(pred_scores['bond_action'])
        
        # Split the scores back into the three action types
        pred_scores['atom_action'] = scores[:L_atom]
        pred_scores['bond_action'] = scores[L_atom:L_atom+L_bond]
        pred_scores['attach_action'] = scores[L_atom+L_bond:]
        return pred_scores
            
        
    
    def forward_step_sparse(self, step_batch: dict, state_dict=Optional[dict],
                     first_step: Optional[List[int]] = None) -> dict:
        
        x, m, x_e, m_e, new_hidden_states = self.encoder(x=step_batch['x'], 
                                  m=step_batch['m'],
                                  x_e = step_batch['x_e'],
                                  m_e = step_batch['m_e'],
                                  hidden_states=step_batch['hidden_states'],
                                  edge_idx=step_batch['edge_idx'])
        
        step_batch['atom_feats'] = x

        node_state, pred_scores = self.decoder(step_batch)
        
        graph_id_unique = step_batch['graph_id'].unique()
        
        
        # -------------------action type prediction
        graph_feat = torch.cat([
                scatter_mean(node_state, step_batch["graph_id"], dim=0),
                scatter_max(node_state, step_batch["graph_id"], dim=0)[0]], 
                dim=1)
        graph_feat = graph_feat[graph_id_unique]
        
        pred_action_type = self.graph_predictor(graph_feat)
        # ------------------------------------------
        
        
        pred_scores = self.patch_softmax(pred_scores, self.useAtomAction, self.useBondAction, self.useAttachAction)
        pred_scores['pred_action_type'] = pred_action_type

        
        result = {
            'state': {
                "x": x, 
                "m": m,
                "x_e": x_e,
                "m_e": m_e,
                "hidden_states": new_hidden_states,
                "graph_id": step_batch["graph_id"],
                "local_node_idx": step_batch["local_node_idx"],
                "local_edge_idx": step_batch["local_edge_idx"],
                "edge_idx": step_batch["edge_idx"]
                }, 
            'pred_scores': pred_scores
        }

        return result

