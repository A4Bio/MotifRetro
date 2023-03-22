"""
MEGAN model for reaction generation
"""
from typing import Tuple, Optional, List

import copy
import torch
# from src.feat import ORDERED_ATOM_OH_KEYS, ORDERED_BOND_OH_KEYS, ORDERED_MOTIF_OH_KEYS
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.MotifRetro5_GNN_module import MeganDecoder
from modules.MotifRetro5_GNN_module import MeganEncoder
import numpy as np
import random
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_softmax, scatter_mean, scatter_max
# from transformers import BertConfig, BertModel
import torch




device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def to_one_hot(x, dims: int):
    one_hot = torch.FloatTensor(*x.shape, dims).zero_().to(device)
    x = torch.unsqueeze(x, -1)
    target = one_hot.scatter_(x.dim() - 1, x.data, 1)

    target = Variable(target)
    return target


default_atom_features = 'is_supernode', 'atomic_num', 'formal_charge', 'chiral_tag', \
                        'num_explicit_hs', 'is_aromatic', 'is_edited'
default_bond_features = 'bond_type', 'bond_stereo', 'is_edited'

default_motif_features = 'is_supernode', 'motif_ids'


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ActionFormer(nn.Module):
    def __init__(self, vocab_size=100, outdim=1024, num_hidden_layers=5) -> None:
        super().__init__()
        # Define BERT configuration
        config = BertConfig(vocab_size=vocab_size, hidden_size=768, num_hidden_layers=num_hidden_layers, num_attention_heads=12, intermediate_size=1024, 
                            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        self.readout = nn.Linear(768, outdim)

        self.bert_model = BertModel(config)
    
    def forward(self, prev_action, position_ids, attention_mask):
        outputs = self.bert_model(prev_action, position_ids=position_ids, attention_mask=attention_mask)
        out = self.readout(outputs[0][:,0,:])
        return out


class MotifRetro_model(nn.Module):
    def __init__(self, 
                 n_atom_actions: int, 
                 n_bond_actions: int, 
                 feat_vocab: dict,
                 bond_emb_dim: int = 8, 
                 hidden_dim: int = 512, 
                 reaction_type_given: bool = False, 
                 n_reaction_types: int = 10,
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
                 scale_up = 4
    ):
        super(MotifRetro_model, self).__init__()
        self.prop2oh = feat_vocab['prop2oh']
        self.feat_vocab = feat_vocab
        self.n_actions = n_atom_actions
        self.n_bond_actions = n_bond_actions
        self.bond_emb_dim = bond_emb_dim
        self.hidden_dim = hidden_dim
        self.reaction_type_given = reaction_type_given
        self.useAtomAction = useAtomAction
        self.useBondAction = useBondAction
        self.useAttachAction = useAttachAction
        self.use_degree_feat = use_degree_feat

        total_atom_oh_len = sum(len(self.prop2oh['atom'][key]) + 1 for key in self.prop2oh['atom'])
        total_bond_oh_len = sum(len(self.prop2oh['bond'][key]) + 1 for key in self.prop2oh['bond'])


        if reaction_type_given:
            self.atom_reaction_type_embedding = nn.Embedding(n_reaction_types, hidden_dim)
            self.bond_reaction_type_embedding = nn.Embedding(n_reaction_types, bond_emb_dim)
            self.atom_embedding = nn.Linear(total_atom_oh_len, hidden_dim)
            self.bond_embedding = nn.Linear(total_bond_oh_len, bond_emb_dim)
            

        else:
            self.reaction_type_embedding = None
            self.atom_embedding = nn.Linear(total_atom_oh_len, hidden_dim)
            self.bond_embedding = nn.Linear(total_bond_oh_len, bond_emb_dim)
        
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
                                    scale_up = scale_up
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
                                    scale_up = scale_up
                                    )
        
        self.graph_predictor = nn.Sequential(nn.Linear(2*hidden_dim, 128),
                                             nn.ReLU(),
                                             nn.Linear(128,4))
        
        self.global_id = n_atom_actions+n_bond_actions
        
        # self.actionformer = ActionFormer()

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
    
    def forward(self, batch, given_reaction_center=False):
        n_steps = len(batch)
        prediction_scores = []

        state_dict = None
        steps_range = range(1, n_steps) if given_reaction_center else range(n_steps)
        
        B = batch[0]['graph_id'].unique().shape[0]
        device = batch[0]['graph_id'].device
        prev_actions = torch.zeros(B, len(batch)+1, device=device).long()-1
        
        for idx, step_batch in enumerate(batch):
            g_id = step_batch['graph_id'].unique()
            prev_actions[g_id, idx+1] = step_batch['action_ind']

        for step_i in steps_range:
            step_batch = batch[step_i]
            prev_action = prev_actions[step_batch['graph_id'].unique(),:step_i+1]
            step_batch["prev_action"] = prev_action
            step_results = self.forward_one_step(step_batch, state_dict=state_dict)
            state_dict = {
                'state': step_results['state'],
            }
            prediction_scores.append(step_results['pred_scores'])
            
        
        return prediction_scores

    def make_state(self, state, state_graph_id, graph_id):
        graph_id_uni = graph_id.unique()
        
        state_list = []
        for id in graph_id_uni:
            mask = state_graph_id == id
            val = state[mask]
            N_new = (graph_id==id).sum()
            if mask.sum() != N_new:
                val = torch.vstack([val, torch.ones(N_new - mask.sum(), val.shape[1]).cuda() * 0])
                # val = F.pad(val, 0,0, 0, N_new-val.shape[0])  # [20, 1024] -> [21, 1024]
            state_list.append(val)
        state_val = torch.cat(state_list, dim=0)
        # state_val = torch.zeros_like(state_val)# remove hidden state
        return state_val
    
    def patch_softmax(self, pred_scores, temperature=1.0, useAtomAction=True, useBondAction=True, useAttachAction=False):

        scores = torch.cat([pred_scores['atom_action'], 
                            pred_scores['bond_action'],
                            pred_scores['attach_action']])
        batch_id = torch.cat([pred_scores['atom_idx'][0], 
                              pred_scores['bond_idx'][0],
                              pred_scores['attach_idx'][0]]).long()
        
        all_actions_mask = torch.cat([pred_scores['atom_mask']*useAtomAction,
                                      pred_scores['bond_mask']*useBondAction,
                                      pred_scores['attach_mask']*useAttachAction])
        soft_mask = (1.0 - all_actions_mask) * -1e9
        
        scores = scatter_softmax(scores / temperature + soft_mask, batch_id) * all_actions_mask
        
        L_atom = len(pred_scores['atom_action'])
        L_bond = len(pred_scores['bond_action'])
        pred_scores['atom_action'] = scores[:L_atom]
        pred_scores['bond_action'] = scores[L_atom:L_atom+L_bond]
        pred_scores['attach_action'] = scores[L_atom+L_bond:]
        return pred_scores
            
        
    
    def forward_one_step(self, step_batch: dict, state_dict=Optional[dict],
                     first_step: Optional[List[int]] = None) -> dict:
        step_batch = self._preprocess_sparse(step_batch)
        
        step_batch = self.encoder(step_batch)
        if  state_dict is not None:
            node_state = state_dict['state']["val"]
            state_graph_id = state_dict['state']["graph_id"]
            graph_id = step_batch['graph_id']
            state_val = self.make_state(node_state, state_graph_id, graph_id) 
            step_batch['temporal_feat'] = state_val
        else:
            step_batch['temporal_feat'] = torch.zeros_like(step_batch['atom_feats'])

        node_state, temporal_feat, pred_scores = self.decoder(step_batch)
        
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
            'state': {"val": node_state, "graph_id": step_batch["graph_id"]}, 
            'pred_scores': pred_scores
        }

        return result


if __name__ == "__main__":
    model = ActionFormer()
    prev_action = torch.randint(0, 10, (32, 4))
    prev_t = torch.arange(4).unsqueeze(0).repeat(32, 1)
    mask = torch.ones(32, 4).bool()
    out = model(prev_action, prev_t, mask)
    