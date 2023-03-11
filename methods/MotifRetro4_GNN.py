import os
import time
import copy
from typing import *

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Mol
from torch_scatter import scatter_max, scatter_mean, scatter_sum
from tqdm import tqdm

from methods.base_method import Base_method
from models.MotifRetro4_GNN_model import Megan
from src.feat.utils import fix_explicit_hs
from src.model.beam_search import  BeamSearch 
from src.model.megan_utils import RdkitCache, get_base_action_masks
from src.utils.retro_utils import mol_to_unmapped_smiles

from collections import Counter


class MotifRetro(Base_method):
    def __init__(self, args, device, steps_per_epoch, feat_vocab, action_vocab):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.use_motif_action = self.args.use_motif_action
        self.feat_vocab = feat_vocab
        self.action_vocab = action_vocab
        self.model = self._build_model(feat_vocab, action_vocab)
        self.mix_precision = False
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)

    def _build_model(self, feat_vocab, action_vocab):
        
        
        model = Megan(n_atom_actions=action_vocab['n_atom_actions'],
                  n_bond_actions=action_vocab['n_bond_actions'],
                  bond_emb_dim = self.args.bond_emb_dim,
                  hidden_dim = self.args.hidden_dim,
                  reaction_type_given = self.args.reaction_type_given,
                  n_reaction_types = self.args.n_reaction_types,
                  reaction_type_emb_dim = self.args.reaction_type_emb_dim,
                  feat_vocab=feat_vocab,
                  useAtomAction = self.args.useAtomAction,
                  useBondAction = self.args.useBondAction,
                  useAttachAction = self.args.useAttachAction,
                  use_past_values = self.args.use_past_values,
                  use_synthon_feat = self.args.use_synthon_feat,
                  use_degree_feat = self.args.use_degree_feat,
                  predict_atom_num = self.args.predict_atom_num,
                  n_encoder_conv = self.args.n_encoder_conv,
                  enc_residual = self.args.enc_residual,
                    enc_dropout = self.args.enc_dropout,
                    n_fc = self.args.n_fc,
                    n_decoder_conv = self.args.n_decoder_conv,
                    dec_residual = self.args.dec_residual,
                    bond_atom_dim = self.args.bond_atom_dim,
                    atom_fc_hidden_dim = self.args.atom_fc_hidden_dim,
                    bond_fc_hidden_dim = self.args.bond_fc_hidden_dim,
                    dec_dropout = self.args.dec_dropout,
                    att_heads = self.args.att_heads,
                    att_dim = self.args.att_dim,
                    attention_dropout=self.args.attention_dropout,
                    temperature = self.args.temperature,
                  ).to(self.device)
        
        model.to(self.device)
        return model
    
    def get_target_sparse(self, batch):
        targets = []
        for t, step_batch in enumerate(batch):
            atom_action = step_batch['atom_action']
            bond_action_exp = torch.zeros(step_batch['edge_idx'].shape[0], atom_action.shape[1], device=atom_action.device)
            
            edge_idx = step_batch['edge_idx']
            bond_action_idx = step_batch['bond_action_idx']
            
            N = edge_idx.max()+1
            mask = (edge_idx[:,0]*N + edge_idx[:,1]).view(-1,1) == (step_batch['bond_action_idx'][:,0]*N + bond_action_idx[:,1]).view(1,-1)
            try:
                bond_action_exp[mask.any(dim=-1)] = step_batch['bond_action'] 
            except:
                bond_action_exp[mask.any(dim=-1)] = step_batch['bond_action'][mask.any(dim=0)] # 新加了化学键导致错误

            targets.append(torch.cat([atom_action, bond_action_exp], dim=0).view(-1))
        
        return targets


    def get_loss_sparse(self, prediction_scores):
        eps = 1e-9
        total_loss = None
        count = 0
        all_losses = []

        for pred_score in prediction_scores:
            # loss_atom = -torch.log2(pred_score['atom_action']+~pred_score['atom_target'].bool()+eps)*pred_score['atom_target']
            
            # loss_bond = -torch.log2(pred_score['bond_action']+~pred_score['bond_target'].bool()+eps)*pred_score['bond_target']
            
            # loss_attach = -torch.log2(pred_score['attach_action']+~pred_score['attach_target'].bool()+eps)*pred_score['attach_target']
            
            loss_atom = -torch.log2(pred_score['atom_action']+eps)*pred_score['atom_target'] - torch.log2(1-pred_score['atom_action']+eps)
            
            loss_bond = -torch.log2(pred_score['bond_action']+eps)*pred_score['bond_target'] - torch.log2(1-pred_score['bond_action']+eps)
            
            loss_attach = -torch.log2(pred_score['attach_action']+eps)*pred_score['attach_target'] - torch.log2(1-pred_score['attach_action']+eps)
            
            
            
            loss = loss_atom.sum()*self.args.useAtomAction\
                   +loss_bond.sum()*self.args.useBondAction\
                   + loss_attach.sum()*self.args.useAttachAction
            all_losses.append(loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            count += 1

        return total_loss / count / self.args.batch_size

    def get_action_type_loss(self, batch, prediction_scores):
        action_type_map = {"change_atom":0, "add_motif":0, 
                           "change_bond":1,
                           "attach_atom":2,
                           "stop":3}
        
        loss_list = []
        for step_patch, pred_score in zip(batch, prediction_scores):
            action_types = [one[2][0] for one in step_patch['action_tuples']]
            action_types_ind = [action_type_map[one] for one in action_types]
            
            label = torch.as_tensor(action_types_ind, device = self.device)
            pred_action_type = pred_score['pred_action_type']
            
            loss_list.append(self.criterion(pred_action_type, label))
        
        return torch.stack(loss_list).mean()
  
    def get_metric_sparse(self, prediction_scores, batch):
        correct_actions = []
        incorrect_actions = []
        
        # (#Graph, #edit_length)
        Max_N_graphs = len(batch[0]['graph_id'].unique())
        correct = torch.zeros(Max_N_graphs, len(batch)).cuda()
        mask = torch.zeros(Max_N_graphs, len(batch)).cuda()
        
        for i, (pred_score, patch) in enumerate(zip(prediction_scores, batch)):
            action_idx_cat = torch.cat([pred_score['atom_idx'], pred_score['bond_idx'], pred_score['attach_idx']], dim=1)
            pred_score_cat = torch.cat([pred_score['atom_action'], pred_score['bond_action'], pred_score['attach_action']])
            target_cat = torch.cat([pred_score['atom_target'], pred_score['bond_target'], pred_score['attach_target']])
            
            y_val_pred, y_pred = scatter_max(pred_score_cat, action_idx_cat[0])
            
            y_val, y_true = scatter_max(target_cat, action_idx_cat[0])
            
            y_val_one = y_val == 1
            compare = ((y_pred == y_true) & y_val_one).float()
            
            existing_mask = (y_true!=action_idx_cat[0].shape[0])
            graph_id = action_idx_cat[0][y_true[existing_mask]]
            
            correct[graph_id, i] = compare[existing_mask]
            mask[graph_id, i] = 1
            
            true_action_idx = action_idx_cat[:, y_true[existing_mask]]
            
            action_types = true_action_idx[4]
            for action_type, ind, cmp in zip(action_types, true_action_idx[3], compare):
                if action_type==0: # 这里也包含了stop action
                    action_tuple = self.action_vocab['atom_ind2action'][int(ind)]
                elif action_type==1:
                    action_tuple = self.action_vocab['bond_ind2action'][int(ind)]
                elif action_type==2:
                    action_tuple = self.action_vocab['bond_ind2action'][int(ind)]
                else:
                    raise "Unknown action type!"
                
                if cmp:
                    correct_actions.append(action_tuple)
                else:
                    incorrect_actions.append(action_tuple)
            
        
        step_correct = torch.sum(correct) / mask.sum()
        metric = {
            "step_acc": step_correct.cpu().detach().numpy()
        }
        
        all_correct = (torch.sum(correct, dim=-1) == mask.sum(dim=-1)).float()
        acc = all_correct.mean()
        metric['acc'] = acc.detach().cpu().numpy()
        
        
        action_type_map = {"change_atom":0, "add_motif":0, 
                           "change_bond":1,
                           "attach_atom":2,
                           "stop":3}
        
        pred_action_type_list = []
        for step_patch, pred_score in zip(batch, prediction_scores):
            action_types = [one[2][0] for one in step_patch['action_tuples']]
            action_types_ind = [action_type_map[one] for one in action_types]
            
            label = torch.as_tensor(action_types_ind, device = self.device)
            pred_action_type = pred_score['pred_action_type']
            
            pred_action_type_list.append(pred_action_type.argmax(dim=1) == label)
        metric['pred_action_type_acc'] = torch.cat(pred_action_type_list, dim=0).float().mean().cpu().numpy()
        
        return metric, correct_actions, incorrect_actions, all_correct, correct
    
    def forward_loss_metric(self, batch):
        prediction_scores = self.model.forward(batch)
                
        batch_metric = {"acc": 0.0, "step_acc": 0.0}
        loss = self.get_loss_sparse(prediction_scores)
        loss_actype = self.get_action_type_loss(batch, prediction_scores)
        
        # loss = loss + loss_actype
        
        batch_metric, correct_actions, incorrect_actions, batch_all_correct, step_correct = self.get_metric_sparse(prediction_scores,  batch)

        batch_metric['loss'] = loss.detach().cpu().numpy()
        return loss, batch_metric, correct_actions, incorrect_actions
             
    def train_one_epoch(self, train_loader):
        scaler = torch.cuda.amp.GradScaler()
        self.model.train()
        self.train_loader = train_loader
        
        train_pbar = tqdm(train_loader)
        metric_list = []

        all_correct = []
        all_in_correct = []

        for idx, (batch, source_smi, target_smi) in enumerate(train_pbar):
            self.optimizer.zero_grad()
            

            
            if self.mix_precision:
                loss, batch_metric, correct_actions, incorrect_actions = self.forward_loss_metric(batch)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss, batch_metric, correct_actions, incorrect_actions = self.forward_loss_metric(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
            
            all_correct += correct_actions
            all_in_correct += incorrect_actions
            metric_list.append(batch_metric)
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
        

        epoch_metric = {}
        for k in batch_metric.keys():
            epoch_metric[k] = np.mean([one[k] for one in metric_list])
            
        self.scheduler.step()
        return epoch_metric

    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        valid_pbar = tqdm(valid_loader)

        all_correct = []
        all_in_correct = []
        metric_list = []
        for idx, (batch, source_smi, target_smi) in enumerate(valid_pbar):
            loss, batch_metric, correct_actions, incorrect_actions = self.forward_loss_metric(batch)

            batch_metric['loss'] = loss.detach().cpu().numpy()

            all_correct += correct_actions
            all_in_correct += incorrect_actions

            metric_list.append(batch_metric)

        all_correct = dict(Counter(all_correct))
        all_in_correct = dict(Counter(all_in_correct))

        epoch_metric = {}
        for action_tuple in self.action_vocab["action_freq"].keys():
            this_correct = 0
            this_in_correct = 0
            if action_tuple in all_correct:
                this_correct = all_correct[action_tuple]
            
            if action_tuple in all_in_correct:
                this_in_correct = all_in_correct[action_tuple]
        
            try:
                epoch_metric[f'act_{str(action_tuple)}'] = this_correct/ (this_correct + this_in_correct)  
            except:
                epoch_metric[f'act_{str(action_tuple)}'] = 0  # [0, 0, None]

        for k in batch_metric.keys():
            epoch_metric[k] = np.mean([one[k] for one in metric_list])
            
        return epoch_metric

    def test_one_epoch(self, test_loader):  # support beam-search
        # prog_bar = tqdm(desc=f'{save_path} beam search on {split_key}', total=len(split_ind))
        def prediction_is_correct(y_pred: str, y_true: str):
            if not y_pred:
                return False

            try:
                y_pred = Chem.CanonSmiles(y_pred)
                y_pred = y_pred.replace("@@", "@")
            except:
                return False
            y_true = Chem.CanonSmiles(y_true)
            y_true = y_true.replace("@@", "@")
            # order of compounds in SMILES does not matter
            pred_mols = Counter(y_pred.split('.'))
            true_mols = Counter(y_true.split('.'))

            for mol_smi in true_mols:
                if pred_mols[mol_smi] != true_mols[mol_smi]:
                    return False
            return True


        def remap_reaction_to_canonical(input_mol: Mol, target_mol: Mol) -> Tuple[Mol, Mol]:
            """
            Re-maps reaction according to order of atoms in RdKit - this makes sure that stereochemical SMILES are canonical.
            Note: this method does not transfer any information from target molecule to the input molecule
            (the input molecule is mapped according to its order of atoms in its canonical SMILES)
            """

            # converting Mol to smiles and again to Mol makes atom order canonical
            input_mol = Chem.MolFromSmiles(Chem.MolToSmiles(input_mol))
            target_mol = Chem.MolFromSmiles(Chem.MolToSmiles(target_mol))

            map2map = {}
            for i, a in enumerate(input_mol.GetAtoms()):
                map2map[int(a.GetAtomMapNum())] = i + 1
                a.SetAtomMapNum(i + 1)

            max_map = max(map2map.values())

            for i, a in enumerate(target_mol.GetAtoms()):
                old_map = int(a.GetAtomMapNum())
                if old_map in map2map:
                    new_map = map2map[old_map]
                else:
                    new_map = max_map + 1
                    max_map += 1
                a.SetAtomMapNum(new_map)

            return input_mol, target_mol
            
        start_time = time.time()
        n_samples, n_gen_reactions = 0, 0
        is_incorrect, is_duplicate = [], []
        n_preds = np.zeros(len(test_loader.dataset), dtype=int)

        top_k = np.zeros(self.args.beam_size, dtype=float)
        accs = np.zeros(self.args.beam_size, dtype=float)
        
        base_action_masks = get_base_action_masks(self.args.n_max_atoms + 1, action_vocab=self.action_vocab)


        for i, batch in tqdm(enumerate(test_loader)):
            
            input_mols = []
            target_mols = []

            for j in range(len(batch['source_smi'])):

                try:
                    input_mol = Chem.MolFromSmiles(batch['source_smi'][j])
                    target_mol = Chem.MolFromSmiles(batch['target_smi'][j])

                    
                    # remap input and target molecules according to canonical SMILES atom order
                    input_mol, target_mol = remap_reaction_to_canonical(input_mol, target_mol)

                    # fix a bug in marking explicit Hydrogen atoms by RdKit
                    input_mol = fix_explicit_hs(input_mol)

                except Exception as e:
                    # logger.warning(f'Exception while input mol to SMILES {str(e)}')
                    input_mol = None
                    target_mol = None

                if input_mol is None or target_mol is None:
                    input_mols.append(None)
                    target_mols.append(None)
                    continue

                input_mols.append(input_mol)
                target_mols.append(target_mol)

            if 'reaction_type_id' in batch:
                reaction_types = batch['reaction_type_id']
            else:
                reaction_types = None
            
            eval_model = BeamSearch([self.model], 
                                    max_steps=self.args.max_gen_steps,
                                    beam_size=self.args.beam_size, batch_size=self.args.eval_batch_size,
                                    base_action_masks=base_action_masks, max_atoms=self.args.n_max_atoms,
                                    reaction_types=reaction_types,
                                    feat_vocab=self.feat_vocab,
                                    action_vocab=self.action_vocab, 
                                    sparse=self.args.sparse)

            with torch.no_grad():
                beam_search_results = eval_model.beamsearch(input_mols)

            # with open(pred_path, 'a') as fp:
            for j in range(len(batch['source_smi'])):
            # for sample_i, ind in enumerate(batch_ind):
                input_mol, target_mol = input_mols[j], target_mols[j]
                try:
                    target_smi = mol_to_unmapped_smiles(target_mol)  # ground truth
                    target_mapped = Chem.MolToSmiles(target_mol)
                except Exception as e:
                    # logger.info(f"Exception while target to smi: {str(e)}")
                    n_samples += 1
                    continue

                has_correct = False
                final_smis = set()

                results = beam_search_results[j]  
                n_preds[n_samples] = len(results)

                for i, path in enumerate(results):
                    if path['final_smi_unmapped']:
                        try:
                            final_mol = Chem.MolFromSmiles(path['final_smi_unmapped'])

                            if final_mol is None:
                                final_smi = path['final_smi_unmapped']
                            else:
                                input_mol, final_mol = remap_reaction_to_canonical(input_mol, final_mol)
                                final_smi = mol_to_unmapped_smiles(final_mol)

                        except Exception as e:
                            final_smi = path['final_smi_unmapped']
                    else:
                        final_smi = path['final_smi_unmapped']

                    is_duplicate.append(final_smi in final_smis)
                    is_incorrect.append(final_smi is None or final_smi == '')
                    final_smis.add(final_smi)
                    # print(final_smi)
                    correct = prediction_is_correct(final_smi, target_smi)

                    # 如果这是第一个预测正确的结果
                    if correct and not has_correct:
                        top_k[i:] += 1
                        accs[i] += 1
                        has_correct = True
                        ranking = i + 1
                    n_gen_reactions += 1

                n_samples += 1
                    

            # if (i > 0 and i % 100 == 0) or i >= len(test_loader) - 1:
            print("^" * 100)
            print(f'Beam search parameters: beam size={self.args.beam_size}, max steps={self.args.max_gen_steps}')
            print()
            for k, top in enumerate(top_k):
                acc = accs[k]
                print('Top {:3d}: {:7.4f}% cum {:7.4f}%'.format(k + 1, acc * 100 / n_samples, top * 100 / n_samples))
            print()
            avg_incorrect = '{:.4f}%'.format(100 * np.sum(is_incorrect) / len(is_incorrect))
            avg_duplicates = '{:.4f}%'.format(100 * np.sum(is_duplicate) / len(is_duplicate))
            avg_n_preds = '{:.4f}'.format(n_gen_reactions / n_samples)
            less_preds = '{:.4f}%'.format(100 * np.sum(n_preds[:n_samples] < self.args.beam_size) / n_samples)
            zero_preds = '{:.4f}%'.format(100 * np.sum(n_preds[:n_samples] == 0) / n_samples)
            print(f'Avg incorrect reactions in Top {self.args.beam_size}: {avg_incorrect}')
            print(f'Avg duplicate reactions in Top {self.args.beam_size}: {avg_duplicates}')
            print(f'Avg number of predictions per target: {avg_n_preds}')
            print(f'Targets with < {self.args.beam_size} predictions: {less_preds}')
            print(f'Targets with zero predictions: {zero_preds}')
            print()
            # break


        total_time = time.time() - start_time
        s_targets = '{:.4f}'.format(total_time / n_samples)
        s_reactions = '{:.4f}'.format(n_gen_reactions / total_time)
        total_time = '{:.4f}'.format(total_time)
        avg_incorrect = '{:.4f}%'.format(100 * np.sum(is_incorrect) / len(is_incorrect))
        avg_duplicates = '{:.4f}%'.format(100 * np.sum(is_duplicate) / len(is_duplicate))
        avg_n_preds = '{:.4f}'.format(n_gen_reactions / n_samples)
        less_preds = '{:.4f}%'.format(100 * np.sum(n_preds < self.args.beam_size) / n_samples)
        zero_preds = '{:.4f}%'.format(100 * np.sum(n_preds == 0) / n_samples)

        return {
            "TOP1": accs[0] * 100 / n_samples,
            "TOP5": accs[4] * 100 / n_samples if self.args.beam_size >= 5 else None,
            "TOP10": accs[9] * 100 / n_samples if self.args.beam_size >= 10 else None,
            "TOP20": accs[19] * 100 / n_samples if self.args.beam_size >= 20 else None,
            "TOP50": accs[49] * 100 / n_samples if self.args.beam_size >= 50 else None,
        }

        
