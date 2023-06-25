import argparse
import os
from datetime import datetime

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--res_dir', default='/gaozhangyang/experiments/MotifRetro/results', type=str)
    
    parser.add_argument('--ex_name', default='motif_trees_0.7802_0.3220_class', type=str)
    parser.add_argument('--dataset_key', default='uspto_50k', type=str, choices=["uspto_50k", "uspto_hard"])
    parser.add_argument('--featurizer_key', default='add_feats', type=str)

    # dataset parameters
    parser.add_argument('--data_path', default='/gaozhangyang/experiments/MotifRetro/data')
    parser.add_argument("--traversal", default="bfs", choices=['bfs', 'dfs', 'mix'])
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--vocab_path', default="motif_trees_0.7802_0.3220", choices=['motif_trees_0.2011_1.0000', 
            'motif_trees_0.2468_0.8299', 
            'motif_trees_0.2763_0.7547', 
            'motif_trees_0.2853_0.7320', 
            'motif_trees_0.3880_0.5834', 
            'motif_trees_0.3953_0.5734', 
            'motif_trees_0.4349_0.5453', 
            'motif_trees_0.6536_0.3793', 
            
            'motif_trees_0.7802_0.3220', 
            'motif_trees_0.7872_0.3201', 
            'motif_trees_0.8348_0.3022', 
            'motif_trees_0.8883_0.2845', 
            'motif_trees_0.9113_0.2777', 
            'motif_trees_0.9254_0.2739', 
            'motif_trees_0.9883_0.2615', 
            'motif_trees_1.0000_0.2605'])
    parser.add_argument('--n_reaction_types', default=10, type=int)
    parser.add_argument('--reaction_type_given', default=1, type=int)
    
    # method parameters
    parser.add_argument('--method', default='MotifRetro_GNN', choices=["MotifRetro", "MotifRetro_GNN"])
    parser.add_argument('--config_file', '-c', default=None, type=str)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--useAtomAction', default=1, type=int)
    parser.add_argument('--useBondAction', default=1, type=int)
    parser.add_argument('--useAttachAction', default=1, type=int)
    
    # Training parameters
    parser.add_argument('--epoch', default=100, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--optim_name', default="AdamW", choices=['AdamW', 'SGD', 'Lion'])
    parser.add_argument('--schdular_name', default="linear_warmup", choices=['linear_warmup', 'onecycle', 'cosine'])
    

    # Evaluation parameters
    parser.add_argument('--n_max_atoms', default=100, type=int)
    parser.add_argument('--beam_size', default=10, type=int)
    parser.add_argument('--max_gen_steps', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--only_test', default=1, type=int)
    parser.add_argument('--only_valid', default=0, type=int)
    
    # GNN parameters
    parser.add_argument('--bond_emb_dim', default=1024, type=int)
    parser.add_argument('--atom_feature_keys', default=['is_supernode', 'atomic_num', 'formal_charge', 'chiral_tag', 'num_explicit_hs', 'is_aromatic'], type=list)
    parser.add_argument('--bond_feature_keys', default=['bond_type', 'bond_stereo'], type=list)
    parser.add_argument('--n_encoder_conv', default=6, type=int)
    parser.add_argument('--scale_up', default=1, type=float)
    parser.add_argument('--enc_residual', default=True, type=bool)
    parser.add_argument('--enc_dropout', default=0.1, type=float)
    parser.add_argument('--n_decoder_conv', default=2, type=int)
    parser.add_argument('--dec_residual', default=True, type=bool)
    parser.add_argument('--n_fc', default=1, type=int)
    parser.add_argument('--atom_fc_hidden_dim', default=128, type=int)
    parser.add_argument('--bond_fc_hidden_dim', default=128, type=int)
    parser.add_argument('--bond_atom_dim', default=128, type=int)
    parser.add_argument('--dec_dropout', default=0.5, type=float)
    parser.add_argument('--att_heads', default=8, type=int)
    parser.add_argument('--att_dim', default=128, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)

    parser.add_argument('--topk', default=50, type=int)
    parser.add_argument('--use_degree_feat', default=1, type=int)
    parser.add_argument('--predict_atom_num', default=0, type=int)
    parser.add_argument('--no_wandb', default=1, type=int)
    parser.add_argument('--align_upper_lower', default=0, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    
    # parser.add_argument('--dropout', default=None, type=float)
    

    args = parser.parse_args()
    print(args.reaction_type_given)
#     args.ex_name += str(datetime.now())
    args.featurizer_key = args.vocab_path
    args.vocab_path = os.path.join("/gaozhangyang/experiments/MotifRetro/data/uspto_50k",  args.vocab_path+".json")
    args.bond_emb_dim = args.hidden_dim
    args.n_decoder_conv = 8-args.n_encoder_conv


    return args