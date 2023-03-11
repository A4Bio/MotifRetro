# CUDA_VISIBLE_DEVICES=1 python main.py --featurizer_key "uspto_50k_frag_now" --add_mask 1 --batch_size 128 --dropout 0.1 --temperature 0.1 --ex_name baseline_new_bpe

CUDA_VISIBLE_DEVICES=4 python main.py --featurizer_key "uspto_50k_motiftree_bfs_new" --add_mask 1 --batch_size 128 --dropout 0.1 --temperature 1 --lr 0.0005 --useTAU 0 --no_wandb 0 --ex_name uspto_50k_motiftree_bfs_new

CUDA_VISIBLE_DEVICES=5 python main.py --featurizer_key "uspto_50k_motiftree_bfs_new" --add_mask 1 --batch_size 128 --dropout 0.1 --temperature 1 --lr 0.0005 --useTAU 0 --no_wandb 0 --ex_name uspto_50k_motiftree_bfs_new_ignore_attachatom


CUDA_VISIBLE_DEVICES=3 python main.py --featurizer_key "uspto_50k_motiftree_bfs" --add_mask 1 --batch_size 128 --dropout 0.1 --temperature 1 --lr 0.0005 --useTAU 0 --no_wandb 0 --ex_name baseline_motiftree_bfs_no_attach_atom


CUDA_VISIBLE_DEVICES=2 python main.py --featurizer_key "uspto_50k_motiftree_dfs" --add_mask 1 --batch_size 128 --dropout 0.1 --temperature 1 --lr 0.0005 --useTAU 0 --no_wandb 0 --ex_name baseline_motiftree_dfs



CUDA_VISIBLE_DEVICES=1 python main.py --featurizer_key "uspto_50k_motiftree_bfs_check_beamsearch" --add_mask 1 --batch_size 128 --dropout 0.1 --temperature 1 --lr 0.0005 --useTAU 0 --no_wandb 0 --ex_name uspto_50k_motiftree_bfs_check_beamsearch


CUDA_VISIBLE_DEVICES=4 python main.py --add_mask 1 --batch_size 128 --dropout 0.1 --temperature 1 --lr 0.0005 --useTAU 0 --no_wandb 0 --ex_name uspto50k_recheck_clean_decoder

CUDA_VISIBLE_DEVICES=1 python main.py --add_mask 1 --batch_size 128 --dropout 0.1 --temperature 1 --lr 0.0005 --useTAU 0 --no_wandb 0 --ex_name pred_action_type_support_attach


CUDA_VISIBLE_DEVICES=1 python main.py  --batch_size 128 --dropout 0.1 --temperature 1 --lr 0.0005 --no_wandb 0 --ex_name DynamicGNN

CUDA_VISIBLE_DEVICES=2 python main.py  --batch_size 128 --dropout 0.1 --temperature 1 --lr 0.0005 --no_wandb 0 --ex_name padding_rm_synthon_attn_



cd /xuyongjie/gaozhangyang/experiments/MotifRetro
conda activate equibind
CUDA_VISIBLE_DEVICES=3 wandb agent motifretro/test-project/tr2bflef