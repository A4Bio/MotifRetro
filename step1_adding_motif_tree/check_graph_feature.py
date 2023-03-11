import sys; sys.path.append("/gaozhangyang/experiments/MotifRetro")
from src.feat.featurize_gzy_psvae import gen_training_samples
from rdkit import Chem
import json

if __name__ == "__main__":
    source_mol = Chem.MolFromSmiles('[c:1]1(-[c:15]2[cH:16][cH:17][c:18](-[n:19]3[cH:20][cH:21][cH:22][n:23]3)[cH:24][cH:25]2)[n:2]([CH3:3])[cH:4][n:5][c:6]1-[c:7]1[cH:8][c:9]([C:10]#[N:11])[cH:12][cH:13][n:14]1')
    target_mol = Chem.MolFromSmiles('[c:15]1([B:28]([OH:27])[OH:29])[cH:16][cH:17][c:18](-[n:19]2[cH:20][cH:21][cH:22][n:23]2)[cH:24][cH:25]1.[c:1]1([Br:26])[n:2]([CH3:3])[cH:4][n:5][c:6]1-[c:7]1[cH:8][c:9]([C:10]#[N:11])[cH:12][cH:13][n:14]1')
    feat_vocab = json.load(open("/gaozhangyang/experiments/MotifRetro/data/uspto_50k/feat/uspto_50k_frag/feat_vocab.json","r"))
    # for key, val in feat_vocab['prop2oh']['atom'].items():
    #     feat_vocab['prop2oh']['atom'][key] = int(val)
    
    feat_vocab['prop2oh']['atom']['is_aromatic'] = {int(key):val for key, val in feat_vocab['prop2oh']['atom']['is_aromatic'].items()}
    
    
    vocab_path = "/gaozhangyang/experiments/MotifRetro/data/uspto_50k/adding_motif_trees.json"
    training_samples, final_smi = gen_training_samples(source_mol,target_mol, n_max_steps=30,feat_vocab=feat_vocab, use_motif_action=1, vocab_path=vocab_path)
    print()