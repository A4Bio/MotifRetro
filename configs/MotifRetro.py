# USPTO-50k (retrosynthesis) - reaction type unknown
dataset_key = 'uspto_50k'

max_n_epochs = 200
learning_rate = 0.0001
gen_lr_factor = 0.05
gen_lr_patience = 6
early_stopping = 16
start_epoch = 0
megan_warmup_epochs = 1
use_hierachical_action = True

reaction_type_given = False
bond_emb_dim = 32
hidden_dim = 1024
stateful = True
n_reaction_types = 10
reaction_type_emb_dim = 16

atom_feature_keys = ['is_supernode', 'atomic_num', 'formal_charge', 'chiral_tag',
                           'num_explicit_hs', 'is_aromatic']
bond_feature_keys = ['bond_type', 'bond_stereo']

n_encoder_conv = 6
enc_residual = True
enc_dropout = 0.0

n_decoder_conv = 2
dec_residual = True
n_fc = 2
atom_fc_hidden_dim = 128
bond_fc_hidden_dim = 128
bond_atom_dim = 128
dec_dropout = 0.0

att_heads = 8
att_dim = 128
