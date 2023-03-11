# project initialization

import torch
import logging
import multiprocessing
import os
import random
import numpy as np


if int(os.environ.get("DEBUG", 0)) >= 1:
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.INFO

USE_MOTIF_ACTION_KEYS = [
    'megan_16_bfs_randat_motif_vocab_20',
    'megan_16_bfs_randat_motif_vocab_500',
    'megan_16_bfs_randat_motif_vocab_100',
    'megan_16_bfs_randat_motif_vocab_500_motif_feat',
]

USE_MOTIF_FEATURES_KEYS = [
    'megan_16_bfs_randat_motif_feat',
    'megan_16_bfs_randat_motif_vocab_500_motif_feat',
]



# def set_random_seed():
#     seed = int(os.environ.get('RANDOM_SEED', 0))
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True


# set_random_seed()
