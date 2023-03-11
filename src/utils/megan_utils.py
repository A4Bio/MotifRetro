import gzip
import logging
import os
import shutil
import zipfile

import requests


def unzip_and_clean(archive_dir: str, file_name: str):
    archive_path = os.path.join(archive_dir, file_name)

    if file_name.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as f:
            f.extractall(path=archive_dir)
    elif file_name.endswith('.gz'):
        output_path = archive_path.replace('.gz', '')
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError(f'Unsupported archive format for file: {archive_path}')

    os.remove(archive_path)


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)



import logging
from functools import reduce
from itertools import islice

from torch import nn
from torch.nn.modules.module import _addindent

import datetime
import logging
import os
import sys
from logging import handlers
from random import shuffle
from typing import Tuple, List

import numpy as np
import torch
from git import Repo
from rdkit import Chem
# disable warnings from rdkit - we will handle errors ourselves
from rdkit import rdBase
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import RenumberAtoms
from tqdm import tqdm

rdBase.DisableLog('rdApp.error')

logger = logging.getLogger(__name__)


def parse_logging_level(logging_level):
    """
    :param logging_level: Logging level as string
    :return: Parsed logging level
    """
    lowercase = logging_level.lower()
    if lowercase == 'debug':
        return logging.DEBUG
    if lowercase == 'info':
        return logging.INFO
    if lowercase == 'warning':
        return logging.WARNING
    if lowercase == 'error':
        return logging.ERROR
    if lowercase == 'critical':
        return logging.CRITICAL
    raise ValueError('Logging level {} could not be parsed.'.format(logging_level))


def configure_logger(logs_dir, name=__name__,
                     console_logging_level=logging.INFO, reset: bool = False):
    if not reset and len(logging.getLogger(name).handlers) != 0:
        print("Already configured logger '{}'".format(name))
        return

    if isinstance(console_logging_level, str):
        console_logging_level = parse_logging_level(console_logging_level)

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    if console_logging_level is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(log_format)
        ch.setLevel(console_logging_level)
        logger.addHandler(ch)

    for lvl, key in [(logging.INFO, 'info'), (logging.DEBUG, 'debug')]:
        file_path = os.path.join(logs_dir, 'logs_{}.txt'.format(key))

        fh = handlers.RotatingFileHandler(file_path, maxBytes=(1048576 * 5), backupCount=7)
        fh.setFormatter(log_format)
        fh.setLevel(lvl)

        logger.addHandler(fh)

    logger.debug("Logging configured!")

    return logger


def get_git_info() -> dict:
    """
    :return: information about current GIT branch/commit
    """
    repo = Repo(search_parent_directories=True)

    commit = repo.head.commit

    return {
        'commit': {
            'message': commit.message,
            'author': commit.author.name,
            'date': datetime.datetime.fromtimestamp(commit.committed_date).isoformat(),
            'sha': commit.hexsha
        },
        'branch': repo.active_branch.name
    }


def get_script_run_info() -> dict:
    """
    :return: meta information about script that is currently run
    """
    script = ' '.join(sys.argv)

    try:
        git_info = get_git_info()
    except:
        git_info = {}

    return {
        'script': script,
        'current_date': datetime.datetime.now().isoformat(),
        'git': git_info,
        'user': os.environ.get("USER", 'UNKNOWN')
    }


def smiles_to_unmapped(smi: str) -> str:
    """
    :param smi: a compound SMILES
    :return: canonical and unmapped version of the compound SMILES
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ''

    for a in mol.GetAtoms():
        a.ClearProp("molAtomMapNumber")
    smi_unmapped = Chem.MolToSmiles(mol, canonical=True)

    return smi_unmapped


def mol_to_unmapped(mol: Mol) -> Mol:
    mol_copy = Chem.Mol(mol)
    for a in mol_copy.GetAtoms():
        a.ClearProp("molAtomMapNumber")
    return mol_copy


def mol_to_unmapped_smiles(mol: Mol) -> str:
    mol_copy = mol_to_unmapped(mol)
    smi_unmapped = Chem.MolToSmiles(mol_copy, canonical=True)
    return smi_unmapped


def renumber_atoms_for_mapping(mol: Mol) -> Mol:
    new_order = []
    for a in mol.GetAtoms():
        new_order.append(a.GetAtomMapNum())
    new_order = [int(a) for a in np.argsort(new_order)]
    return RenumberAtoms(mol, new_order)


def complete_mappings(subs: str, prod: str) -> Tuple[str, str]:
    """
    Adds map number to incompletely mapped reaction
    :param subs: smiles for substrates of the reaction (mapped)
    :param prod: smiles for product of the reaction (mapped)
    :return: tuple of new SMILES for (smi, prod)
    """
    subs = Chem.MolFromSmiles(subs)
    prod = Chem.MolFromSmiles(prod)

    max_num = 0
    for mol in (subs, prod):
        max_num = max(max(a.GetAtomMapNum() for a in mol.GetAtoms()), max_num)

    curr_num = max_num + 1

    for mol in (subs, prod):
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() == 0:
                a.SetAtomMapNum(curr_num)
                curr_num += 1

    return Chem.MolToSmiles(subs), Chem.MolToSmiles(prod)


def convert_arg(arg):
    try:
        i = int(arg)
        f = float(arg)
        if i == f:
            return i
        return f
    except ValueError:
        return arg


def to_torch_tensor(arr, long: bool = False) -> torch.Tensor:
    if not isinstance(arr, np.ndarray):
        arr = arr.toarray()
    # noinspection PyUnresolvedReferences
    ten = torch.from_numpy(arr)
    if long:
        ten = ten.long()
    else:
        ten = ten.float()

    # if torch.cuda.is_available():
    #     return ten.cuda()
    return ten


def loop():
    show_bar = int(os.environ.get('DEBUG', 0)) != 0
    if show_bar:
        loop_ = tqdm
    else:
        def loop_(x):
            return x
    return loop_


def lists_to_tuple(x):
    if type(x) != list:
        return x
    elif len(x) == 0:
        return ()
    elif len(x) == 1:
        return (lists_to_tuple(x[0]), )

    a, b = x[0], x[1:]
    return (lists_to_tuple(a),) + lists_to_tuple(b)


def randomize_mol_order(smi: str) -> str:
    mols = smi.split(".")
    shuffle(mols)
    return '.'.join(mols)


def mark_reactants(source_mol: Mol, target_mol: Mol):
    target_atoms = set(a.GetAtomMapNum() for a in reversed(target_mol.GetAtoms()))
    for a in source_mol.GetAtoms():
        m = a.GetAtomMapNum()
        if m is not None and m > 0 and m in target_atoms:
            a.SetBoolProp('in_target', True)


def filter_reactants(sub_mols: List[Mol], prod_mol: Mol) -> Mol:
    mol_maps = set(a.GetAtomMapNum() for a in prod_mol.GetAtoms())
    reactants = []
    for mol in sub_mols:
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() in mol_maps:
                reactants.append(mol)
                break
    return Chem.MolFromSmiles('.'.join([Chem.MolToSmiles(m) for m in reactants]))


def get_metric_order(metric_key: str) -> int:
    """
    Returns 'order' of a metric (how to compare it)
    :param metric_key: key of the metric
    :return: -1 if 'smaller is better' (e.g. loss) and +1 if 'greater is better' (e.g. accuracy)
    """
    key = metric_key.strip().lower()
    if key.endswith('loss'):
        return -1
    if key.endswith('acc') or key.endswith('accuracy') or key.endswith('auc'):
        return 1
    raise ValueError("Could not define ordering of a metric: {}, please provide it manually".format(metric_key))


def to_binary_one_hot(dataset):
    return np.array([[1 - i, i] for i in dataset], dtype=float)


def save_weights(filename: str, model: nn.Module, optimizer=None, **kwargs):
    """
    Save all weights necessary to resume training
    """
    state = {
        'model': model.state_dict()
    }
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, filename)


def load_state_dict(path: str) -> dict:
    """
    Loads saved model dictionary
    """
    try:
        return torch.load(path)
    except RuntimeError:
        # probably user tries to load on CPU model trained on GPU
        return torch.load(path, map_location=lambda storage, loc: storage)


def summary(model, file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if p is not None:
                total_params += reduce(lambda x, y: x * y, p.shape)
            else:
                logger.debug(name + " param is None")

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file=file)
    return count


def has_finished_training(model_dir: str) -> bool:
    if os.path.exists(os.path.join(model_dir, "FINISHED")):
        logger.info("Finished")
        return True
    return False


def select_batch_from_dict(data: dict, batch_ind: np.ndarray) -> dict:
    """
    Selects a batch of samples by their index from a dictionary of features
    :param data: dictionary of features, where each value is an array of same length (number of samples)
    :param batch_ind: vector of indices to select
    :return: new dictionary with same keys, but values that are features of the selected samples
    """
    return dict((k, v[batch_ind]) for k, v in data.items())


def add_supervisor_logger(logging_level=logging.INFO):
    fh = logging.FileHandler('/var/log/supervisor/ml_api.log')
    fh.setLevel(logging_level)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(format)

    logger.addHandler(fh)
    logger.debug("Added supervisor logging handler!")


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


from typing import Tuple

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
import numpy as np

from src.feat.graph_features import try_get_atom_feature
# from src.feat import ATOM_EDIT_TUPLE_KEYS


def fix_incomplete_mappings(sub_mol: Mol, prod_mol: Mol) -> Tuple[Mol, Mol]:
    max_map = max(a.GetAtomMapNum() for a in sub_mol.GetAtoms())
    max_map = max(max(a.GetAtomMapNum() for a in prod_mol.GetAtoms()), max_map)

    for mol in (sub_mol, prod_mol):
        for a in mol.GetAtoms():
            map_num = a.GetAtomMapNum()
            if map_num is None or map_num < 1:
                max_map += 1
                a.SetAtomMapNum(max_map)
    return sub_mol, prod_mol


def add_map_numbers(mol: Mol) -> Mol:
    # converting to smiles to mol and again to smiles makes atom order canonical
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    map_nums = np.arange(mol.GetNumAtoms()) + 1
    np.random.shuffle(map_nums)

    for i, a in enumerate(mol.GetAtoms()):
        a.SetAtomMapNum(int(map_nums[i]))
    return mol


def reac_to_canonical(sub_mol, prod_mol): # converting to smiles to mol and again to smiles makes atom order canonical
    sub_mol = Chem.MolFromSmiles(Chem.MolToSmiles(sub_mol))
    prod_mol = Chem.MolFromSmiles(Chem.MolToSmiles(prod_mol))

    # in RdKit chirality can be marked different depending on order of atoms in molecule list
    # here we remap atoms so the map order is consistent with atom list order

    map2map = {}
    for i, a in enumerate(prod_mol.GetAtoms()):
        map2map[a.GetAtomMapNum()] = i + 1
        a.SetAtomMapNum(i + 1)

    max_map = max(map2map.values())
    for i, a in enumerate(sub_mol.GetAtoms()):
        m = a.GetAtomMapNum()
        if m in map2map:
            a.SetAtomMapNum(map2map[m])
        else:
            max_map += 1
            a.SetAtomMapNum(max_map)

    return sub_mol, prod_mol


def get_bond_tuple(bond) -> Tuple[int, int, int, int]:
    a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
    bt = int(bond.GetBondType())
    st = int(bond.GetStereo())
    if a1 > a2:
        a1, a2 = a2, a1
    return a1, a2, bt, st


def atom_to_edit_tuple(atom) -> Tuple:  # 考察原子的四个 property： ['formal_charge', 'chiral_tag', 'num_explicit_hs', 'is_aromatic']
    feat = [try_get_atom_feature(atom, key) for key in ATOM_EDIT_TUPLE_KEYS]
    return tuple(feat)


# rdkit has a problem with implicit hs. By default there are only explicit hs.
# This is a hack to fix this error
def fix_explicit_hs(mol: Mol) -> Mol:
    for a in mol.GetAtoms():
        a.SetNoImplicit(False)

    mol = Chem.AddHs(mol, explicitOnly=True)
    mol = Chem.RemoveHs(mol)

    Chem.SanitizeMol(mol)
    return mol


def get_atom_ind(mol: Mol, atom_map: int) -> int:
    for i, a in enumerate(mol.GetAtoms()):
        if a.GetAtomMapNum() == atom_map:
            return i
    raise ValueError(f'No atom with map number: {atom_map}')


def add_motif_action_vocab(action_vocab):
    action_vocab['atom2motif'] = {}
    action_vocab['atom_actions_ind'] = []
    action_vocab['motif_actions_ind'] = []
    for action, ind in action_vocab['action2ind'].items():  # construct motif_actions_ind and atom_actions_ind
        if action[0] == 'add_motif':
            action_vocab['motif_actions_ind'].append(ind)
        elif action[0] == 'add_atom':
            action_vocab['atom_actions_ind'].append(ind)
            action_vocab['atom2motif'][ind] = {'atom_action': action, 'motif_actions_ind': [-1]}
    
    for action, ind in action_vocab['action2ind'].items():  # construct atom2motif
        if action[0] == 'add_motif':
            for k, v in action_vocab['atom2motif'].items():
                if action[1][:2] == v['atom_action'][1]:
                    v['motif_actions_ind'].append(ind)
    
    # create motif2atom
    action_vocab['motif2atom'] = {}
    for k, v in action_vocab['atom2motif'].items():
        for motif_ind in v['motif_actions_ind']:
            if motif_ind != -1 and motif_ind not in action_vocab['motif2atom']:
                action_vocab['motif2atom'][motif_ind] = k

    action_vocab['non_empty_atom2motif_ind'] = list()
    for i, (k, v) in enumerate(action_vocab['atom2motif'].items()):
        if len(v['motif_actions_ind']) > 1:
            action_vocab['non_empty_atom2motif_ind'].append(k)

    # max atom2motif
    action_vocab['max_atom2motif'] = 0
    for k, v in action_vocab['atom2motif'].items():
        action_vocab['max_atom2motif'] = max(action_vocab['max_atom2motif'], len(v['motif_actions_ind']))
    
    return action_vocab


def add_motif_feature_vocab(action_vocab, vocab_path):
    action_vocab['prop2oh']['motif'] = {'is_supernode': action_vocab['prop2oh']['atom']['is_supernode'], 'motif_ids': {'supernode': 0}}
    with open(vocab_path, 'r') as fin:
        for i, line in enumerate(fin.readlines()[1:]):
            # if line.strip().split('\t')[0] in prop2oh['motif']['motif_ids']:
                # raise ValueError
            action_vocab['prop2oh']['motif']['motif_ids'][line.strip().split('\t')[0]] = len(action_vocab['prop2oh']['motif']['motif_ids'])
    
    extra_motif = ["Zn", "Mg", "Se", "Cu", "Sn", "Si"]
    for motif in extra_motif:
        action_vocab['prop2oh']['motif']['motif_ids'][motif] = len(action_vocab['prop2oh']['motif']['motif_ids'])
    return action_vocab


from src.feat.megan_graph import MeganTrainingSamplesFeaturizer
from src.split.basic_splits import DefaultSplit

N_JOBS = 128
DEFAULT_SPLIT = DefaultSplit()

FEATURIZER_INITIALIZERS = {
    # 5 variants of action ordering tested on UPSTO-50k, 'megan_16_bfs_randat' is the one we use for final evaluation
    'megan_16_dfs_cano': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                split=DEFAULT_SPLIT, key='megan_16_dfs_cano',
                                                                action_order='dfs'),
    'megan_16_bfs_cano': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                split=DEFAULT_SPLIT, key='megan_16_bfs_cano',
                                                                action_order='bfs'),
    'megan_16_dfs_randat': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                  split=DEFAULT_SPLIT, key='megan_16_dfs_randat',
                                                                  action_order='dfs_randat'),
    'megan_16_bfs_randat': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                  split=DEFAULT_SPLIT, key='megan_16_bfs_randat',
                                                                  action_order='bfs_randat'),
    'megan_16_bfs_randat_motif_vocab_500': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                  split=DEFAULT_SPLIT, key='megan_16_bfs_randat_motif_vocab_500',
                                                                  action_order='bfs_randat',
                                                                  use_motif_action=True, vocab_path='/gaozhangyang/experiments/retrosynthesis_prediction/data/uspto_50k/vocab_500.txt'),
    'megan_16_bfs_randat_motif_vocab_100': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                  split=DEFAULT_SPLIT, key='megan_16_bfs_randat_motif_vocab_100',
                                                                  action_order='bfs_randat',
                                                                  use_motif_action=True, vocab_path='/gaozhangyang/experiments/retrosynthesis_prediction/data/uspto_50k/vocab_100.txt'),
    'megan_16_bfs_randat_motif_vocab_20': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                  split=DEFAULT_SPLIT, key='megan_16_bfs_randat_motif_vocab_20',
                                                                  action_order='bfs_randat',
                                                                  use_motif_action=True, vocab_path='/gaozhangyang/experiments/retrosynthesis_prediction/data/uspto_50k/vocab_20.txt'),
    'megan_16_bfs_randat_motif_vocab_500_motif_feat': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                  split=DEFAULT_SPLIT, key='megan_16_bfs_randat_motif_vocab_500_motif_feat',
                                                                  action_order='bfs_randat',
                                                                  use_motif_action=True, use_motif_feature=True, vocab_path='/gaozhangyang/experiments/retrosynthesis_prediction/data/uspto_50k/vocab_500.txt'),

    'megan_16_random': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16, key='megan_16_random',
                                                              split=DEFAULT_SPLIT, action_order='random'),

    # variant that we use for USPTO-FULL
    'megan_32_bfs_randat': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=32,
                                                                  split=DEFAULT_SPLIT, key='megan_32_bfs_randat',
                                                                  action_order='bfs_randat'),

    # variant that we use for USPTO-MIT (forward synthesis prediction)
    'megan_for_8_dfs_cano': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=8,
                                                                   split=DEFAULT_SPLIT, forward=True,
                                                                   action_order='dfs'),
}