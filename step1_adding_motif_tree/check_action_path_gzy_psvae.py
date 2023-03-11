import sys; sys.path.append("/gaozhangyang/experiments/MotifRetro")
import pandas as pd
from rdkit import Chem
import json
import copy
from src.feat.utils import atom_to_edit_tuple, get_bond_tuple, fix_incomplete_mappings, reac_to_canonical, fix_explicit_hs, renumber_atoms_for_mapping, mark_reactants
from src.utils.chem_utils import mol2smi, smi2mol, compare_mol, compare_smi
from rdkit.Chem import Draw
from src.feat.reaction_actions import StopAction, AddMotifAction
from tqdm import tqdm
import numpy as np
from src.feat.featurize_gzy_psvae import ReactionSampleGenerator as ReactionSampleGenerator_gzy
from src.utils.retro_utils import preprocess_mols


# utils
feat_vocab = json.load(open("/gaozhangyang/experiments/MotifRetro/data/uspto_50k/feat/uspto_50k_frag/feat_vocab.json", 'r'))

props = feat_vocab['prop2oh']
prop2oh = {'atom': {}, 'bond': {}}

for type_key in prop2oh.keys():
    oh_dict = props[type_key]
    for key, values in oh_dict.items():
        converted_values = {}
        for prop_val, val_oh in values.items():
            try:
                prop_val = int(prop_val)
            except ValueError:
                pass
            converted_values[prop_val] = val_oh
        prop2oh[type_key][key] = converted_values

feat_vocab['prop2oh'] = prop2oh

# visualization
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, MolsToGridImage

drawOptions = rdMolDraw2D.MolDrawOptions()
drawOptions.prepareMolsBeforeDrawing = False
drawOptions.bondLineWidth = 4
drawOptions.minFontSize = 12


def prepare_mol(mol, new_am):
    highlight_idx = []
    for i, atom in enumerate(mol.GetAtoms()):
        am = atom.GetAtomMapNum()
        if am in new_am:
            highlight_idx.append(i)
            
    try:
        mol_draw = rdMolDraw2D.PrepareMolForDrawing(mol)
    except Chem.KekulizeException:
        mol_draw = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
        # Chem.SanitizeMol(mol_draw, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    
    
    return mol_draw, highlight_idx

def plot_states(states, target_mol, source_mol):
    target_am = set([atom.GetAtomMapNum() for atom in  target_mol.GetAtoms()])
    source_am = set([atom.GetAtomMapNum() for atom in  source_mol.GetAtoms()])
    new_am = list(target_am - source_am)

    mol_list = []
    highlightAtomLists = []
    for one in states:
        mol, highlight = prepare_mol(one, new_am)
        mol_list.append(mol)
        highlightAtomLists.append(highlight)

    return MolsToGridImage(mol_list, molsPerRow=5,  subImgSize=(500, 500), drawOptions=drawOptions, highlightBondLists = highlightAtomLists)


# get edit states
def get_edit_sates(sample_generator):
    state = [copy.deepcopy(sample_generator.source_mol)]

    for i in range(100):
        reaction_action = sample_generator.generate_gen_action()
        print(reaction_action)
        if type(reaction_action)==StopAction:
            break
        
        sample_generator.source_mol = reaction_action.apply(sample_generator.source_mol) # 这是关键部分
        latent_mol = copy.deepcopy(sample_generator.source_mol)
        latent_mol.UpdatePropertyCache(strict=False)
        state.append(latent_mol)
    state.append(sample_generator.target_mol)
    return state

def fix_explicit_hs(mol):
    for a in mol.GetAtoms():
        a.SetNoImplicit(False)

    mol = Chem.AddHs(mol, explicitOnly=True)
    mol = Chem.RemoveHs(mol)

    Chem.SanitizeMol(mol)
    return mol



data = pd.read_csv("/gaozhangyang/experiments/MotifRetro/data/uspto_50k/raw_train.csv", index_col=0)

failed = 0
error = 0
long_path_idx = []
for idx in tqdm(range(0, len(data))):


    target, source = data.iloc[idx]["reactants>reagents>production"].split(">>")
    # target, source = data.iloc[7]["rxn_smiles"].split(">>")

    target_mol, source_mol = preprocess_mols(target, source)


    vocab_path = "/gaozhangyang/experiments/MotifRetro/data/uspto_50k/adding_motif_trees.json"



    motifretro_sample_generator = ReactionSampleGenerator_gzy(Chem.rdchem.RWMol(source_mol), target_mol,  feat_vocab=feat_vocab,  use_motif_action=True, vocab_path=vocab_path)
    
    # motifretro_states = get_edit_sates(motifretro_sample_generator)

    try:
        motifretro_states = get_edit_sates(motifretro_sample_generator)
    except:
        failed+=1
        continue
    
    # motifretro_states = get_edit_sates(motifretro_sample_generator)
    
    if len(motifretro_states)>5:
        long_path_idx.append(idx)
        
    # assert compare_mol(motifretro_states[-2], motifretro_states[-1])
    
    smi_pred = mol2smi(motifretro_states[-2])
    smi_true = mol2smi(motifretro_states[-1])
    try:
        assert compare_smi(smi_pred, smi_true)
    except:
        error+=1
        print(idx)

# idx=39139报错, failed: 33, error:1

# 25 invalid samples on USPTO-50k
print(f"Failed Num: {failed}")
print(f"Error Num: {error}")