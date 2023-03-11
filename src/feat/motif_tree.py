import networkx as nx
import json
from networkx import json_graph
import sys; sys.path.append("/gaozhangyang/experiments/MotifRetro")
from src.feat.reaction_actions import AddMotifAction_with_dummySMI
from rdkit import Chem
from src.utils.chem_utils import smi2mol, mol2smi
from src.feat.utils import  fix_explicit_hs, fix_valence

class MotifTree:
    def __init__(self, dictionary, am_mapping=None) -> None:
        self.tree = self.from_dict(dictionary)
        self.am_mapping = am_mapping
        if self.am_mapping is not None:
            self.update_ams(am_mapping)
    
    def from_dict(self, dictionary):
        tree = json_graph.tree_graph(dictionary['tree'])
        tree.graph = dictionary['graph']
        return tree
    
    def update_ams(self, am_mapping):
        mol = smi2mol(self.tree.graph['am_smi'], sanitize=False)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(am_mapping[atom.GetAtomMapNum()])
        self.tree.graph['am_smi'] = mol2smi(mol)
        
        for t, idx in enumerate(self.traversal()):
            node = self.tree.nodes[idx]
            mol = smi2mol(node['am_smi'])
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(am_mapping[atom.GetAtomMapNum()])
            node['am_smi'] = mol2smi(mol)
            
            
    
    def traversal(self, order = "bfs"):
        if order=="dfs":
            T = nx.dfs_tree(self.tree, source=self.tree.graph['root'])
        if order=="bfs":
            T = nx.bfs_tree(self.tree, source=self.tree.graph['root'])
        for node_idx in T.nodes:
            yield node_idx
    
    def to_molecule(self):
        # mol = Chem.MolFromSmiles("*")
        for t, idx in enumerate(self.traversal()):
            node = self.tree.nodes[idx]
            
            if t==0:
                mol = Chem.MolFromSmiles(node["am_smi"], sanitize=False)
                continue
            
            action = AddMotifAction_with_dummySMI(node["am_smi"])
            mol = action.apply(mol)
        return mol
    
if __name__ == "__main__":
    data = json.load(open("/gaozhangyang/experiments/MotifRetro/data/uspto_50k/adding_motif_trees.json", "r"))
    
    ##-------------------check reconstruction
    for idx, (key, val) in enumerate(data.items()):
        print(key)
        MT = MotifTree(val)
        mol = MT.to_molecule()

        true_smi = Chem.CanonSmiles(key.replace("@@", "@").replace("@", ""))
        pred_smi = Chem.CanonSmiles(mol2smi(mol, rm_am=True).replace("@@", "@").replace("@", ""))

        assert true_smi==pred_smi
    
    
    # ##-------------------convert to action path
    # decomposed_path = {}
    # for idx, (key, val) in enumerate(data.items()):
    #     MT = MotifTree(val)
        
    #     decomposed_path[key] = {"raw_info":
    #             {"raw_freq": MT.tree.graph['freq'],
    #              "start_attach":0,
    #              "raw_am_smi": MT.tree.graph['am_smi']},
    #             "path":{}}
                                
    #     for idx, node_idx in enumerate(MT.traversal()):
    #         node = MT.tree.nodes[node_idx]
    #         mol = Chem.MolFromSmiles(node['am_smi'])
    #         for atom in mol.GetAtoms():
    #             if atom.GetSymbol()=="*":
    #                 attach = atom.GetAtomMapNum()
    #         decomposed_path[key]['path'][str(idx)] = [
    #             {node['smi']:{
    #             "attach":attach,
    #             "am_smi":node['am_smi'],
    #             "freq":node["freq"]}
    #             }]
    # json.dump(decomposed_path, open("/gaozhangyang/experiments/MotifRetro/data/uspto_50k/decomposed_path_from_motiftree.json", "w"), indent=10)
    # print()