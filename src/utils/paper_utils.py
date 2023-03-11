from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import Image
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import rdScaffoldNetwork
from urllib import parse

def draw_smiles(smi, name, path="figs"):
    m = Chem.MolFromSmiles(smi)
    d = rdMolDraw2D.MolDraw2DCairo(800, 500)
    do = rdMolDraw2D.MolDrawOptions()
    do.bondLineWidth = 3
    do.fixedBondLength = 30
    do.clearBackground = False
    d.SetDrawOptions(do)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    d.WriteDrawingText(f'{path}/{name}')
    Image(filename=f'{path}/{name}', width=300)
    


def smi2svg(smi):
    mol = Chem.MolFromSmiles(smi)
    try:
        Chem.rdmolops.Kekulize(mol)
    except:
        pass
    drawer = rdMolDraw2D.MolDraw2DSVG(690, 400)
    AllChem.Compute2DCoords(mol)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")
    return svg
 
def smi2image(smi):
    svg_string = smi2svg(smi)
    impath = 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe="")
    return impath


def get_ams_attach(smi):
    ams = []
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.GetSymbol()=="*":
            attach = atom.GetAtomMapNum()
            continue
        ams.append(atom.GetAtomMapNum())
    return ams, attach



