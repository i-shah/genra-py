from matplotlib.patches import Ellipse, Circle, FancyArrowPatch,FancyBboxPatch
from textwrap import wrap
import pandas as pd
from rdkit import Chem
import io 
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D, MolToImage
from rdkit.Chem.rdDepictor import GenerateDepictionMatching2DStructure
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
import re

from genra.utl.svg import *

class ChemBase:
  def __init__(self,*args,kekulize=True,wedgeBonds=True, template=None, bondWidth=2,
               molsize=(200,200), **kwargs):
    self._kekulize = kekulize
    self._wedgeBonds= wedgeBonds
    self._template = template
    self._molsize  = molsize
    self._bondWidth= bondWidth

    dopt = DrawingOptions()
    dopt.bondLineWidth=bondWidth
    dopt.bondWidth = bondWidth
    dopt.colorBonds=False
    dopt.kekulize=kekulize
    self._dopt=dopt
    
  def __call__(self,smi,**kwargs):
    return self.smi2mol(smi,**kwargs)
  
  def smi2mol(self,smi):
    molSize=self._molsize
    
    mol=Chem.MolFromSmiles(smi)
    mc = Chem.Mol(mol.ToBinary())
    if self._kekulize:
      try:
          Chem.Kekulize(mc)
      except:
          mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
 
    Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^\
                  Chem.SanitizeFlags.SANITIZE_KEKULIZE^\
                  Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)

    if self._template:
      GenerateDepictionMatching2DStructure(mol,self._template,acceptFailure=True)
    
    return mol

class ChemImage(ChemBase):
  def __init__(self,*args,**kwargs):
      super().__init__(self,*args,**kwargs)

  def __call__(self,smi,**kwargs):
    return self.smi2img(smi,**kwargs)
  
  def smi2img(self,smi,**kwargs):
      mol = self.smi2mol(smi,**kwargs)
      return MolToImage(mol,kekulize=self._kekulize,bondLineWidth=self._bondWidth,
                        wedgebonds=self._wedgeBonds,options=self._dopt)
      
class ChemSVG(ChemBase):
  
  def __init__(self,*args,**kwargs):
      ChemBase.__init__(self,*args,**kwargs)
      
  def __call__(self,smi,**kwargs):
    return self.smi2svg(smi,**kwargs)
  
  def smi2svg(self,smi,opacity=None):
    molSize=self._molsize    
    mol = self.smi2mol(smi)
    mc = Chem.Mol(mol.ToBinary())    
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    if opacity:
      svg = re.sub("rect style='opacity:[^\;]+;",
                   "rect style='opacity:{};".format(opacity),svg)
    return svg
  
class ChemJson(ChemSVG,Svg2Json):
  def __init__(self,*args,**kwargs):
      ChemSVG.__init__(self,*args,**kwargs)
      Svg2Json.__init__(self,*args,**kwargs)
  
  def __call__(self,smi,**kwargs):
    return self.smi2json(smi,**kwargs)
  
  def smi2json(self,smi,**kwargs):
    svg = self.smi2svg(smi,**kwargs)
    return self.read_svg(svg)
    
  