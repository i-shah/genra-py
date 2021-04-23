import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.text as text
import matplotlib.font_manager as fm 
import pylab as pl
import pandas as pd
from textwrap import wrap

from .chem import *

class ChemView(ChemImage):
  def __init__(self,*args,ax=None,z0=1,fs=1.2,xyoff=[0,0],lw=1.0,dbg=False,
               **kwargs):
    ChemImage.__init__(self,*args,**kwargs)
    self.dbg=dbg
    self.HA = dict(middle='center',end='right',start='left')
    self.ax = ax
    self.z0 = z0
    self.fs = fs
    self.lw = lw
    self.xyoff=xyoff

  def __call__(self,smi,x0=0.0,y0=0.0,h=None,w=None,scale=1.0):
    try:
      Img = self.smi2img(smi)
    except:
      pass
    else:
      w = w if w else self._molsize[0]
      h = h if h else self._molsize[1]
      hw,hh = 0.5 * w, 0.5 * h
      self.ax.imshow(Img,extent=(x0-hw,x0+hw,y0-hw,y0+hw))
    
  
class ChemJSView(ChemJson):
  def __init__(self,*args,ax=None,z0=1,fs=1.2,xyoff=[0,0],lw=1.0,dbg=False,
               **kwargs):
    ChemJson.__init__(self,*args,**kwargs)
    self.dbg=dbg
    self.HA = dict(middle='center',end='right',start='left')
    self.ax = ax
    self.z0 = z0
    self.fs = fs
    self.lw = lw
    self.xyoff=xyoff
    
  def __call__(self,smi,x0=0.0,y0=0.0,ball_stick=False,atom_r=5.0):
    J = self.smi2json(smi) if type(smi)==str else smi
      
    if 'text' in J:
      if ball_stick:
        self.drawBS(J['text'],x0=x0,y0=y0,atom_r=atom_r)
      else:
        self.drawText(J['text'],x0=x0,y0=y0)
    if 'path' in J:
      self.drawPath(J['path'],x0=x0,y0=y0)
      
  def drawText(self,G,x0=0.0,y0=0.0):
    xyoff=self.xyoff
    for g in G:
      if self.dbg: 
        print("Text: {x:.1f} {y:.1f} {text} {fill} {dominant_baseline} {text_anchor}".format(**g))
      horiz_anchor = self.HA[g['text_anchor']]
      txt = text.Text(g['x']+x0+xyoff[0],g['y']+y0+xyoff[1],g['text'],color=g['fill'],zorder=self.z0,
                        ha = horiz_anchor, va = 'center',
                        fontproperties=fm.FontProperties(size=g['font-size']*self.fs,
                                                         weight=g['font-weight'],
                                                         style=g['font-style'],
                                                         family=g['font-family']))
      self.ax.add_artist(txt)

  def drawPath(self,G,x0=0.0,y0=0.0):
    xyoff=self.xyoff
    for g in G:
      L = pd.DataFrame(g['lines'])
      L.loc[:,'x'] = L.x.astype(np.float)+x0
      L.loc[:,'y'] = L.y.astype(np.float)+y0

      poly = mpatches.Polygon(L[['x','y']],lw=float(g['stroke-width'])*self.lw,
                              facecolor=g['fill'],
                              edgecolor=g['stroke'],
                              zorder=self.z0,fill=g['fill']!='none',visible=True)
      pl.gca().add_patch(poly)
    
  def drawBS(self,G,x0=0.0,y0=0.0,atom_r=5):
    xyoff=self.xyoff
    for g in G:
      if self.dbg: 
        print("Text: {x:.1f} {y:.1f} {text} {fill} {dominant_baseline} {text_anchor}".format(**g))
      atoms = mpatches.Ellipse([x0+g['x']+xyoff[0],y0+g['y']+xyoff[1]],atom_r,atom_r,
                               color=g['fill'],zorder=self.z0+1)
      self.ax.add_artist(atoms)

    
class ChemJsonBS:
  def __init__(self,*args,dbg=False,**kwargs):
    ChemSVG.__init__(*args,**kwargs)    
    self.dbg=dbg
    self.HA = dict(middle='center',end='right',start='left')
    
  def draw(self,Json,ax,xmax=100,ymax=100,
           x0=0.0,y0=0.0,z0=1,lw=1.0,atom_r=2,
           fs=1.2,xyoff=[2,-5],text_align=dict(ha='center',va='top')):
    if 'text' in Json:
      for g in Json['text']:
        if self.dbg: 
          print("Text: {x:.1f} {y:.1f} {text} {fill} {dominant_baseline} {text_anchor}".format(**g))
        #horiz_anchor = self.HA[g['text_anchor']]
        atoms = mpatches.Ellipse([x0+g['x']+xyoff[0],y0+g['y']+xyoff[1]],atom_r,atom_r,
                                 color=g['fill'],zorder=z0+1)
        ax.add_artist(atoms)

    if 'path' in Json:
        for g in Json['path']:
            #P1 = np.array([map(float,p[1:].split(',')) for p in g['points'].split(' ') if len(p)>1])
            #P1[:,0] += x0
            #P1[:,1] += y0
            L = pd.DataFrame(g['lines'])
            L.loc[:,'x'] = L.x.astype(np.float)+x0
            L.loc[:,'y'] = L.y.astype(np.float)+y0
            
            poly = mpatches.Polygon(L[['x','y']],lw=float(g['stroke-width'])*lw,
                                    facecolor=g['fill'],
                                    edgecolor=g['stroke'],
                                    zorder=z0,fill=g['fill']!='none',visible=True)
            pl.gca().add_patch(poly)

    
    