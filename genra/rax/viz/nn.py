from matplotlib.collections import PatchCollection
from textwrap import wrap
import pandas as pd
import math

from genra.utl.ixml import *
from genra.rax.skl.reg import GenRAPredValue
from genra.chm.viz import *

class GenRAView:
  
  def __init__(self,*args,X=None, Y=None, 
               knn=25,kekulize=True,wedgeBonds=True,metric='jaccard',
               dt='c',text_wrap_len=30,chm_name_font_size=8,
               **kwargs):
    self._kekulize = kekulize
    self._wedgeBonds= wedgeBonds
    self._X = X
    self._Y = Y
    self._dt=dt
    self._txt_wrap=text_wrap_len
    self._chm_fnsz=chm_name_font_size
    self._GV = GenRAPredValue(n_neighbors=knn,metric=metric,n_jobs=-1)

  def loadData(self,X,Y=[],Info=None):
    self._X = X
    self._Y = Y
    self._I = Info
    
    I2I = dict(zip(range(X.shape[0]),X.index))
    
    self._GV.fit(X,Y if len(Y)>0 else np.ones(X.shape[0]))
    Sim, Ind = self._GV.kneighbors_sim(self._X)
    
    # Store as dicts for lookup using the index 
    self._NN_sim = dict(zip(X.index,Sim))
    self._NN_ind = dict(zip(X.index,[[I2I[i] for i in I] for I in Ind]))
    
  def getKNN(self,cid,k=10):
    Info = self._I
    Sim,Ind = self._NN_sim[cid],self._NN_ind[cid]    
    NNi= pd.DataFrame(dict(ID=Ind,sim=Sim)).iloc[:(k+1)]
    self._NNi = NNi.merge(self._I,on='ID')
  
class GenRAViewNN(GenRAView):
  def __init__(self,th_tot=math.pi,th0=0.5*math.pi,
                 BBox=(-500,-500,500,500),Origin=[0,0],
                 pred = False,t0=None,r_min=100,rs=10,
                 chm_sz=(100,100),
                 ax = None,
                 use_svg=False,
                 **kwargs):
    GenRAView.__init__(self,**kwargs)
    self._th_tot = th_tot
    self._th0    = th0
    self._BBox   = BBox
    self._O      = Origin
    self._pred   = pred
    self._r_min  = r_min
    self._rs     = rs
    self._chm_sz = chm_sz
    self._ax      = ax
    self._CV = ChemJSView(ax=ax,xyoff=[0,0],fs=1.5,molsize=chm_sz) \
                if use_svg else  ChemView(ax=ax,xyoff=[0,0],fs=1.5,molsize=chm_sz)
    
  def circLayout(self,cid,k=10):    
    dth = self._th_tot/k
    xo,yo = self._O
    xmin,ymin,xmax,ymax= self._BBox
    r_max = abs(xmax-xo)-self._r_min
    #print(r_max)
    self.getKNN(cid,k=k)
    NN_c = self._NNi.iloc[1:]
    NN_c.loc[:,'r'] = self._r_min+r_max*(1-NN_c.sim)*self._rs
    NN_c.loc[:,'th'] = self._th0-(np.arange(0,NN_c.shape[0]) * dth)
    NN_c.loc[:,'x'] = NN_c.r*np.cos(NN_c.th)
    NN_c.loc[:,'y'] = NN_c.r*np.sin(NN_c.th)
    NN_c.loc[:,'lx']= 0.5*(NN_c.x+xo)
    NN_c.loc[:,'ly']= 0.5*(NN_c.y+yo)
    
    self._NN_c = NN_c
    
  def draw(self,cid,k=10,pred=False,t0=None):
    self.circLayout(cid,k=k)
    self.drawArcs()
    self.drawChems()
    
  def drawArcs(self):
    ch_w,ch_h = self._chm_sz
    ch_hw,ch_hh= 0.5*ch_w,0.5*ch_h
    xo,yo = self._O
    dsc_lab= self._dt
    
    ax = self._ax

    for i,C in self._NN_c.iterrows():
      ax.add_patch(FancyArrowPatch(posB=(C.x,C.y),posA=(xo,yo),
                                   color='grey',arrowstyle='->',
                                   #shrinkA=60,shrinkB=60,
                                   linewidth=2,zorder=0))
      ax.add_artist(text.Text(C.lx,C.ly,'{:3.2f}{}'.format(C.sim,dsc_lab),
                              color=(1.0,0.1,0.1),
                              ha='center',va='top',zorder=10,
                              backgroundcolor='white',
                              fontproperties=fm.FontProperties(size=8,weight='normal')))

  def drawChems(self):
    ch_w,ch_h = self._chm_sz
    ch_hw,ch_hh= 0.5*ch_w,0.5*ch_h
    xo,yo = self._O
    CV = self._CV
    ax = self._ax
    
    BGs=[]
    
    # Target
    BGs.append(mpatches.Rectangle((xo-ch_hw,xo-ch_hh),ch_w,ch_h,zorder=1))
    Tgt = self._NNi.iloc[0]
    CV(Tgt.smiles,x0=xo,y0=yo)
    txt = '\n'.join(wrap(Tgt.chemical_name,self._txt_wrap))
    ax.add_artist(text.Text(xo,yo+ch_hh,txt,
                            color=(0.2,0.3,0.2),
                            ha='center',va='top',zorder=10,
                            backgroundcolor='white',
                            fontproperties=fm.FontProperties(size=self._chm_fnsz,
                                                             weight='normal')))

    
    for i,C in self._NN_c.iterrows():
      x0, y0 = C.x-ch_hw,C.y-ch_hh
      BGs.append(mpatches.Rectangle((C.x-ch_hw,C.y-ch_hh),ch_w,ch_h,zorder=1))
      try:
        CV(C.smiles,x0=C.x,y0=C.y)
      except:
        print("Failed: {}".format(C.chemical_name))
      ax.add_artist(text.Text(C.x,C.y+ch_hh,
                              '\n'.join(wrap(C.chemical_name,self._txt_wrap)),
                              color=(0.2,0.3,0.2),
                              ha='center',va='top',zorder=10,
                              backgroundcolor='white',
                              fontproperties=fm.FontProperties(size=self._chm_fnsz,
                                                             weight='normal')))

      
      
    PC = PatchCollection(BGs,alpha=0.0)
    ax.add_collection(PC)
    
