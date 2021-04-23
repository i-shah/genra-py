import re

from .ixml import *

class Svg2Json:
  def __init__(self,*args,**kwargs):
    self._xmlparser = Xml2Obj()
    self._svgtypes = ['polygon','text','line']  
    self._rx=re.compile(r'\d+')
    self._rxp=re.compile(r'(?P<command>\w+)\s+(?P<x>\d+\.\d+),(?P<y>\d+\.\d+)')
  def __call__(self,svg_str,**kwargs):
    return self.read_svg(svg_str)
    
  def read_svg(self,svg_str,fontsize=6.0,**kwargs):
    J = elemDict(self._xmlparser.Parse(svg_str))
    K = self._svgtypes
    
    #return J
    rx = self._rx
    rxp= self._rxp
   
    if 'text' in J:
      J['text'] = [self.getText(j) for j in J['text']]
      
    if 'line' in J:
      J['line'] = [self.getLine(i) for i in J['line']]

    if 'polygon' in J:
      J['polygon'] = [self.getPolygon(i) for i in J['polygon']]
      
    if 'path' in J:
      J['path'] = [self.getPath(i) for i in J['path']]

    return J

  def getText(self,In):
    Out = {}
    Out.update({k.replace('-','_'):self.fix(k,v) for k,v in In.items()})
    Out['text'] = Out.pop('tspan')
    if 'style' in Out: Out.update(self.getStyles(Out['style']))
      
    return Out
 
  def getLine(self,In):
    Out = {}
    Out.update({k.replace('-','_'):self.fix(k,v) for k,v in In.items()})
    
    return Out
  
  def getPolygon(self,In):
    Out = {}
    Out.update({k.replace('-','_'):self.fix(k,v) for k,v in In.items()})
    
    return Out
  
  def getPath(self,In):
    Out = {}
    Out.update({k:self.fix(k,v) for k,v in In.items()})
    Out['lines'] = [m.groupdict() for m in self._rxp.finditer(Out['d'])]
    if 'style' in Out: Out.update(self.getStyles(Out['style']))

    return Out
 
  def fixpoly(self,x):
    p = rxp.findall(x)
    if not p: return x
    P = "M%s Z" % " L".join(p)
    return P

  def fixrgb(self,x):
    a = rx.findall(x)
    if len(a) != 3:
      return "#ffffff"
    else:
      return "#%(r)02X%(g)02X%(b)02X" % dict(zip(('r','g','b'),map(int,a)))

  def fix(self,k,v):
    if k=='tspan':
      v = v[0]['cdata']
    elif k in ['x','y','font-size']:
      v = float(v)
      try:
        v = float(v)
      except:
        pass
    return v
      
  def getStyles(self,x):
    Y = {k:v for k,v in [i.split(':') for i in x.split(';')]}
    for atr in ['font-size','stroke-width','stroke-opacity','fill-opacity']:
      if atr in Y: 
        val = Y[atr]
        try:
          val = float(re.sub('\D+$','',val))
        except:
          pass
        else:
          Y[atr]=val
    return Y
      
  def old(self,x):
    if 'fill' in x:
      x['fill']=self.fixrgb(x['fill'])
    if 'color' in x:
      x['color']=self.fixrgb(x['color'])
    if 'x' in x:
      x['x']=float(x['x'])
    if 'y' in x:
      x['y']=float(x['y'])
    if 'points' in x:
      x['points']=self.fixpoly(x['points'])
    if 'font-size' in x:
      try:
        x['font-size'] = float(x['font-size'])
      except:
        x['font-size'] = fontsize
    return x

    