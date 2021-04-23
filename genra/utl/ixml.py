# $Id: ixml.py 9 2012-05-18 18:04:29Z ishah $

import sys
import string
from lxml import etree
from lxml.etree import XMLParser
from lxml.etree import ElementTree
#from xml.parsers import *
from xml.parsers import expat
from xml.etree.ElementTree import *
import re
from copy import *
from types import *


class Element:
    'A parsed XML element'
    def __init__(self,name,attributes):
        'Element constructor'
        # The element's tag name
        self.name = name
        # The element's attribute dictionary
        self.attributes = attributes
        # The element's cdata
        self.cdata = ''
        # The element's child element list (sequence)
        self.children = []
        
    def AddChild(self,element):
        'Add a reference to a child element'
        self.children.append(element)
        
    def getAttribute(self,key):
        'Get an attribute value'
        return self.attributes.get(key)
    
    def getData(self):
        'Get the cdata'
        return self.cdata
        
    def getElements(self,name=''):
        'Get a list of child elements'
        #If no tag name is specified, return the all children
        if not name:
            return self.children
        else:
            # else return only those children with a matching tag name
            elements = []
            for element in self.children:
                if element.name == name:
                    elements.append(element)
            return elements

    def searchElement(self,name=None,elems=[]):
        if name==None: return
        for element in self.children:
            if element.name == name:
                elems.append(element)
            else:
                element.searchElement(name,elems)
        
class Xml2Obj:
    'XML to Object'
    def __init__(self):
        self.root = None
        self.nodeStack = []
        
    def StartElement(self,name,attributes):
        'SAX start element even handler'
        # Instantiate an Element object
        element = Element(name.encode(),attributes)
        
        # Push element onto the stack and make it a child of parent
        if len(self.nodeStack) > 0:
            parent = self.nodeStack[-1]
            parent.AddChild(element)
        else:
            self.root = element
        self.nodeStack.append(element)
        
    def EndElement(self,name):
        'SAX end element event handler'
        self.nodeStack = self.nodeStack[:-1]

    def CharacterData(self,data):
        'SAX character data event handler'
        if data.strip():
            #data = data.encode()
            element = self.nodeStack[-1]
            element.cdata += data
            return

    def Parse(self,text=None,filename=None):
        # Create a SAX parser
        Parser = expat.ParserCreate()

        # SAX event handlers
        Parser.StartElementHandler = self.StartElement
        Parser.EndElementHandler = self.EndElement
        Parser.CharacterDataHandler = self.CharacterData

        # Parse the XML File
        if filename != None:
            ParserStatus = Parser.Parse(open(filename,'r').read(), 1)
        elif text != None:
            ParserStatus = Parser.Parse(text, 1)                 
        return self.root
    

            
def elemDict(elem):
    E={}
    def my_decode(x):
        try:
            y=x.decode('ascii')
        except:
            y=x

        return y

    if len(elem.attributes) >0:
        for k,v in elem.attributes.items():
            if v != '':
                E[my_decode(k)]=my_decode(v)

    if elem.cdata != '':
        #print elem.cdata
        E['cdata']=my_decode(elem.cdata)

    for e in elem.children:
        #print e.name
        k = my_decode(e.name)
        if k not in E:
            E[k]=[elemDict(e)]
        else:
            if isinstance(E[k],list):
                E[k].append(elemDict(e))
            else:
                old=E[k]
                E[k]=[old,elemDict(e)]
    return E


def cleanElemDict(E,tostr=True):
    Out = {}
    def toStr(a):
        if tostr: return str(a)
        else: return a
    
    if type(E)==dict:
        for k1,v1 in E.items():
            if k1=='cdata':
                return v1
            else:
                Out[str(k1)]=cleanElemDict(v1)
        
    elif type(E)==list:
        if len(E)==1:
            return cleanElemDict(E[0])
        else:
            return [cleanElemDict(e) for e in E]

    else:
        return E
    
    return Out
