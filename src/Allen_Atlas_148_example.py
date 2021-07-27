import numpy as np
import matplotlib.pyplot as plt  
import scipy.io
import networkx as nx

#The Virtual Brain
from tvb.simulator.lab import *

def listToDict(lst):
    op = { i : lst[i] for i in range(0, len(lst) ) }
    return op
    
def drawMouseFunctionalgraph(matr,w,nodecolor,nsize,al,vmi,vma,colormap,rimmel,sv):
    G=nx.from_numpy_matrix(matr)
    fon=[]
    fon.extend('w' for i in range(nregions))
    for i in rimmel:
        fon[i]='k'
    wline=[]
    wline.extend(5. for i in range(nregions))
    for i in rimmel:
        wline[i]=15.5
        
    lab=listToDict(range(nregions))
    
    xrange=range
    node_list = sorted(G.nodes())
    
    # figsize is intentionally set small to condense the graph
    fig, ax = plt.subplots(figsize=(16,16))
    margin=0.33
    fig.subplots_adjust(margin, margin, 1.-margin, 1.-margin)
    ax.axis('equal')
    print(Nposition[0])

    nx.draw(G, width=w, pos=Nposition, edge_color='#525050', ax=ax,  linewidths=wline, node_color=nodecolor, node_size=nsize, node_shape='o',alpha=al,cmap=colormap,vmin=vmi,vmax=vma,with_labels=True, font_size=18)
    #description = nx.draw_networkx_labels(G, pos=Nposition)

    ax= plt.gca()
    ax.collections[0].set_edgecolor(fon)
    plt.tight_layout()
    plt.savefig(sv+'net.png', transparent=True)    
    plt.show()

    
# Import the anatomical structural connectivity of the Allen Institute
path='/Users/giovanni/Desktop/Neuro/tvb_codes/connectivity/'

A148_con = connectivity.Connectivity.from_file(path+"Allen_148.zip")   

centres=np.loadtxt(path+"Allen_148/centres.txt")
x, y, z = centres
  
with open(path+"Allen_148/region_labels.txt") as f:
    content = f.readlines()   

# you may also want to remove whitespace characters like `\n` at the end of each line
SC_labels = [ix.strip() for ix in content] 
nregions = len(SC_labels)     #number of regions

Nposition={}
for i in range(nregions):
    Nposition[i]=(x[i],y[i])
#     if i==19:
#         Nposition[i]=(x[i]-2.3,y[i]-0.5)
#     elif i==71:
#         Nposition[i]=(x[i]+2.3,y[i]-0.5)
#     elif i==21:
#         Nposition[i]=(x[i],y[i]+1.5)
#     elif i==73:
#         Nposition[i]=(x[i],y[i]+1.5)
#     elif i==27:
#         Nposition[i]=(x[i]+1,y[i])
#     elif i==79:
#         Nposition[i]=(x[i]-1,y[i])
#     else:
#         Nposition[i]=(x[i],y[i])


SC=A148_con.weights
SC[np.where(SC == 0.)[0],np.where(SC == 0.)[1]]=1e-7
np.fill_diagonal(SC, 0.)

weights = 2.5*SC[np.triu_indices(nregions,1)]

fonW=[]
fonW.extend('w' for i in range(nregions))
fon=[]
fon.extend('black' for i in range(nregions))
nodecolor=[]
nodecolor.extend('darkgray' for i in range(nregions))