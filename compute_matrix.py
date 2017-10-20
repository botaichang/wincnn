import wincnn
import scipy.signal as spy 
from numpy.fft import * 
import numpy as np

#F(2,3)
#=======================================
# 1. direct convolution 
#=======================================
d = np.array([1,0,1,3]).T
g = np.array([10,20,30]).T

directConv = spy.correlate(d,g,'valid')

#=======================================
#2. winograd convolution
#=======================================
AT,G,BT,_ = wincnn.cookToomFilter((0,1,-1),2,3,0)
AT = np.array(AT)
G = np.array(G)
BT = np.array(BT)

Gg = G.dot(g)
BTd = BT.dot(d) 
C = Gg*(BTd)
winout = AT.dot(C)

print winout
print directConv
