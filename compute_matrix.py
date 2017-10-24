import wincnn
import scipy.signal as spy 
from numpy.fft import * 
import numpy as np
from sympy import pprint
#=======================================
#For 1D Convolution  Y = AT(Gg.*BTd)
#For 2D Convolution  Y = AT(GgGt.*BTdB)A
#=======================================
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
AT,G,BT,_ = wincnn.cookToomFilter((0,1,-1),2,3,2)
AT = np.array(AT)
G = np.array(G)
BT = np.array(BT)

Gg = G.dot(g)
BTd = BT.dot(d) 
C = Gg*(BTd)
winout = AT.dot(C)

print 'directConv = \n',directConv
print 'winout = \n',winout


#F(4,3) input = 4+3-1 = 6
#=======================================
# 1. direct convolution 
#=======================================
d = np.array([1,0,1,3,2,3]).T
g = np.array([10,20,30]).T
directConv = spy.correlate(d,g,'valid')
#=======================================
#2. winograd convolution
#=======================================
AT,G,BT,_ = wincnn.cookToomFilter((0,1,-1,2,-2),4,3,2)
AT = np.array(AT)

G = np.array(G)
BT = np.array(BT)

Gg = G.dot(g)
BTd = BT.dot(d) 
C = Gg*(BTd)
winout = AT.dot(C)

print 'directConv = \n',directConv
print 'winout = \n',winout
#=======================================
# 1. direct convolution 
#=======================================
d = np.array([1,0,1,3,2,3,3]).T # m + r - 1 = 7, r=4,m = 4
g = np.array([10,20,30,10]).T
directConv = spy.correlate(d,g,'valid')
#=======================================
#2. winograd convolution
#=======================================
AT,G,BT,_ = wincnn.cookToomFilter((0,1,-1,2,-2,3),4,4,2)
AT = np.array(AT)
G = np.array(G)
BT = np.array(BT)
Gg = G.dot(g)
BTd = BT.dot(d) 
C = Gg*(BTd)
winout = AT.dot(C)
print 'directConv = \n',directConv
print 'winout = \n',winout

#=======================================
# 1. 2D direct convolution 
# Y = AT(GgGt.*BTdB)A
#=======================================
d = np.array([[1,0,1,3,2,3],[1,2,3,4,5,6],[0,0,1,0,1,0],[1,1,0,1,0,0],[1,1,1,1,1,1],[2,1,1,0,0,0]]).T
g = np.array([[10,20,30],[1,2,3],[4,5,6]]).T
directConv = spy.correlate(d,g,'valid')
print directConv
#=======================================
#2. winograd convolution
#=======================================
AT,G,BT,_ = wincnn.cookToomFilter((0,1,-1,2,-2),4,3,2)
AT = np.array(AT)
G = np.array(G)
BT = np.array(BT)
GgGt = G.dot(g).dot(G.T)
BTdB = BT.dot(d).dot(BT.T)
C = GgGt*(BTdB)
print C.shape
winout = AT.dot(C).dot(AT.T)
print 'directConv = \n',directConv
print 'winout = \n',winout

