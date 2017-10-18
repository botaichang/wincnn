import scipy.signal as spy 
from numpy.fft import * 

f = [0,0,1,2]

g = [10,20,30]


#=======================================
# 1. direct convolution
#=======================================
directConv = spy.correlate(f,g,'valid')

print directConv 

#=======================================
# 2. fft convolution
#=======================================
f_0 = f[0:3]
f_1 = f[1:4]
g_reverse = g
g_reverse.reverse()

print ifft(fft(f_0)* fft(g_reverse))
print ifft(fft(f_1)* fft(g_reverse))

#=======================================
# 3. winograd convolution
#=======================================

