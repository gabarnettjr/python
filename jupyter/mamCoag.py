import numpy as np
import matplotlib.pyplot as plt

def f( lnD, N_i, sigma_i, D_i ) :
    y = N_i / np.sqrt(2*np.pi) / np.log(sigma_i)
    y = y * np.exp( -1./2. * ( lnD - np.log(D_i) )**2. /  np.log(sigma_i)**2. )
    return y

# lnD = np.log( np.linspace( 1e-8, 1e-4, 1001 ) )
lnD = np.linspace( -20, -13, 1001 )
# lnD = np.logspace( -8, -4, 1001 )

#accumulation:
N_1 = 1e8
sigma_1 = 1.8
D_1 = .2e-6
f_1 = f( lnD, N_1, sigma_1, D_1 )

#aitken:
N_2 = 1e9
sigma_2 = 1.6
D_2 = .04e-6
f_2 = f( lnD, N_2, sigma_2, D_2 )

#primary carbon:
N_3 = 2e8
sigma_3 = 1.6
D_3 = .08e-6
f_3 = f( lnD, N_3, sigma_3, D_3 )

# print(lnD)
# print
# print( f( lnD, N_i, sigma_i, D_i ) )

help(plt.subplots)

fig, ax = plt.subplots( 1, 1, figsize=(10,4.5) )

plt.plot( lnD, f_1, lnD, f_2, lnD, f_3  )
plt.legend( [ 'accumulation', 'aitken', 'primary carbon' ]  )
plt.title( 'the three modes' )
plt.xlabel( 'log of particle diameter' )
plt.show()

# plt.plot( lnD, f_1+f_2+f_3 )
# plt.show()