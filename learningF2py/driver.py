from numpy import f2py

# f2py.run_main(['-m','add','add.f'])

with open("add.f") as sourcefile :
    sourcecode = sourcefile.read()
f2py.compile( sourcecode.encode(), modulename='add' )

import numpy as np
import add
x = np.arange(0.,5.,1.)
y = np.arange(0.,5.,1.)
z = np.zeros((5))
print(x)
print(y)
print(z)
add.zadd(x,y,z,len(x))
print(z)
