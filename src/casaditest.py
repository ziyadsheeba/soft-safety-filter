from casadi import *
import numpy as np
import ipdb
opti = casadi.Opti()
opti.solver('ipopt')

x = opti.variable(2,1)

opti.minimize( -x[0]**2 - x[1]**2)
opti.subject_to( x[0] <= 1)
opti.subject_to( x[0] >= -1)
opti.subject_to(x[1] >= -1)
opti.subject_to(x[1] <= 1 )
sol = opti.solve()
print('x: ', sol.value(x))

