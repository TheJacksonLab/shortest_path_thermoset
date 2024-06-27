import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from SP_thermoset import SP_tools, SP_AA


# O21 
SP_AA.create_NN_lammps(lammps_file='./ck98.dat',
                  new_file='./ck98_NN.dat') 

SPL_O21,_ = SP_tools.get_SP('./ck98_NN.dat')

# O100 
SP_AA.create_NN_lammps(lammps_file='./ck88.dat',
                  new_file='./ck88_NN.dat') 
SPL_O100,_ = SP_tools.get_SP('./ck88_NN.dat')

# O150
SP_AA.create_NN_lammps(lammps_file='./ck83.dat',
                  new_file='./ck83_NN.dat') 
SPL_O150,_ = SP_tools.get_SP('./ck83_NN.dat')

# O200
SP_AA.create_NN_lammps(lammps_file='./ck77.dat',
                  new_file='./ck77_NN.dat') 
SPL_O200,_ = SP_tools.get_SP('./ck77_NN.dat')

plt.figure(dpi=300)
a,b = np.histogram(SPL_O21,np.arange(30))
plt.plot((b[1:]+b[:-1])/2,a,'-o',ms=3,label='98\%')
a,b = np.histogram(SPL_O100,np.arange(30))
plt.plot((b[1:]+b[:-1])/2,a,'-s',ms=3,label='88\%')
a,b = np.histogram(SPL_O150,np.arange(30))
plt.plot((b[1:]+b[:-1])/2,a,'-v',ms=3,label='83\%')
a,b = np.histogram(SPL_O200,np.arange(30))
plt.plot((b[1:]+b[:-1])/2,a,'-^',ms=3,label='77\%')
plt.xlabel('$D^g$')
plt.ylabel('Counts')
plt.legend()
plt.xlim(5,30)
plt.ylim(0,)
plt.tight_layout()
plt.savefig('./disp_ck_AA.png')
plt.show()