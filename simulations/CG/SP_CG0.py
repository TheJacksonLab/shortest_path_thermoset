import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys 
sys.path.append('../../')
import tools_lammps as tool_lmp
import SP_thermoset.SP_tools as sp
from tqdm import tqdm

def collect_data(dir_list,image_number1=2,image_numbe2=8):
    SPL_M400 = []
    hs_M400 = []
    size_M400 = []
    for i, dir_path in tqdm(enumerate(dir_list)):
        try:
            SPL,_ = sp.get_SP('{}/cool.dat'.format(dir_path),image_number1,image_numbe2,slice_x_max=2)
            num_BB, strain = sp.count_BB_CG(file_bonds='{}/cool.dat'.format(dir_path),
                                            file_trj='{}/dump_relax.data'.format(dir_path),threshold=1.5)
            strain = (np.array(strain)[1:]+np.array(strain)[:-1])/2
            num_BB = np.array(num_BB)
            hs_M400.append([(strain[np.diff(num_BB)>0][-1]+strain[np.diff(num_BB)>0][0])/2,strain[np.diff(num_BB)>0][0],strain[np.diff(num_BB)>0][-1]])
            lmp = tool_lmp.read_lammps_full('{}/cool.dat'.format(dir_path))
            size_M400.append(lmp.x[1]-lmp.x[0])
            SPL_M400.append(SPL)
        except:
            print(dir_path)
            continue
    hs_M400 = np.array(hs_M400)
    min_M400 = np.array([np.min(SPL_M400[i]) for i in range(len(SPL_M400))])
    return SPL_M400, hs_M400, min_M400, size_M400

dir_list = ['./M40_ck75_v{}'.format(i) for i in range(10)]
SPL_M40_ck75, hs_M40_ck75, min_M40_ck75, size_M40_ck75 = collect_data(dir_list)

sp.count_BB_CG(file_bonds='{}/cool.dat'.format(dir_list[0]),file_trj='{}/dump_relax.data'.format(dir_list[0]),threshold=1.5)

def plot_min_SPL(ax,min_M20,size_M20,hs_M20,label1,**kwargs):
    ax.plot(min_M20/size_M20,np.exp(hs_M20)-1,'o',label='{0:s}'.format(label1),**kwargs)

plt.figure(dpi=300, figsize=np.array([2.3, 1.8]))
ax = plt.gca()
xrange = np.arange(1.2,4,0.1)
plt.plot(xrange,1.2*1.25*xrange-1,'k-',lw=1.5,label=r'Upper limit')
xx = np.concatenate([min_M40_ck75/size_M40_ck75])
yy = np.concatenate([np.exp(hs_M40_ck75[:,1])-1])
p1 = np.linalg.lstsq(xx.reshape(-1,1),yy.reshape(-1,1)+1,rcond=None)[0]
plt.plot(xrange,(p1*xrange-1).squeeze(),'--',color='violet',lw=1.0,label=r'Fitting')

plot_min_SPL(ax,min_M40_ck75,size_M40_ck75,hs_M40_ck75[:,1],'simulation',marker='o',ms=3,color='C2')

plt.legend(fontsize=5,ncol=1)
plt.tight_layout()
plt.xlabel(r'$D^g_\mathrm{min}/L_0$')
plt.ylabel(r'$\epsilon_n$')
plt.tight_layout() 
plt.savefig('min_strain0.png')
plt.show()