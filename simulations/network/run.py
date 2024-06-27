import numpy as np
from generate_network import create_ini_lammps,lmp_add_edge
import os
import sys
sys.path.append('../../')
from SP_thermoset import SP_tools as sp
from tqdm import tqdm

run_lammps = 'lmp_serial'
def SP_create(lmp_file,size,ck_degree,density,max_degree,sigma,if_generate=True):
    if if_generate:
        create_ini_lammps('network.dat',size,density=density)
        os.system('{} -in in.relax'.format(run_lammps))
        lmp = lmp_add_edge('relax.dat',lmp_file,ck_degree=ck_degree,max_degree=max_degree,sigma=sigma)
    SPL, path = sp.get_SP(lmp_file,2,6)
    return SPL, path

sizes_list = [40,50]
SPL_sizes = []
for size in sizes_list:
    SPL_runs = []
    for i in tqdm(range(10)):
        SPL, path = SP_create('network_s{}_v{}.dat'.format(size,i),size,ck_degree=0.95,density=0.0015026296018031556*1,max_degree=3,sigma=15,if_generate=True)
        if SPL is not None:
            SPL_runs.append(SPL)
    SPL_sizes.append(SPL_runs)

# SP_full('network_test.dat',70,ck_degree=0.95,density=0.0015026296018031556*1,max_degree=3,sigma=10,if_generate=True)

min_40 = np.array([np.min(i) for i in SPL_sizes[0]])/40
min_50 = np.array([np.min(i) for i in SPL_sizes[1]])/50

mean_40 = np.array([np.mean(i) for i in SPL_sizes[0]])/40
mean_50 = np.array([np.mean(i) for i in SPL_sizes[1]])/50

print(f'minimum SP length for size of 40 = {np.mean(min_40)}')
print(f'Average SP length for size of 40 = {np.mean(mean_40)}')

print(f'minimum SP length for size of 50 = {np.mean(min_50)}')
print(f'Average SP length for size of 50 = {np.mean(mean_50)}')






