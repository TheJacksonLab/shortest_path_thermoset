import numpy as np
import matplotlib.pyplot as plt
import my_common as mc
import extract_local_str as els
import networkx as nx
import scienceplots
plt.style.use(['science','ieee'])
import tools_lammps as tool_lmp
import shortest_path as sp

def length_evo(path,result,t,L):
    path_0 = path[:,0]
    wavelength = path[-1,1]
    length_contour_list = []
    length_e2e = []
    for itime in range(len(t)):
        flag=False
        length_contour = 0
        for i in range(len(path_0)-1):
            coors = result[itime].loc[:,'x':'z'].to_numpy() \
                    - np.array([L[itime][0][0],L[itime][1][0],L[itime][2][0]])
            box = np.array([
                [L[itime][0][1]-L[itime][0][0],0,0],
                [0,L[itime][1][1]-L[itime][1][0],0],
                [0,0,L[itime][2][1]-L[itime][2][0]],
            ])
            length = mc.pbc_distance(coors[path_0[i]],coors[path_0[i+1]],box)
            if length>1.5:
                flag = True
                break
            length_contour += length
        if flag==True:
            break
        length_e2e.append(L[itime][0][1]-L[itime][0][0])
        length_contour_list.append(length_contour/wavelength)
    return length_contour_list, length_e2e

dir_path = '../simulations/M40'
SPL,path = sp.get_SP_new('{}/cool.dat'.format(dir_path),2,6,slice_x_max=2)

file_trj = '{}/dump_relax.data'.format(dir_path)
result, t, L = tool_lmp.read_lammps_dump_custom(file_trj)

# make a plot
plt.figure(dpi=300, figsize=[2.4,1.8])
colormap = plt.cm.jet
colors = colormap(np.linspace(0, 1, 11))
for i in range(11):
    path_0 = path[np.argsort(SPL)[i]]
    lc_0, le_0 = length_evo(path_0,result,t,L)
    plt.plot(lc_0,le_0,'-',color=colors[i])
plt.plot([10,20],[10,20],'--')
plt.xlabel('Contour length')
plt.ylabel('System length $L$')
plt.tight_layout()
plt.savefig('./length_evolution.png')
plt.show()



