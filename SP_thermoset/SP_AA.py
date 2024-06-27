import networkx as nx
import numpy as np
import pandas as pd
import tools_lammps as tool_lmp
import copy
from scipy.signal import savgol_filter
import re

def read_MLABT_log(file):
    log_read = pd.read_csv(file)
    idx_collect = []
    step_unique = np.unique((log_read.Step.to_numpy()))
    last_step = int(step_unique[-1])
    step_interval = int(step_unique[-1]) - int(step_unique[-2])
    for istep in range(0,last_step,step_interval):
        if np.sum(log_read.to_numpy()[:,0]==istep)>0:
            idx_collect.append(np.squeeze(np.argwhere(log_read.to_numpy()[:,0]==istep)[0]))
    log_refine = log_read.loc[idx_collect,:].reset_index()
    return log_refine, log_read

def read_MLABT_BBIr(file):
    
    with open(file, 'r') as file:
        content = file.read()
    
    # pattern = r'\[([\d\.e+-]+) ([\d\.e+-]+) ([\d\.e+-]+) ([\d\.e+-]+)\]'
    pattern = r'\[\s*([\d]+\.?[\deE+-]*)\s+([\d]+\.?[\deE+-]*)\s+([\d]+\.?[\deE+-]*)\s+([\d]+\.?[\deE+-]*)\s*\]'
    matches = re.findall(pattern, content)
    return np.array(matches).astype(float)

def read_lammps(file, lmp_mode='charge'):
    """
    shift the original point to 0 0 0
    """

    f=open(file,'r')
    L=f.readlines()
    f.close()

    isxyxzyz = 0
    for iline in range(len(L)):
        if 'atoms' in L[iline]:
            natoms = int(L[iline].split()[0])
        if 'xlo' in L[iline]:
            xlo=float(L[iline].split()[0])
            xhi=float(L[iline].split()[1])
        if 'ylo' in L[iline]:
            ylo=float(L[iline].split()[0])
            yhi=float(L[iline].split()[1])
        if 'zlo' in L[iline]:
            zlo=float(L[iline].split()[0])
            zhi=float(L[iline].split()[1])

        if 'xy' in L[iline]:
            isxyxzyz=1
            xy=float(L[iline].split()[0])
            xz=float(L[iline].split()[1])
            yz=float(L[iline].split()[2])

        if 'Atoms #' in L[iline]:
            latom = iline+2

    if isxyxzyz==0:
        xy=0; xz=0; yz=0

    box = np.array([[xhi-xlo,0,0],[xy,yhi-ylo,0],[xz,yz,zhi-zlo]])

    index = np.empty(natoms)
    atom_type = np.empty(natoms)
    coors = np.empty([natoms,3])

    for i in range(natoms):
        index[i] = int(L[latom+i].split()[0])
        if lmp_mode =='charge':
            atom_type[i] = int(L[latom+i].split()[1])
            coors[i,:] = np.array([float(L[latom+i].split()[3])-xlo,float(L[latom+i].split()[4])-ylo,float(L[latom+i].split()[5])-zlo])
        elif lmp_mode =='full':
            atom_type[i] = int(L[latom+i].split()[2])
            coors[i,:] = np.array([float(L[latom+i].split()[4])-xlo,float(L[latom+i].split()[5])-ylo,float(L[latom+i].split()[6])-zlo])

    if atom_type[-1]>0:
        return natoms,box,index,atom_type,coors
    else:
        print("error")

def create_NN_lammps(lammps_file,new_file,select_idx=[],natom_DGEBA=49,natom_MDA=29):
    """
    inputs: lammps_file,new_file,select_idx (optional for making the selected N atoms another type in visualization)
    output: print new lammps file 
    
    """
    lmp_tmp = els.read_lammps_full(lammps_file)
    bond_info = lmp_tmp.bond_info

    natoms,box,index,atom_type,coors = read_lammps(lammps_file,lmp_mode='full')
    coors = coors[np.argsort(index)]
    atom_type = atom_type[np.argsort(index)]
    index = index[np.argsort(index)]
    idx_N = np.squeeze(np.argwhere(atom_type==8))
    
    n_mol = int(natoms/(natom_DGEBA*2+natom_MDA)*3) # total number of molecules 
    idx_N_mol = 3*((idx_N+1)//(natom_DGEBA*2+natom_MDA))+2 # mol index (MDA) of which the N is in

    e = [] # index of neighbor molecule of N 
    CN_N = []
    for iN in range(len(idx_N)):
        tmp = np.concatenate((bond_info[bond_info[:,2] == idx_N[iN]+1,3]-1,bond_info[bond_info[:,3] == idx_N[iN]+1,2]-1)) # neighbor atom
        CN_N.append(len(tmp))

        tmp_idx1 = (tmp)//(natom_DGEBA*2+natom_MDA)
        tmp_idx2 = (tmp)%(natom_DGEBA*2+natom_MDA) // natom_DGEBA
        idx_neigh_mol = tmp_idx1*3+tmp_idx2
        # print(idx_neigh_mol,idx_N_mol[iN])
        idx_neighbor = idx_neigh_mol
        # idx_neighbor = idx_neigh_mol[idx_neigh_mol!=idx_N_mol[iN]] ## ignore self
        # print(idx_neighbor,idx_N_mol[iN])
        e.append(idx_neighbor)
        # for ie in range(len(idx_neighbor)):
        #     e.append((int(idx_neighbor[ie])))
    # e = np.array(e)
    CN_N=np.array(CN_N) # all CN_N should be 3 
    
    ue = [np.unique(e[i]) for i in range(len(e))]
    bond_info_N = []
    n_bond_N = 0
    idx_mol_betweenNN= []

    for i in range(len(ue)):
        for j in range(i+1,len(ue)):
            if np.size(np.intersect1d(ue[i],ue[j]))>=1:
                n_bond_N += 1
                idx_mol_betweenNN.append(np.intersect1d(ue[i],ue[j]))
                if np.size(np.intersect1d(ue[i],ue[j]))>=2:
                    bond_info_N.append([n_bond_N,3,i+1,j+1])
                    # print(np.intersect1d(ue[i],ue[j])%3)  # number of self-loops
                else:
                    if np.intersect1d(ue[i],ue[j])%3==2:
                        bond_info_N.append([n_bond_N,2,i+1,j+1])
                    else:
                        bond_info_N.append([n_bond_N,1,i+1,j+1])
    bond_info_N = np.array(bond_info_N)
    num_mol_ue = np.array([np.sum(np.concatenate(ue) == i) for i in range(n_mol)])
    
    atom_info_N = lmp_tmp.atom_info[np.argsort(lmp_tmp.atom_info[:,0]),:][idx_N]
    atom_info_N[:,2] = 1
    atom_info_N[:,0] = np.arange(1,len(atom_info_N)+1)
    lmp_N = tool_lmp.lammps(len(idx_N),1,lmp_tmp.x,lmp_tmp.y,lmp_tmp.z,
                            mass=np.array([[1,14.007,'N']]).reshape(-1,3),
                            atom_info = atom_info_N,
                            )
    lmp_N.bond_info = bond_info_N
    lmp_N.nbond_types = 3
    lmp_N.nbonds = len(bond_info_N)
    
    if len(select_idx)>0:
        lmp_N.atom_info[select_idx,2]=2
        lmp_N.natom_types = 2 
        lmp_N.mass = np.array([[1,14.007,'N'],[2,15.999,'O']]).reshape(-1,3)
        
    tool_lmp.write_lammps_full(new_file,lmp_N)
    return ue

# visualize local stress 
def concatenate_xyz(filenames, output_filename="combined.xyz"):
    """
    Concatenate multiple XYZ files into a single file.

    Parameters:
    - filenames: List of names of XYZ files to concatenate.
    - output_filename: Name of the output concatenated XYZ file.
    """
    with open(output_filename, 'w') as outfile:
        for fname in filenames:
            with open(fname, 'r') as infile:
                # Copy content of infile to outfile
                for line in infile:
                    outfile.write(line)

def write_xyz_from_density(density, threshold, filename="output.xyz"):
    """
    Write an XYZ file from a density array.

    Parameters:
    - density: 3D numpy array of density values.
    - threshold: minimum density value to include in the output.
    - filename: name of the output XYZ file.
    """
    # Identify points exceeding the threshold
    indices = np.where(density > threshold)
    densities_above_threshold = density[indices]
    
    # Open file for writing
    with open(filename, 'w') as f:
        # Write the number of atoms (pseudo-atoms)
        f.write(f"{len(densities_above_threshold)}\n")
        f.write("Atoms. Generated from density data.\n")
        
        # Loop through and write pseudo-atoms to the file
        for (ix, iy, iz), dens_value in zip(zip(*indices), densities_above_threshold):
            atom_type = "C"  # Using Carbon as a placeholder. Can be modified based on need.
            f.write(f"{atom_type} {ix*bin_size[0]:.5f} {iy*bin_size[1]:.5f} {iz*bin_size[2]:.5f} {dens_value:.5f}\n")

def midpoint_pbc(coords, idx_a, box):
    """
    Calculate the midpoint between two atoms in a bond considering PBC.

    Parameters:
    - coords: Array containing atomic coordinates (N x 3 where N is the number of atoms).
    - idx_a: Indices of atoms forming bonds (M x 2 where M is the number of bonds).
    - box: 3x3 matrix representing the simulation box.

    Returns:
    - Array containing midpoints (M x 3).
    """
    midpoints = []
    box_diag = np.diag(box)
    
    for idx in idx_a:
        r1 = coords[idx[0]]
        r2 = coords[idx[1]]
        # Calculate bond vector
        dr = r2 - r1
        # Apply minimum image convention
        dr = dr - np.round(dr / box_diag) * box_diag
        # Calculate midpoint
        midpoint = r1 + dr / 2
        # Make sure the midpoint is within the box
        midpoint = (midpoint + box_diag) % box_diag
        midpoints.append(midpoint)
    return np.array(midpoints)

def output_data_afterBB(location, threshold_BB = 1.5):
    # write dump files after each bond breaking to lammps data file with modified topology
    # location = './length/M40_v1'
    file_bonds = '{}/cool.dat'.format(location)
    file_trj = '{}/dump_relax.data'.format(location)

    # SPL,path = get_SP(file_bonds,2,8)
    lmp = tool_lmp.read_lammps_full(file_bonds)
    lmp_tmp = copy.copy(lmp)
    result, t, L = tool_lmp.read_lammps_dump_custom(file_trj)

    bond_length = []
    idx_bead_NN_list = []
    strain_list = []
    for i in range(len(t)):
        idx_bonded_atoms = (lmp_tmp.bond_info[:,2:]-1).astype(int)
        box = np.array([[L[i][0][1]-L[i][0][0],0,0],
                        [0,L[i][1][1]-L[i][1][0],0],
                        [0,0,L[i][2][1]-L[i][2][0]]])
        coors = result[i].loc[:,'x':'z'].to_numpy()
        
        # for ibond in range(len(idx_bonded_atoms)):
        dist = tool_lmp.pbc_distance(coors[idx_bonded_atoms[:,0]],
                        coors[idx_bonded_atoms[:,1]],
                        box)
        bond_length.append(dist)
        if np.sum(dist>threshold_BB)>0:
            strain_list.append(np.log(box[0,0]/(L[0][0][1]-L[0][0][0])))
            idx_bead_NN = (idx_bonded_atoms[np.argwhere(dist>1.5).squeeze()]).reshape(-1,2)
            idx_bead_NN_list.append(idx_bead_NN)
            for ibb in idx_bead_NN:
                A = np.argwhere((lmp_tmp.bond_info[:,2]==ibb[0]+1) & (lmp_tmp.bond_info[:,3]==ibb[1]+1))
                if len(A)>0:
                    lmp_tmp.bond_info = np.delete(lmp_tmp.bond_info, A.squeeze(), axis=0)
                    print(len(lmp_tmp.bond_info)+1,len(lmp_tmp.bond_info))
                    lmp_tmp.nbonds = lmp_tmp.nbonds -1
                else:
                    print('error on the bond')
            atom_info = lmp_tmp.atom_info[np.argsort(lmp_tmp.atom_info[:,0])]
            atom_info[:,4:7] = coors
            lmp_tmp.atom_info = atom_info
            lmp_tmp.x = [np.min(coors[:,0]),np.min(coors[:,0])+box[0,0]]
            lmp_tmp.y = [np.min(coors[:,1]),np.min(coors[:,1])+box[1,1]]
            lmp_tmp.z = [np.min(coors[:,2]),np.min(coors[:,2])+box[2,2]]
            tool_lmp.write_lammps_full('{}/t{}.dat'.format(location,i),lmp_tmp)
    
    return idx_bead_NN_list,strain_list

# def first_BB_segment(file_trj,SPL,path,file_bonds,threshold):

#     # 1st bond breakage 
#     lmp = tool_lmp.read_lammps_full(file_bonds)
#     idx_bonded_atoms = (lmp.bond_info[:,2:]-1).astype(int)

#     # distances of bonds in the trajectory 
#     result, t = els.read_multiple_xyz(file_trj)

#     bond_length = []
#     strain = []
#     for i in range(len(t)):
#         box = result[i][0]
#         coors = result[i][4]
#         strain.append(np.log(box[0,0]/result[0][0][0,0]))
#         # for ibond in range(len(idx_bonded_atoms)):
#         dist = tool_lmp.pbc_distance(coors[idx_bonded_atoms[:,0]],
#                         coors[idx_bonded_atoms[:,1]],
#                         box)
#         bond_length.append(dist)
#         if np.sum(dist>threshold)>0:
#             break

#     # print(t[i])
#     idx_bead_NN = (idx_bonded_atoms[np.argwhere(dist>1.5).squeeze()]).reshape(-1,2)
#     # print(idx_bead_NN)

#     idx_onpath_list = []
#     for i in range(idx_bead_NN.shape[0]):
#         for j in range(len(path)):
#             if idx_bead_NN[i] in path[np.argsort(SPL)[j]]:
#                 path_length = np.sort(SPL)[j]
#                 idx_path = np.argwhere(np.sort(SPL)==path_length)[0]
#                 print(idx_path)
#                 print('0th broken bond in {}th SP (length), {}th SP'.format(idx_path, j),path_length-1)
#                 if j==idx_path:
#                     # try:
#                     which_edge = np.min([np.argwhere(path[np.argsort(SPL)[j]][:-1,0]==idx_bead_NN[i][0]).squeeze(),
#                                         np.argwhere(path[np.argsort(SPL)[j]][:-1,0]==idx_bead_NN[i][1]).squeeze()])
#                     # except:
#                     print(len(path[np.argsort(SPL)[j]][:-1,0])-1,which_edge,idx_bead_NN[i])
#                     idx_onpath_list.append([j,which_edge])
#                 break
#     idx_onpath_list = np.concatenate([idx_onpath_list])
#     return idx_onpath_list