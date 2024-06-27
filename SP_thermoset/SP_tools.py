import networkx as nx
import numpy as np
import tools_lammps as tool_lmp
from scipy.signal import savgol_filter

def create_periodic_image(G, image_count, box_size, coors):
    G_new = nx.Graph()
    
    # Helper function: Get translated coordinates for nodes
    def get_translated_coordinate(coordinate, translation_vector):
        return tuple(np.add(coordinate, translation_vector))
    
    # Helper function: Check if an edge crosses PBC inside an image
    def edge_crosses_pbc(u_coor, v_coor, box_size):
        # print(u_coor,v_coor,box_size)
        if abs(u_coor[0] - v_coor[0]) > box_size / 2:
            return True
        return False
    
    # 1. Duplicate nodes
    for i in range(image_count):
        for node, data in G.nodes(data=True):
            translated_coor = get_translated_coordinate(coors[node], [i * box_size, 0, 0])
            # print(translated_coor)
            G_new.add_node((node, i), coordinate=translated_coor)  # Using a tuple (node, i) to uniquely identify nodes in different images
            
    # 2. Add edges within each image, avoiding those that cross PBC
    for i in range(image_count):
        for u, v in G.edges():
            u_coor = G_new.nodes[(u, i)]['coordinate']
            v_coor = G_new.nodes[(v, i)]['coordinate']
            
            if not edge_crosses_pbc(u_coor, v_coor, box_size):
                G_new.add_edge((u, i), (v, i), capacity=1)
    # print(len(G_new.edges))
    
   # 3. Add edges between images considering PBC
    for i in range(image_count - 1):  # No PBC between the last image and the first one for G2 and G3
        for u in G.nodes():
            u_coor = G_new.nodes[(u, i)]['coordinate']
            for v in G.nodes():
                v_coor = G_new.nodes[(v, i+1)]['coordinate']
                if G.has_edge(u, v):  # Checking if nodes are connected in original G1
                    if (v_coor[0]-u_coor[0])<box_size/2:
                        G_new.add_edge((u, i), (v, i+1), capacity=1)  # Connect the nodes between adjacent images
                    
    return G_new

def find_shortest_path_across_PB(lammps_file,image_number,slice_x_max=None,direction='x'):
    """
    find the shortest paths across the PB (might be multiple),

    lammps_file needs to be a file based on NN !!!
    """
    lmp_new = tool_lmp.read_lammps_full(lammps_file)
    if slice_x_max is None:
        slice_x_max = lmp_new.x[1]-lmp_new.x[0]
    lmp_new.atom_info = lmp_new.atom_info[np.argsort(lmp_new.atom_info[:,0])]
    coors = lmp_new.atom_info[:,4:7]-np.array([lmp_new.x[0],lmp_new.y[0],lmp_new.z[0]])
    G1 = nx.Graph()
    G1.add_nodes_from(np.arange(0,lmp_new.natoms))
    for ib in range(len(lmp_new.bond_info)):
        G1.add_edges_from([lmp_new.bond_info[ib,2:].astype(int)-1])
    if direction=='x':
        box_size = float(np.diff(lmp_new.x))
        Gn = create_periodic_image(G1, image_number, box_size,coors)
        
    # idx_slice = np.argwhere((coors[:,0]<lmp_new.x[0]+slice_x_max)).squeeze()
    idx_slice = np.argwhere((coors[:,0]<slice_x_max)).squeeze()
    SP_10 = []
    path_10 = []
    for i in idx_slice:
        if nx.has_path(Gn,(i,0),(i,image_number-1)):
            # SP_10.append(nx.shortest_path_length(Gn,(i,0),(i,image_number-1)))
            path_tmp = np.array(nx.shortest_path(Gn,(i,0),(i,image_number-1)))
            SP_10.append(len(path_tmp)-1)
            path_10.append(path_tmp)
    return SP_10, path_10

def get_SP(NN_file,image_min=2,image_max=15,remove_case3=True,slice_x_max=15):
    SPL_list = []
    SPL_all = []
    path_all = []
    for image_number in range(image_min,image_max):
        SPL, path = find_shortest_path_across_PB(NN_file,slice_x_max=slice_x_max,image_number=image_number)
        # np.sort(np.array(SPL)/12)
        SPL_true = []
        path_true = []
        for ipath in range(len(SPL)):
            if len(path[ipath][:,0]) - len(np.unique(path[ipath][:,0])) -1 == 0:
                SPL_true.append(SPL[ipath])
                path_true.append(path[ipath])
                SPL_all.append(SPL[ipath]/(image_number-1))
                path_all.append(path[ipath])
        if len(SPL_true)>0:
            SPL_list.append([image_number-1,np.sort(np.array(SPL_true)/(image_number-1))[0]])
        else:
            break
    SPL_all_unique = []
    path_all_unique = []
    for ipath in range(len(SPL_all)):
        if SPL_all[ipath] not in SPL_all_unique:
            SPL_all_unique.append(SPL_all[ipath])
            path_all_unique.append(path_all[ipath])
        else:
            if remove_case3:
                idx_given_paths = (np.argwhere(np.array(SPL_all_unique)==SPL_all[ipath]))
                n=0
                for jpath in idx_given_paths:
                        if (len(np.intersect1d(path_all_unique[jpath.squeeze()][:-1,0],
                            path_all[ipath][:-1,0])) == len(path_all[ipath][:-1,0])):
                            n+=1
                if n==0:
                    SPL_all_unique.append(SPL_all[ipath])
                    path_all_unique.append(path_all[ipath])
            else:
                SPL_all_unique.append(SPL_all[ipath])
                path_all_unique.append(path_all[ipath])
                
    return SPL_all_unique,path_all_unique

# def get_SP_new(NN_file,image_min=2,image_max=15,remove_case3=True,slice_x_max=15):
#     """
#     only count one path for one pair of identical atoms
#     """
#     SPL_list = []
#     SPL_all = []
#     path_all = []
#     for image_number in range(image_min,image_max):
#         SPL, path = find_shortest_path_across_PB(NN_file,slice_x_max=slice_x_max,image_number=image_number)
#         # np.sort(np.array(SPL)/12)
#         SPL_true = []
#         path_true = []
#         for ipath in range(len(SPL)):
#             if len(path[ipath][:,0]) - len(np.unique(path[ipath][:,0])) -1 == 0:
#                 SPL_true.append(SPL[ipath])
#                 path_true.append(path[ipath])
#                 SPL_all.append(SPL[ipath]/(image_number-1))
#                 path_all.append(path[ipath])
#         if len(SPL_true)>0:
#             SPL_list.append([image_number-1,np.sort(np.array(SPL_true)/(image_number-1))[0]])
#             # print(SPL_list[-1])
#         else:
#             break
#     SPL_all_unique = []
#     path_all_unique = []
#     for ipath in range(len(SPL_all)):
#         if SPL_all[ipath] not in SPL_all_unique:
#             SPL_all_unique.append(SPL_all[ipath])
#             path_all_unique.append(path_all[ipath])
#         else:
#             if remove_case3:
#                 idx_given_paths = (np.argwhere(np.array(SPL_all_unique)==SPL_all[ipath]))
#                 n=0
#                 for jpath in idx_given_paths:
#                     if (len(np.intersect1d(path_all_unique[jpath.squeeze()][:-1,0],
#                         path_all[ipath][:-1,0])) == len(path_all[ipath][:-1,0])):
#                         n+=1
#                 if n==0:
#                     SPL_all_unique.append(SPL_all[ipath])
#                     path_all_unique.append(path_all[ipath])
#             else:
#                 SPL_all_unique.append(SPL_all[ipath])
#                 path_all_unique.append(path_all[ipath])

#     # path_flat = [path_all_unique[i][:-1,0] for i in range(len(path_all_unique))]
#     starting_point = np.unique([path[0,0] for path in path_all_unique])
#     spl_final = []
#     path_final = []
#     for i in range(len(starting_point)):
#         idx_starting = [path[0,0]==starting_point[i] for path in path_all_unique]
#         idx_tmp = np.argwhere(idx_starting).flatten()
#         if len(idx_tmp)==1:
#             spl_final.append(SPL_all_unique[idx_tmp[0]])
#             path_final.append(path_all_unique[idx_tmp[0]])
#         else:
#             spl_values = np.array(SPL_all_unique)[idx_tmp]
#             min_spl_idx = idx_tmp[np.argmin(spl_values)]

#             spl_final.append(spl_values.min())
#             path_final.append(path_all_unique[min_spl_idx])
#     return spl_final,path_final

def read_multiple_xyz(file):
    """
    read multiple frames in output xyz of lammps, 
    input: dump_file, !!!! now change to 'custum'
    output: result (index,type_atoms, coors) and t 
    """
    f = open(file)
    lft = list(f)
    f.close()
    lt=[]
    t=[]
    natom = int(lft[3].split()[0])
    for il in range(len(lft)):
        if 'ITEM: TIMESTEP' in lft[il]:
            lt.append(il)
            t.append(lft[il+1].split()[0])

    def read_lf(lf):
        box = np.zeros([3,3])      ###### now only orthogonal box 
        coors = np.zeros([natom,3])
        mol = []
        type_atom = []
        index = []
        l=0
        
        xlo = float(lf[5].split()[0]); xhi = float(lf[5].split()[1]);
        ylo = float(lf[6].split()[0]); yhi = float(lf[6].split()[1]);
        zlo = float(lf[7].split()[0]); zhi = float(lf[7].split()[1]);
        box[0,0] = xhi-xlo
        box[1,1] = yhi-ylo
        box[2,2] = zhi-zlo
        
        for ia in lf[9:9+natom]:
            coors[l,:] = np.array(ia.split()[3:6:1]).astype('float')
            coors[l,:] = coors[l,:] - np.array([xlo,ylo,zlo])
            type_atom.append(int(ia.split()[2]))
            mol.append(int(ia.split()[1]))
            index.append(int(ia.split()[0]))
            l+=1

        type_atom = np.array(type_atom)
        mol = np.array(mol)
        index = np.array(index)
        
        mol = mol[np.argsort(index)]
        type_atom = type_atom[np.argsort(index)]
        coors = coors[np.argsort(index)]
        index = index[np.argsort(index)]
        
        return box,index,mol,type_atom,coors

    # lf=[]
    result = []
    for it in range(len(lt)):
        if it==len(lt)-1:
    #         lf.append(lft[lt[it]:])
            result.append(read_lf(lft[lt[it]:]))
        else:
    #         lf.append(lft[lt[it]:lt[it+1]-1])
            result.append(read_lf(lft[lt[it]:lt[it+1]]))
    
    t=np.array(t)
    return result,t


def count_BB_CG(file_bonds,file_trj,threshold):
    lmp = tool_lmp.read_lammps_full(file_bonds)
    idx_bonded_atoms = (lmp.bond_info[:,2:]-1).astype(int)

    # distances of bonds in the trajectory 
    result, t = read_multiple_xyz(file_trj)

    bond_length = []
    strain = []
    for i in range(len(t)):
        box = result[i][0]
        coors = result[i][4]
        strain.append(np.log(box[0,0]/result[0][0][0,0]))
        # for ibond in range(len(idx_bonded_atoms)):
        dist = tool_lmp.distance_pbc(coors[idx_bonded_atoms[:,0]],
                        coors[idx_bonded_atoms[:,1]],
                        box)
        bond_length.append(dist)

    # number of broken bonds, if length>1.5
    num_BB = []
    for i in range(len(t)):
        num_BB.append(np.sum(bond_length[i]>threshold))
    return num_BB, np.array(strain)

