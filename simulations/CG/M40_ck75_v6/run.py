import sys
import os
sys.path.append('../')
from CG_DGEBA import generate_lattice, crosslink_AB 

generate_lattice(num_A=80, num_B=40, box_size=200, min_distance=0.8, output_file='test.dat')

os.system('mpirun ~/lammps-23Jun2022/build/lmp -in in.relax')

crosslink_AB('relax.dat','test_ck.dat','mpirun ~/lammps-23Jun2022/build/lmp -in in.anneal',percentage_ck=0.75, functionality=4, crosslink_distance=1.5)

os.system('mpirun ~/lammps-23Jun2022/build/lmp -in in.cool > out_cool')

os.system('mpirun ~/lammps-23Jun2022/build/lmp -in in.deform')
