import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools_lammps as tool_lmp
import sys 
sys.path.append('../../')
import SP_thermoset.SP_tools as sp
from tqdm import tqdm

def collect_data0(dir_list,image_number1=2,image_numbe2=8):
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

# dir_list = ['./length/old/M20_v{}'.format(i) for i in range(10)]
# SPL_M20, hs_M20, min_M20, size_M20 = collect_data0(dir_list)

dir_list = ['../../simulations/CG/M40_v{}'.format(i) for i in range(10)]
SPL_M40, hs_M40, min_M40, size_M40 = collect_data0(dir_list)

dir_list = ['../../simulations/CG/M40_ck75_v{}'.format(i) for i in range(10)]
SPL_M40_ck75, hs_M40_ck75, min_M40_ck75, size_M40_ck75 = collect_data0(dir_list)

def plot_min_SPL(ax,min_M20,size_M20,hs_M20,label1,**kwargs):
    ax.plot(min_M20/size_M20,np.exp(hs_M20)-1,'o',label='{0:s}'.format(label1),**kwargs) # , rmse={1:.2f}, r2={2:.2f}

plt.figure(dpi=300, figsize=np.array([2.3, 1.8]))
ax = plt.gca()
xrange = np.arange(1.2,4,0.1)
plt.plot(xrange,1.2*1.25*xrange-1,'k-',lw=1.5,label=r'Upper limit')
xx = np.concatenate([min_M40/size_M40,
                     min_M40_ck75/size_M40_ck75])
yy = np.concatenate([np.exp(hs_M40[:,1])-1,
                     np.exp(hs_M40_ck75[:,1])-1])
p1 = np.linalg.lstsq(xx.reshape(-1,1),yy.reshape(-1,1)+1,rcond=None)[0]
plt.plot(xrange,(p1*xrange-1).squeeze(),'--',color='violet',lw=1.0,label=r'Fitting, $\alpha \sim 0.86$')

plot_min_SPL(ax,min_M40, size_M40, hs_M40[:,1],'CG1',marker='o',ms=3,color='C1') # 200 beads
plot_min_SPL(ax,min_M40_ck75,size_M40_ck75,hs_M40_ck75[:,1],'CG3',marker='^',ms=3,color='C3') # less crosslinking

plt.legend(fontsize=5,ncol=1)
plt.tight_layout()
plt.xlabel(r'$D^g_\mathrm{min}/L_0$')
plt.ylabel(r'$\epsilon_n$')
plt.tight_layout() 
plt.savefig('min_strain0.png')
plt.show()

# plt.figure(dpi=300, figsize=np.array([3.3, 2.4])/1.25)
# # plt.plot(value_M20 ,  np.exp(hs_M20[:,1])-1,'*',label='20')
# # plt.plot(value_M40 ,  np.exp(hs_M40[:,1])-1,'*',label='40')
# # # plt.plot(value_M40_sst ,  np.exp(hs_M40_sst[:,1])-1,'*',label='40_sst')
# # # plt.plot(value_M80 ,  np.exp(hs_M80[:,1])-1,'*',label='80')
# plt.plot(value_M100,np.exp(hs_M100[:,1])-1,'*',label='100')
# plt.plot(value_M200,np.exp(hs_M200[:,1])-1,'*',label='200')
# plt.plot(value_M400,np.exp(hs_M400[:,1])-1,'*',color='y',label='400')
# plt.plot(value_M1000,np.exp(hs_M1000[:,1])-1,'*',color='lime',label='1000')
# # plt.plot([3,5],[1,3],'--')
# plt.legend(fontsize=4,ncol=2)
# plt.xlabel(r'$D^g_\mathrm{eff}$')
# plt.ylabel(r'$\epsilon_0$')
# plt.tight_layout()
# # plt.savefig('fig_eff_epsilon0.png')
# plt.show()

# xx = np.concatenate((value_M200,value_M400,value_M1000))
# # xx = np.concatenate((value_M400,value_M1000))
# yy= np.concatenate([
#                     # np.exp(hs_M40[:,1])-1,
#                     # np.exp(hs_M100[:,1])-1,
#                     np.exp(hs_M200[:,1])-1,
#                     np.exp(hs_M400[:,1])-1,
#                     np.exp(hs_M1000[:,1])-1
#                     ])
# print(np.polyfit(xx,yy,1))

# plt.figure(dpi=300, figsize=np.array([3.3, 2.4])/1.25)
# plt.plot(value_M20,  np.exp(hs_M20[:,0])-1,'*',label='20')
# plt.plot(value_M40,  np.exp(hs_M40[:,0])-1,'*',label='40')
# plt.plot(value_M100,np.exp(hs_M100[:,0])-1,'*',label='100')
# plt.plot(value_M200,np.exp(hs_M200[:,0])-1,'*',label='200')
# plt.plot(value_M400,np.exp(hs_M400[:,0])-1,'*',color='y',label='400')
# plt.plot(value_M1000,np.exp(hs_M1000[:,0])-1,'*',color='lime',label='1000')
# plt.legend(fontsize=4,ncol=2)
# plt.xlabel(r'$D^g_\mathrm{eff}$')
# plt.ylabel(r'$\epsilon_h$')
# plt.tight_layout()
# # plt.savefig('fig_eff_epsilon0.png')
# plt.show()

# plt.figure(dpi=300, figsize=np.array([3.3, 2.4])/1.25)
# plt.plot(mean_M20/size_M20,  np.exp(hs_M20[:,0])-1,'*',label='20')
# plt.plot(mean_M40/size_M40,  np.exp(hs_M40[:,0])-1,'*',label='40')
# plt.plot(mean_M100/size_M100,np.exp(hs_M100[:,0])-1,'*',label='100')
# plt.plot(mean_M200/size_M200,np.exp(hs_M200[:,0])-1,'*',label='200')
# plt.plot(mean_M400/size_M400,np.exp(hs_M400[:,0])-1,'*',color='y',label='400')
# plt.plot(mean_M1000/size_M1000,np.exp(hs_M1000[:,0])-1,'*',color='lime',label='1000')
# plt.legend(fontsize=4,ncol=2)
# plt.xlabel(r'$D^g_\mathrm{eff}$')
# plt.ylabel(r'$\epsilon_0$')
# plt.tight_layout()
# # plt.savefig('fig_eff_epsilon0.png')
# plt.show()

# r2_list = []
# for alpha in np.arange(0.01,2,0.01):
#     # value_M20 = compute_effective_length(SPL_M20,size_M20,alpha)
#     value_M40 = compute_effective_length(SPL_M40,size_M40,alpha)
#     value_M60 = compute_effective_length(SPL_M60,size_M60,alpha)
#     value_M80 = compute_effective_length(SPL_M80,size_M80,alpha)
#     value_M100 = compute_effective_length(SPL_M100,size_M100,alpha)
#     value_M200 = compute_effective_length(SPL_M200,size_M200,alpha)
#     value_M400 = compute_effective_length(SPL_M400,size_M400,alpha)
#     value_M1000 = compute_effective_length(SPL_M1000,size_M1000,alpha)
#     # xx = np.concatenate((value_M40,value_M60,value_M80,value_M100,value_M200,value_M400,value_M1000))
#     # yy = np.concatenate((np.exp(hs_M40[:,1]),np.exp(hs_M60[:,1]),np.exp(hs_M80[:,1]),
#     #                      np.exp(hs_M100[:,1]),np.exp(hs_M200[:,1]),np.exp(hs_M400[:,1]),np.exp(hs_M1000[:,1])))
#     xx = np.concatenate((value_M100,value_M200,value_M400,value_M1000))
#     yy = np.concatenate((np.exp(hs_M100[:,1]),np.exp(hs_M200[:,1]),np.exp(hs_M400[:,1]),np.exp(hs_M1000[:,1])))
#     r2, rmse = evaluate_linear_fit_np(xx,yy)
#     r2_list.append(r2)
# plt.figure(dpi=300, figsize=np.array([3.3, 2.4])/1.25)
# plt.plot(np.arange(0.01,2,0.01),r2_list,'-o',ms=1.5)
# plt.show()

# fig, ax = plt.subplots(1,2,dpi=300, figsize=np.array([4, 3]))
# ax[0].errorbar([np.cbrt(20),np.cbrt(40),np.cbrt(100),np.cbrt(200),np.cbrt(400),np.cbrt(1000)],
#          [
#           np.mean(min_M20/size_M20),
#           np.mean(min_M40/size_M40),
#           np.mean(min_M100/size_M100),
#           np.mean(min_M200/size_M200),
#           np.mean(min_M400/size_M400),
#           np.mean(min_M1000/size_M1000),]
#           ,yerr= [
#                 np.std(min_M20/size_M20),
#                 np.std(min_M40/size_M40),
#                 np.std(min_M100/size_M100),
#                 np.std(min_M200/size_M200),
#                 np.std(min_M400/size_M400),
#                 np.std(min_M1000/size_M1000),],marker='o',label='Min')

# ax[0].plot([np.cbrt(20),np.cbrt(40),np.cbrt(100),np.cbrt(200),np.cbrt(400),np.cbrt(1000)],
#          [
#           np.mean(mean_M20/size_M20),
#           np.mean(mean_M40/size_M40),
#           np.mean(mean_M100/size_M100),
#           np.mean(mean_M200/size_M200),
#           np.mean(mean_M400/size_M400),
#           np.mean(mean_M400/size_M1000),],'-s',label='Mean')

# ax[0].plot([np.cbrt(20),np.cbrt(40),np.cbrt(100),np.cbrt(200),np.cbrt(400),np.cbrt(1000)],
#          [
#           np.mean(value_M20),
#           np.mean(value_M40),
#           np.mean(value_M100),
#           np.mean(value_M200),
#           np.mean(value_M400),
#           np.mean(value_M1000),],'-*',label='Eff')

# ax[0].legend()

# ax[1].plot([np.cbrt(20),np.cbrt(40),np.cbrt(100),np.cbrt(200),np.cbrt(400),np.cbrt(1000)],
#          [np.mean(hs_M20[:,1]),
#           np.mean(hs_M40[:,1]),
#           np.mean(hs_M100[:,1]),
#           np.mean(hs_M200[:,1]),
#           np.mean(hs_M400[:,1]),
#           np.mean(hs_M1000[:,1])],'-s',label=r'$\epsilon_0$')

# ax[1].plot([np.cbrt(20),np.cbrt(40),np.cbrt(100),np.cbrt(200),np.cbrt(400),np.cbrt(1000)],
#          [np.mean(hs_M20[:,0]),
#           np.mean(hs_M40[:,0]),
#           np.mean(hs_M100[:,0]),
#           np.mean(hs_M200[:,0]),
#           np.mean(hs_M400[:,0]),
#           np.mean(hs_M1000[:,0])],'-o',label=r'$\epsilon_h$')
# ax[0].set_xlabel('size')
# ax[1].set_xlabel('size')
# ax[0].set_ylabel(r'$G^g$')
# ax[1].set_ylabel(r'$\epsilon$')
# ax[1].legend()
# plt.tight_layout()
# plt.show()

# plt.figure(dpi=300, figsize=np.array([3.3, 2.4])/1.25)
# xx = [np.mean(value_M20),
#       np.mean(value_M40),
#       np.mean(value_M60),
#     #   np.mean(value_M80),
#       np.mean(value_M100),
#       np.mean(value_M200),
#       np.mean(value_M400),
#       np.mean(value_M1000),]
# yy =  [ np.mean(np.exp(hs_M20[:,1])),
#         np.mean(np.exp(hs_M40[:,1])),
#         np.mean(np.exp(hs_M60[:,1])),
#         # np.mean(np.exp(hs_M80[:,1])),
#         np.mean(np.exp(hs_M100[:,1])),
#         np.mean(np.exp(hs_M200[:,1])),
#         np.mean(np.exp(hs_M400[:,1])),
#         np.mean(np.exp(hs_M1000[:,1])),]
# yyerr = [np.std(np.exp(hs_M20[:,1])),
#          np.std(np.exp(hs_M40[:,1])),
#          np.std(np.exp(hs_M60[:,1])),
#         #  np.std(np.exp(hs_M80[:,1])),
#          np.std(np.exp(hs_M100[:,1])),
#          np.std(np.exp(hs_M200[:,1])),
#          np.std(np.exp(hs_M400[:,1])),
#          np.std(np.exp(hs_M1000[:,1])),]
# xxerr = [np.std(value_M20),
#          np.std(value_M40),
#          np.std(value_M60),
#         #  np.std(value_M80),
#          np.std(value_M100),
#          np.std(value_M200),
#          np.std(value_M400),
#          np.std(value_M1000),]
# # plt.errorbar(xx-1,yy,yerr=yyerr,xerr=xxerr,capsize=1.5,marker='o')
# plt.plot(np.array(xx)/2-1,np.array(yy)-1,'o')
# # r2,rmse = evaluate_linear_fit_np(xx,yy)

# # p2 = np.polyfit(yy,xx,2)
# # yrange= np.arange(1,3.5,0.1)
# # plt.plot(np.polyval(p2,yrange),yrange,'r--')
# plt.xlabel(r'$D^g_\mathrm{eff}$')
# plt.ylabel(r'$\epsilon_0$')
# plt.title('R2={0:.3f} RMSE={1:.3f}'.format(r2,rmse))
# plt.tight_layout()
# plt.show()

# from scipy.optimize import curve_fit
# def custom_func(x, beta, D0):
#     return D0+np.exp(-beta*x**2)

# range_size = [np.cbrt(40),np.cbrt(100),np.cbrt(200),np.cbrt(400),np.cbrt(1000)]
# D_size = [
#         #   np.mean(value_M20),
#           np.mean(value_M40),
#           np.mean(value_M100),
#           np.mean(value_M200),
#           np.mean(value_M400),
#           np.mean(value_M1000),]
# params, covariance = curve_fit(custom_func, range_size, D_size, p0=[0.1,2])

# plt.figure(dpi=200)
# plt.plot(range_size,D_size,'-*',label='Eff')
# xrange = np.arange(3,12,1)
# plt.plot(xrange,custom_func(xrange,*params),'--',label='Eff')
# # plt.yscale('log')
# plt.show()
