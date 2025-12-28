#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm

def plot_wall(wall,ax):
    phii = np.linspace(0,2*np.pi,32)

    PHI, ZZ = np.meshgrid(phii, wall['Z'].values)
    RR = np.array([wall['R'].values for _ in phii]).T 
    XX = RR * np.cos(PHI)
    YY = RR * np.sin(PHI)
    
    ax.plot_wireframe(XX,YY,ZZ,color='gray', alpha=0.2, linewidth=0.1)
    return
    
def plot_data(name,iplane,df,fig,is_plot_2d = 0,log = 0, with_neg=0.,lim=[]):
    
    len = df['R'].shape[0]
    row = int(iplane)
    col = int(len//iplane)
    print(row,col)

    Rdf1 = df['R'].values
    Rdf=np.reshape(Rdf1,(row,col),order='C')
    Zdf1 = df['Z'].values
    Zdf = np.reshape(Zdf1,(row,col),order='C')
    PHIdf1 = df['phi'].values
    PHIdf = np.reshape(PHIdf1,(row,col),order='C')
    thetadf1 = df['theta'].values
    thetadf = np.reshape(thetadf1,(row,col),order='C')
    namedf1 = df[name].values
    namedf = np.reshape(namedf1,(row,col),order='C')
    if lim != []:
        namedf[namedf<lim[0]] = lim[0]
        namedf[namedf>lim[1]] = lim[1]

    if is_plot_2d == 0:
        ax = fig.add_subplot(111, projection='3d')
        Xdf = Rdf * np.cos(PHIdf)
        Ydf = Rdf * np.sin(PHIdf)

        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(namedf.min(), namedf.max())
        norm = LogNorm(vmin=namedf.min(), vmax=namedf.max())
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(namedf) 
        
        sc = ax.plot_surface(
            Xdf,Ydf,Zdf,
            facecolors=cm.viridis(norm(namedf)),
            cmap=cmap,
            alpha=0.8,
            lw=0.001,
        )
        
        cbar = fig.colorbar(sm,ax=ax)
        cbar.set_label('Value Intensity', rotation=270, labelpad=15)
        ax.set_xlabel('X Axis', fontsize=10)
        ax.set_ylabel('Y Axis', fontsize=10)
        ax.set_zlabel('Z Axis', fontsize=10)
        ax.set_title('3D Value Distribution in Cylindrical Coordinates', pad=20)



        ax.view_init(elev=30, azim=45)

    elif is_plot_2d == 1:
        ax = fig.add_subplot(111)
        data = namedf
        
        pos_mask = data > 0
        neg_mask = data < 0
        
        pos_data = np.where(pos_mask, data, 1.e-1)  # 正值数据
        neg_data = -np.where(neg_mask, data, np.nan)  # 非正值数据
        
        
        if log == 0:
            pos_norm = plt.Normalize(0,vmax=np.nanmax(pos_data))
            neg_norm = plt.Normalize(0,vmax=np.nanmax(neg_data))
        elif log ==1:
            pos_norm = LogNorm(vmin=np.nanmin(pos_data), vmax=np.nanmax(pos_data))
            if with_neg == 1:
                neg_norm = LogNorm(vmin=np.nanmin(neg_data), vmax=np.nanmax(neg_data))
        if lim != []:
            pos_norm = LogNorm(vmin=max(lim[0],0), vmax=max(lim[1],0))
            neg_norm = LogNorm(vmin=min(lim[0],0), vmax=min(lim[1],0))
            
        
        pos_cmap = cm.get_cmap('GnBu')
        neg_cmap = cm.get_cmap('OrRd')
        if with_neg ==0:
            pos_cmap = cm.get_cmap('viridis')
        #cmap = cm.get_cmap('Greys')
        phi = PHIdf[:,0]
        theta = thetadf[0,:]
        #theta_grid,phi_grid  = np.meshgrid(theta,phi)
        extent=(0,1080,0,1080)
        '''heatmap = ax.pcolormesh(
            phi_grid,theta_grid,pos_data,
            shading='auto',  # 自动处理数据边缘
            cmap=pos_cmap,
            norm=pos_norm
        )'''
        
        heatmap_ims = ax.imshow(
            np.rot90(pos_data),
            extent=extent,
            cmap = pos_cmap,
            norm = pos_norm,
            interpolation='bicubic'
        )
        '''neg_heatmap = ax.pcolormesh(
            phi_grid,theta_grid,neg_data,
            shading='auto',  # 自动处理数据边缘
            cmap=neg_cmap,
            norm=neg_norm
        )'''
        if with_neg ==1:
            neg_heatmap_ims = ax.imshow(
                np.rot90(neg_data),
                extent=extent,
                cmap=neg_cmap,
                norm=neg_norm
            )
            neg_cbar = fig.colorbar(neg_heatmap_ims, ax=ax, pad=0.02)
            neg_cbar.set_label('neg_'+name, rotation=270, labelpad=15)
    # 添加颜色条
        cbar = fig.colorbar(heatmap_ims, ax=ax, pad=0.02)
        #cbar.set_label(name, rotation=270, labelpad=15)
        
        
        # 坐标轴设置
        ax.set_xlabel('phi', fontsize=16)
        ax.set_ylabel('theta', fontsize=16)
        #ax.set_title(name + ' heatmap', fontsize=14)
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{x*2/1080-1:.1f}π"
        ))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda y, _: f"{y*2/1080:.1f}π"
        ))

if __name__ == '__main__':
    wall = pd.read_csv('/home/ac_desktop/XL50-U/XL50-U_1.4.5/wallcontour_updated.dat', sep='\s+', names=['R','Z'])
    iplane = 1080
    name = 'heatF_tot_cd'
    name1 = 'heatF_par_cd'
    name2,name3 = 'heatF_prp_cd','heatF_tot_cv'
    df = pd.read_csv('/home/ac_desktop/syncfiles/postproc_145/boundary_quantities_s04650.dat', skiprows=2,sep='\s+', names=['R','Z','phi','theta',name,name1,name2,name3])

    fig = plt.figure(figsize=(8, 6),dpi=150)
    plt.rcParams['xtick.labelsize'] = 14 # X轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 14
    

    plot_data(name,iplane,df,fig,is_plot_2d=0,log=1,with_neg=0,lim=[1e5,3e8])
    #plot_wall(wall,ax)

    plt.tight_layout()
    plt.show()