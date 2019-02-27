#!/usr/bin/env python

#%%
import sys
import numpy as np

#sys.path.append('/Users/ifenty/git_repo_mine/simplegrid/')
sys.path.append('/home/owang/CODE/projects/modules/simplegrid-master/')
import simplegrid as sg


def extract_G_grid_fields_from_mitgrid_files_as_tiles(llc, grid_dir):

    tile1_name = 'tile001.mitgrid'
    tile2_name = 'tile002.mitgrid'
    tile3_name = 'tile003.mitgrid'
    tile4_name = 'tile004.mitgrid'
    tile5_name = 'tile005.mitgrid'
    
    XG_t = np.ones((13,llc+1,llc+1))*np.nan
    YG_t = np.ones((13,llc+1,llc+1))*np.nan
    RAZ_t = np.ones((13,llc+1,llc+1))*np.nan    
    
    i=np.arange(llc*3 + 1)
    i1=i[llc*0  :llc+llc*0+1]
    i2=i[llc*1:llc+llc*1+1]
    i3=i[llc*2:llc+llc*2+1]
    
    print i1.shape, i2.shape, i3.shape
    print i1[0:3], i1[-3:]
    print i2[0:3], i2[-3:]
    print i3[0:3], i3[-3:]
    
    #%% TILE 1
    mg1 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile1_name,
                                     llc,llc*3, verbose=True)
    
    XG_t[0,:,:] = mg1['XG'][np.ix_(np.arange(91),i1)].T
    XG_t[1,:,:] = mg1['XG'][np.ix_(np.arange(91),i2)].T
    XG_t[2,:,:] = mg1['XG'][np.ix_(np.arange(91),i3)].T
    
    YG_t[0,:,:] = mg1['YG'][np.ix_(np.arange(91),i1)].T
    YG_t[1,:,:] = mg1['YG'][np.ix_(np.arange(91),i2)].T
    YG_t[2,:,:] = mg1['YG'][np.ix_(np.arange(91),i3)].T
    
    RAZ_t[0,:,:] = mg1['RAZ'][np.ix_(np.arange(91),i1)].T
    RAZ_t[1,:,:] = mg1['RAZ'][np.ix_(np.arange(91),i2)].T
    RAZ_t[2,:,:] = mg1['RAZ'][np.ix_(np.arange(91),i3)].T
    
    mg1=[]
    
    #%% TILE 2
    mg2 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile2_name,
                                     llc,3*llc)
    
    XG_t[3,:,:] = mg2['XG'][np.ix_(np.arange(91),i1)].T
    XG_t[4,:,:] = mg2['XG'][np.ix_(np.arange(91),i2)].T
    XG_t[5,:,:] = mg2['XG'][np.ix_(np.arange(91),i3)].T
    
    YG_t[3,:,:] = mg2['YG'][np.ix_(np.arange(91),i1)].T
    YG_t[4,:,:] = mg2['YG'][np.ix_(np.arange(91),i2)].T
    YG_t[5,:,:] = mg2['YG'][np.ix_(np.arange(91),i3)].T

    RAZ_t[3,:,:] = mg2['RAZ'][np.ix_(np.arange(91),i1)].T
    RAZ_t[4,:,:] = mg2['RAZ'][np.ix_(np.arange(91),i2)].T
    RAZ_t[5,:,:] = mg2['RAZ'][np.ix_(np.arange(91),i3)].T
    
    mg2=[]
    
    #%% TILE 3
    mg3 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile3_name,
                                     llc,llc)
    XG_t[6,:,:]  = mg3['XG'].T
    YG_t[6,:,:]  = mg3['YG'].T
    RAZ_t[6,:,:] = mg3['RAZ'].T
    mg3=[]
    
    #%% TILE 4
    mg4 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile4_name,
                                     llc*3,llc)
    XG_t[7,:,:] = mg4['XG'][np.ix_(i1,np.arange(91))].T
    XG_t[8,:,:] = mg4['XG'][np.ix_(i2,np.arange(91))].T
    XG_t[9,:,:] = mg4['XG'][np.ix_(i3,np.arange(91))].T
    
    YG_t[7,:,:] = mg4['YG'][np.ix_(i1,np.arange(91))].T
    YG_t[8,:,:] = mg4['YG'][np.ix_(i2,np.arange(91))].T
    YG_t[9,:,:] = mg4['YG'][np.ix_(i3,np.arange(91))].T

    RAZ_t[7,:,:] = mg4['RAZ'][np.ix_(i1,np.arange(91))].T
    RAZ_t[8,:,:] = mg4['RAZ'][np.ix_(i2,np.arange(91))].T
    RAZ_t[9,:,:] = mg4['RAZ'][np.ix_(i3,np.arange(91))].T
    
    mg4 = []
    
    #%% TILE 5
    mg5 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile5_name,
                                     llc*3,llc)
    XG_t[10,:,:] = mg5['XG'][np.ix_(i1,np.arange(91))].T
    XG_t[11,:,:] = mg5['XG'][np.ix_(i2,np.arange(91))].T
    XG_t[12,:,:] = mg5['XG'][np.ix_(i3,np.arange(91))].T
    
    YG_t[10,:,:] = mg5['YG'][np.ix_(i1,np.arange(91))].T
    YG_t[11,:,:] = mg5['YG'][np.ix_(i2,np.arange(91))].T
    YG_t[12,:,:] = mg5['YG'][np.ix_(i3,np.arange(91))].T
    
    RAZ_t[10,:,:] = mg5['RAZ'][np.ix_(i1,np.arange(91))].T
    RAZ_t[11,:,:] = mg5['RAZ'][np.ix_(i2,np.arange(91))].T
    RAZ_t[12,:,:] = mg5['RAZ'][np.ix_(i3,np.arange(91))].T
    

    mg5 = []

    return XG_t, YG_t, RAZ_t

#%%
if __name__ == '__main__':

    llc = 90
    grid_dir = '/Volumes/ECCO_BASE/ECCO_v4r3/grid_llc90'

    XG, YG, RAZ = extract_G_grid_fields_from_mitgrid_files_as_tiles(llc, grid_dir)
    
    #%%
    ds = xr.Dataset({'XG': (['tile','j_g','i_g'], XG),
                     'YG': (['tile','j_g','i_g'], YG),
                     'RAZ': (['tile','j_g','i_g'], RAZ)}, 
                    coords={'tile': range(1,14),
                            'j_g': (('j_g',), range(1,92), {'axis':'Y',}),
                            'i_g': (('i_g',), range(1,92), {'axis':'X',})})
    

    ds.to_netcdf(grid_dir + '/' + 'G_grid_fields_as_tiles.nc')

    
    #%%
    sys.path.append('/Users/ifenty/git_repo_mine/ECCOv4-py/')
    import ecco_v4_py as ecco
    ecco.plot_tiles(ds.YG)
