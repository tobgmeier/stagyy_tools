import matplotlib.pyplot as plt
import numpy as np
from stagpy import stagyydata
from stagpy import field
from pathlib import Path
import math
import sys
import argparse
from annulus_slice import *
import glob
import os
from scipy.interpolate import interp1d
from scipy import ndimage
import time
import pickle as pkl
import f90nml
import h5py

def angles_transform_deg(phi):
    phi_return = []
    phi = (180/np.pi)*np.array(phi)
    #phi = np.where(phi < 270, 270 - phi, phi - 270)
    for phi_i in phi:
        if phi_i < 270:
            phi_return.append(270-phi_i)
        elif phi_i >= 270:
            phi_return.append(270+360-phi_i)
        else:
            print('not good', phi_i)

    return list(phi_return)


def remove_outliers(n, phi, p_threshold=0.05):
    n = np.array(n)
    phi = np.array(phi)
    median = np.median(n)
    delete_indices=[]
    for i in range(0,len(n)):
        if n[i] < int(p_threshold*median+2):
            delete_indices.append(i)

    n = np.delete(n, delete_indices)
    phi = np.delete(phi, delete_indices)

    return n, phi


def get_nb_phi_histogram(T_field, p_mesh, n_bins = 128):
    phi_all = p_mesh[T_field > 0]
    x_mesh,y_mesh,_,_ = field.get_meshes_fld(step,'T')
    phi_hist, phi_hist_edges = np.histogram(phi_all, bins=n_bins, range=(0,2*np.pi))

    n_plumes = []
    phi_plumes = []
    count = 0 
    phi_count = 0
    for i, n_i in enumerate(phi_hist):
        if (n_i == 0 and count ==  0):
            continue
        if n_i != 0:
            count += n_i
            phi_count += n_i*(phi_hist_edges[i]+phi_hist_edges[i+1])/2
        if (n_i == 0 and count  != 0):
            n_plumes.append(count)
            phi_plumes.append(phi_count/count)
            count = 0
            phi_count = 0

    if count != 0:
        n_plumes.append(count) 
        phi_plumes.append(phi_count/count)   
    return n_plumes, phi_plumes

def get_nb_plumes(step,slice_name='total', n_sigma=1, n_thresh=10, count_cold = True,n_bins = 128,internal_heating = False):
    x_mesh,y_mesh,T_field_temp,_ = field.get_meshes_fld(step,'T')
    pcoord, rcoord = step.geom.p_centers, step.geom.r_centers
    p_mesh, r_mesh = np.meshgrid(step.geom.p_centers, step.geom.r_centers, indexing="ij")
    T_field_temp_temp = slices_dict[slice_name].field_mask(r_mesh=r_mesh, p_mesh=p_mesh,field_mesh=T_field_temp)
    selection_mask = np.logical_not(slices_dict[slice_name].area_selection(r_mesh, p_mesh))
    T_field = T_field_temp_temp.copy()
    T_field_c = T_field_temp_temp.copy()


    n_bins = n_bins
    nml = f90nml.read('par')
    r_cmb = nml['geometry']['r_cmb']
    
    r_bins = np.linspace(np.min(r_mesh),np.max(r_mesh), n_bins)

    mean_tot = []
    std_tot = []
    r_mid_tot = []


    for i in range(0,len(r_bins)-1):
        dr = 0 
        r_min = r_bins[i]
        r_max = r_bins[i+1]
        r_mid_tot.append((r_min+r_max)/2)

        ring_cond = np.logical_or(r_mesh < r_min-dr, r_mesh > r_max+dr)
        fld = np.ma.masked_where(ring_cond,T_field)

        mean_i = np.ma.mean(fld)

        std_i = np.ma.std(fld)
        mean_tot.append(mean_i)
        std_tot.append(std_i)


    
    if(n_bins > 2):
        l = len(std_tot)
        f = interp1d(std_tot[0:int(l/2)], r_mid_tot[0:int(l/2)],fill_value="extrapolate")
        f_c = interp1d(std_tot[int(l/2):int(l)], r_mid_tot[int(l/2):int(l)],fill_value="extrapolate")
        

        r_bl = f(min(np.ma.mean(std_tot)+0.5*np.ma.std(std_tot),np.ma.max(std_tot)))  
        r_bl_c = f_c(np.ma.mean(std_tot)-0.5*np.ma.std(std_tot))

    else:
        r_bl = r_cmb+0.12*r_cmb
    

    r_bl = r_cmb+0.15*r_cmb   #0.1 seems ok
    r_bl_bottom = r_cmb+0.06*r_cmb
    r_bl_top = r_cmb+0.15*r_cmb


        
    mean_std = np.ma.mean(std_tot)

    #p_h = 0.3   #Vary these parameters below to see which one is best, 0.1 from Leng et al. 2012, 0.4 seems to be working fine
    #p_h = 0.05
    if internal_heating == True:
        print("internal heating parameters")
        p_c = 0.05
        p_h = 0.1
        rand_amplitude = 150
    else: 
        p_c = 0.1
        p_h = 0.2
        rand_amplitude=350
    #rand_amplitude = 0
    no_mask_count = 0

    for i in range(0,len(r_bins)-1):
            r_min = r_bins[i]
            r_max = r_bins[i+1]

            
            ring_cond = np.logical_or(r_mesh < r_min, r_mesh > r_max)
            fld = np.ma.masked_where(ring_cond,T_field)
            fld_c = np.ma.masked_where(ring_cond,T_field_c)
            mean_i = np.ma.mean(fld)
            max_i = np.ma.max(fld)
            min_i = np.ma.min(fld)
            std_i = np.ma.std(fld)

            

            T_thresh = mean_i+p_h*(max_i-mean_i)+rand_amplitude

            T_thresh_c = mean_i+p_c*(min_i-mean_i)-rand_amplitude

            #Convert field to 1 (part of plume) and 0 (background)x
            T_field[np.logical_and(np.logical_and(fld < T_thresh, np.logical_not(ring_cond)),selection_mask)] = int(0) #put area selection here
            T_field[np.logical_and(np.logical_and(fld > T_thresh,np.logical_not(ring_cond)),selection_mask)] = int(1)
            T_field[r_mesh < r_bl_bottom] = int(0)
            T_field[r_mesh > np.amax(r_mesh)-r_bl_top+r_cmb] =int(0)

            #Downwellings detection
            if count_cold:
                T_field_c[np.logical_and(np.logical_and(fld_c < T_thresh_c, np.logical_not(ring_cond)),selection_mask)] = int(1) #put area selection here
                T_field_c[np.logical_and(np.logical_and(fld_c > T_thresh_c,np.logical_not(ring_cond)),selection_mask)] = int(0)
                T_field_c[r_mesh < r_cmb + 0.2*r_cmb] = int(0)
                T_field_c[r_mesh > np.amax(r_mesh)-0.02*r_cmb] = int(0)

            no_mask_count = no_mask_count + fld.count()



    T_field = np.ma.filled(T_field, fill_value=0) #The masked values have to be filled with 0s
    T_field_c = np.ma.filled(T_field_c, fill_value=0)


    ####check the annulus-border####
    T_field_b = T_field.copy()
    T_field_b = np.ma.filled(T_field_b, fill_value=0)
    T_field_b = slices_dict['annulus-border'].field_mask(r_mesh=r_mesh, p_mesh=p_mesh,field_mesh=T_field_b)
    T_field_b = np.ma.filled(T_field_b, fill_value=0)
    T_field_b= T_field_b.astype(int)

    phi_b = p_mesh[T_field_b > 0]
    split_bool = False
    
    if phi_b.size:
        phi_0 = phi_b[0]
        phi_b_hist, _ = np.histogram(phi_b, bins=n_bins, range=(0,2*np.pi))
        for phi_b_i in phi_b:
            if phi_b_i > phi_0 + 0.4:
                split_bool = True
                if np.count_nonzero(phi_b_hist)==1:
                    split_bool = False #In this case, the splitting is already corrected by the histogram binning
                continue
            else:
                phi_0 = phi_b_i

        if split_bool == True:
            print('SPLITTING (v2)')

    if count_cold:
        T_field_cb = T_field_c.copy()
        T_field_cb = np.ma.filled(T_field_cb, fill_value=0)
        T_field_cb = slices_dict['annulus-border'].field_mask(r_mesh=r_mesh, p_mesh=p_mesh,field_mesh=T_field_cb)
        T_field_cb = np.ma.filled(T_field_cb, fill_value=0)
        T_field_cb= T_field_cb.astype(int)
        phi_bc = p_mesh[T_field_cb > 0]
        split_bool_c = False
        if phi_bc.size:
            phi_0 = phi_bc[0]
            phi_bc_hist, _ = np.histogram(phi_bc, bins=n_bins, range=(0,2*np.pi))
            for phi_bc_i in phi_bc:
                if phi_bc_i > phi_0 + 0.4:
                    split_bool_c = True
                    if np.count_nonzero(phi_bc_hist)==1:
                        split_bool_c = False #In this case, the splitting is already corrected by the histogram binning
                    continue
                else:
                    phi_0 = phi_bc_i

            if split_bool_c == True:
                print('SPLITTING - Downwelling (v2)')

            



    #################################

    #Now calculate the number of plumes
    #labels,nb = measure.label(T_field, background=0,neighbors=4,return_num=True) #not needed if only doing histograms

    #PLUMES
    n_plumes, phi_plumes = get_nb_phi_histogram(T_field, p_mesh, n_bins = n_bins)
    if len(n_plumes)==1:
        split_bool = False
    if split_bool == True:
        phi_plumes[0] = (phi_plumes[0]*n_plumes[0]+(phi_plumes[-1]-2*np.pi)*n_plumes[-1])/(n_plumes[0]+n_plumes[-1]) #weighting the angles, needs to be done before the elements are removed from n_plumes
        n_plumes[0] = n_plumes[0]+n_plumes.pop(-1)
        phi_plumes.pop(-1)
        if phi_plumes[0] < 0:
            phi_plumes[0] = phi_plumes[0]+2*np.pi



    nb_corr = len(n_plumes)

    if count_cold:
        n_plumes_c, phi_plumes_c = get_nb_phi_histogram(T_field_c, p_mesh, n_bins = n_bins)
        if len(n_plumes_c)==1:
            split_bool_c = False
        if split_bool_c == True:
            print('n_plumes',n_plumes_c)
            print('phi_plumes', phi_plumes_c)
            phi_plumes_c[0] = (phi_plumes_c[0]*n_plumes_c[0]+(phi_plumes_c[-1]-2*np.pi)*n_plumes_c[-1])/(n_plumes_c[0]+n_plumes_c[-1]) #weighting the angles, needs to be done before the elements are removed from n_plumes
            n_plumes_c[0] = n_plumes_c[0]+n_plumes_c.pop(-1)
            phi_plumes_c.pop(-1)
            if phi_plumes_c[0] < 0:
                phi_plumes_c[0] = phi_plumes_c[0]+2*np.pi

        nb_corr_c = len(n_plumes_c)


        return n_plumes, phi_plumes, n_plumes_c, phi_plumes_c

    return n_plumes, phi_plumes


rep = '.'
start = time.time()
sdat = stagyydata.StagyyData(rep)
n_snapshots = sdat.snaps[-1].isnap

print('Starting plume and downwelling detection program')

nwrite = int(sdat.par['timein']['nwrite'])

last_step_id = sdat.snaps[-1].istep

parser = argparse.ArgumentParser()
parser.add_argument("--smin", help="lowest snapshot number (starting at 0)",type=int,default=0)
parser.add_argument("--smax", help="highest snapshot number",type=int,default=n_snapshots)

args = parser.parse_args()

plumes_d = [] 
plumes_n = [] 
time_all = []


ds = 5 #every nth ds, the plumes/downwelling are counted
nml = f90nml.read('par')
rh = nml['refstate']['rh']
yld = nml['viscosity']['stressy_eta']

if rh > 1e-13 and yld > 99.0e6: #This is for internal heating with high yield stress where some parameters need to be adjusted to get a good detection
    internal_heating = True
else:
    internal_heating = False


try: #This will check whether there's already a zebra_data file. If yes, it will continue the counting from the last snapshot
    with open("zebra_data.p", "rb") as f:
        time_phi_all_plume, time_phi_all_downwelling = pkl.load(f)
    smin = len(time_phi_all_plume)*ds
    smax = args.smax
    print('STARTING FROM :', smin)
except (OSError, IOError) as e:
    time_phi_all_plume, time_phi_all_downwelling = [],[]
    smin = args.smin
    smax = args.smax
    with open("zebra_data.p", "wb") as f:
        pkl.dump((time_phi_all_plume, time_phi_all_downwelling),f)


smin = args.smin
smax = args.smax
n_bins = int(nml['geometry']['nytot']/2)


for j,i in enumerate(range(smin,smax+1,ds)):


    time_phi_i_plume = []
    time_phi_i_downwelling = []
    print('i : ', i )
    step = sdat.snaps[i]

    time_step = step.timeinfo['t']/(60*60*24*365.25)/1e9

    print('time step', time_step)
    time_all.append(time_step)
    time_phi_i_plume.append(time_step)
    time_phi_i_downwelling.append(time_step)


    n_plumes, phi_plumes, n_plumes_c, phi_plumes_c = get_nb_plumes(step,n_sigma=1,slice_name='total', internal_heating =  internal_heating, n_bins = n_bins)
    n_h = len(n_plumes)
    n_c = len(n_plumes_c)
    n_plumes, phi_plumes = remove_outliers(n_plumes, phi_plumes)
    n_plumes_c, phi_plumes_c = remove_outliers(n_plumes_c, phi_plumes_c)
    print('final detection of plumes ', n_plumes, phi_plumes)
    print('final detection of downwellings ', n_plumes_c, phi_plumes_c)
    print('number of detected plumes and downwellings: ', len(n_plumes), len(n_plumes_c))
    print('number of phi s ', len(phi_plumes), len(phi_plumes_c))
    phi_plumes = angles_transform_deg(phi_plumes)
    time_phi_i_plume.append(phi_plumes)
    phi_plumes_c = angles_transform_deg(phi_plumes_c)
    time_phi_i_downwelling.append(phi_plumes_c)
    time_phi_all_plume.append(time_phi_i_plume)
    time_phi_all_downwelling.append(time_phi_i_downwelling)
    with open("zebra_data.p", "wb") as f:
        pkl.dump((time_phi_all_plume,time_phi_all_downwelling),f)

stop = time.time()
print('time of execution ', stop - start)






