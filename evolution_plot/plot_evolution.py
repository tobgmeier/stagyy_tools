import matplotlib.pyplot as plt
import numpy as np
from stagpy import stagyydata
from stagpy import rprof
from stagpy import field
from pathlib import Path
import math
import sys
import argparse
from annulus_slice import *
import glob
import os
import string
import matplotlib.gridspec as gridspec
import pickle as pkl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from stagpy import stagyyparsers as sp
import pandas as pd
import matplotlib as mpl

#Parameters for plotting
file_ending = ".png"
background_color = "white"
text_color = "black"

#Parameters for latex like plots
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',size=11)

def plot_zebra(pickle_file, ax = None, text_size = 10, time_max = np.inf, current_time = True, markersize = '3', vline_file = "no_vfile", darkmode = False,transit_time = 0, y_label=True,box_label=None, ann_daynight = True, t_shift = 0.0,no_legend=True):
    ds = 1
    with open(pickle_file, "rb") as f:
        time_phi_all_plume, time_phi_all_downwelling = pkl.load(f)
        a = np.array(time_phi_all_plume, dtype=object)
    time_phi_all_plume, time_phi_all_downwelling = time_phi_all_plume[::ds], time_phi_all_downwelling[::ds]
    if ax is None:
        ax = plt.gca()

    time_all = []
    first_step = True
    maximum_time = np.max(np.array(time_phi_all_plume,dtype=object)[:,0])
    for i in range(0,len(time_phi_all_plume)):
        time_phi_i_plume = time_phi_all_plume[i]
        time_phi_i_downwelling = time_phi_all_downwelling[i]

        time_step_p = time_phi_i_plume.pop(0) #This removes the first element of the list, which is the timestep
        phi_plumes = time_phi_i_plume[0] # because the remaining array is [[1,2,3,4]]
        
        time_step_d = time_phi_i_downwelling.pop(0)
        phi_downwellings = time_phi_i_downwelling[0]
        if current_time == True:
            maximum_time = time_step_p

        if (time_step_p != time_step_d): #check
            raise NameError('time steps not the same, cannot continue')

        else:
            time_step = time_step_p

            if time_step >= time_max:
                break
            time_all.append(time_step)


        n_plumes = len(phi_plumes)
        n_downwellings = len(phi_downwellings)

        if transit_time > 0:
            time_step = time_step / transit_time
            if first_step == True:
                t_shift = t_shift / transit_time
                first_step = False


        dayside_color = "red"
        nightside_color = "blue"
        if darkmode == True:            
            dayside_color = (0/255,199/255,255/255)
            nightside_color = (255/255,56/255,0/255)
        ax.plot(phi_plumes,[time_step-t_shift]*n_plumes,'.',color=dayside_color,markersize=markersize, label="Upwelling" if i == 0 else "")
        ax.plot(phi_downwellings,[time_step-t_shift]*n_downwellings,'.',color=nightside_color,markersize=markersize,label="Downwelling" if i == 0 else "")

    

    ax.set_xlabel('Longitude ($\degree$)', fontsize = text_size)

    if y_label == True:
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position('left') 
        ax.set_ylabel('Time (Gyrs)', fontsize = text_size)
    if y_label == False:
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position('left')
        ax.set_yticks(color='w') #white color to make them invisible
    plt.subplots_adjust(bottom=0.2)
    ax.fill_between(np.array([0.0,0.5]),0,1,facecolor='white',transform=ax.transAxes)
    ax.fill_between(np.array([0.5,1.0]),0,1,facecolor='lightgrey',transform=ax.transAxes)
    if darkmode == True:
        ax.fill_between(np.array([0.0,0.5]),0,1,facecolor='black',transform=ax.transAxes)
    dt = (max(time_all)-min(time_all))/100
    if no_legend == True:
         print('NO LEGEND')
         lgd = None
    else: 
         lgd = ax.legend(loc="center",bbox_to_anchor=(0.5,-0.07),ncol=2,prop={'size': text_size},markerscale=5.0)
    ax.tick_params(labelsize = text_size)
    ax.set_xlim([0,360])
    ax.set_xticks([0,90,180,270,360])
    ax.set_xticklabels([-90,0,90,180,270])

    if transit_time > 0:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right') 
        ax.set_ylabel('Number of overturns', fontsize = text_size)


    if transit_time > 0:
        ax.set_ylim(max(min(list((np.array(time_all))/transit_time-t_shift)),0),(maximum_time/transit_time-t_shift)) 
    else:
        ax.set_ylim(max(min(np.array(time_all)-t_shift),0),maximum_time-t_shift)

    if ann_daynight:
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("top", size="4%", pad=+0.0)
        cax1.axvline(x=0.5,ymin=0,ymax=1.0,linestyle='dashed',color='black')
        cax1.text(0.48, 0.25, 'Day', horizontalalignment='right', verticalalignment='center', size = text_size,transform=cax1.transAxes)
        cax1.text(0.52, 0.25, 'Night', horizontalalignment='left', verticalalignment='center', size = text_size, transform=cax1.transAxes)
        cax1.axis('off')

    if box_label != None:
        bbox_props = dict(boxstyle="square", fc="white", ec="white", alpha=0.9, pad = 0.1)
        ax.text(0.1,0.97, box_label, ha="center", va="center", size=18,fontweight='bold',
            bbox=bbox_props, transform=ax.transAxes)


    return (lgd,)

fig, axis = plt.subplots(1,1,figsize=(4,12))
extra_artists = plot_zebra("zebra_data.p", ax = axis, markersize = '2', text_size = 13,darkmode=False, no_legend = False)
plt.tight_layout(h_pad=1.2,w_pad=1.2)
plt.savefig("evolution"+file_ending,dpi=300,bbox_inches='tight',pad_inches=0.17,facecolor="white", edgecolor = background_color)
plt.close()
plt.clf()
