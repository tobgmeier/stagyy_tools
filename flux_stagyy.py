import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from stagpy import stagyydata
from stagpy import field
from pathlib import Path
import argparse
import os
import pickle as pkl
from stagpy import stagyyparsers as sp
import pickle as pkl
import pandas as pd
from scipy.interpolate import interp1d
import h5py
pwd = Path('.')


#IMPORTANT: Add this function to field.py (e.g. at the very end of the file) in stagpy: 
'''
def get_sfield(step, var):
    sfld = step.sfields[var].values[0,:,0]
    return sfld
'''


#Parameters for latex like plots
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',size=11)

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def angles_transform_deg(phi):
    """
    Function that transforms angles from stag into degrees and converts it so that dayside is [0,180] and nightside is [180,360]
    """
    phi_return = []
    phi = (180/np.pi)*np.array(phi)
    for phi_i in phi:
        if phi_i < 270:
            phi_return.append(270-phi_i)
        elif phi_i >= 270:
            phi_return.append(270+360-phi_i)
        else:
            print('something is wrong', phi_i)

    return list(phi_return)

def calculate_optimal_ylim(data, exclude_percentile=90, std_multiplier=5):
    """
    Calculate optimal ylim for heatflux data

    Parameters:
    - data (array-like): heatflux values
    - exclude_percentile (float): percentile threshold to exclude initial high values
    - std_multiplier (float): Multiplier for standard deviation to determine ylim

    Returns:
    - tuple: (ylim_lower, ylim_upper) 
    """

    data = np.array(data)
    threshold = np.percentile(data, exclude_percentile)
    filtered_data = data[data < threshold]

    mean_value = np.mean(filtered_data)
    std_value = np.std(filtered_data)

    ylim_lower = mean_value - std_multiplier*std_value
    ylim_upper = mean_value + std_multiplier*std_value

    return ylim_lower, ylim_upper


#Default is png, can be changed to pdf using options
file_ending = ".png"

#need to be in StagYY run directory
rep = '.'
sdat = stagyydata.StagyyData(rep)
n_snapshots = sdat.snaps[-1].isnap
parser = argparse.ArgumentParser()


parser.add_argument('--flux', dest='flux', help="If true, surface heat flux is plotted",default=False, action='store_true')
parser.add_argument('--pdf', dest='pdf', help="If true, pdf files instead of pngs are produced",default=False, action='store_true')
parser.add_argument("--smin", help="lowest snapshot number (starting at 0)",type=int,default=0)
parser.add_argument("--smax", help="highest snapshot number",type=int,default=n_snapshots)
parser.add_argument('--darkmode', dest='darkmode', help="If true, darkmode is enabled (for presentations)",default=False, action='store_true')
parser.add_argument('--skip', dest='skip', help="If true, calculation is skipped (works if flux_data.p already exists",default=False, action='store_true')
parser.add_argument("--transit_time", help="plot overturn time on top instead of physical time",type=float,default=-1.0)
parser.add_argument("--max_time", help="maximum time for xaxis in Gyrs",type=float,default=-1.0)
parser.add_argument("--time_window", help="time window for roll mean",type=float,default=-1.0)
parser.add_argument('--legend', dest='legend', help="If true, legend is plotted",default=False, action='store_true')
parser.add_argument('--no_cmbflux', dest='no_cmbflux', help="If true, cmb flux is not plotted",default=False, action='store_true')
parser.add_argument('--cmb_separate', dest='cmb_separate', help="If true, cmb flux is plotted on separate axis (probably not needed without magma oceans)",default=False, action='store_true')
parser.add_argument('--daymelt', dest='daymelt', help="If true, dayside is plotted on W/m^2 and nightside on mW/m^2",default=False, action='store_true')
parser.add_argument('--all_solid', dest='all_solid', help="If true, dayside and nightside are on mW/m^2",default=False, action='store_true')
parser.add_argument('--ns_molten', dest='ns_molten', help="If true, nightside flux is plotted in W/m2",default=False, action='store_true')

args = parser.parse_args()

file_ending = ".png"
if args.pdf == True:
    file_ending = ".pdf"



time_tot = []

topflux_day = []
topflux_night = []
topflux_tot = []

botflux_day = []
botflux_night = []
botflux_tot = []


#Flux is calculated for every ds-th snapshot (every 5th by default)
ds = 10


#Here, the flux is calculated (this step can be skipped if the flux data was previously saved in flux_data.p)
if args.skip == False:
    for i in range(args.smin,args.smax+1,ds):
        try:
            print("Snapshot number: ", i, n_snapshots+1)
            step = sdat.snaps[i]
        except KeyError:
            print(f"Snapshot {i} does not exist. Skipping...")
            continue  # Skip to the next iteration if snapshot does not exist
        p_mesh, r_mesh = np.meshgrid(step.geom.p_centers, step.geom.r_centers, indexing="ij")
        phase = p_mesh[:,-1]
        #Transform into astro degrees
        phase = np.array(angles_transform_deg(phase))


        #Alternative way to determine the angles (phase information is needed to distinguish between dayside and nightside)
        phase_0 = p_mesh[np.where(r_mesh == np.max(r_mesh))]
        phase_1 = phase_0[:-1]
        phase_2 = phase_0[1:]
        phase_f = (phase_1+phase_2)/2.0
        phase_f = np.array(angles_transform_deg(phase_f))


        time_error = False
        try: 
            if time_error == False:
                time_tot.append(step.timeinfo['t']/(60*60*24*365.25)/1e9)
                print('Time', step.timeinfo['t']/(60*60*24*365.25)/1e9)
            else:
                hf = h5py.File('+hdf5/time_botT.h5', 'r')
                ks = list(hf.keys())
                n_i = hf[ks[i]]
                time_step = n_i[0]/(60*60*24*365.25)/1e9 #index needs to be 1 if updated stag is used (but time error seemed to only have happend with old version)
                time_tot.append(time_step)
                print('Time (time error)', time_step)
        except KeyError:
            print(f"Error accessing time information for snapshot {i}. Skipping to next snapshot.")
            continue
        
        if args.flux == True and args.skip == False:
            phase = phase_f

            #Get surface fluxes
            sflux_tot = field.get_sfield(step, 'ftop')*1000.0
            sflux_day = sflux_tot[np.where(phase < 180.0)]
            sflux_night = sflux_tot[np.where(phase >= 180.0)]
            #Median rather than mean because sometimes can have large outliers
            topflux_day.append(np.median(sflux_day))
            topflux_night.append(np.median(sflux_night))
            topflux_tot.append(np.median(sflux_tot))

            #Get cmb fluxes
            cmbflux_tot = field.get_sfield(step, 'fbot')*1000.0
            cmbflux_day = cmbflux_tot[np.where(phase < 180.0)]
            cmbflux_night = cmbflux_tot[np.where(phase >= 180.0)]
            #There's usually less outliers for the CMB flux, but could also take median here
            botflux_day.append(np.mean(cmbflux_day))
            botflux_night.append(np.mean(cmbflux_night))
            botflux_tot.append(np.mean(cmbflux_tot))

            cmbflux_total = np.array(cmbflux_tot)
            topflux_total = np.array(sflux_tot)

    if args.flux == True:
        with open("flux_data.p", "wb") as f:
            pkl.dump((time_tot,topflux_tot,topflux_day,topflux_night,botflux_tot,botflux_day,botflux_night),f)


if args.skip == True:
    #If flux already calculated before and stored in flux_data.p, can just load it here. 
    with open("flux_data.p", "rb") as f:
        time_tot,topflux_tot,topflux_day,topflux_night,botflux_tot,botflux_day,botflux_night = pkl.load(f)


#Plotting part
dayside_color = "red"
nightside_color = "blue" 

if args.darkmode:
    plt.style.use('dark_background')
    nightside_color = (0/255,199/255,255/255)
    dayside_color = (255/255,56/255,0/255) 


if args.flux == True:

    if args.max_time < 0.0:
        max_time = max(time_tot)
    else:
        max_time = min(args.max_time,max(time_tot))

    if args.darkmode:
        plt.style.use('dark_background')
        nightside_color = (0/255,199/255,255/255)
        dayside_color = (255/255,56/255,0/255)

    if args.time_window < 0.0:
        #Default rolling average window is 20
        window = 20
    else:
        #Window given as input in options (convert from time to int)
        window = int(args.time_window/(max(time_tot)/len(time_tot)))
        print('Rolling average window: ',window)

    ####
    #This is just to print flux at end of of run, can be commented out.    
    ftopday = interp1d(time_tot, topflux_day)
    ftopnight = interp1d(time_tot, topflux_night)
    fbottot = interp1d(time_tot, botflux_tot)
    print('Flux in mW/m^2 at day, night, cmb : ', ftopday(max_time), ftopnight(max_time), fbottot(max_time))
    ####


    #Moving average calculations, needed for the 'bands' around the flux
    ma_topflux_tot = moving_average(np.asarray(topflux_tot),window)
    mmin_topflux_tot = pd.Series(topflux_tot).rolling(window).min().dropna().tolist()
    mmax_topflux_tot = pd.Series(topflux_tot).rolling(window).max().dropna().tolist() 

    ma_topflux_day = moving_average(np.asarray(topflux_day),window)
    mmin_topflux_day = pd.Series(topflux_day).rolling(window).min().dropna().tolist()
    mmax_topflux_day = pd.Series(topflux_day).rolling(window).max().dropna().tolist() 

    ma_topflux_night = moving_average(np.asarray(topflux_night),window)
    mmin_topflux_night = pd.Series(topflux_night).rolling(window).min().dropna().tolist()
    mmax_topflux_night = pd.Series(topflux_night).rolling(window).max().dropna().tolist()

    ma_botflux_tot = moving_average(np.asarray(botflux_tot),window)
    mmin_botflux_tot = pd.Series(botflux_tot).rolling(window).min().dropna().tolist()
    mmax_botflux_tot = pd.Series(botflux_tot).rolling(window).max().dropna().tolist() 

    ma_botflux_day = moving_average(np.asarray(botflux_day),window)
    mmin_botflux_day = pd.Series(botflux_day).rolling(window).min().dropna().tolist()
    mmax_botflux_day = pd.Series(botflux_day).rolling(window).max().dropna().tolist() 

    ma_botflux_night = moving_average(np.asarray(botflux_night),window)
    mmin_botflux_night = pd.Series(botflux_night).rolling(window).min().dropna().tolist()
    mmax_botflux_night = pd.Series(botflux_night).rolling(window).max().dropna().tolist()

    ma_time = moving_average(np.asarray(time_tot), window)


    #Plotting the figure
    fig, ax1 = plt.subplots(1,1)
    if args.transit_time > 0:
        ma_time = ma_time / args.transit_time
    #Plot surface flux    
    print('time',ma_time)
    print(ma_topflux_day)
    ax1.plot(ma_time, ma_topflux_day,'-',color=dayside_color,label='Surface flux dayside')
    ax1.fill_between(ma_time,mmin_topflux_day, mmax_topflux_day,facecolor=dayside_color,alpha=0.2)
    ax1.plot(ma_time, ma_topflux_night,'-',color=nightside_color,label='Surface flux nightside')
    ax1.fill_between(ma_time,mmin_topflux_night, mmax_topflux_night,facecolor=nightside_color,alpha=0.2)
    #Could also plot total flux (dayside and nightside)
    #ax1.plot(ma_time, ma_topflux_tot,'-',color='black',label='Surface flux total')

    #Plot CMB flux
    if args.no_cmbflux == False:
        ax1.plot(ma_time, ma_botflux_day,'--',color='lightsalmon',label='CMB flux dayside')
        ax1.fill_between(ma_time,mmin_botflux_day, mmax_botflux_day,facecolor='salmon',alpha=0.2)
        ax1.plot(ma_time, ma_botflux_night,'--',color='skyblue',label='CMB flux nightside')
        ax1.fill_between(ma_time,mmin_botflux_night, mmax_botflux_night,facecolor='aqua',alpha=0.2)
        ax1.plot(ma_time, ma_botflux_tot,'--',color='black',label='CMB flux total')

    if args.legend:
        #ax1.legend(loc="upper right", markerscale=4, bbox_to_anchor=(0.5, 1.02), ncol=2)
        ax1.legend(loc="upper right", markerscale=4, ncol=2)
    if args.transit_time > 0:
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top') 
        ax1.set_xlabel('Number of overturns',fontsize=13)
        ax1.set_xlim(0,max_time/args.transit_time)
    else:
        ax1.set_xlabel('Time (Gyrs)',fontsize=13)
    ax1.set_ylabel('Heat flux (mW/m$^2$) ',fontsize=13)
    ax1.set_ylim(-20,200) #This is arbitrary, change this 
    ax1.tick_params(labelsize=13)
    plt.savefig("flux_plot"+file_ending,dpi=900,bbox_inches='tight',transparent=False)

    
    #Below are some options for different plots (first one includes CMB heat flux)
    if args.cmb_separate:
        #ax1 is in W/m^2 (high flux), ax2 is in mW/m^2 (CMB heat flux)
        fig, ax1 = plt.subplots(1,1)
        l0 = ax1.plot(ma_time, ma_topflux_day/1000,'-',markersize=2.0,color=dayside_color,label='Surface flux dayside')
        #need to divide by 1000 to get flux in W/m^2
        ax1.fill_between(ma_time,np.array(mmin_topflux_day)/1000, np.array(mmax_topflux_day)/1000,facecolor=dayside_color,alpha=0.2)
        ax1.tick_params(axis='y',labelcolor='red')
        ax2 = ax1.twinx() #Create a y-axis on the right side (for CMB fluxes in mW/m^2, and potentially the nightside if it's not molten)
        if args.ns_molten == True:
            #If nightside is molten (high flux), will be plotted on the left axis in W/m^2
            l1 = ax1.plot(ma_time, ma_topflux_night/1000,'-.',markersize=2.0,color=nightside_color,label='Surface flux nightside')
            ax1.fill_between(ma_time,np.array(mmin_topflux_night)/1000, np.array(mmax_topflux_night)/1000,facecolor=nightside_color,alpha=0.2)
        else:
            #Nightside not molten, so will be plotted on the right side in mW/m^2
            l1 = ax2.plot(ma_time, ma_topflux_night,'-.',markersize=2.0,color=nightside_color,label='Surface flux nightside')
            ax2.fill_between(ma_time,np.array(mmin_topflux_night), np.array(mmax_topflux_night),facecolor=nightside_color,alpha=0.2)
        l2 = ax2.plot(ma_time, ma_botflux_day,'--',color='lightsalmon',label='CMB flux dayside')
        ax2.fill_between(ma_time,mmin_botflux_day, mmax_botflux_day,facecolor='salmon',alpha=0.2)
        l3 = ax2.plot(ma_time, ma_botflux_night,'--',color='skyblue',label='CMB flux nightside')
        ax2.fill_between(ma_time,mmin_botflux_night, mmax_botflux_night,facecolor='aqua',alpha=0.2)
        l4 = ax2.plot(ma_time, ma_botflux_tot,'--',color='green',label='CMB flux total')
        ax2.tick_params(axis ='y', labelcolor = 'black') 
        if args.legend:
            #ax1.legend(loc="upper right", markerscale=4, bbox_to_anchor=(0.5, 1.2), ncol=1)
            #ax2.legend(loc="upper left", markerscale=4, bbox_to_anchor=(0.5, 1.2), ncol=2)
            lns = l0 + l1 + l2 + l3 + l4 
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper right',facecolor='white')
        ax1.set_ylim(-80,50) #This is quite arbitrary and might have to be changed
        ax2.set_ylim(-80,100) 
        if args.transit_time > 0:
            ax1.xaxis.tick_top()
            ax1.xaxis.set_label_position('top') 
            ax1.set_xlabel('Number of overturns',fontsize=13)
            ax1.set_xlim(0,max_time/args.transit_time)
        else:
            ax1.set_xlabel('Time (Gyrs)',fontsize=13)
            ax1.set_xlim(0,max_time) 

        ax1.set_ylabel('Heat flux (W/m$^2$) ',fontsize=13,color='red')
        ax2.set_ylabel('Heat flux (mW/m$^2$) ',fontsize=13,color='black')

        ax1.tick_params(labelsize=13)
        plt.savefig("flux_plot"+file_ending,dpi=900,bbox_inches='tight',transparent=False)

    if args.daymelt:
        #this is for dayside magma ocean and nightside solid
        #ax1 is in W/m^2 (high flux), ax2 is in mW/m^2 (nightside unless using ns_molten)
        fig, ax1 = plt.subplots(1,1)
        l0 = ax1.plot(ma_time, ma_topflux_day/1000,'-',markersize=2.0,color=dayside_color,label='Surface flux dayside')
        #need to divide by 1000 to get flux in W/m^2
        print('time:', ma_time)
        print('nigthside flux:', ma_topflux_night)
        print('dayside flux in W/m^2: ', ma_topflux_day/1000)
        #ax1.fill_between(ma_time,np.array(mmin_topflux_day)/1000, np.array(mmax_topflux_day)/1000,facecolor=dayside_color,alpha=0.2)
        ax1.tick_params(axis='y',labelcolor='red')
        ax2 = ax1.twinx() #Create a y-axis on the right side (for CMB fluxes in mW/m^2, and potentially the nightside if it's not molten)
        if args.ns_molten == True:
            #If nightside is molten (high flux), will be plotted on the left axis in W/m^2
            l1 = ax1.plot(ma_time, ma_topflux_night/1000,'-',markersize=2.0,color=nightside_color,label='Surface flux nightside')
            ax1.fill_between(ma_time,np.array(mmin_topflux_night)/1000, np.array(mmax_topflux_night)/1000,facecolor=nightside_color,alpha=0.2)
        else:
            #Nightside not molten, so will be plotted on the right side in mW/m^2
            l1 = ax2.plot(ma_time, ma_topflux_night,'-.',markersize=2.0,color=nightside_color,label='Surface flux nightside')
            #ax2.fill_between(ma_time,np.array(mmin_topflux_night), np.array(mmax_topflux_night),facecolor=nightside_color,alpha=0.2)
        ax2.tick_params(axis ='y', labelcolor = 'blue') 
        if args.legend:
            #ax1.legend(loc="upper right", markerscale=4, bbox_to_anchor=(0.5, 1.2), ncol=1)
            #ax2.legend(loc="upper left", markerscale=4, bbox_to_anchor=(0.5, 1.2), ncol=2)
            lns = l0 + l1 
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper right',facecolor='white')
        ax1.set_ylim(-80,5000) #This is quite arbitrary and might have to be changed
        ax2.set_ylim(30,40) 
        #ax1.set_ylim(calculate_optimal_ylim(ma_topflux_day/1000,std_multiplier=8))
        #ax2.set_ylim(calculate_optimal_ylim(ma_topflux_night,std_multiplier=20))
        if args.transit_time > 0:
            ax1.xaxis.tick_top()
            ax1.xaxis.set_label_position('top') 
            ax1.set_xlabel('Number of overturns',fontsize=13)
            ax1.set_xlim(0,max_time/args.transit_time)
        else:
            ax1.set_xlabel('Time (Gyrs)',fontsize=13)
            ax1.set_xlim(0,max_time) 

        ax1.set_ylabel('Heat flux (W/m$^2$) ',fontsize=13,color='red')
        ax2.set_ylabel('Heat flux (mW/m$^2$) ',fontsize=13,color='black')

        ax1.tick_params(labelsize=13)
        plt.savefig("flux_plot"+file_ending,dpi=900,bbox_inches='tight',transparent=False)

    if args.all_solid:
        #this is for dayside solid and nightside solid
        #ax1 is in W/m^2 (high flux), ax2 is in mW/m^2 (nightside unless using ns_molten)
        fig, ax1 = plt.subplots(1,1)
        l0 = ax1.plot(ma_time, ma_topflux_day,'-',markersize=2.0,color=dayside_color,label='Surface flux dayside')
        #need to divide by 1000 to get flux in W/m^2
        print('time:', ma_time)
        print('nigthside flux:', ma_topflux_night)
        print('dayside flux in W/m^2: ', ma_topflux_day/1000)
        #ax1.fill_between(ma_time,np.array(mmin_topflux_day)/1000, np.array(mmax_topflux_day)/1000,facecolor=dayside_color,alpha=0.2)
        ax1.tick_params(axis='y',labelcolor='red')
        ax2 = ax1.twinx() #Create a y-axis on the right side (for CMB fluxes in mW/m^2, and potentially the nightside if it's not molten)
        l1 = ax2.plot(ma_time, ma_topflux_night,'-',markersize=2.0,color=nightside_color,label='Surface flux nightside')
        #ax2.fill_between(ma_time,np.array(mmin_topflux_night), np.array(mmax_topflux_night),facecolor=nightside_color,alpha=0.2)
        ax2.tick_params(axis ='y', labelcolor = 'blue') 
        if args.legend:
            #ax1.legend(loc="upper right", markerscale=4, bbox_to_anchor=(0.5, 1.2), ncol=1)
            #ax2.legend(loc="upper left", markerscale=4, bbox_to_anchor=(0.5, 1.2), ncol=2)
            lns = l0 + l1 
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper right',facecolor='white')
        ax1.set_ylim(-80,5000) #This is quite arbitrary and might have to be changed
        ax2.set_ylim(30,40) 
        ax1.set_ylim(calculate_optimal_ylim(ma_topflux_day,std_multiplier=8))
        #ax2.set_ylim(calculate_optimal_ylim(ma_topflux_night,std_multiplier=20))
        if args.transit_time > 0:
            ax1.xaxis.tick_top()
            ax1.xaxis.set_label_position('top') 
            ax1.set_xlabel('Number of overturns',fontsize=13)
            ax1.set_xlim(0,max_time/args.transit_time)
        else:
            ax1.set_xlabel('Time (Gyrs)',fontsize=13)
            ax1.set_xlim(0,max_time) 

        ax1.set_ylabel('Heat flux (mW/m$^2$) ',fontsize=13,color='red')
        ax2.set_ylabel('Heat flux (mW/m$^2$) ',fontsize=13,color='black')

        ax1.tick_params(labelsize=13)
        plt.savefig("flux_plot"+file_ending,dpi=900,bbox_inches='tight',transparent=False)









