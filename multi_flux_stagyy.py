import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
import pickle as pkl
import seaborn as sns
import platform

if platform.system() == 'Darwin':  # macOS
    font_family = 'Futura'
else:  # For Linux or other systems
    font_family = 'Futura Md BT'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rc('font',size=15)
mpl.rcParams['font.family'] = font_family
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['ytick.major.width'] = 1
plt.rc('axes', unicode_minus=False)

# Define function to check and create directory
def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

groups = {
    "negative": ['test01', 'test02'],
    "low": ["test04", "test05", "test07", "test09", "test15", "test16"],
    "high": ["test06", "test08","test13","test14"],
    "variable": ["test10", "test12"],       
}

ylims = {
    "negative": (-500, 50), #in mW
    "low": (0,100),          #in mW
    "high": (1000,10000), #in W
    "variable": (0,1.0e4)       #in W
}

scaling = {
    "negative": 1.0,
    "low": 1.0,
    "high": 1000.0,
    "variable": 1000.0
}

# Loop through subfolders (test01, test02, etc.) and process each
data_root = Path(".")  # Points to the current working directory
subfolders = [data_root / "test01", data_root / "test02"]
subfolders = [data_root / f"test{str(i).zfill(2)}" for i in range(1, 17) if i != 3]

colors = sns.color_palette("Set2", n_colors=len(subfolders))

figs_dayside, axes_dayside = {}, {} #creating empty dictionaries for the figure and axes (each dictionary element will have a separate figure)
fig_nightside, ax_nightside = plt.subplots(figsize=(6.4,4.8))  
for group in groups: #adding figsize for each element
    figs_dayside[group], axes_dayside[group] = plt.subplots(figsize=(6.4,4.8)) 

for group, subfolders in groups.items(): #this will loop through each key and value (here each key contains the subfolders)
    for i, folder in enumerate(subfolders):
        print(f"Processing folder: {folder}")
        pkl_file = Path(folder) / "flux_data.p"  # Adjust file name if needed
        
        if not pkl_file.exists():
            print(f"No pickle file found in {folder}, skipping...")
            continue
        
        # Load pickle data and convert to numpy arrays
        with open(pkl_file, "rb") as f:
            time_tot, topflux_tot, topflux_day, topflux_night, botflux_tot, botflux_day, botflux_night = map(np.array, pkl.load(f))
        
        # Add to dayside plot (dividing by 1000 for correct scaling)
        axes_dayside[group].plot(time_tot, topflux_day / scaling[group], label=f'{folder}',color=colors[i % len(colors)])
        ax_nightside.plot(time_tot, topflux_night, label=f'{folder}',color=colors[i % len(colors)])
        
        # Add to nightside plot
        #axes_nightside.plot(time_tot, topflux_night, label=f'{folder}',color=colors[i % len(colors)])

    # Finalize Dayside Plot
    #ax_dayside.set_ylim(-1,1)
    if group == 'variable':
        axes_dayside[group].set_yscale('log')
    #ax_dayside.set_yscale('log')
    axes_dayside[group].set_xlabel("Time (Gyrs)")
    if scaling[group] == 1000.0:
        axes_dayside[group].set_ylabel("Heat flux (W/m$^2$)")
    else:
        axes_dayside[group].set_ylabel("Heat flux (mW/m$^2$)")
    axes_dayside[group].set_ylim(ylims[group])
    axes_dayside[group].set_xlim(0,8)
    #ax_dayside.set_ylim(0,0.1)
    axes_dayside[group].set_title(f"{group.capitalize()} dayside heat flux cases")
    axes_dayside[group].legend()
    check_directory("plots")
    figs_dayside[group].tight_layout()
    figs_dayside[group].savefig(f"plots/flux_dayside_{group}.pdf", dpi=300)

    
    #Nightside Plot (no groups here)
    ax_nightside.set_xlabel("Time (Gyrs)")
    ax_nightside.set_ylabel("Heat Flux (mW/m²)")
    ax_nightside.set_title("Nightside Heat Flux for All Runs")
    ax_nightside.set_ylim(25,80)
    ax_nightside.legend()
    fig_nightside.savefig("plots/flux_nightside_all.png", dpi=300)
    plt.close(fig_nightside)
    
