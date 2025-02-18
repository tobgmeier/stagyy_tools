import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os
import pickle as pkl
import seaborn as sns
# Parameters for latex-like plots
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font', size=11)

# Define function to check and create directory
def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Initialize plots
fig_dayside, ax_dayside = plt.subplots(figsize=(8, 5))
fig_nightside, ax_nightside = plt.subplots(figsize=(8, 5))


# Loop through subfolders (test01, test02, etc.) and process each
data_root = Path(".")  # Points to the current working directory
subfolders = [data_root / "test01", data_root / "test02"]
subfolders = [data_root / f"test{str(i).zfill(2)}" for i in range(1, 17) if i != 3]

colors = sns.color_palette("tab20", n_colors=len(subfolders))

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
    ax_dayside.plot(time_tot, topflux_day / 1000.0, label=f'{folder}',color=colors[i % len(colors)])
    
    # Add to nightside plot
    ax_nightside.plot(time_tot, topflux_night, label=f'{folder}',color=colors[i % len(colors)])

# Finalize Dayside Plot
#ax_dayside.set_ylim(-1,1)
ax_dayside.set_yscale('symlog', linthresh=1.0e-2)
#ax_dayside.set_yscale('log')
ax_dayside.set_xlabel("Time (Gyrs)")
ax_dayside.set_ylabel("Heat Flux (W/m²)")
#ax_dayside.set_ylim(0,0.1)
ax_dayside.set_title("Dayside Heat Flux for All Runs")
ax_dayside.legend()
check_directory("plots")
fig_dayside.savefig("plots/flux_dayside_all.png", dpi=300)

# Finalize Nightside Plot
ax_nightside.set_xlabel("Time (Gyrs)")
ax_nightside.set_ylabel("Heat Flux (mW/m²)")
ax_nightside.set_title("Nightside Heat Flux for All Runs")
ax_nightside.set_ylim(25,80)
ax_nightside.legend()
fig_nightside.savefig("plots/flux_nightside_all.png", dpi=300)

plt.close(fig_dayside)
plt.close(fig_nightside)

