
import numpy as np
from scipy.signal import welch, csd
import matplotlib.pyplot as plt
p1_dir = '/home/arian/dd_waves/m2_internal_tide_gen_prop/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt,windows
from cartopy.feature import NaturalEarthFeature, COLORS
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from scipy import signal

dzs = '/data1/roms_dd_waves/analysis_outs/dz/'


#######################################################################################################################
################### ----- A -----####################################################################################
etas = '/data1/roms_dd_waves/analysis_outs/NIW/displacement/'
zcross_A = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=905,xi_rho=slice(0,None)).z_rho.values


eta_A_1 = xr.open_mfdataset(etas + 'eta_slice*_1.nc').isel(eta_rho=905,xi_rho=380).eta
eta_A_2 = xr.open_mfdataset(etas + 'eta_slice*_2.nc').isel(eta_rho=905,xi_rho=380).eta
eta_A_3 = xr.open_mfdataset(etas + 'eta_slice*_3.nc').isel(eta_rho=905,xi_rho=380).eta
eta_A_4 = xr.open_mfdataset(etas + 'eta_slice*_4.nc').isel(eta_rho=905,xi_rho=380).eta

eta_A = xr.concat([eta_A_1, eta_A_2, eta_A_3, eta_A_4], dim='ocean_time')
eta_A = eta_A.sortby('ocean_time')
eta_A = eta_A.sel(ocean_time=eta_A.ocean_time.drop_duplicates(dim='ocean_time')).compute()

time = eta_A.ocean_time












fig = plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(nrows=1, ncols=1, width_ratios=[1], height_ratios=[1])
gs.update(left=0.08, right=0.98, wspace=0.25, hspace=0.35, top=0.99, bottom=0.1)

#cmap_N = plt.cm.cubehelix_r
cmap_N = cmo.cm.balance

vmin_N = -50
vmax_N = 50
norm_N = mpl.colors.Normalize(vmin=vmin_N, vmax=vmax_N)

axNa1 = plt.subplot(gs[0, 0])
axNa1.set_ylim(-100, 0) 
axNa1.contourf(time, zcross_A[:,380], eta_A.T, cmap=cmap_N, levels=50,norm=norm_N)
axNa1.set_xlabel(None,size='x-small')
axNa1.set_ylabel('m',size='x-small')
axNa1.tick_params(axis='both', labelsize='x-small')
#ax.tick_params(axis='x', labelsize='x-small', rotation=25)
#ax.tick_params(axis='y', labelsize='x-small')
axNa1.set_xticks([])  # Remove x-axis ticks

axNa1.text(0.02, 0.95, '(a2)', transform=axNa1.transAxes, fontsize='x-small', fontweight='demibold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))


plt.savefig('test_eta.png',dpi = 200)