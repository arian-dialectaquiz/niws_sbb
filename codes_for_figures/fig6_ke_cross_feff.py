

import numpy as np
from scipy.signal import welch, csd
import matplotlib.pyplot as plt
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
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
from gsw import *


A =  [376,1239]
B =  [241,957]
C =  [299,570]
D =  [463,225]


date_ranges_normal = [
	("2004-03-31", "2004-04-09")
	
]

date_ranges_cf = [
	("2004-03-19", "2004-03-24"),
	("2004-04-09", "2004-04-11")
	
]


date_ranges_hurricane = [
	("2004-03-23", "2004-03-30")
	
]


def create_date_mask(dates_array, date_ranges):
	"""
	Creates a single boolean mask for all time steps that fall sshthin
	any of the specified date ranges.
	"""
	# Initialize a mask of False for all time steps
	combined_mask = np.zeros_like(dates_array, dtype=bool)

	for start_date, end_date in date_ranges:
		# Convert strings to Pandas Timestamps for comparison
		start_ts = pd.to_datetime(start_date)
		end_ts = pd.to_datetime(end_date)

		# Create a mask for the current range: (start <= date <= end)
		range_mask = (dates_array >= start_ts) & (dates_array <= end_ts)

		# Update the combined mask using an 'OR' operation
		combined_mask = combined_mask | range_mask

	return combined_mask



############################################################################
#################-------> feff and nike <------#############################
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'
f_p = '/data1/roms_dd_waves/analysis_outs/NIW/vorticity/'
e_p = '/data1/roms_dd_waves/analysis_outs/NIW/eke/'



nike = xr.open_mfdataset(e_p + 'NIKE_slice_*.nc').nike.isel(ocean_time=slice(96,None)) * 1025

vort = xr.open_mfdataset(f_p + 'f_eff_slice_*.nc').f_eff.isel(ocean_time=slice(96,None))

dates = vort.ocean_time.to_index()

zs = xr.open_mfdataset(dzs + f'dz_*.nc').z_rho
zcross_A = zs.isel(eta_rho=A[1],xi_rho=slice(0,None)).values
zcross_B = zs.isel(eta_rho=B[1],xi_rho=slice(0,None)).values
zcross_C = zs.isel(eta_rho=C[1],xi_rho=slice(0,None)).values
zcross_D = zs.isel(eta_rho=D[1],xi_rho=slice(0,None)).values



path_pot ='/data1/roms_dd_waves/analysis_outs/NIW/pot_energy/'
ml = xr.open_mfdataset(path_pot + 'mld_slice_*.nc').isel(ocean_time=slice(96,None)).mld

###########------ Normal
mask_normal = create_date_mask(dates, date_ranges_normal)
vort_normal = vort.isel(ocean_time=mask_normal)

#--A
vort_normal_A = vort_normal.isel(eta_rho=A[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()
f_A = gsw.f(vort_normal_A.lat_rho.values).reshape((610, 1))
rel_A = np.absolute(vort_normal_A)/np.absolute(f_A)


nike_normal_A = nike.isel(ocean_time=mask_normal).isel(eta_rho=A[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()

ml_normal_A = ml.isel(ocean_time=mask_normal,eta_rho=A[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()



###########------cf
mask_cf = create_date_mask(dates, date_ranges_cf)
vort_cf = vort.isel(ocean_time=mask_cf)


#--B
vort_cf_B = vort_cf.isel(eta_rho=B[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()
f_B = gsw.f(vort_cf_B.lat_rho.values).reshape((610, 1))
rel_B = np.absolute(vort_cf_B)/np.absolute(f_B)

nike_cf_B = nike.isel(ocean_time=mask_cf).isel(eta_rho=B[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()

ml_cf_B = ml.isel(ocean_time=mask_cf,eta_rho=B[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()

#######-----#hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
vort_hurricane = vort.isel(ocean_time=mask_hurricane)

#--D
vort_hurricane_D = vort_hurricane.isel(eta_rho=D[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()
f_D = gsw.f(vort_hurricane_D.lat_rho.values).reshape((610, 1))
rel_D = np.absolute(vort_hurricane_D)/np.absolute(f_D)

nike_hurricane_D = nike.isel(ocean_time=mask_hurricane).isel(eta_rho=D[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()

ml_hurricane_D = ml.isel(ocean_time=mask_hurricane,eta_rho=D[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()


km_cross = np.arange(0,610,1)



###########_---> Plotting it <-------------#######

fig = plt.figure(figsize=(9, 5))
gs = gridspec.GridSpec(nrows=2, ncols=4, width_ratios=[33,33,33,1], height_ratios=[1, 1])
gs.update(left=0.07, right=0.95, wspace=0.1, hspace=0.1, top=0.95, bottom=0.08)

#---> fratio colormap

cmap_f = plt.cm.gist_stern
vmin_f = 0.8
vmax_f = 1.2
norm_f = mpl.colors.Normalize(vmin=vmin_f, vmax=vmax_f)

cbar_f = plt.subplot(gs[0, 3])
cbf = mpl.colorbar.ColorbarBase(cbar_f, cmap=cmap_f, norm=norm_f, extend='both', orientation='vertical')
cbf.set_label(None)
cbar_f.yaxis.set_ticks_position('left')  # Ticks on the left side
cbar_f.tick_params(axis='y', labelsize='x-small', rotation=25)


#---> nike colormap
cmap_ke = plt.cm.gist_ncar_r

vmin_ke = 0
vmax_ke = 15
norm_ke = mpl.colors.Normalize(vmin=vmin_ke, vmax=vmax_ke)
bar_title_ke = r'J.m$^{-3}$'

cbar_ke = plt.subplot(gs[1, 3])
cbke = mpl.colorbar.ColorbarBase(cbar_ke, cmap=cmap_ke, norm=norm_ke, extend='max', orientation='vertical')
cbke.set_label(bar_title_ke, size='x-small', labelpad=5)
cbar_ke.yaxis.set_ticks_position('left')  # Ticks on the left side
cbar_ke.tick_params(axis='y', labelsize='x-small', rotation=25)

##----> Normal cross A

#---> fratio
ax1_A = plt.subplot(gs[0, 0])
for i in range(len(km_cross)):
	ax1_A.contourf(km_cross, zcross_A[:,i], rel_A.isel(s_rho=slice(1,None)).T, levels=100,cmap=cmap_f, norm=norm_f)

ax1_A.fill_between(km_cross, np.min(zcross_A, axis=0), y2=ax1_A.get_ylim()[0], color='grey', zorder = 2)
ax1_A.set_title('Normal', loc='left', fontsize=10)

ax1_A.plot(km_cross, -1 * ml_normal_A, color='black', linestyle='--', linewidth=1.5, label='MLD', zorder=5)
ax1_A.set_xlabel('km from coast',size='x-small')
ax1_A.set_ylabel('m',size='x-small')
ax1_A.tick_params(axis='both', labelsize='x-small')
ax1_A.set_ylim(-400, 0) 
ax1_A.set_xlim(230, 609) 
ax1_A.text(0.02, 0.05, '(a)', transform=ax1_A.transAxes, fontsize=8, fontweight='bold')


#-----> NIKE
ax2_A = plt.subplot(gs[1, 0])
for i in range(len(km_cross)):
	ax2_A.contourf(km_cross, zcross_A[:,i], nike_normal_A.isel(s_rho=slice(1,None)), levels=100,cmap=cmap_ke, norm=norm_ke)

ax2_A.fill_between(km_cross, np.min(zcross_A, axis=0), y2=ax2_A.get_ylim()[0], color='grey', zorder = 2)

ax2_A.plot(km_cross, -1 * ml_normal_A, color='black', linestyle='--', linewidth=1.5, label='MLD', zorder=5)
ax2_A.set_xlabel('km from coast',size='x-small')
ax2_A.set_ylabel('m',size='x-small')
ax2_A.tick_params(axis='both', labelsize='x-small')
ax2_A.set_ylim(-400, 0) 
ax2_A.set_xlim(230, 609) 
ax2_A.text(0.02, 0.05, '(b)', transform=ax2_A.transAxes, fontsize=8, fontweight='bold')


##----> CF cross B

#---> fratio
ax1_B = plt.subplot(gs[0, 1])
for i in range(len(km_cross)):
	ax1_B.contourf(km_cross, zcross_B[:,i], rel_B.isel(s_rho=slice(1,None)).T, levels=100,cmap=cmap_f, norm=norm_f)

ax1_B.fill_between(km_cross, np.min(zcross_B, axis=0), y2=ax1_B.get_ylim()[0], color='grey', zorder = 2)
ax1_B.set_title('Cold Front', loc='left', fontsize=10)

ax1_B.plot(km_cross, -1 * ml_cf_B, color='black', linestyle='--', linewidth=1.5, label='MLD', zorder=5)
ax1_B.set_xlabel('km from coast',size='x-small')
ax1_B.set_yticks([])
ax1_B.set_ylabel('')
ax1_B.tick_params(axis='x', labelsize='x-small')
ax1_B.set_ylim(-400, 0) 
ax1_B.set_xlim(70, 609) 
ax1_B.text(0.02, 0.05, '(c)', transform=ax1_B.transAxes, fontsize=8, fontweight='bold')


#-----> NIKE
ax2_B = plt.subplot(gs[1, 1])
for i in range(len(km_cross)):
	ax2_B.contourf(km_cross, zcross_B[:,i], nike_cf_B.isel(s_rho=slice(1,None)), levels=100,cmap=cmap_ke, norm=norm_ke)

ax2_B.fill_between(km_cross, np.min(zcross_B, axis=0), y2=ax2_B.get_ylim()[0], color='grey', zorder = 2)

ax2_B.plot(km_cross, -1 * ml_cf_B, color='black', linestyle='--', linewidth=1.5, label='MLD', zorder=5)
ax2_B.set_yticks([])
ax2_B.set_ylabel('')
ax2_B.tick_params(axis='x', labelsize='x-small')
ax2_B.set_ylim(-400, 0) 
ax2_B.set_xlim(70, 609) 
ax2_B.text(0.02, 0.05, '(d)', transform=ax2_B.transAxes, fontsize=8, fontweight='bold')
ax2_B.set_xlabel('km from coast',size='x-small')

##----> hurricane cross D

#---> fratio
ax1_D = plt.subplot(gs[0, 2])
for i in range(len(km_cross)):
	ax1_D.contourf(km_cross, zcross_D[:,i], rel_D.isel(s_rho=slice(1,None)).T, levels=100,cmap=cmap_f, norm=norm_f)

ax1_D.fill_between(km_cross, np.min(zcross_D, axis=0), y2=ax1_D.get_ylim()[0], color='grey', zorder = 2)
ax1_D.set_title('Hurricane', loc='left', fontsize=10)

ax1_D.plot(km_cross, -1 * ml_hurricane_D, color='black', linestyle='--', linewidth=1.5, label='MLD', zorder=5)
ax1_D.set_yticks([])
ax1_D.set_ylabel('')
ax1_D.tick_params(axis='x', labelsize='x-small')
ax1_D.set_ylim(-400, 0) 
ax1_D.set_xlim(250, 609) 
ax1_D.text(0.02, 0.05, '(e)', transform=ax1_D.transAxes, fontsize=8, fontweight='bold')


#-----> NIKE
ax2_D = plt.subplot(gs[1, 2])
for i in range(len(km_cross)):
	ax2_D.contourf(km_cross, zcross_D[:,i], nike_hurricane_D.isel(s_rho=slice(1,None)), levels=100,cmap=cmap_ke, norm=norm_ke)

ax2_D.fill_between(km_cross, np.min(zcross_D, axis=0), y2=ax2_D.get_ylim()[0], color='grey', zorder = 2)

ax2_D.plot(km_cross, -1 * ml_hurricane_D, color='black', linestyle='--', linewidth=1.5, label='MLD', zorder=5)
ax2_D.set_yticks([])
ax2_D.set_ylabel('')
ax2_D.tick_params(axis='x', labelsize='x-small')
ax2_D.set_ylim(-400, 0) 
ax2_D.set_xlim(250, 609) 
ax2_D.text(0.02, 0.05, '(f)', transform=ax2_D.transAxes, fontsize=8, fontweight='bold')
ax2_D.set_xlabel('km from coast',size='x-small')

plt.savefig('fig_6_NIKE_ratio.png', dpi = 300)




















