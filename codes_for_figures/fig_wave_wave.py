
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

from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, NullFormatter, FixedLocator



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
	Creates a single boolean mask for all time steps that fall within
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




out_path = '/data1/roms_dd_waves/analysis_outs/NIW/coarse/'

ww = xr.open_mfdataset(out_path+'PI_wave_zsum_slice_*.nc').PI_wave_zsum

dates = ww.ocean_time.to_index()


mask_normal = create_date_mask(dates, date_ranges_normal)
ww_normal = ww.isel(ocean_time=mask_normal).mean(dim='ocean_time').compute()
ww_normal = xr.where(ww_normal > 0, ww_normal / 4, ww_normal)

###########------cf
mask_cf = create_date_mask(dates, date_ranges_cf)
ww_cf = ww.isel(ocean_time=mask_cf).mean(dim='ocean_time').compute()
ww_cf = xr.where(ww_cf > 0, ww_cf / 4, ww_cf)


#######-----#hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
ww_hurricane = ww.isel(ocean_time=mask_hurricane).mean(dim='ocean_time').compute()
ww_hurricane = xr.where(ww_hurricane > 0, ww_hurricane / 4, ww_hurricane)


lon_rho = ww_hurricane.lon_rho.compute()
lat_rho = ww_hurricane.lat_rho.compute()

##################-----> Modes <---- ###################

import re
def natural_keys(text):
	"""Sorts strings numerically rather than alphabetically."""
	return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

m_path = '/data1/roms_dd_waves/analysis_outs/NIW/speed/modes/'

files_modes = sorted(glob.glob(m_path + 'velocity_eof_modes_slice_*.nc'), key=natural_keys)
modes = xr.open_mfdataset(files_modes, combine='nested', concat_dim='eta_rho', parallel=True)

dates = modes.ocean_time.to_index()

#######################################################################################################################
################### ----- Cross-sections -----##########################################################################
dzs = '/data1/roms_dd_waves/analysis_outs/dz/niws/'

z = xr.open_mfdataset(dzs + f'dz_*.nc').isel(s_rho = 0).z_rho #bottom

h = z.compute()*-1

Ha = -1*h.isel(eta_rho=A[1],xi_rho=slice(0,None))
#-Bs
Hb = -1*h.isel(eta_rho=B[1],xi_rho=slice(0,None))
#-C
Hc = -1*h.isel(eta_rho=C[1],xi_rho=slice(0,None))
#-D
Hd = -1*h.isel(eta_rho=C[1],xi_rho=slice(0,None))

zs = xr.open_mfdataset(dzs + f'dz_*.nc').z_rho
zcross_A = zs.isel(eta_rho=A[1],xi_rho=slice(0,None)).values
zcross_B = zs.isel(eta_rho=B[1],xi_rho=slice(0,None)).values
zcross_C = zs.isel(eta_rho=C[1],xi_rho=slice(0,None)).values
zcross_D = zs.isel(eta_rho=D[1],xi_rho=slice(0,None)).values

#########_----> KE 2 per depth (J/m³)
ke2_normal_A = modes.ke_mode2.isel(ocean_time=mask_normal).isel(eta_rho=A[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()
ke2_normal_A = ke2_normal_A*1025

ke2_cf_B = modes.ke_mode2.isel(ocean_time=mask_cf).isel(eta_rho=B[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()
ke2_cf_B = ke2_cf_B*1025

ke2_hurricane_D = modes.ke_mode2.isel(ocean_time=mask_hurricane).isel(eta_rho=D[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()
ke2_hurricane_D = ke2_hurricane_D*1025

km_cross = np.arange(0,602,1)




#########_----> wave_wave per class of depth (W/m²)
def calculate_depth_distribution(data, depth, bins):
	"""
	Calculates sum of 'data' within vertical 'depth' bins.
	"""
	# 1. Force conversion to simple Numpy arrays
	#    (If they are xarray objects, .values extracts the array)
	d_vals = data.values if hasattr(data, 'values') else data
	z_vals = depth.values if hasattr(depth, 'values') else depth

	# 2. Flatten arrays to 1D
	d_flat = d_vals.flatten()
	z_flat = z_vals.flatten()
	
	# 3. Remove NaNs (Crucial for clean histograms)
	valid = ~np.isnan(d_flat) & ~np.isnan(z_flat)
	
	# 4. Compute histogram
	hist, _ = np.histogram(z_flat[valid], bins=bins, weights=d_flat[valid])
	
	return hist


# Setup Bins
depth_bins = np.arange(-2700, 15, 50)  
bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2

#-------------A
ww_normal_a = ww_normal.isel(eta_rho=A[1],xi_rho=slice(0,None))
D_normal_A = calculate_depth_distribution(ww_normal_a, Ha[1:-1], depth_bins)

ke2_hurricane_D = modes.ke_mode2.isel(ocean_time=mask_hurricane).isel(eta_rho=D[1],xi_rho=slice(0,None)).mean(dim='ocean_time').compute()

#-----------------B
ww_cf_b = ww_cf.isel(eta_rho=B[1],xi_rho=slice(0,None))
D_cf_B = calculate_depth_distribution(ww_cf_b, Hb[1:-1], depth_bins)

#---------------D
ww_hurricane_d = ww_hurricane.isel(eta_rho=D[1],xi_rho=slice(0,None))
D_hurricane_D = calculate_depth_distribution(ww_hurricane_d, Hd[1:-1], depth_bins)











#######################################################################################################################
################### ----- MAP -----####################################################################################


###########------normal

mask_normal = create_date_mask(dates, date_ranges_normal)

ke_1_normal = modes.ke_sum_mode1.isel(ocean_time=mask_normal).mean(dim='ocean_time').compute()
ke_2_normal = modes.ke_sum_mode2.isel(ocean_time=mask_normal).mean(dim='ocean_time').compute()
ke_3_normal = modes.ke_sum_mode3.isel(ocean_time=mask_normal).mean(dim='ocean_time').compute()
ratio_normal = (ke_2_normal+ke_3_normal)/ke_1_normal


###########------cf
mask_cf = create_date_mask(dates, date_ranges_cf)
ww_cf = ww.isel(ocean_time=mask_cf).mean(dim='ocean_time').compute()

ke_1_cf = modes.ke_sum_mode1.isel(ocean_time=mask_cf).mean(dim='ocean_time').compute()
ke_2_cf = modes.ke_sum_mode2.isel(ocean_time=mask_cf).mean(dim='ocean_time').compute()
ke_3_cf = modes.ke_sum_mode3.isel(ocean_time=mask_cf).mean(dim='ocean_time').compute()
ratio_cf = (ke_2_cf+ke_3_cf)/ke_1_cf

#######-----#hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
ww_hurricane = ww.isel(ocean_time=mask_hurricane).mean(dim='ocean_time').compute()

ke_1_hurricane = modes.ke_sum_mode1.isel(ocean_time=mask_hurricane).mean(dim='ocean_time').compute()
ke_2_hurricane = modes.ke_sum_mode2.isel(ocean_time=mask_hurricane).mean(dim='ocean_time').compute()
ke_3_hurricane = modes.ke_sum_mode3.isel(ocean_time=mask_hurricane).mean(dim='ocean_time').compute()
ratio_hurricane = (ke_2_hurricane+ke_3_hurricane)/ke_1_hurricane



#Negative values are transfer to higher harmonics. Positive values are trasnfer to mean flow 


colors_c = ['Orange', 'cyan', 'Lime', 'grey']  
labels_c = ['A', 'B', 'C','D']

name = f"fig_9_wave_wave.png" 
fig = plt.figure(figsize=(9, 6)) # Slightly increased height for the stacked plots

# Main Grid: 2 Rows, 4 Columns (3 Data Cols + 1 Colorbar Col)
gs = gridspec.GridSpec(nrows=2, ncols=4, width_ratios=[33,33,33,1], height_ratios=[1, 1.2])
gs.update(left=0.05, right=0.97, wspace=0.15, hspace=0.1, top=0.95, bottom=0.07)

# =============================================================================
# ROW 1: MAPS (Normal, Cold Front, Hurricane)
# =============================================================================
# (This section remains largely unchanged, just ensuring it points to gs[0, x])

# --- 1. Normal Map ---
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.text(0.03, 0.08, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax.set_title('Normal', loc='left', fontsize=10)

cmap_map = plt.cm.bwr_r
vmin_map, vmax_map = -0.5, 0.5
norm_map = mpl.colors.Normalize(vmin=vmin_map, vmax=vmax_map)

ax.set_ylim(bottom=-32.6, top=-20)
ax.set_xlim(left=-52, right=-39.2)
ax.contourf(ww_normal.lon_rho, ww_normal.lat_rho, ww_normal*1e3, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0)

# Isobaths & Features
levels = [50, 200, 1000, 2000]
styles = ['dotted', 'dotted', 'dashed', 'solid']
colors_iso = ['gray', 'silver', 'dimgrey', 'gray']
for l, s, c in zip(levels, styles, colors_iso):
    ax.contour(h.lon_rho, h.lat_rho, h, levels=[l], zorder=3, colors=c, linestyles=s, linewidths=1)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}

# Colorbar (Normal)
cbar_1 = inset_axes(ax, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb1.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)



# --- 2. Cold Front Map ---
ax_cf = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax_cf.text(0.03, 0.08, '(b)', transform=ax_cf.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax_cf.set_title('Cold Fronts', loc='left', fontsize=10)
ax_cf.set_ylim(bottom=-32.6, top=-20)
ax_cf.set_xlim(left=-52, right=-39.2)
ax_cf.contourf(ww_cf.lon_rho, ww_cf.lat_rho, ww_cf*1e3, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0)

# Points A, B, C, D
points_indices = [A[1], B[1], C[1], D[1]]
points_labels = ['A', 'B', 'C', 'D']
for idx, lbl, clr in zip(points_indices, points_labels, colors_c):
    # Scatter line
    ax_cf.scatter(h.lon_rho[idx, :], h.lat_rho[idx, :], c=clr, s=0.05, marker='.')
    # Label at end
    ax_cf.text(h.lon_rho[idx, 609]+0.2, h.lat_rho[idx, 609]-1, lbl, color=clr, zorder=5, fontsize='x-small', va='bottom')

for l, s, c in zip(levels, styles, colors_iso):
    ax_cf.contour(h.lon_rho, h.lat_rho, h, levels=[l], zorder=3, colors=c, linestyles=s, linewidths=1)

ax_cf.coastlines()
ax_cf.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
gl = ax_cf.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = gl.right_labels = gl.left_labels = False
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}

cbar_2 = inset_axes(ax_cf, width="60%", height="3%", loc=2)
cbar_2.set_facecolor('lightgray')
cb2 = mpl.colorbar.ColorbarBase(cbar_2, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb2.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)
cbar_2.xaxis.set_ticks_position('bottom')
cbar_2.tick_params(axis='x', labelsize='x-small', rotation=25)

# --- 3. Hurricane Map ---
ax_hc = plt.subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax_hc.text(0.03, 0.08, '(c)', transform=ax_hc.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax_hc.set_title('Hurricane', loc='left', fontsize=10)
ax_hc.set_ylim(bottom=-32.6, top=-20)
ax_hc.set_xlim(left=-52, right=-39.2)
ax_hc.contourf(ww_hurricane.lon_rho, ww_hurricane.lat_rho, ww_hurricane*1e3, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0)

for l, s, c in zip(levels, styles, colors_iso):
    ax_hc.contour(h.lon_rho, h.lat_rho, h, levels=[l], zorder=3, colors=c, linestyles=s, linewidths=1)

ax_hc.coastlines()
ax_hc.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
gl = ax_hc.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = gl.right_labels = gl.left_labels = False
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}


cbar_3 = inset_axes(ax_hc, width="60%", height="3%", loc=2)
cbar_3.set_facecolor('lightgray')
cb3 = mpl.colorbar.ColorbarBase(cbar_3, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb3.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)
cbar_3.xaxis.set_ticks_position('bottom')
cbar_3.tick_params(axis='x', labelsize='x-small', rotation=25)

# =============================================================================
# ROW 2: CROSS SECTIONS (Split into Top: Wave-Wave, Bottom: KE)
# =============================================================================

# Common Cross-section Params
cmap_ke = plt.cm.gist_ncar_r
vmin_ke, vmax_ke = 0, 7
norm_ke = mpl.colors.Normalize(vmin=vmin_ke, vmax=vmax_ke)
bar_title_ke = r'J.m$^{-3}$'

# --- 1. Normal Cross Section (A) ---
# Split gs[1, 0] into two vertical subplots
gs_A = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], height_ratios=[1, 3], hspace=0.08)

# Top: Wave-Wave
ax_ww_A = plt.subplot(gs_A[0])
ax_ww_A.plot(km_cross, ww_normal_a[0:602]*10e3, color='black', linewidth=1, linestyle='-', alpha=0.9)
ax_ww_A.set_ylim(-5, 2)
ax_ww_A.axhline(0, color='grey', linewidth=0.5, linestyle=':')
ax_ww_A.set_ylabel(r'W.m$^{-2}$ $\times$ 10$^{3}$', size='xx-small') # Adjusted label based on scaling
ax_ww_A.tick_params(axis='y', labelsize='xx-small')
ax_ww_A.tick_params(axis='x', labelbottom=False) # Hide x labels
ax_ww_A.set_xlim(230, 609)
ax_ww_A.text(0.02, 0.05, labels_c[0], transform=ax_ww_A.transAxes, fontsize=8, fontweight='bold', color=colors_c[0])

# Bottom: KE Contour (Shares X with Top)
ax_ke_A = plt.subplot(gs_A[1], sharex=ax_ww_A)
for i in range(len(km_cross)):
    ax_ke_A.contourf(km_cross, zcross_A[:,i], ke2_normal_A.isel(s_rho=slice(1,None)), levels=100, cmap=cmap_ke, norm=norm_ke)
ax_ke_A.fill_between(km_cross, np.min(zcross_A[:,0:602], axis=0), y2=ax_ke_A.get_ylim()[0], color='grey', zorder=2)

ax_ke_A.set_ylim(-600, 0)
ax_ke_A.set_xlim(230, 609)
ax_ke_A.set_ylabel('m', size='x-small')
ax_ke_A.set_xlabel('km from coast', size='x-small')
ax_ke_A.tick_params(axis='both', labelsize='x-small')
ax_ke_A.text(0.02, 0.05, '(d)', transform=ax_ke_A.transAxes, fontsize=8, fontweight='bold', color='black')


# --- 2. Cold Front Cross Section (B) ---
gs_B = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[1, 3], hspace=0.08)

# Top: Wave-Wave
ax_ww_B = plt.subplot(gs_B[0], sharey=ax_ww_A) # Share Y with first plot to align scales
ax_ww_B.plot(km_cross, ww_cf_b[0:602]*10e3, color='black', linewidth=1, linestyle='-', alpha=0.9)
ax_ww_B.set_ylim(-5, 2)
ax_ww_B.axhline(0, color='grey', linewidth=0.5, linestyle=':')
ax_ww_B.tick_params(axis='y', labelleft=False) # Hide Y labels
ax_ww_B.tick_params(axis='x', labelbottom=False)
ax_ww_B.set_xlim(70, 609)
ax_ww_B.text(0.02, 0.05, labels_c[1], transform=ax_ww_B.transAxes, fontsize=8, fontweight='bold', color=colors_c[1])

# Bottom: KE Contour
ax_ke_B = plt.subplot(gs_B[1], sharex=ax_ww_B)
for i in range(len(km_cross)):
    ax_ke_B.contourf(km_cross, zcross_B[:,i], ke2_cf_B.isel(s_rho=slice(1,None)), levels=100, cmap=cmap_ke, norm=norm_ke)
ax_ke_B.fill_between(km_cross, np.min(zcross_B[:,0:602], axis=0), y2=ax_ke_B.get_ylim()[0], color='grey', zorder=2)

ax_ke_B.set_ylim(-600, 0)
ax_ke_B.set_xlim(70, 609)
ax_ke_B.set_xlabel('km from coast', size='x-small')
ax_ke_B.tick_params(axis='x', labelsize='x-small')
ax_ke_B.set_yticklabels([]) # Hide Y labels for middle plot
ax_ke_B.text(0.02, 0.05, '(e)', transform=ax_ke_B.transAxes, fontsize=8, fontweight='bold', color='black')


# --- 3. Hurricane Cross Section (D) ---
gs_D = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 2], height_ratios=[1, 3], hspace=0.08)

# Top: Wave-Wave
ax_ww_D = plt.subplot(gs_D[0], sharey=ax_ww_A)
ax_ww_D.plot(km_cross, ww_hurricane_d[0:602]*10e3, color='black', linewidth=1, linestyle='-', alpha=0.9)
ax_ww_D.set_ylim(-5, 2)
ax_ww_D.axhline(0, color='grey', linewidth=0.5, linestyle=':')
ax_ww_D.tick_params(axis='y', labelleft=False)
ax_ww_D.tick_params(axis='x', labelbottom=False)
ax_ww_D.set_xlim(250, 609)
ax_ww_D.text(0.02, 0.05, labels_c[3], transform=ax_ww_D.transAxes, fontsize=8, fontweight='bold', color=colors_c[3])

# Bottom: KE Contour
ax_ke_D = plt.subplot(gs_D[1], sharex=ax_ww_D)
for i in range(len(km_cross)):
    ax_ke_D.contourf(km_cross, zcross_D[:,i], ke2_hurricane_D.isel(s_rho=slice(1,None)), levels=100, cmap=cmap_ke, norm=norm_ke)
ax_ke_D.fill_between(km_cross, np.min(zcross_D[:,0:602], axis=0), y2=ax_ke_D.get_ylim()[0], color='grey', zorder=2)

ax_ke_D.set_ylim(-600, 0)
ax_ke_D.set_xlim(250, 609)
ax_ke_D.set_xlabel('km from coast', size='x-small')
ax_ke_D.tick_params(axis='x', labelsize='x-small')
ax_ke_D.set_yticklabels([])
ax_ke_D.text(0.02, 0.05, '(f)', transform=ax_ke_D.transAxes, fontsize=8, fontweight='bold', color='black')


# --- Colorbar for KE (Rightmost Column) ---
cbar_ke_ax = plt.subplot(gs[1, 3])
cbke = mpl.colorbar.ColorbarBase(cbar_ke_ax, cmap=cmap_ke, norm=norm_ke, extend='max', orientation='vertical')
cbke.set_label(bar_title_ke, size='x-small', labelpad=5)
cbar_ke_ax.yaxis.set_ticks_position('left')
cbar_ke_ax.tick_params(axis='y', labelsize='x-small')


plt.savefig(name, dpi=300)
