

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

####################################################################################################
##################--------> MAPS of dissipation <-------------------################################


d_p = '/data1/roms_dd_waves/analysis_outs/NIW/viscous/'
diss = xr.open_mfdataset(d_p + 'visc_loss_*.nc').viscous.isel(ocean_time=slice(1,None))[:,:,22:,:-23]

dzs = '/data1/roms_dd_waves/analysis_outs/dz/niws/'
z = xr.open_mfdataset(dzs + f'dz_*.nc').isel(s_rho = 0).z_rho[22:,:-23] #bottom
lon_rho = z.lon_rho.compute()
lat_rho = z.lat_rho.compute()
h = z.compute()*-1



dates = diss.ocean_time.to_index()

###########------normal

mask_normal = create_date_mask(dates, date_ranges_normal)
diss_normal = diss.isel(ocean_time=mask_normal).sum(dim='s_rho').mean(dim='ocean_time').compute()

###########------cf
mask_cf = create_date_mask(dates, date_ranges_cf)
diss_cf = diss.isel(ocean_time=mask_cf).sum(dim='s_rho').mean(dim='ocean_time').compute()


#######-----#hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
diss_hurricane = diss.isel(ocean_time=mask_hurricane).sum(dim='s_rho').mean(dim='ocean_time').compute()





####################################################################################################
##################--------> Norm vorticity <-------------------################################


f_p = '/data1/roms_dd_waves/analysis_outs/NIW/vorticity/'
vort = xr.open_mfdataset(f_p + 'relative_vorticity_*.nc').zeta.isel(s_rho=-1,ocean_time=slice(0,-1,3))[:,22:,:-23]


f_sbb = gsw.f(lat_rho.values) #s⁻1


###########------normal

mask_normal = create_date_mask(dates, date_ranges_normal)
vort_normal = vort.isel(ocean_time=mask_normal).mean(dim='ocean_time').compute()/np.absolute(f_sbb)

###########------cf
mask_cf = create_date_mask(dates, date_ranges_cf)
vort_cf = vort.isel(ocean_time=mask_cf).mean(dim='ocean_time').compute()/np.absolute(f_sbb)


#######-----#hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
vort_hurricane = vort.isel(ocean_time=mask_hurricane).mean(dim='ocean_time').compute()/np.absolute(f_sbb)





####################################################################################################
##################--------> wp at mld <-------------------################################


wp_normal = xr.open_dataset('wp_mld_normal.nc').wp[22:,:-23]
wp_cf = xr.open_dataset('wp_mld_cf.nc').wp[22:,:-23]
wp_hurricane = xr.open_dataset('wp_mld_hurricane.nc').wp[22:,:-23]




####################################################################################################
##################--------> Wave-Mean_flow at mld <-------------------##############################
#
#wm_p = '/data1/roms_dd_waves/analysis_outs/NIW/EP_flux/'
#wm = xr.open_mfdataset(wm_p + 'mean_flow_wave_slice_*.nc').mean_flow_wave.isel(ocean_time=slice(0,-1,3))
#
#dates = wm.ocean_time.to_index()
#
############------normal
#
#mask_normal = create_date_mask(dates, date_ranges_normal)
#wm_normal = wm.isel(ocean_time=mask_normal).mean(dim='ocean_time').compute()
#
#mask_pos_normal_wm = (wm_normal > 0)
#mask_neg_normal_wm = (wm_normal < 0)
#
#
#
############------cf
#mask_cf = create_date_mask(dates, date_ranges_cf)
#wm_cf = wm.isel(ocean_time=mask_cf).mean(dim='ocean_time').compute()
#
#
########-----#hurricane
#mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
#wm_hurricane = wm.isel(ocean_time=mask_hurricane).mean(dim='ocean_time').compute()


####################################################################################################
##################-------->conditions  <-------------------################################


###---> Normal
mask_pos_normal = (vort_normal > 0)
mask_neg_normal = (vort_normal < 0)

# --- Positive Vorticity (Cyclonic) ---
diss_pos_masked_normal = diss_normal.where(mask_pos_normal).values.flatten()
wp_pos_masked_normal = wp_normal.where(mask_pos_normal).values.flatten()

# Remove NaNs introduced by the masking
diss_pos_normal = diss_pos_masked_normal[~np.isnan(diss_pos_masked_normal)]
wp_pos_normal = wp_pos_masked_normal[~np.isnan(wp_pos_masked_normal)]

# --- Negative Vorticity (Anticyclonic) ---
diss_neg_masked_normal = diss_normal.where(mask_neg_normal).values.flatten()
wp_neg_masked_normal = wp_normal.where(mask_neg_normal).values.flatten()

# Remove NaNs introduced by the masking
diss_neg_normal = diss_neg_masked_normal[~np.isnan(diss_neg_masked_normal)]
wp_neg_normal = wp_neg_masked_normal[~np.isnan(wp_neg_masked_normal)]


# 3. Create a DataFrame for CSV output (for easy use of the extracted data)
df_pos_normal = pd.DataFrame({
	'vorticity_sign': 'positive',
	'dissipation': diss_pos_normal,
	'wp_prime': wp_pos_normal
})
df_neg_normal = pd.DataFrame({
	'vorticity_sign': 'negative',
	'dissipation': diss_neg_normal,
	'wp_prime': wp_neg_normal
})
df_combined_normal = pd.concat([df_pos_normal, df_neg_normal], ignore_index=True)



###--------------------------------------> Cold front
mask_pos_cf = (vort_cf > 0)
mask_neg_cf = (vort_cf < 0)

# --- Positive Vorticity (Cyclonic) ---
diss_pos_masked_cf = diss_cf.where(mask_pos_cf).values.flatten()
wp_pos_masked_cf = wp_cf.where(mask_pos_cf).values.flatten()

# Remove NaNs introduced by the masking
diss_pos_cf = diss_pos_masked_cf[~np.isnan(diss_pos_masked_cf)]
wp_pos_cf = wp_pos_masked_cf[~np.isnan(wp_pos_masked_cf)]

# --- Negative Vorticity (Anticyclonic) ---
diss_neg_masked_cf = diss_cf.where(mask_neg_cf).values.flatten()
wp_neg_masked_cf = wp_cf.where(mask_neg_cf).values.flatten()

# Remove NaNs introduced by the masking
diss_neg_cf = diss_neg_masked_cf[~np.isnan(diss_neg_masked_cf)]
wp_neg_cf = wp_neg_masked_cf[~np.isnan(wp_neg_masked_cf)]


# 3. Create a DataFrame for CSV output (for easy use of the extracted data)
df_pos_cf = pd.DataFrame({
	'vorticity_sign': 'positive',
	'dissipation': diss_pos_cf,
	'wp_prime': wp_pos_cf
})
df_neg_cf = pd.DataFrame({
	'vorticity_sign': 'negative',
	'dissipation': diss_neg_cf,
	'wp_prime': wp_neg_cf
})
df_combined_cf = pd.concat([df_pos_cf, df_neg_cf], ignore_index=True)


###--------------------------------------> Hurricane
mask_pos_hurricane = (vort_hurricane > 0)
mask_neg_hurricane = (vort_hurricane < 0)


# --- Positive Vorticity (Cyclonic) ---
diss_pos_masked_hurricane = diss_hurricane.where(mask_pos_hurricane).values.flatten()
wp_pos_masked_hurricane = wp_hurricane.where(mask_pos_hurricane).values.flatten()

# Remove NaNs introduced by the masking
diss_pos_hurricane = diss_pos_masked_hurricane[~np.isnan(diss_pos_masked_hurricane)]
wp_pos_hurricane = wp_pos_masked_hurricane[~np.isnan(wp_pos_masked_hurricane)]

# --- Negative Vorticity (Anticyclonic) ---
diss_neg_masked_hurricane = diss_hurricane.where(mask_neg_hurricane).values.flatten()
wp_neg_masked_hurricane = wp_hurricane.where(mask_neg_hurricane).values.flatten()

# Remove NaNs introduced by the masking
diss_neg_hurricane = diss_neg_masked_hurricane[~np.isnan(diss_neg_masked_hurricane)]
wp_neg_hurricane = wp_neg_masked_hurricane[~np.isnan(wp_neg_masked_hurricane)]


# 3. Create a DataFrame for CSV output (for easy use of the extracted data)
df_pos_hurricane = pd.DataFrame({
	'vorticity_sign': 'positive',
	'dissipation': diss_pos_hurricane,
	'wp_prime': wp_pos_hurricane
})
df_neg_hurricane = pd.DataFrame({
	'vorticity_sign': 'negative',
	'dissipation': diss_neg_hurricane,
	'wp_prime': wp_neg_hurricane
})
df_combined_hurricane = pd.concat([df_pos_hurricane, df_neg_hurricane], ignore_index=True)














##################-----> Plotting <------###################################
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, NullFormatter, FixedLocator

name = f"fig_7_diss_norm_vort.png"  # Create a unique filename

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1,1,1], height_ratios=[1,1])
gs.update(left=0.1, right=0.97, wspace=0.20, hspace=0.1, top=0.96, bottom=0.1)


######----> Normal
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.text(0.03, 0.08, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax.set_title('Normal', loc='left', fontsize=10)
# Define colormap and normalization
cmap = plt.cm.cubehelix_r

vmin = 0
vmax = 1
# Changed to LogNorm
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

#norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax) 

# Define other parameters
bar_title = r'x10$^{-3}$ W.m$^{-2}$'

ax.set_ylim(bottom=-32.6, top=-20)
ax.set_xlim(left=-52, right=-39.2)

# Contourf with logarithmic normalization
ax.contourf(lon_rho, lat_rho, diss_normal*50*1000, levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)
# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='xx-small',
	title_fontsize='xx-small',
	loc='center',
	bbox_to_anchor=(0.3, 0.65)
)

# Coastlines and gridlines
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax.patch.set_edgecolor('black')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


# Colorbar
cbar_1 = inset_axes(
	ax, 
	width="90%", 
	height="100%", 
	loc=2, 
	bbox_to_anchor=[0.05, 0.95, 0.60, 0.03],
	bbox_transform=ax.transAxes
)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)

#---> scatter
# The first line is correct for creating the Axes object (sc_normal)
sc_normal = plt.subplot(gs[1, 0]) 
sc_normal.text(0.03, 0.08, '(b)', transform=sc_normal.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

sc_normal.scatter(wp_pos_normal*500, diss_pos_normal*50, marker='D',s=5, alpha=0.5, label='($\zeta$/f > 0)', color='firebrick')
sc_normal.scatter(wp_neg_normal*500, diss_neg_normal*50, s=5, alpha=0.3, label=' ($\zeta$/f < 0)', color='royalblue')

sc_normal.set_xlabel(r"$w'p'$ @ mld  W.m$^{-2}$ ",fontsize='x-small')
sc_normal.set_ylabel(r"Dissipation  W.m$^{-2}$ ",fontsize='x-small')
sc_normal.legend(loc='best', fontsize='x-small')
sc_normal.grid(True, linestyle='--', alpha=0.6)
sc_normal.set_xlim(left=1e-9, right=3e-2)

sc_normal.set_xscale('log')
sc_normal.set_yscale('log')
sc_normal.tick_params(axis='both', labelsize='x-small', rotation=25)

######----> CF
ax_cf = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax_cf.text(0.03, 0.08, '(c)', transform=ax_cf.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax_cf.set_title('Cold Front', loc='left', fontsize=10)
# Define colormap and normalization
cmap = plt.cm.cubehelix_r
vmin = 0
vmax = 1
# Changed to LogNorm
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

#norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax) 

# Define other parameters
bar_title = r'x10$^{-3}$ W.m$^{-2}$'

ax_cf.set_ylim(bottom=-32.6, top=-20)
ax_cf.set_xlim(left=-52, right=-39.2)

# Contourf with logarithmic normalization
ax_cf.contourf(lon_rho, lat_rho, diss_cf*50*1000, levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax_cf.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax_cf.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax_cf.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax_cf.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='xx-small',
	title_fontsize='xx-small',
	loc='center',
	bbox_to_anchor=(0.3, 0.65)
)

# Coastlines and gridlines
ax_cf.coastlines()
ax_cf.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax_cf.patch.set_edgecolor('black')
gl = ax_cf.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


# Colorbar
cbar_1 = inset_axes(
	ax_cf, 
	width="90%", 
	height="100%", 
	loc=2, 
	bbox_to_anchor=[0.05, 0.95, 0.60, 0.03],
	bbox_transform=ax_cf.transAxes
)


cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)

#---> scatter
# The first line is correct for creating the Axes object (sc_cf)
sc_cf = plt.subplot(gs[1, 1]) 
sc_cf.text(0.03, 0.08, '(d)', transform=sc_cf.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

sc_cf.scatter(wp_pos_cf*500, diss_pos_cf*50, marker='D',s=5, alpha=0.5, label='($\zeta$/f > 0)', color='firebrick')
sc_cf.scatter(wp_neg_cf*500, diss_neg_cf*50, s=5, alpha=0.3, label=' ($\zeta$/f < 0)', color='royalblue')

sc_cf.set_xlabel(r"$w'p'$ @ mld  W.m$^{-2}$ ",fontsize='x-small')
#sc_normal.set_ylabel(r"Dissipation  W.m$^{-2}$' ",fontsize='x-small')
sc_cf.set_ylabel(None)
#sc_cf.legend(loc='best', fontsize='x-small')
sc_cf.grid(True, linestyle='--', alpha=0.6)
sc_cf.set_xlim(left=1e-9, right=3e-2)

sc_cf.set_xscale('log')
sc_cf.set_yscale('log')
sc_cf.tick_params(axis='both', labelsize='x-small', rotation=25)


######----> Hurricane
ax_hr = plt.subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax_hr.text(0.03, 0.08, '(e)', transform=ax_hr.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax_hr.set_title('Hurricane', loc='left', fontsize=10)
# Define colormap and normalization
cmap = plt.cm.cubehelix_r
vmin = 0
vmax = 1
# Changed to LogNorm
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

#norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax) 

# Define other parameters
bar_title = r'x10$^{-3}$ W.m$^{-2}$'

ax_hr.set_ylim(bottom=-32.6, top=-20)
ax_hr.set_xlim(left=-52, right=-39.2)

# Contourf with logarithmic normalization
ax_hr.contourf(lon_rho, lat_rho, diss_hurricane*50*1000, levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)


# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax_hr.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax_hr.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax_hr.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax_hr.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='xx-small',
	title_fontsize='xx-small',
	loc='center',
	bbox_to_anchor=(0.3, 0.65)
)

# Coastlines and gridlines
ax_hr.coastlines()
ax_hr.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax_hr.patch.set_edgecolor('black')
gl = ax_hr.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


# Colorbar
cbar_1 = inset_axes(
	ax_hr, 
	width="90%", 
	height="100%", 
	loc=2, 
	bbox_to_anchor=[0.05, 0.95, 0.60, 0.03],
	bbox_transform=ax_hr.transAxes
)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)


#---> scatter
# The first line is correct for creating the Axes object (sc_hr)
sc_hr = plt.subplot(gs[1, 2]) 
sc_hr.text(0.03, 0.08, '(f)', transform=sc_hr.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

sc_hr.scatter(wp_pos_hurricane*500, diss_pos_hurricane*50, marker='D',s=5, alpha=0.5, label='($\zeta$/f > 0)', color='firebrick')
sc_hr.scatter(wp_neg_hurricane*500, diss_neg_hurricane*50, s=5, alpha=0.3, label=' ($\zeta$/f < 0)', color='royalblue')

sc_hr.set_xlabel(r"$w'p'$ @ mld  W.m$^{-2}$ ",fontsize='x-small')
sc_hr.set_ylabel(None)
#sc_hr.legend(loc='best', fontsize='x-small')
sc_hr.grid(True, linestyle='--', alpha=0.6)
sc_hr.set_xlim(left=1e-9, right=3e-2)

sc_hr.set_xscale('log')
sc_hr.set_yscale('log')
sc_hr.tick_params(axis='both', labelsize='x-small', rotation=25)

plt.savefig(name, dpi=300)

