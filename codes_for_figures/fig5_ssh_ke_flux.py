

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
#################------->Normalised vorticity <------########################


f_p = '/data1/roms_dd_waves/analysis_outs/NIW/vorticity/'
vort = xr.open_mfdataset(f_p + 'relative_vorticity_*.nc').zeta.isel(s_rho=-1,ocean_time=slice(96,None))

dzs = '/data1/roms_dd_waves/analysis_outs/dz/niws/'

z = xr.open_mfdataset(dzs + f'dz_*.nc').isel(s_rho = 0).z_rho #bottom
lon_rho = z.lon_rho.compute()
lat_rho = z.lat_rho.compute()
h = z.compute()*-1


f_sbb = gsw.f(lat_rho.values) #s⁻1


dates = vort.ocean_time.to_index()

###########------normal

mask_normal = create_date_mask(dates, date_ranges_normal)
vort_normal = vort.isel(ocean_time=mask_normal).mean(dim='ocean_time').compute()/f_sbb

###########------cf
mask_cf = create_date_mask(dates, date_ranges_cf)
vort_cf = vort.isel(ocean_time=mask_cf).mean(dim='ocean_time').compute()/f_sbb


#######-----#hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
vort_hurricane = vort.isel(ocean_time=mask_hurricane).mean(dim='ocean_time').compute()/f_sbb


############################################################################
#################-------> EKE arrays <------################################
e_p = '/data1/roms_dd_waves/analysis_outs/NIW/eke/'

nike = xr.open_mfdataset(e_p + 'nike_mld_*.nc').nike_mld.isel(ocean_time=slice(96,None)) * 1025
mld_ke = nike.s_rho.isel(ocean_time=slice(96,None)).mean(dim='ocean_time').compute()

#Normal
mask_normal = create_date_mask(dates, date_ranges_normal)
nike_normal = nike.isel(ocean_time=mask_normal).mean(dim='ocean_time').compute()*200

#cf
mask_cf = create_date_mask(dates, date_ranges_cf)
nike_cf = nike.isel(ocean_time=mask_cf).mean(dim='ocean_time').compute()*200

#hurricane

mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
nike_hurricane = nike.isel(ocean_time=mask_hurricane).mean(dim='ocean_time').compute()*200





############################################################################
#################------->Effective vorticity <------########################

f_p = '/data1/roms_dd_waves/analysis_outs/NIW/vorticity/'
fe = xr.open_mfdataset(f_p + 'f_eff_slice_*.nc').f_eff.isel(ocean_time=slice(96,None))


############################################################################
##############------->Cross-sections per class of depht J/m² <------########

Ha = -1*h.isel(eta_rho=A[1],xi_rho=slice(0,None))
#-Bs
Hb = -1*h.isel(eta_rho=B[1],xi_rho=slice(0,None))
#-C
Hc = -1*h.isel(eta_rho=C[1],xi_rho=slice(0,None))
#-D
Hd = -1*h.isel(eta_rho=C[1],xi_rho=slice(0,None))


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
nike_normal_a = nike_normal.isel(eta_rho=A[1],xi_rho=slice(0,None))
D_normal_A = calculate_depth_distribution(nike_normal_a, Ha, depth_bins)

nike_cf_a = nike_cf.isel(eta_rho=A[1],xi_rho=slice(0,None))
D_cf_A = calculate_depth_distribution(nike_cf_a, Ha, depth_bins)

nike_hurricane_a = nike_hurricane.isel(eta_rho=A[1],xi_rho=slice(0,None))
D_hurricane_A = calculate_depth_distribution(nike_hurricane_a, Ha, depth_bins)

#-----------------B
nike_normal_b = nike_normal.isel(eta_rho=B[1],xi_rho=slice(0,None))
D_normal_B = calculate_depth_distribution(nike_normal_b, Ha, depth_bins)

nike_cf_b = nike_cf.isel(eta_rho=B[1],xi_rho=slice(0,None))
D_cf_B = calculate_depth_distribution(nike_cf_b, Ha, depth_bins)

nike_hurricane_b = nike_hurricane.isel(eta_rho=B[1],xi_rho=slice(0,None))
D_hurricane_B = calculate_depth_distribution(nike_hurricane_b, Ha, depth_bins)

#---------------C
nike_normal_c = nike_normal.isel(eta_rho=C[1],xi_rho=slice(0,None))
D_normal_C = calculate_depth_distribution(nike_normal_c, Ha, depth_bins)

nike_cf_c = nike_cf.isel(eta_rho=C[1],xi_rho=slice(0,None))
D_cf_C = calculate_depth_distribution(nike_cf_c, Ha, depth_bins)

nike_hurricane_c = nike_hurricane.isel(eta_rho=C[1],xi_rho=slice(0,None))
D_hurricane_C = calculate_depth_distribution(nike_hurricane_c, Ha, depth_bins)

#---------------D
nike_normal_d = nike_normal.isel(eta_rho=D[1],xi_rho=slice(0,None))
D_normal_D = calculate_depth_distribution(nike_normal_d, Ha, depth_bins)

nike_cf_d = nike_cf.isel(eta_rho=D[1],xi_rho=slice(0,None))
D_cf_D = calculate_depth_distribution(nike_cf_d, Ha, depth_bins)

nike_hurricane_d = nike_hurricane.isel(eta_rho=D[1],xi_rho=slice(0,None))
D_hurricane_D = calculate_depth_distribution(nike_hurricane_d, Ha, depth_bins)








"""




##################-----> Plotting <------###################################
colors = ['Orange', 'fuchsia', 'Lime', 'dodgerblue']  
labels = ['A', 'B', 'C','D']
name = f"fig_5_vort_ke.png"  # Create a unique filename

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1,1,1], height_ratios=[1,1])
gs.update(left=0.1, right=0.97, wspace=0.20, hspace=0.1, top=0.96, bottom=0.1)

######----> Normal
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.text(0.03, 0.08, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax.set_title('Normal', loc='left', fontsize=10)
# Define colormap and normalization
cmap = plt.cm.bwr

#cmap = plt.cm.gist_ncar_r
vmin = -0.15
vmax = 0.15
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
#bar_title = r'W.m$^{-2}$'
bar_title = ''

ax.set_ylim(bottom=-32.6, top=-20)
ax.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax.contourf(lon_rho, lat_rho, vort_normal, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

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
cbar_1 = inset_axes(ax, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)

######----> Cold Front
ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.text(0.03, 0.08, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax2.set_title('Cold Front', loc='left', fontsize=10)
# Define colormap and normalization
cmap = plt.cm.bwr

#cmap = plt.cm.gist_ncar_r
vmin = -0.15
vmax = 0.15
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
#bar_title = r'W.m$^{-2}$'
bar_title = ''

ax2.set_ylim(bottom=-32.6, top=-20)
ax2.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax2.contourf(lon_rho, lat_rho, vort_cf, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

ax2.scatter(lon_rho[A[1],A[0]], lat_rho[A[1],A[0]], zorder=5, s=20, marker='*', color='k', label='A')
ax2.scatter(lon_rho[B[1],B[0]], lat_rho[B[1],B[0]], zorder=5, s=20, marker='+', color='blue', label='B')
ax2.scatter(lon_rho[C[1],C[0]], lat_rho[C[1],C[0]], zorder=5, s=20, marker='s', color='orange', label='C')
ax2.scatter(lon_rho[D[1],D[0]], lat_rho[D[1],D[0]], zorder=5, s=20, marker='D', color='red', label='D')

legend = ax2.legend(loc=4, fontsize='xx-small', markerscale=0.7)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax2.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax2.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax2.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax2.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
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
ax2.coastlines()
ax2.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax2.patch.set_edgecolor('black')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
cbar_1 = inset_axes(ax2, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)



######----> Cold Front
ax3 = plt.subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax3.text(0.03, 0.08, '(c)', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax3.set_title('Hurricane', loc='left', fontsize=10)
# Define colormap and normalization
cmap = plt.cm.bwr

#cmap = plt.cm.gist_ncar_r
vmin = -0.15
vmax = 0.15
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
#bar_title = r'W.m$^{-2}$'
bar_title = ''

ax3.set_ylim(bottom=-32.6, top=-20)
ax3.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax3.contourf(lon_rho, lat_rho, vort_hurricane, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)


ax3.scatter(lon_rho[A[1],:], lat_rho[A[1],:],c='grey',s=0.05,marker='.')
ax3.scatter(lon_rho[B[1],:], lat_rho[B[1],:],c='grey',s=0.05,marker='.')
ax3.scatter(lon_rho[C[1],:], lat_rho[C[1],:],c='grey',s=0.05,marker='.')
ax3.scatter(lon_rho[D[1],:], lat_rho[D[1],:],c='grey',s=0.05,marker='.')

ax3.text(lon_rho[A[1], 609]+0.2, lat_rho[A[1], 609]-1, 'A', color='grey',zorder=5, fontsize='x-small', verticalalignment='bottom')
ax3.text(lon_rho[[B[1]], 609]+0.2, lat_rho[[B[1]], 609]-1, 'B', color='grey', zorder=5,fontsize='x-small', verticalalignment='bottom')
ax3.text(lon_rho[C[1], 609]+0.2, lat_rho[C[1], 609]-1, 'C', color='grey', zorder=5,fontsize='x-small', verticalalignment='bottom')
ax3.text(lon_rho[D[1], 609]+0.2, lat_rho[D[1], 609]-1, 'D', color='grey', zorder=5,fontsize='x-small', verticalalignment='bottom')

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax3.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax3.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax3.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax3.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
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
ax3.coastlines()
ax3.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax3.patch.set_edgecolor('black')
gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
cbar_1 = inset_axes(ax3, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)



######----> Normal
ax4 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax4.text(0.03, 0.08, '(d)', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization
#cmap = plt.cm.ocean_r
cmap = plt.cm.gist_ncar_r
vmin = 0
vmax = 3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'kJ.m$^{-2}$'

ax4.set_ylim(bottom=-32.6, top=-20)
ax4.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax4.contourf(lon_rho, lat_rho, nike_normal/1000, levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax4.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax4.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax4.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax4.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
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
ax4.coastlines()
ax4.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax4.patch.set_edgecolor('black')
gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
cbar_1 = inset_axes(ax4, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)


######----> cf
ax5 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax5.text(0.03, 0.08, '(e)', transform=ax5.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization
#cmap = plt.cm.ocean_r
cmap = plt.cm.gist_ncar_r
vmin = 0
vmax = 3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'kJ.m$^{-2}$'

ax5.set_ylim(bottom=-32.6, top=-20)
ax5.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax5.contourf(lon_rho, lat_rho, nike_cf/1000, levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax5.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax5.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax5.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax5.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
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
ax5.coastlines()
ax5.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax5.patch.set_edgecolor('black')
gl = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
cbar_1 = inset_axes(ax5, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)

######----> hurricane
ax6 = plt.subplot(gs[1, 2], projection=ccrs.PlateCarree())
ax6.text(0.03, 0.08, '(f)', transform=ax6.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization
#cmap = plt.cm.ocean_r
cmap = plt.cm.gist_ncar_r
vmin = 0
vmax = 3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'kJ.m$^{-2}$'

ax6.set_ylim(bottom=-32.6, top=-20)
ax6.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax6.contourf(lon_rho, lat_rho, nike_hurricane/1000, levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax6.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax6.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax6.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax6.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
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
ax6.coastlines()
ax6.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax6.patch.set_edgecolor('black')
gl = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
cbar_1 = inset_axes(ax6, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)


plt.savefig(name, dpi=300)



