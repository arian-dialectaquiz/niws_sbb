
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
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, NullFormatter, FixedLocator

#######################################################################################################################
################### ----- MAP -----####################################################################################

dzs = '/data1/roms_dd_waves/analysis_outs/dz/'
path = '/data1/roms_dd_waves/analysis_outs/NIW/EP_flux/'

dz = xr.open_mfdataset(dzs + f'dz_*.nc')


z = xr.open_mfdataset(dzs + f'dz_*.nc').isel(s_rho = 0).z_rho #bottom
lon_rho = z.lon_rho.compute()
lat_rho = z.lat_rho.compute()

## EP flux
EP1 = xr.open_mfdataset(path + 'mean_flow_wave*_1.nc').mean_flow_wave
EP2 = xr.open_mfdataset(path + 'mean_flow_wave*_2.nc').mean_flow_wave
EP3 = xr.open_mfdataset(path + 'mean_flow_wave*_3.nc').mean_flow_wave
EP4 = xr.open_mfdataset(path + 'mean_flow_wave*_4.nc').mean_flow_wave


EP = xr.concat([EP1, EP2, EP3, EP4], dim='ocean_time')
EP = EP.sortby('ocean_time')
EP = EP.sel(ocean_time=EP.ocean_time.drop_duplicates(dim='ocean_time'))


#EP_int = EP.sum(dim='s_rho').compute()
#EP_int_file = EP_int.to_dataset(name='EP_z')		
#EP_int_file.to_netcdf('EP_dz_sum.nc')
#EP_time = EP_int.mean(dim='ocean_time').compute()
#EP_tfile = EP_time.to_dataset(name='EP_tz')		
#EP_tfile.to_netcdf('EP_time_dz_sum.nc')



EP_int = xr.open_dataset('EP_dz_sum.nc').EP_z
EP_time = xr.open_dataset('EP_time_dz_sum.nc').EP_tz

EP = -1* EP
EP_time = -1* EP_time
EP_int = EP_int *-1


#######################################################################################################################
################### ----- A -----####################################################################################

ii, jj = 905, 380

## - point per time
EP_A_p = EP_int.isel(eta_rho=ii,xi_rho=jj)


## - point per time per z
EP_A_z = EP.isel(eta_rho=ii,xi_rho=jj,s_rho=slice(1,40)).compute()

z_a = dz['z_rho'].isel(eta_rho=ii, xi_rho=jj)  

### - cross section
#EP_A = EP.isel(eta_rho=905,xi_rho=slice(0,None),s_rho=slice(0,-1)).compute().mean(dim='ocean_time')
#zcross_A = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=905,xi_rho=slice(0,-1)).z_rho.values



#######################################################################################################################
################### ----- B -----####################################################################################
ii, jj = 500, 238


## - point per time
EP_B_p = EP_int.isel(eta_rho=ii,xi_rho=jj)

EP_B_z = EP.isel(eta_rho=ii,xi_rho=jj,s_rho=slice(1,40)).compute()

z_b = dz['z_rho'].isel(eta_rho=ii, xi_rho=jj)  

### - cross section
#EP_B = EP.isel(eta_rho=500,xi_rho=slice(0,None),s_rho=slice(0,-1)).compute().mean(dim='ocean_time')
#zcross_B = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=500,xi_rho=slice(0,-1)).z_rho.values





#######################################################################################################################
################### ----- C -----####################################################################################

ii, jj = 280, 221

## - point per time
EP_C_p = EP_int.isel(eta_rho=ii,xi_rho=jj)

EP_C_z = EP.isel(eta_rho=ii,xi_rho=jj,s_rho=slice(1,40)).compute()

z_c = dz['z_rho'].isel(eta_rho=ii, xi_rho=jj)  

## - cross section
#EP_C = EP.isel(eta_rho=280,xi_rho=slice(0,None),s_rho=slice(0,-1)).compute().mean(dim='ocean_time')
#zcross_C = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=280,xi_rho=slice(0,-1)).z_rho.values


#######################################################################################################################
################### ----- D -----####################################################################################
ii, jj = 120, 275


## - point per time
EP_D_p = EP_int.isel(eta_rho=ii,xi_rho=jj)

EP_D_z = EP.isel(eta_rho=ii,xi_rho=jj,s_rho=slice(1,40)).compute()

z_d = dz['z_rho'].isel(eta_rho=ii, xi_rho=jj) 
## - cross section
#EP_D = EP.isel(eta_rho=120,xi_rho=slice(0,None),s_rho=slice(0,-1)).compute().mean(dim='ocean_time')
#zcross_D = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=120,xi_rho=slice(0,-1)).z_rho.values



##################-----> Plotting <------###################################
#km_cross_A = np.arange(0,429,1)
#km_cross_B = km_cross_A 
#km_cross_C = km_cross_A 
#km_cross_D = km_cross_A 
zcross_A = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=905,xi_rho=slice(0,-1)).z_rho.values

zcross_B = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=500,xi_rho=slice(0,-1)).z_rho.values

zcross_C = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=280,xi_rho=slice(0,-1)).z_rho.values

zcross_D = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=120,xi_rho=slice(0,-1)).z_rho.values

time = EP_A_z.ocean_time


##################-----> Cold fronts <------###################################

date_ranges = [
	("2001-12-10", "2001-12-16"),
	("2001-12-23", "2001-12-27"),
	("2001-12-30", "2002-01-02"),
	("2002-01-06", "2002-01-10"),
	("2002-01-15", "2002-01-20"),
	("2002-01-24", "2002-01-25"),
	("2002-01-28", "2002-01-30"),
	("2002-02-14", "2002-02-22"),
	("2002-03-19", "2002-03-22"),
]

dates = time.to_index()

# Get indices and datetime values that fall in each range
selected_idx = []
selected_dates = []

for start, end in date_ranges:
	mask = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
	idx = np.where(mask)[0]        # positions
	vals = dates[mask]             # datetime values
	
	selected_idx.append(idx)
	selected_dates.append(vals)
	

all_idx = np.concatenate(selected_idx)
dates_cf = time.isel(ocean_time=all_idx)



#########_----> plotting
name = f"FIG_5_EP_MAP_time.png"  # Create a unique filename

fig = plt.figure(figsize=(6, 7))
gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[50,50], height_ratios=[1,1,1,1])
gs.update(left=0.08, right=0.99, wspace=0.20, hspace=0.2, top=0.98, bottom=0.08)

ax = plt.subplot(gs[0:2, 0], projection=ccrs.PlateCarree())
# Define colormap and normalization
cmap = plt.cm.bwr

#cmap = cmo.cm.balance
vmin = -1
vmax = 1
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'W.m$^{-2}$'
ax.set_ylim(bottom=-30.5, top=-22)
ax.set_xlim(left=-49.2, right=-40.9)
# Contourf with normalization
ax.contourf(EP_time.lon_rho, EP_time.lat_rho, EP_time*1000, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

ax.scatter(z.lon_rho[905,380].values.mean(), z.lat_rho[905,380].values.mean(), zorder=5, s=30, marker='*', color='green', label='A')
ax.scatter(z.lon_rho[500,238].values.mean() , z.lat_rho[500,238].values.mean(), zorder=5, s=30, marker='+', color='k', label='B')
ax.scatter(z.lon_rho[280,221].values.mean(), z.lat_rho[280,221].values.mean(), zorder=5, s=30, marker='s', color='orange', label='C')
ax.scatter(z.lon_rho[120,275].values.mean(), z.lat_rho[120,275].values.mean(), zorder=5, s=30, marker='D', color='cyan', label='D')

legend = ax.legend(loc=4, fontsize='xx-small', markerscale=0.7)

# Depth contour levels
levels_1 = [-50]
levels_2 = [-200]
levels_3 = [-1000]
levels_4 = [-2000]
# Contour lines for isobaths
ax.contour(lon_rho, lat_rho, z, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax.contour(lon_rho, lat_rho, z, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax.contour(lon_rho, lat_rho, z, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax.contour(lon_rho, lat_rho, z, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
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
	bbox_to_anchor=(0.443, 0.72)
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


ax.text(0.90, 0.95, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))
# Colorbar
cbar_1 = inset_axes(ax, width="50%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb1.set_label(bar_title, size='x-small')
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)
cb1.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)

##############################################################################
####--- Per time at midpoints
ax1 = plt.subplot(gs[2:4, 0])

ax1.plot(EP_A_p.ocean_time, EP_A_p*100, zorder=2,color='green',label='A')  
ax1.plot(EP_A_p.ocean_time, EP_B_p*100, zorder=2,color='k',label='B')            
ax1.plot(EP_A_p.ocean_time, EP_C_p*100, zorder=2,color='gold',label='C')      
ax1.plot(EP_A_p.ocean_time, EP_D_p*100, color='cyan',label='D')       

ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))  # reduce number of ticks

ax1.tick_params(axis='y', labelsize='x-small')
ax1.text(0.90, 0.95, 'b', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))
# X ticks: concise dates, rotated small
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=20))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(ax1.get_xticklabels(), rotation=15, ha='center', fontsize='x-small')

# Grid & spines
ax1.grid(True, which='major', alpha=0.3, linewidth=0.6)
ax1.grid(True, which='minor', alpha=0.15, linewidth=0.5)
ax1.minorticks_on()
for spine in ['top', 'right']:
	ax1.spines[spine].set_visible(False)

for start, end in date_ranges:
	ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end),
				color='purple', alpha=0.2, zorder=1, label="_nolegend_")

# Add a legend entry only once
ax1.axvspan(pd.to_datetime(date_ranges[0][0]), pd.to_datetime(date_ranges[0][1]),
			color='purple', alpha=0.2, zorder=1, label='Cold Fronts')
ax1.legend(loc='best', fontsize='x-small', frameon=True)

# Label sizes
#ax1.set_ylabel(r'W.m$^{-2}$', fontsize='small')  # (same units as colorbar)
ax1.set_ylabel(fr'{bar_title} $\times$ 10$^{{-2}}$', size='x-small')

plt.rcParams['font.family'] = 'Times New Roman'


##############################################################################
####--- depth x time

cmap_N = plt.cm.bwr
vmin_N = -0.5
vmax_N = 0.5
norm_N = mpl.colors.Normalize(vmin=vmin_N, vmax=vmax_N)
bar_title =  r'W.m$^{-3}$'
#A
axa = plt.subplot(gs[0, 1])
axa.set_ylim(-250, 0) 
axa.contourf(time, zcross_A[:,380], EP_A_z.T*1e3, cmap=cmap_N, levels=50,norm=norm_N)
axa.set_xlabel(None,size='x-small')
axa.set_ylabel('m',size='x-small')
axa.tick_params(axis='both', labelsize='x-small')
axa.set_ylim(-200, 0) 
#ax.tick_params(axis='x', labelsize='x-small', rotation=25)
#ax.tick_params(axis='y', labelsize='x-small')
axa.set_xticks([])  # Remove x-axis ticks

axa.text(0.02, 0.95, 'c', transform=axa.transAxes, fontsize='x-small', fontweight='demibold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))
# Colorbar
cbar_3 = axa.inset_axes([0.035, 0.14, 0.4, 0.06])
cbar_3.set_facecolor('lightgrey')
cb3 = mpl.colorbar.ColorbarBase(cbar_3, cmap=cmap_N, norm=norm_N, extend='both', orientation='horizontal')
cb3.set_label(bar_title, size='x-small')
cbar_3.xaxis.set_ticks_position('bottom')
cbar_3.tick_params(axis='x', labelsize='x-small')
#cb3.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)
cb3.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)


#B
axb = plt.subplot(gs[1, 1])
axb.set_ylim(-250, 0) 
axb.contourf(time, zcross_B[:,238], EP_B_z.T*1e3, cmap=cmap_N, levels=50,norm=norm_N)
axb.set_xlabel(None,size='x-small')
axb.set_ylabel('m',size='x-small')
axb.tick_params(axis='both', labelsize='x-small')
axb.set_ylim(-200, 0) 
#ax.tick_params(axis='x', labelsize='x-small', rotation=25)
#ax.tick_params(axis='y', labelsize='x-small')
axb.set_xticks([])  # Remove x-axis ticks

axb.text(0.02, 0.95, 'd', transform=axb.transAxes, fontsize='x-small', fontweight='demibold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))
#C
axc = plt.subplot(gs[2, 1])
axc.set_ylim(-250, 0) 
axc.contourf(time, zcross_C[:,221], EP_C_z.T*1e3, cmap=cmap_N, levels=50,norm=norm_N)
axc.set_xlabel(None,size='x-small')
axc.set_ylabel('m',size='x-small')
axc.tick_params(axis='both', labelsize='x-small')
axc.set_ylim(-200, 0) 
#ax.tick_params(axis='x', labelsize='x-small', rotation=25)
#ax.tick_params(axis='y', labelsize='x-small')
axc.set_xticks([])  # Remove x-axis ticks

axc.text(0.02, 0.95, 'e', transform=axc.transAxes, fontsize='x-small', fontweight='demibold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))

#D
axd = plt.subplot(gs[3, 1])
axd.set_ylim(-250, 0) 
axd.contourf(time, zcross_D[:,275], EP_D_z.T*1e3, cmap=cmap_N, levels=50,norm=norm_N)
axd.set_xlabel(None,size='x-small')
axd.set_ylabel('m',size='x-small')
axd.tick_params(axis='both', labelsize='x-small')
axd.set_ylim(-180, 0) 
axd.xaxis.set_major_locator(mdates.DayLocator(interval=12))
axd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(axd.get_xticklabels(), rotation=15, ha='center', fontsize='x-small')

#axd.tick_params(axis='x', labelsize='x-small', rotation=25)
axd.tick_params(axis='y', labelsize='x-small')
#axd.set_xticks([])  # Remove x-axis ticks

axd.text(0.02, 0.95, 'f', transform=axd.transAxes, fontsize='x-small', fontweight='demibold', va='top', ha='left',
		 bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='lightgray'))



plt.savefig(name, dpi=300)
