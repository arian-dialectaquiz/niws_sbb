

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



date_ranges_normal = [
	("2004-03-31", "2004-04-09")
	
]

date_ranges_cf = [
	("2004-03-19", "2004-03-24"),
	("2004-04-09", "2004-04-11")
	
]


date_ranges_hurricane = [
	("2004-03-23", "2004-03-30")	]
##############################################################################################
###############_---> zs <----################################################################

dzs = '/data1/roms_dd_waves/analysis_outs/dz/'
zz =  xr.open_mfdataset(dzs + f'dz_*.nc').z_rho


#################------> A <---#############################
zcross_A = zz.isel(eta_rho=A[1],xi_rho=slice(0,None))#.values
zpos = -200
abs_diff = np.abs(zcross_A.isel(s_rho=0) - zpos)
z_slope_A = abs_diff.argmin().values


dT_dz_A = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/theta/dT_dz_slice_*.nc').isel(eta_rho=A[1],xi_rho=slice(0,None)).dT_dz
akt_A = xr.open_dataset('/data1/roms_dd_waves/analysis_outs/NIW/theta/AKt_eta_1239.nc').AKt.isel(s_rho=slice(1,None),xi_rho=slice(0,None))
flux_A = -1*akt_A * dT_dz_A

flux_comp_vol_A = (flux_A / zcross_A[:,0:z_slope_A]).sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)*z_slope_A*1000/3600

dates = flux_A.ocean_time.to_index()

#---> Normal
mask_normal = create_date_mask(dates, date_ranges_normal)
flux_normal_A = flux_A.isel(ocean_time=mask_normal).compute()
flux_comp_vol_normal_A = flux_comp_vol_A.isel(ocean_time=mask_normal).compute()
flux_comp_vol_normal_A_t = flux_comp_vol_normal_A.mean()

#---> cf
mask_cf = create_date_mask(dates, date_ranges_cf)
flux_cf_A = flux_A.isel(ocean_time=mask_cf).compute()
flux_comp_vol_cf_A = flux_comp_vol_A.isel(ocean_time=mask_cf).compute()
flux_comp_vol_cf_A_t = flux_comp_vol_cf_A.mean()

#---> hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
flux_hurricane_A = flux_A.isel(ocean_time=mask_hurricane).compute()
flux_comp_vol_hurricane_A = flux_comp_vol_A.isel(ocean_time=mask_hurricane).compute()
flux_comp_vol_hurricane_A_t = flux_comp_vol_hurricane_A.mean()

#################------> B <---#############################

zcross_B = zz.isel(eta_rho=B[1],xi_rho=slice(0,None))#.values
zpos = -200
abs_diff = np.abs(zcross_B.isel(s_rho=0) - zpos)
z_slope_B = abs_diff.argmin().values 


dT_dz_B = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/theta/dT_dz_slice_*.nc').isel(eta_rho=B[1],xi_rho=slice(0,None)).dT_dz
akt_B = xr.open_dataset('/data1/roms_dd_waves/analysis_outs/NIW/theta/AKt_eta_957.nc').AKt.isel(s_rho=slice(1,None),xi_rho=slice(0,None))
flux_B = -1*akt_B * dT_dz_B

flux_comp_vol_B = (flux_B / zcross_B[:,0:z_slope_B]).sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)*z_slope_B*1000/3600

dates = flux_B.ocean_time.to_index()

#---> Normal
mask_normal = create_date_mask(dates, date_ranges_normal)
flux_normal_B = flux_B.isel(ocean_time=mask_normal).compute()
flux_comp_vol_normal_B = flux_comp_vol_B.isel(ocean_time=mask_normal).compute()
flux_comp_vol_normal_B_t = flux_comp_vol_normal_B.mean()

#---> cf
mask_cf = create_date_mask(dates, date_ranges_cf)
flux_cf_B = flux_B.isel(ocean_time=mask_cf).compute()
flux_comp_vol_cf_B = flux_comp_vol_B.isel(ocean_time=mask_cf).compute()
flux_comp_vol_cf_B_t = flux_comp_vol_cf_B.mean()

#---> hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
flux_hurricane_B = flux_B.isel(ocean_time=mask_hurricane).compute()
flux_comp_vol_hurricane_B = flux_comp_vol_B.isel(ocean_time=mask_hurricane).compute()
flux_comp_vol_hurricane_B_t = flux_comp_vol_hurricane_B.mean()


#################------> C <---#############################

zcross_C = zz.isel(eta_rho=C[1],xi_rho=slice(0,None))#.values
zpos = -200
abs_diff = np.abs(zcross_C.isel(s_rho=0) - zpos)
z_slope_C = abs_diff.argmin().values 

dT_dz_C = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/theta/dT_dz_slice_*.nc').isel(eta_rho=C[1],xi_rho=slice(0,None)).dT_dz
akt_C = xr.open_dataset('/data1/roms_dd_waves/analysis_outs/NIW/theta/AKt_eta_570.nc').AKt.isel(s_rho=slice(1,None),xi_rho=slice(0,None))
flux_C = -1*akt_C * dT_dz_C

flux_comp_vol_C = (flux_C / zcross_C[:,0:z_slope_C]).sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)*z_slope_C*1000/3600

dates = flux_C.ocean_time.to_index()

#---> Normal
mask_normal = create_date_mask(dates, date_ranges_normal)
flux_normal_C = flux_C.isel(ocean_time=mask_normal).compute()
flux_comp_vol_normal_C = flux_comp_vol_C.isel(ocean_time=mask_normal).compute()
flux_comp_vol_normal_C_t = flux_comp_vol_normal_C.mean()

#---> cf
mask_cf = create_date_mask(dates, date_ranges_cf)
flux_cf_C = flux_C.isel(ocean_time=mask_cf).compute()
flux_comp_vol_cf_C = flux_comp_vol_C.isel(ocean_time=mask_cf).compute()
flux_comp_vol_cf_C_t = flux_comp_vol_cf_C.mean()

#---> hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
flux_hurricane_C = flux_C.isel(ocean_time=mask_hurricane).compute()
flux_comp_vol_hurricane_C = flux_comp_vol_C.isel(ocean_time=mask_hurricane).compute()
flux_comp_vol_hurricane_C_t = flux_comp_vol_hurricane_C.mean()


#################------> D <---#############################
zcross_D = zz.isel(eta_rho=D[1],xi_rho=slice(0,None))#.values
zpos = -200
abs_diff = np.abs(zcross_D.isel(s_rho=0) - zpos)
z_slope_D = abs_diff.argmin().values 

dT_dz_D = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/theta/dT_dz_slice_*.nc').isel(eta_rho=D[1],xi_rho=slice(0,None)).dT_dz
akt_D = xr.open_dataset('/data1/roms_dd_waves/analysis_outs/NIW/theta/AKt_eta_225.nc').AKt.isel(s_rho=slice(1,None),xi_rho=slice(0,None))
flux_D = -1*akt_D * dT_dz_D

flux_comp_vol_D = (flux_D / zcross_D[:,0:z_slope_D]).sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)*z_slope_D*1000/3600

dates = flux_D.ocean_time.to_index()

#---> Normal
mask_normal = create_date_mask(dates, date_ranges_normal)
flux_normal_D = flux_D.isel(ocean_time=mask_normal).compute()
flux_comp_vol_normal_D = flux_comp_vol_D.isel(ocean_time=mask_normal).compute()
flux_comp_vol_normal_D_t = flux_comp_vol_normal_D.mean()

#---> cf
mask_cf = create_date_mask(dates, date_ranges_cf)
flux_cf_D = flux_D.isel(ocean_time=mask_cf).compute()
flux_comp_vol_cf_D = flux_comp_vol_D.isel(ocean_time=mask_cf).compute()
flux_comp_vol_cf_D_t = flux_comp_vol_cf_D.mean()

#---> hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)
flux_hurricane_D = flux_D.isel(ocean_time=mask_hurricane).compute()
flux_comp_vol_hurricane_D = flux_comp_vol_D.isel(ocean_time=mask_hurricane).compute()
flux_comp_vol_hurricane_D_t = flux_comp_vol_hurricane_D.mean()


#############---------->>> Plotting it <<<---########################

full_flux_data = {
    'A': flux_comp_vol_A,
    'B': flux_comp_vol_B,
    'C': flux_comp_vol_C,
    'D': flux_comp_vol_D
}

# Ensure your time array is common and cleaned
# (Assuming all sections share the same ocean_time index)
time = flux_comp_vol_A.ocean_time.values


# --- 1. Setup Figure ---
fig, axes = plt.subplots(1, 4, figsize=(10, 5), sharey=True)
plt.subplots_adjust(bottom=0.2, top=0.85, wspace=0.1, left=0.05, right=0.98)

sections = ['A', 'B', 'C', 'D']
section_names = ['North', 'Central-N', 'Central-S', 'South']

# Styling Dictionary
styles = {
	'normal': {'color': 'k', 'ls': ':', 'alpha': 1, 'label': 'Normal'},
	'cf': {'color': 'royalblue', 'ls': '--', 'lw': 1.5, 'label': 'Cold Front'},
	'hurricane': {'color': 'crimson', 'ls': '-', 'lw': 1.5, 'label': 'Hurricane'}
}

# --- 2. Plotting Loop ---
for i, sec in enumerate(sections):
	ax = axes[i]
	
	# 1. Get full data and time
	# (Assuming flux_comp_vol_A, time, and masks are defined)
	full_data = eval(f"flux_comp_vol_{sec}")
	
	# 2. Plot Scenario Segments
	# We plot each scenario separately using the masks. 
	# Because the masks are boolean, gaps between events will not be connected.
	
	# --- Normal ---
	ax.plot(time, full_data.where(mask_normal), **styles['normal'])
	
	# --- Cold Fronts ---
	ax.plot(time, full_data.where(mask_cf), **styles['cf'])
	
	# --- Hurricane ---
	ax.plot(time, full_data.where(mask_hurricane), **styles['hurricane'])

	# 3. Add Mean Value Text (f_vals)
	# Accessing the means calculated in your previous step
	f_norm = eval(f"flux_comp_vol_normal_{sec}_t").values
	f_cf = eval(f"flux_comp_vol_cf_{sec}_t").values
	f_hur = eval(f"flux_comp_vol_hurricane_{sec}_t").values
	
	text_str = (f"Means\n"
				f"Normal: {f_norm:.2f}\n"
				f"Cold Front: {f_cf:.2f}\n"
				f"Hurricane: {f_hur:.2f}")
	
	ax.text(0.45, 0.98, text_str, transform=ax.transAxes, fontsize=7,
			va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

	# 4. Formatting
	ax.set_title(f"Section {sec}", fontweight='demibold',fontsize=8)
	ax.grid(alpha=0.3)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
	ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
	plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
	plt.setp(ax.get_yticklabels(), fontsize=8)
	
	if i == 0:
		ax.set_ylabel(r'$^\circ$C m$^2$ s$^{-1}$', fontweight='normal')

	
	# Add subplot label (a, b, c, d)
	ax.text(0.9, 0.05, f"({chr(97+i)})", transform=ax.transAxes, fontweight='bold', fontsize=8)

# Add a single legend for the whole figure
axes[0].legend(loc='upper left', bbox_to_anchor=(1.2, 1.18), ncol=3, fontsize=9, frameon=False)

plt.savefig("flux_timeseries.png", dpi=300, bbox_inches='tight')

"""
km_cross = np.arange(0,610,1)


fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(nrows=5, ncols=4, 
					   width_ratios=[1, 1, 1, 1.3], 
					   height_ratios=[1, 1, 1, 1, 0.15])

gs.update(left=0.07, right=0.93, wspace=0.12, hspace=0.4, top=0.95, bottom=0.08)

cmap = plt.cm.bwr
vmin, vmax = -0.03, 0.03
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# --- 2. Data Mapping for Loops ---
sections = ['A', 'B', 'C', 'D']
scenarios = ['Normal', 'Cold Front', 'Hurricane']

# Mapping display scenarios to your variable suffixes
# Note: Using your variable names like flux_normal_A, flux_cf_A, flux_hurricane_A
data_map = {
	'Normal': 'normal',
	'Cold Front': 'cf',
	'Hurricane': 'hurricane'
}
def apply_spatial_perks(ax, km, z, data, f_val, label, is_left=False, is_bottom=False, tri_x=None):
	# The actual data plot
	z_finite = np.nan_to_num(z.values, nan=0)
	for i in range(len(km)):

		ax.pcolormesh(km, z_finite[:, i], data, cmap=cmap, norm=norm)

	
	# Bathymetry
	ax.fill_between(km, np.min(z_finite, axis=0), y2=-300, color='grey', zorder=2)
	
	# Gold Slope Marker
	if tri_x is not None:
		ax.axvline(tri_x, color='gold', lw=1.5, alpha=0.9, zorder=3)
		ax.scatter(tri_x, -248, color='gold', marker='^', s=45, zorder=4)

	# Flux Text Box
	flux_text = fr'$F = {f_val:.2f}$'
	ax.text(0.01, 0.35, flux_text, transform=ax.transAxes, fontsize='x-small', 
			bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round'))
	
	# Subplot Label (e.g., a1)
	ax.text(0.02, 0.95, label, transform=ax.transAxes, fontweight='bold', va='top')

	# Axis tweaks
	ax.set_ylim(-250, 0)
	ax.tick_params(labelsize='x-small')
	if not is_left: ax.set_yticklabels([])
	if not is_bottom: ax.set_xticklabels([])

# --- Row 0: Section A ---
ax_a1 = fig.add_subplot(gs[0, 0])
apply_spatial_perks(ax_a1, km_cross, zcross_A, flux_normal_A.mean('ocean_time')*10, flux_comp_vol_normal_A_t, '(a1)', is_left=True, tri_x=km_cross[z_slope_A])
ax_a1.set_title("Normal", fontweight='bold')
ax_a1.set_ylabel("North (m)", fontweight='bold')

ax_a2 = fig.add_subplot(gs[0, 1])
apply_spatial_perks(ax_a2, km_cross, zcross_A, flux_cf_A.mean('ocean_time')*10, flux_comp_vol_cf_A_t, '(a2)', tri_x=km_cross[z_slope_A])
ax_a2.set_title("Cold Front", fontweight='bold')

ax_a3 = fig.add_subplot(gs[0, 2])
apply_spatial_perks(ax_a3, km_cross, zcross_A, flux_hurricane_A.mean('ocean_time')*10, flux_comp_vol_hurricane_A_t, '(a3)', tri_x=km_cross[z_slope_A])
ax_a3.set_title("Hurricane", fontweight='bold')

# Timeseries A
ax_a4 = fig.add_subplot(gs[0, 3])
ax_a4.plot(flux_comp_vol_normal_A.ocean_time, flux_comp_vol_normal_A, color='royalblue', label='Normal')
ax_a4.plot(flux_comp_vol_cf_A.ocean_time, flux_comp_vol_cf_A, color='purple', label='CF')
ax_a4.plot(flux_comp_vol_hurricane_A.ocean_time, flux_comp_vol_hurricane_A, color='crimson', label='Hurricane')
ax_a4.set_ylabel(r'$^\circ\text{C}\,\text{m}^2\,\text{s}^{-1}$', size='x-small')
ax_a4.yaxis.set_label_position("right")
ax_a4.yaxis.tick_right()
ax_a4.set_xticklabels([])
ax_a4.grid(alpha=0.3)
ax_a4.legend(loc='upper right', fontsize='xx-small', ncol=3, bbox_to_anchor=(1.1, 1.3))

# --- Row 1: Section B ---
ax_b1 = fig.add_subplot(gs[1, 0])
apply_spatial_perks(ax_b1, km_cross, zcross_B, flux_normal_B.mean('ocean_time')*10, flux_comp_vol_normal_B_t, '(b1)', is_left=True, tri_x=km_cross[z_slope_B])
ax_b1.set_ylabel("Central-N (m)", fontweight='bold')

ax_b2 = fig.add_subplot(gs[1, 1])
apply_spatial_perks(ax_b2, km_cross, zcross_B, flux_cf_B.mean('ocean_time')*10, flux_comp_vol_cf_B_t, '(b2)', tri_x=km_cross[z_slope_B])

ax_b3 = fig.add_subplot(gs[1, 2])
apply_spatial_perks(ax_b3, km_cross, zcross_B, flux_hurricane_B.mean('ocean_time'), flux_comp_vol_hurricane_B_t, '(b3)', tri_x=km_cross[z_slope_B])

# Timeseries B
ax_b4 = fig.add_subplot(gs[1, 3])
ax_b4.plot(flux_comp_vol_normal_B.ocean_time, flux_comp_vol_normal_B, color='royalblue')
ax_b4.plot(flux_comp_vol_cf_B.ocean_time, flux_comp_vol_cf_B, color='purple')
ax_b4.plot(flux_comp_vol_hurricane_B.ocean_time, flux_comp_vol_hurricane_B, color='crimson')
ax_b4.yaxis.set_label_position("right")
ax_b4.yaxis.tick_right()
ax_b4.set_xticklabels([])
ax_b4.grid(alpha=0.3)


# --- Row 2: Section C ---
ax_c1 = fig.add_subplot(gs[2, 0])
apply_spatial_perks(ax_c1, km_cross, zcross_C, flux_normal_C.mean('ocean_time')*10, flux_comp_vol_normal_C_t, '(c1)', is_left=True, tri_x=km_cross[z_slope_C])
ax_c1.set_ylabel("Central-S (m)", fontweight='bold')

ax_c2 = fig.add_subplot(gs[2, 1])
apply_spatial_perks(ax_c2, km_cross, zcross_C, flux_cf_C.mean('ocean_time')*10, flux_comp_vol_cf_C_t, '(c2)', tri_x=km_cross[z_slope_C])

ax_c3 = fig.add_subplot(gs[2, 2])
apply_spatial_perks(ax_c3, km_cross, zcross_C, flux_hurricane_C.mean('ocean_time')*10, flux_comp_vol_hurricane_C_t, '(c3)', tri_x=km_cross[z_slope_C])

# Timeseries C
ax_c4 = fig.add_subplot(gs[2, 3])
ax_c4.plot(flux_comp_vol_normal_C.ocean_time, flux_comp_vol_normal_C, color='royalblue')
ax_c4.plot(flux_comp_vol_cf_C.ocean_time, flux_comp_vol_cf_C, color='purple')
ax_c4.plot(flux_comp_vol_hurricane_C.ocean_time, flux_comp_vol_hurricane_C, color='crimson')
ax_c4.yaxis.set_label_position("right")
ax_c4.yaxis.tick_right()
ax_c4.set_xticklabels([])
ax_c4.grid(alpha=0.3)

# --- Row 3: Section D ---
ax_d1 = fig.add_subplot(gs[3, 0])
apply_spatial_perks(ax_d1, km_cross, zcross_D, flux_normal_D.mean('ocean_time')*10, flux_comp_vol_normal_D_t, '(d1)', is_left=True, is_bottom=True, tri_x=km_cross[z_slope_D])
ax_d1.set_ylabel("South (m)", fontweight='bold')
ax_d1.set_xlabel("km", fontsize='small')

ax_d2 = fig.add_subplot(gs[3, 1])
apply_spatial_perks(ax_d2, km_cross, zcross_D, flux_cf_D.mean('ocean_time')*10, flux_comp_vol_cf_D_t, '(d2)', is_bottom=True, tri_x=km_cross[z_slope_D])
ax_d2.set_xlabel("km", fontsize='small')

ax_d3 = fig.add_subplot(gs[3, 2])
apply_spatial_perks(ax_d3, km_cross, zcross_D, flux_hurricane_D.mean('ocean_time')*10, flux_comp_vol_hurricane_D_t, '(d3)', is_bottom=True, tri_x=km_cross[z_slope_D])
ax_d3.set_xlabel("km", fontsize='small')

# Timeseries D
ax_d4 = fig.add_subplot(gs[3, 3])
ax_d4.plot(flux_comp_vol_normal_D.ocean_time, flux_comp_vol_normal_D, color='royalblue')
ax_d4.plot(flux_comp_vol_cf_D.ocean_time, flux_comp_vol_cf_D, color='purple')
ax_d4.plot(flux_comp_vol_hurricane_D.ocean_time, flux_comp_vol_hurricane_D, color='crimson')
ax_d4.yaxis.set_label_position("right")
ax_d4.yaxis.tick_right()
ax_d4.tick_params(axis='x', rotation=25, labelsize='x-small')
ax_d4.grid(alpha=0.3)

cbar_ax = fig.add_subplot(gs[4, 0:3])
cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb.set_label(r'($^\circ\text{C} \, \text{s}^{-1}$)', size='small')

plt.savefig("heat_flux.png", dpi=300, bbox_inches='tight')