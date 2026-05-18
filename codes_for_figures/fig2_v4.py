
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

def rotary_cospectrum(tau_u, tau_v, u, v, fs, nperseg=256, noverlap=None, window='blackmanharris'):
	"""
	Compute the rotary co-spectrum (real part of complex CSD) between 
	complex wind stress and complex velocity.
	"""
	# Form complex vectors
	tau_c = np.asarray(tau_u, dtype=complex) + 1j * np.asarray(tau_v, dtype=complex)
	w_c = np.asarray(u, dtype=complex) + 1j * np.asarray(v, dtype=complex)

	# Align finite samples
	mask = np.isfinite(tau_c) & np.isfinite(w_c)
	tau_c, w_c = tau_c[mask], w_c[mask]
	
	if tau_c.size < 4:
		raise ValueError("Not enough finite samples after masking.")

	# Segmenting/window
	nper = int(min(nperseg, tau_c.size))
	if noverlap is None:
		noverlap = nper // 2

	win = windows.get_window(window, nper)

	# Cross-spectral density (two-sided for rotary)
	# return_onesided=False gives negative (CW) and positive (CCW) frequencies
	freqs, Pxy = signal.csd(
		tau_c, w_c, fs=fs, window=win, nperseg=nper, noverlap=noverlap,
		detrend='constant', scaling='density', return_onesided=False
	)

	# Shift zero frequency to center
	freqs = np.fft.fftshift(freqs)
	Pxy = np.fft.fftshift(Pxy)

	# Co-spectrum is the real part of the Cross-Spectral Density
	Co = np.real(Pxy)

	# Separate CW (negative freqs) and CCW (positive freqs)
	cw_mask = freqs < 0
	ccw_mask = freqs > 0

	# Return absolute frequencies for CW so they can be plotted on the same log-x axis
	return freqs[ccw_mask], Co[ccw_mask], np.abs(freqs[cw_mask]), Co[cw_mask]
	
############################################################################################################

winds = '/data1/roms_dd_waves/analysis_outs/NIW/wind_input/' #already filtered data
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
T_fixed = 12.42  # [hours] fixed boundary between "super" and "mid-high"
fs = 1


avg = xr.open_dataset('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/inputs/paper_2_1km_closed_cropped_smooth_sponge.nc')

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

##--> Virtual points <--###
A =  [376,1239]
B =  [241,957]
C =  [299,570]
D =  [463,225]


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

#######################################################################################################################
################### ----- MAP -----####################################################################################
dzs = '/data1/roms_dd_waves/analysis_outs/dz/niws/'

z = xr.open_mfdataset(dzs + f'dz_*.nc').isel(s_rho = 0).z_rho #bottom
lon_rho = z.lon_rho.compute()
lat_rho = z.lat_rho.compute()
h = z.compute()*-1


#WU
#wu = xr.open_mfdataset(winds + 'WU_*.nc').WU.isel(ocean_time=slice(120,None)).compute()
##WV
#wv = xr.open_mfdataset(winds + 'WV_*.nc').WV.isel(ocean_time=slice(120,None)).compute()
#
#WI = wu + wv
#WI_tfile = WI.to_dataset(name='WI')		
#WI_tfile.to_netcdf('WI_full.nc')

WI = xr.open_dataset('WI_full.nc').WI
dates = WI.ocean_time.to_index()

#Normal
mask_normal = create_date_mask(dates, date_ranges_normal)

WI_normal = WI.isel(ocean_time=mask_normal)

indices_normal = np.where(mask_normal)[0]
dates_normal = WI_normal.ocean_time


#CF
mask_cf = create_date_mask(dates, date_ranges_cf)

WI_cf = WI.isel(ocean_time=mask_cf)

indices_cf = np.where(mask_cf)[0]
dates_cf = WI_cf.ocean_time

#Hurricane
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)

WI_hurricane = WI.isel(ocean_time=mask_hurricane)

indices_hurricane = np.where(mask_hurricane)[0]
dates_hurricane = WI_hurricane.ocean_time


#----> Means <----##
WI_map_normal = WI_normal.mean(dim='ocean_time')

WI_map_cf = WI_cf.mean(dim='ocean_time')

WI_map_hurricane = WI_hurricane.mean(dim='ocean_time')



#######------> Points and covariances <-------############
f_mean = np.absolute(avg.f.mean()) 
T_mean = (2*np.pi/f_mean)/3600 #inertial period in hours
T1 = 0.8 * T_mean
T2 = 1.5 * T_mean
f1 = 1/( 0.8 * T_mean)
f2 = 1/( 1.5 * T_mean)

# f_mean is the Coriolis frequency in cph (cycles/hour)
f_mean_rad_s = np.absolute(avg.f.mean()).item() # Assuming 'avg.f' is in rad/s
f_mean_cph = f_mean_rad_s * 3600 / (2 * np.pi) # Convert rad/s to cycles/hour

# Define shading frequencies (cph)
f1 = f_mean_cph * 1/1.5 # T2 = 1.5*T_mean -> f2 = f_mean / 1.5 
f2 = f_mean_cph * 1/0.8 # T1 = 0.8*T_mean -> f1 = f_mean / 0.8

#--A
tu_A = xr.open_mfdataset(winds + 'tu_slice*.nc').isel(eta_rho=A[1],xi_rho=A[0]).tu

tv_A = xr.open_mfdataset(winds + 'tv_slice*.nc').isel(eta_rho=A[1],xi_rho=A[0]).tv

u_A = xr.open_mfdataset(speeds + 'u_slice_*.nc').isel(eta_rho=A[1],xi_rho=A[0]).u.isel(s_rho=-1)

v_A = xr.open_mfdataset(speeds + 'v_slice_*.nc').isel(eta_rho=A[1],xi_rho=A[0]).v.isel(s_rho=-1)


Co_T_A_u, f_A_u = cospectrum_per_frequency(tu_A, u_A,nperseg=256, fs=1)

Co_T_A_v, f_A_v = cospectrum_per_frequency(tv_A, v_A,nperseg=256, fs=1)

f_A = f_A_u
Co_T_A = (Co_T_A_u + Co_T_A_v) * f_A


#--B
tu_B = xr.open_mfdataset(winds + 'tu_slice*.nc').isel(eta_rho=B[1],xi_rho=B[0]).tu

tv_B = xr.open_mfdataset(winds + 'tv_slice*.nc').isel(eta_rho=B[1],xi_rho=B[0]).tv

u_B = xr.open_mfdataset(speeds + 'u_slice_*.nc').isel(eta_rho=B[1],xi_rho=B[0]).u.isel(s_rho=-1)

v_B = xr.open_mfdataset(speeds + 'v_slice_*.nc').isel(eta_rho=B[1],xi_rho=B[0]).v.isel(s_rho=-1)

Co_T_B_u, f_B_u = cospectrum_per_frequency(tu_B, u_B,nperseg=256, fs=1)

Co_T_B_v, f_B_v = cospectrum_per_frequency(tv_B, v_B,nperseg=256, fs=1)

f_B = f_B_u
Co_T_B = (Co_T_B_u + Co_T_B_v) * f_B


#--C
tu_C = xr.open_mfdataset(winds + 'tu_slice*.nc').isel(eta_rho=C[1],xi_rho=C[0]).tu

tv_C = xr.open_mfdataset(winds + 'tv_slice*.nc').isel(eta_rho=C[1],xi_rho=C[0]).tv

u_C = xr.open_mfdataset(speeds + 'u_slice_*.nc').isel(eta_rho=C[1],xi_rho=C[0]).u.isel(s_rho=-1)

v_C = xr.open_mfdataset(speeds + 'v_slice_*.nc').isel(eta_rho=C[1],xi_rho=C[0]).v.isel(s_rho=-1)

Co_T_C_u, f_C_u = cospectrum_per_frequency(tu_C, u_C,nperseg=256, fs=1)

Co_T_C_v, f_C_v = cospectrum_per_frequency(tv_C, v_C,nperseg=256, fs=1)

f_C = f_C_u
Co_T_C = (Co_T_C_u + Co_T_C_v) * f_C



#--D
tu_D = xr.open_mfdataset(winds + 'tu_slice*.nc').isel(eta_rho=D[1],xi_rho=D[0]).tu

tv_D = xr.open_mfdataset(winds + 'tv_slice*.nc').isel(eta_rho=D[1],xi_rho=D[0]).tv

u_D = xr.open_mfdataset(speeds + 'u_slice_*.nc').isel(eta_rho=D[1],xi_rho=D[0]).u.isel(s_rho=-1)

v_D = xr.open_mfdataset(speeds + 'v_slice_*.nc').isel(eta_rho=D[1],xi_rho=D[0]).v.isel(s_rho=-1)

Co_T_D_u, f_D_u = cospectrum_per_frequency(tu_D, u_D,nperseg=256, fs=1)

Co_T_D_v, f_D_v = cospectrum_per_frequency(tv_D, v_D,nperseg=256, fs=1)


f_D = f_D_u

Co_T_D = (Co_T_D_u + Co_T_D_v) * f_D


############################################################################################################
#########################-----> Defining the control volumes <--------######################################

dv_S_eta, dv_S_xi = [40,430], [300,600]

dv_C_eta, dv_C_xi = [450,700],[115,350]

dv_N_eta, dv_N_xi = [800,1200],[430,600]



# 2. Smoothing function (same as yours)
def smooth_spectrum(f, co, window_len=6):
	s = np.r_[co[window_len-1:0:-1], co, co[-2:-window_len-1:-1]]
	w = np.ones(window_len, 'd')
	y = np.convolve(w/w.sum(), s, mode='valid')
	return y[int(window_len/2):len(co)+int(window_len/2)]


# 3. Process the data
points_dict = {'A': A, 'B': B, 'C': C, 'D': D}
rotary_results = {}

for name, coords in points_dict.items():
	eta, xi = coords[1], coords[0]
	
	# Load data
	tu = xr.open_mfdataset(winds + 'tu_slice*.nc').isel(eta_rho=eta, xi_rho=xi).tu
	tv = xr.open_mfdataset(winds + 'tv_slice*.nc').isel(eta_rho=eta, xi_rho=xi).tv
	u = xr.open_mfdataset(speeds + 'u_slice_*.nc').isel(eta_rho=eta, xi_rho=xi).u.isel(s_rho=-1)
	v = xr.open_mfdataset(speeds + 'v_slice_*.nc').isel(eta_rho=eta, xi_rho=xi).v.isel(s_rho=-1)
	
	# Compute rotary co-spectrum
	f_ccw, co_ccw, f_cw, co_cw = rotary_cospectrum(tu, tv, u, v, nperseg=256, fs=1)
	
	# Multiply by frequency (variance preserving form, if that's what your original `* f_A` did)
	co_ccw = co_ccw * f_ccw
	co_cw = co_cw * f_cw
	
	# Smooth
	co_ccw_s = smooth_spectrum(f_ccw, co_ccw)
	co_cw_s = smooth_spectrum(f_cw, co_cw)
	
	rotary_results[name] = {
		'f_ccw': f_ccw, 'co_ccw': co_ccw_s,
		'f_cw': f_cw, 'co_cw': co_cw_s
	}
#
## Apply to your results
#Co_T_A_s = smooth_spectrum(f_A, Co_T_A)
#Co_T_B_s = smooth_spectrum(f_B, Co_T_B)
#Co_T_C_s = smooth_spectrum(f_C, Co_T_C)
#Co_T_D_s = smooth_spectrum(f_D, Co_T_D)

"""
A standard co-spectrum tells you at what frequencies the wind is actively transferring kinetic energy into the ocean. 
A rotary co-spectrum takes this a step further and tells you in what rotational direction that energy transfer is happening.

Because your domain is the South Brazil Bight (Southern Hemisphere), the Coriolis force dictates that near-inertial 
currents rotate counter-clockwise (CCW).

When a storm or frontal system passes over the ocean, the wind stress vector also rotates. If the wind stress rotates 
CCW at roughly the local inertial frequency, it creates resonant forcing. The wind pushes the water exactly in phase with 
its natural inertial rotation (like perfectly timing pushes on a swing).

By separating the co-spectrum:

	CCW Co-spectrum (Positive Frequencies): You will likely see a massive positive peak in the near-inertial band. 
	This proves that the wind is resonantly dumping energy into the CCW-rotating inertial currents.

	CW Co-spectrum (Negative Frequencies): This will likely be much lower, as CW rotating winds are "anti-resonant" in 
	the Southern Hemisphere and do not efficiently transfer energy to the ocean at the inertial frequency.

	Positive vs. Negative Values: A positive value means the wind is doing work on the ocean (adding energy).
	A negative value (the dots in your plot) means the ocean is doing work on the wind (losing energy to the atmosphere), 
	which can happen if the current is flowing against the wind.

"""






##################-----> Plotting <------###################################
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, NullFormatter, FixedLocator
from matplotlib.lines import Line2D

name = f"fig_2_cospectrum_covariance_wind_v3.png"  # Create a unique filename

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1,1,1], height_ratios=[1,1])
gs.update(left=0.1, right=0.97, wspace=0.20, hspace=0.1, top=0.96, bottom=0.1)


######----> Normal
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.text(0.03, 0.08, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax.set_title('Normal', loc='left', fontsize=10)
# Define colormap and normalization
cmap = plt.cm.cubehelix_r

#cmap = cmo.cm.thermal
vmin = 0
vmax = 6
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'W.m$^{-2}$'
ax.set_ylim(bottom=-32.6, top=-20)
ax.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax.contourf(lon_rho, lat_rho, WI_map_normal*1e3, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

ax.scatter(lon_rho[A[1],A[0]], lat_rho[A[1],A[0]], zorder=5, s=20, marker='*', color='k', label='A')
ax.scatter(lon_rho[B[1],B[0]], lat_rho[B[1],B[0]], zorder=5, s=20, marker='+', color='blue', label='B')
ax.scatter(lon_rho[C[1],C[0]], lat_rho[C[1],C[0]], zorder=5, s=20, marker='s', color='orange', label='C')
ax.scatter(lon_rho[D[1],D[0]], lat_rho[D[1],D[0]], zorder=5, s=20, marker='D', color='red', label='D')

#legend = ax.legend(loc=4, fontsize='xx-small', markerscale=0.7)

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
cb1.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)


######----> CF
ax_cf = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax_cf.text(0.03, 0.08, '(b)', transform=ax_cf.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax_cf.set_title('Cold Fronts', loc='left', fontsize=10)

# Define colormap and normalization
cmap = plt.cm.cubehelix_r

#cmap = cmo.cm.thermal
vmin = 0
vmax = 6
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'W.m$^{-2}$'
ax_cf.set_ylim(bottom=-32.6, top=-20)
ax_cf.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax_cf.contourf(lon_rho, lat_rho, WI_map_cf*1e3, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

ax_cf.scatter(lon_rho[A[1],A[0]], lat_rho[A[1],A[0]], zorder=5, s=20, marker='*', color='k')#, label='A')
ax_cf.scatter(lon_rho[B[1],B[0]], lat_rho[B[1],B[0]], zorder=5, s=20, marker='+', color='blue')#, label='B')
ax_cf.scatter(lon_rho[C[1],C[0]], lat_rho[C[1],C[0]], zorder=5, s=20, marker='s', color='orange')#, label='C')
ax_cf.scatter(lon_rho[D[1],D[0]], lat_rho[D[1],D[0]], zorder=5, s=20, marker='D', color='red')#, label='D')


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
cbar_2 = inset_axes(ax_cf, width="60%", height="3%", loc=2)
cbar_2.set_facecolor('lightgray')
cb2 = mpl.colorbar.ColorbarBase(cbar_2, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb2.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)
cbar_2.xaxis.set_ticks_position('bottom')
cbar_2.tick_params(axis='x', labelsize='x-small', rotation=25)

def plot_slice_box(ax_cf, lon_rho, lat_rho, eta_slice, xi_slice, color, label, linestyle='--', linewidth=1, fill=False, alpha=0.1):
	"""
	Plots a box defined by ROMS grid indices (eta, xi) onto the map axes.
	"""
	# 1. Extract boundary indices
	eta_start, eta_end = eta_slice
	xi_start, xi_end = xi_slice

	# 2. Get the corner and edge coordinates (using the rho grid indices)
	# The box is defined by the four corners:
	# (xi_start, eta_start), (xi_end, eta_start), (xi_end, eta_end), (xi_start, eta_end)

	# We need to trace the boundary. We take the coordinates at the four edges.
	
	# 2a. Bottom Edge (eta_start, xi_start to xi_end)
	lon_bottom = lon_rho[eta_start, xi_start:xi_end + 1]
	lat_bottom = lat_rho[eta_start, xi_start:xi_end + 1]

	# 2b. Right Edge (xi_end, eta_start to eta_end) - reverse order for seamless plot
	lon_right = lon_rho[eta_start:eta_end + 1, xi_end]
	lat_right = lat_rho[eta_start:eta_end + 1, xi_end]

	# 2c. Top Edge (eta_end, xi_end to xi_start) - reverse order
	lon_top = lon_rho[eta_end, xi_start:xi_end + 1][::-1]
	lat_top = lat_rho[eta_end, xi_start:xi_end + 1][::-1]

	# 2d. Left Edge (xi_start, eta_end to eta_start) - reverse order
	lon_left = lon_rho[eta_start:eta_end + 1, xi_start][::-1]
	lat_left = lat_rho[eta_start:eta_end + 1, xi_start][::-1]

	# 3. Concatenate all coordinates to form the closed path
	lon_path = np.concatenate([lon_bottom, lon_right, lon_top, lon_left])
	lat_path = np.concatenate([lat_bottom, lat_right, lat_top, lat_left])
	
	# 4. Plot the boundary line
	ax_cf.plot(lon_path, lat_path, color=color, linestyle=linestyle, linewidth=linewidth, 
			transform=ccrs.PlateCarree(), zorder=10, label=label)
			
	# 5. Optionally fill the box
	if fill:
		ax_cf.fill(lon_path, lat_path, color=color, alpha=alpha, transform=ccrs.PlateCarree(), zorder=9)


# --- Plotting the defined boxes ---
# South Box (dv_S) - Let's make it green
plot_slice_box(ax_cf, lon_rho, lat_rho, dv_S_eta, dv_S_xi, 
			   color='purple', label='South Box', fill=True, alpha=0.15)

# North Box (dv_N) - Let's make it purple
plot_slice_box(ax_cf, lon_rho, lat_rho, dv_N_eta, dv_N_xi, 
			   color='cyan', label='North Box', fill=True, alpha=0.15)

# Coastal Box (dv_C) - Let's make it cyan
plot_slice_box(ax_cf, lon_rho, lat_rho, dv_C_eta, dv_C_xi, 
			   color='gold', label='Central Box', fill=True, alpha=0.15)


legend = ax_cf.legend(loc=4, fontsize='xx-small', markerscale=0.7)

######----> Hurricane
ax_hc = plt.subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax_hc.text(0.03, 0.08, '(c)', transform=ax_hc.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax_hc.set_title('Catarina Hurricane', loc='left', fontsize=10)

# Define colormap and normalization
cmap = plt.cm.cubehelix_r

#cmap = cmo.cm.thermal
vmin = 0
vmax = 6
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'W.m$^{-2}$'
ax_hc.set_ylim(bottom=-32.6, top=-20)
ax_hc.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax_hc.contourf(lon_rho, lat_rho, WI_map_hurricane*1e3, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)

ax_hc.scatter(lon_rho[A[1],A[0]], lat_rho[A[1],A[0]], zorder=5, s=20, marker='*', color='k', label='A')
ax_hc.scatter(lon_rho[B[1],B[0]], lat_rho[B[1],B[0]], zorder=5, s=20, marker='+', color='blue', label='B')
ax_hc.scatter(lon_rho[C[1],C[0]], lat_rho[C[1],C[0]], zorder=5, s=20, marker='s', color='orange', label='C')
ax_hc.scatter(lon_rho[D[1],D[0]], lat_rho[D[1],D[0]], zorder=5, s=20, marker='D', color='red', label='D')

legend = ax_hc.legend(loc=4, fontsize='xx-small', markerscale=0.7)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax_hc.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax_hc.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax_hc.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax_hc.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

# Coastlines and gridlines
ax_hc.coastlines()
ax_hc.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax_hc.patch.set_edgecolor('black')
gl = ax_hc.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
cbar_3 = inset_axes(ax_hc, width="60%", height="3%", loc=2)
cbar_3.set_facecolor('lightgray')
cb3 = mpl.colorbar.ColorbarBase(cbar_3, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb3.set_label(fr'{bar_title} $\times$ 10$^{{-3}}$', size='x-small', labelpad=2)
cbar_3.xaxis.set_ticks_position('bottom')
cbar_3.tick_params(axis='x', labelsize='x-small', rotation=25)


#############   ----> Cospectrum <--------################
from scipy.stats import chi2

def get_conf_interval(data, nperseg, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    
    # 1. Get the scalar length of the data (this prevents the broadcasting error!)
    n_total = len(np.asarray(data)[np.isfinite(data)]) 
    
    # 2. Calculate segments and degrees of freedom
    step = nperseg - noverlap
    n_segments = (n_total - noverlap) // step
    nu = 2 * n_segments 
    
    # 3. Get 95% CI factors
    lower = nu / chi2.ppf(0.975, nu)
    upper = nu / chi2.ppf(0.025, nu)
    
    return lower, upper


colors = {'A': 'k', 'B': 'blue', 'C': 'orange', 'D': 'red'}
markers = {'A': '*', 'B': '+', 'C': 's', 'D': 'D'}
ax_g.set_xlim(left=10**-2,right=4*10**-1) 
ax_g.set_ylim(bottom=10**-7,top=2*10**-2)
# 5. Plotting
colors = {'A': 'k', 'B': 'blue', 'C': 'orange', 'D': 'red'}

ax_g = plt.subplot(gs[1, :])
ax_g.text(0.03, 0.08, '(d)', transform=ax_g.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax_g.set_xlim(left=10**-2, right=4*10**-1) 
ax_g.set_ylim(bottom=10**-7, top=2*10**-2)
ax_g.set_xscale('log')
ax_g.set_yscale('log')

# Plot continuous lines for each point
for name in ['A', 'B', 'C', 'D']:
    res = rotary_results[name]
    color = colors[name]
    
    # --- Plot CCW (Solid lines) ---
    f_ccw = res['f_ccw']
    co_ccw = res['co_ccw']
    
    ax_g.plot(f_ccw, np.abs(co_ccw), color=color, linestyle='-', linewidth=1.5, label=f'{name} (CCW)')

    # --- Plot CW (Dashed lines) ---
    f_cw = res['f_cw']
    co_cw = res['co_cw']
    
    ax_g.plot(f_cw, np.abs(co_cw), color=color, linestyle='--', linewidth=1.5, label=f'{name} (CW)')

# --- Explicit Shading Blocks for CCW ---
# Get the scalar factors using point A's data
low_fact, upp_fact = get_conf_interval(tu_A, 256, 128)

ax_g.fill_between(rotary_results['A']['f_ccw'], np.abs(rotary_results['A']['co_ccw']) * low_fact, np.abs(rotary_results['A']['co_ccw']) * upp_fact, color='k', alpha=0.1, zorder=0)
ax_g.fill_between(rotary_results['B']['f_ccw'], np.abs(rotary_results['B']['co_ccw']) * low_fact, np.abs(rotary_results['B']['co_ccw']) * upp_fact, color='blue', alpha=0.1, zorder=0)
ax_g.fill_between(rotary_results['C']['f_ccw'], np.abs(rotary_results['C']['co_ccw']) * low_fact, np.abs(rotary_results['C']['co_ccw']) * upp_fact, color='orange', alpha=0.1, zorder=0)
ax_g.fill_between(rotary_results['D']['f_ccw'], np.abs(rotary_results['D']['co_ccw']) * low_fact, np.abs(rotary_results['D']['co_ccw']) * upp_fact, color='red', alpha=0.1, zorder=0)
# Manually shade the CW components for each point
ax_g.fill_between(rotary_results['A']['f_cw'], np.abs(rotary_results['A']['co_cw']) * low_fact, np.abs(rotary_results['A']['co_cw']) * upp_fact, color='k', alpha=0.1, zorder=0)
ax_g.fill_between(rotary_results['B']['f_cw'], np.abs(rotary_results['B']['co_cw']) * low_fact, np.abs(rotary_results['B']['co_cw']) * upp_fact, color='blue', alpha=0.1, zorder=0)
ax_g.fill_between(rotary_results['C']['f_cw'], np.abs(rotary_results['C']['co_cw']) * low_fact, np.abs(rotary_results['C']['co_cw']) * upp_fact, color='orange', alpha=0.1, zorder=0)
ax_g.fill_between(rotary_results['D']['f_cw'], np.abs(rotary_results['D']['co_cw']) * low_fact, np.abs(rotary_results['D']['co_cw']) * upp_fact, color='red', alpha=0.1, zorder=0)



# Reference Lines
ax_g.axvline(f_mean_cph, color='gray', linestyle='-', linewidth=1.5, label=r'$f$')
ax_g.axvspan(f1, f2, color='salmon', alpha=0.3, label='[0.8f, 1.5f]')

ax_g.set_xlabel('Frequency [cph]')
ax_g.set_ylabel(r'$\text{Pa} \cdot \text{m} \cdot \text{s}^{-1}$')
ax_g.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.6)

# Legend management (Cleaned up to match continuous lines)
legend_elements = [
    Line2D([0], [0], color='k', linestyle='-', label='CCW (+) Freqs'),
    Line2D([0], [0], color='k', linestyle='--', label='CW (-) Freqs')
]
handles, labels = ax_g.get_legend_handles_labels()
# Extract only the reference line elements, ignore the plotted lines to prevent duplicates
clean_handles = handles[-2:] + legend_elements 

ax_g.legend(handles=clean_handles, loc='upper right', fontsize='small', ncol=2)

plt.tight_layout()
plt.savefig(name, dpi=300)