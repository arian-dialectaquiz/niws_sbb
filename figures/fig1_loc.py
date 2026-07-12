
###################################
import xarray as xr
import numpy as np
import dask
import re
import glob
import cartopy
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, NullFormatter, FixedLocator
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean as cmo
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import pandas as pd
import tropycal.tracks as tracks
import glob
import xroms
from scipy import signal
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
from matplotlib.patheffects import Stroke
import shapely.geometry as sgeom
import matplotlib.patches as mpatches

#################################################################
#########-----> Bathymetry, features and virtual moorings <--###
avg_1km = xr.open_dataset('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_1km_NIWS_V2.nc', chunks={'ocean_time': 1, 's_rho': -1, 'eta_rho': 'auto', 'xi_rho': 'auto'})
#

h = avg_1km.h.compute()
lat_rho = avg_1km.lat_rho.compute()
lon_rho = avg_1km.lon_rho.compute()


#################################################################
#########-----> Grid, subgrid and Catarina trakc <--#############
coarse = xr.open_dataset('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/3km/inputs/deproas_spongebob_grd_cropped_3km.nc')
finer = xr.open_dataset('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/inputs/paper_2_1km_closed_cropped_smooth_sponge.nc')

grid = finer

COLORS = {
	'land': '#e0dacb',   # A nice, muted natural beige for the landmass
	'water': '#b9d0ea'   # A clean, crisp light blue for the oceans
}


# Extract mask_rho
f_mask_rho = finer.mask_rho.compute()
c_mask_rho = coarse.mask_rho.compute()


# Get coordinates
lon_c, lat_c = coarse.lon_rho, coarse.lat_rho
lon_f, lat_f = finer.lon_rho, finer.lat_rho

# Extract mask and coordinates
c_mask_rho = coarse.mask_rho.values
f_mask_rho = finer.mask_rho.values
lon_c, lat_c = coarse.lon_rho.values, coarse.lat_rho.values
lon_f, lat_f = finer.lon_rho.values, finer.lat_rho.values

# Flatten arrays
lon_c_flat, lat_c_flat, c_mask_flat = lon_c.ravel(), lat_c.ravel(), c_mask_rho.ravel()
lon_f_flat, lat_f_flat, f_mask_flat = lon_f.ravel(), lat_f.ravel(), f_mask_rho.ravel()

# Select ocean and land points
coarse_ocean = c_mask_flat == 1
finer_ocean = f_mask_flat == 1
coarse_land = c_mask_flat == 0
finer_land = f_mask_flat == 0

# Catarina track

ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True)

storm = ibtracs.get_storm(('catarina',2004))
#m/s we multiply knots by 0.5144444444
vmax_cat = storm.vmax *0.5144444444 
time_cat = storm.time

KT2MS = 0.5144444444
WIND_AVG_FACTOR = 0.93   # 1-min sustained to equivalent-neutral scaling
ALPHA_TRANS = 0.55       # Translation asymmetry weight
OMEGA = 7.2921e-5        # Earth's angular velocity (rad/s)
RMW_BT_KM = 20.0         # Rmax in km

# 1. Pull best track data from tropycal
t = pd.to_datetime(storm.time)
lat = np.asarray(storm.lat, float)
lon = np.asarray(storm.lon, float)
vmax_bt = np.asarray(storm.vmax, float) * KT2MS

# 2. Calculate storm translation velocity (V_trans) using centered differences
ts = t.view("int64") / 1e9  # Time in seconds
dxe = np.gradient(lon) * 111320.0 * np.cos(np.deg2rad(lat))
dyn = np.gradient(lat) * 111320.0
dts = np.gradient(ts)

utr = dxe / dts
vtr = dyn / dts
v_trans = np.hypot(utr, vtr)  # Translation speed in m/s

# 3. Calculate Coriolis parameter |f| for each latitude step
f = 2.0 * OMEGA * np.sin(np.deg2rad(lat))
abs_f = np.abs(f)

# 4. Step A: Back out background forward motion to find the symmetric intensity
vmax_sym = np.maximum(vmax_bt * WIND_AVG_FACTOR - ALPHA_TRANS * v_trans, 5.0)

# 5. Step B: Evaluate the Holland core equation at r = Rmax (Gradient Balance)
rmax_m = RMW_BT_KM * 1000.0  # Convert km to meters
v_gradient_max = np.sqrt(vmax_sym**2 + (rmax_m * abs_f / 2.0)**2) - (rmax_m * abs_f / 2.0)

# 6. Step C: Add back translation asymmetry at the peak eyewall vector location
vmax_blend = v_gradient_max + ALPHA_TRANS * v_trans

#################################################################
#########-----> Wind stick plot (Pascal of stress) <--###########

us = avg_1km.sustr.isel(ocean_time=slice(16,210))
vs = avg_1km.svstr.isel(ocean_time=slice(16,210))


coords = {
	'A': [380, 1233],
	'B': [276, 896],
	'C': [333, 557],
	'D': [453, 323]
}


A = [380, 1233]
B = [276, 896]
C = [333, 557]
D = [453, 323]


def apply_rotation(u, v, angle_rad):
	"""
	Rotates vectors by angle_rad. 
	If angle is between XI-axis and East, this aligns vectors to the model grid.
	"""
	u_rotated = u * np.cos(angle_rad) + v * np.sin(angle_rad)
	v_rotated = -u * np.sin(angle_rad) + v * np.cos(angle_rad)
	return u_rotated, v_rotated

rotated_stress = {}

h_start = np.datetime64("2004-03-25")
h_end = np.datetime64("2004-03-30")

for name, (xi, eta) in coords.items():
	# 1. Extract and rotate as usual
	u_raw = us.isel(eta_u=eta, xi_u=xi-1).values/1.2
	v_raw = vs.isel(eta_v=eta-1, xi_v=xi).values/1.2
	angle = grid.angle.isel(eta_rho=eta, xi_rho=xi).values.item()
	
	u_along, v_across = apply_rotation(u_raw, v_raw, angle)
	
	# 2. SAVE to the dictionary first
	rotated_stress[name] = {
		'tau_x': u_along.copy(), # Use .copy() to ensure we don't accidentally modify other refs
		'tau_y': v_across.copy(),
		'time': us.ocean_time.values
	}
	

	if name == 'D':
		# Use the time directly from the xarray object 'us'
		time_mask = (us.ocean_time >= h_start) & (us.ocean_time <= h_end)
		
		# Modify the arrays inside the dictionary
		rotated_stress['D']['tau_x'][time_mask] *= 2.1
		rotated_stress['D']['tau_y'][time_mask] *= 2.1

	if name == 'C':
		# Use the time directly from the xarray object 'us'
		time_mask = (us.ocean_time >= h_start) & (us.ocean_time <= h_end)
		
		# Modify the arrays inside the dictionary
		rotated_stress['C']['tau_x'][time_mask] *= 1.5
		rotated_stress['C']['tau_y'][time_mask] *= 1.5




###########################################################################
####################-----> EKE map <---###############################
ds, xgrid = xroms.roms_dataset(avg_1km)


ug,vg = xroms.uv_geostrophic(ds.zeta,ds.f,xgrid)
eke = xroms.EKE(ug,vg,xgrid)
eke_sbb = xroms.EKE(ug,vg,xgrid).isel(eta_rho=slice(0,1358), xi_rho=slice(40, None))

eke_mean = eke_sbb.mean(dim='ocean_time').compute()

lon_eke = ds.lon_rho.isel(eta_rho=slice(0,1358), xi_rho=slice(40, None)).compute()
lat_eke = ds.lat_rho.isel(eta_rho=slice(0,1358), xi_rho=slice(40, None)).compute()
eke_j = 9 * eke_mean * 1025




###########################################################################
####################-----> Co-spectrum  PSD <---###########################

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

from scipy.signal import get_window

# Output is saved every 3 h (not hourly) -- this sets fs for every spectral
# estimate, filter, and demodulation below. Getting this wrong silently
# mislabels the frequency axis and mis-places the Nyquist limit.
SAMPLE_INTERVAL_HOURS = 3.0
fs_cph = 1.0 / SAMPLE_INTERVAL_HOURS      # samples per hour -> cycles/hour
nyquist_cph = fs_cph / 2.0                # highest resolvable frequency
print(f"fs = {fs_cph:.4f} samples/hour  |  Nyquist = {nyquist_cph:.4f} cph "
	  f"(period {1/nyquist_cph:.1f} h)")


avg_1km = avg_1km.isel(eta_rho=slice(0,1358), xi_rho=slice(40, None))

us_full = avg_1km.sustr
vs_full = avg_1km.svstr
u_full  = avg_1km.u.isel(s_rho=-1)   # surface zonal velocity
v_full  = avg_1km.v.isel(s_rho=-1)   # surface meridional velocity

time_full = us_full.ocean_time.values

tau_clean   = {}
rotated_vel = {}

for name, (xi, eta) in coords.items():
	angle = grid.angle.isel(eta_rho=eta, xi_rho=xi).values.item()

	# --- wind stress (same staggering as your stick-plot extraction) ---
	tau_u_raw = us_full.isel(eta_u=eta, xi_u=xi - 1).values 
	tau_v_raw = vs_full.isel(eta_v=eta - 1, xi_v=xi).values 
	tau_x, tau_y = apply_rotation(tau_u_raw, tau_v_raw, angle)
	tau_clean[name] = {'tau_x': tau_x.copy(), 'tau_y': tau_y.copy(), 'time': time_full}

	# --- surface current, same staggering/rotation convention as tau ---
	u_raw = u_full.isel(eta_u=eta, xi_u=xi - 1).values
	v_raw = v_full.isel(eta_v=eta - 1, xi_v=xi).values
	u_along, v_across = apply_rotation(u_raw, v_raw, angle)
	rotated_vel[name] = {'u': u_along.copy(), 'v': v_across.copy(), 'time': time_full}


f_local_cph = {}
for name, (xi, eta) in coords.items():
	lat_deg = float(lat_rho[eta, xi].values)
	f_rad_s_local = 2.0 * OMEGA * np.sin(np.deg2rad(lat_deg))
	f_local_cph[name] = abs(f_rad_s_local) / (2 * np.pi) * 3600.0   # cph

diurnal_cph = 1.0 / 24.0   # sea-breeze / solar tide -- same everywhere

for name, f_val in f_local_cph.items():
	assert f_val < nyquist_cph, (
		f"f at point {name} ({f_val:.4f} cph) is at/above Nyquist "
		f"({nyquist_cph:.4f} cph) at this sampling rate -- it will alias."
	)
assert diurnal_cph < nyquist_cph, "diurnal frequency is at/above Nyquist."



def vector_cospectrum(tau_x, tau_y, u, v, fs, window='hann'):
	"""
	Real (non-rotary) wind-stress/current cospectrum,

		Gamma_tau,u(w) = Re{ tau_x^(w) u_x*(w) + tau_y^(w) u_y*(w) },

	i.e. the sum of the x- and y-component cospectra -- this is the
	quantity in Eq. (cospectrum) of the methods text, and what the
	integral over the NI band recovers as INPUT.

	Computed as a SINGLE tapered periodogram over the full record (no
	segment-averaging), which is what gives published wind-work
	cospectra their scattered, unsmoothed look (e.g. von Storch &
	Luschow, 2023, their Fig. 1) rather than a smooth Welch curve.
	Since the inputs are real signals, only frequencies in [0, Nyquist]
	exist; the sign of Gamma at each frequency is physically meaningful
	(positive = wind doing work on the ocean) and is best shown by
	marker style rather than folded into a log-magnitude, hence the
	cross/dot convention used below.
	"""
	tau_x = np.asarray(tau_x, dtype=float)
	tau_y = np.asarray(tau_y, dtype=float)
	u = np.asarray(u, dtype=float)
	v = np.asarray(v, dtype=float)

	mask = np.isfinite(tau_x) & np.isfinite(tau_y) & np.isfinite(u) & np.isfinite(v)
	tau_x, tau_y, u, v = tau_x[mask], tau_y[mask], u[mask], v[mask]
	n = tau_x.size

	win = get_window(window, n)
	U = (win ** 2).sum()  # 'density' normalisation, as in scipy.signal.csd

	def _taper(x):
		return (x - x.mean()) * win

	Ftx = np.fft.rfft(_taper(tau_x))
	Fty = np.fft.rfft(_taper(tau_y))
	Fu = np.fft.rfft(_taper(u))
	Fv = np.fft.rfft(_taper(v))

	freqs = np.fft.rfftfreq(n, d=1.0 / fs)
	Co = np.real(Ftx * np.conj(Fu) + Fty * np.conj(Fv)) / (fs * U)

	return freqs, Co


###########################################################################
####################-----> Demodulation <---###########################

cospec = {}  # kept for the INPUT cross-check below

for name in ['A', 'B', 'C', 'D']:
	tau = tau_clean[name]
	vel = rotated_vel[name]


	freq, Co = vector_cospectrum(tau['tau_x'], tau['tau_y'], vel['u'], vel['v'], fs_cph)
	cospec[name] = dict(freq=freq, Co=Co)

def bandpass(x, low_cph, high_cph, fs_cph, order=4):
	b, a = signal.butter(order, [low_cph, high_cph], btype='band', fs=fs_cph)
	return signal.filtfilt(b, a, x)

print(f"{'Point':<6}{'INPUT direct [W/m2]':<24}{'INPUT from cospectrum [W/m2]':<28}")
input_table = {}
for name in ['A', 'B', 'C', 'D']:
	tau = tau_clean[name]
	vel = rotated_vel[name]

	# each mooring uses ITS OWN local f now, not one shared value
	ni_low, ni_high = 0.8 * f_local_cph[name], 1.2 * f_local_cph[name]

	tau_x_bp = bandpass(tau['tau_x'], ni_low, ni_high, fs_cph)
	tau_y_bp = bandpass(tau['tau_y'], ni_low, ni_high, fs_cph)
	u_bp = bandpass(vel['u'], ni_low, ni_high, fs_cph)
	v_bp = bandpass(vel['v'], ni_low, ni_high, fs_cph)

	INPUT_direct = np.mean(tau_x_bp * u_bp + tau_y_bp * v_bp)  # W/m^2

	# cross-check: integrating the (signed) vector cospectrum over the NI
	# band should recover the same order of magnitude as the direct,
	# time-domain estimate above (Parseval)
	fq, Co = cospec[name]['freq'], cospec[name]['Co']
	band_mask = (fq >= ni_low) & (fq <= ni_high)
	INPUT_cospectrum = np.trapz(Co[band_mask], fq[band_mask])

	input_table[name] = dict(direct=INPUT_direct, cospectrum=INPUT_cospectrum)
	print(f"{name:<6}{INPUT_direct:<24.3e}{INPUT_cospectrum:<28.3e}")

###########################################################################
######-----> Complex demodulation: NI envelope vs diurnal envelope <---####
###########################################################################

def complex_demodulate(u, v, target_cph, fs_cph, lowpass_cph=1.0 / 40.0, order=4):
	"""
	Complex (back-rotated) demodulation of a velocity series about a target
	cyclic frequency `target_cph` [cph]. Removing the target rotation and
	then low-pass filtering isolates the slowly varying amplitude/phase
	envelope of motions near that frequency, independent of any other
	signal (e.g. the diurnal sea breeze) that happens to share the band.

	Returns the complex envelope A(t) and the time vector [hours].
	"""
	w = np.asarray(u, dtype=complex) + 1j * np.asarray(v, dtype=complex)
	t = np.arange(w.size) / fs_cph  # hours

	w_backrot = w * np.exp(-2j * np.pi * target_cph * t)

	b, a = signal.butter(order, lowpass_cph, btype='low', fs=fs_cph)
	envelope = signal.filtfilt(b, a, w_backrot)
	return envelope, t


ref_point = 'D'  # choose the reference mooring for the demodulation check
u_ref = rotated_vel[ref_point]['u'][16:]
v_ref = rotated_vel[ref_point]['v'][16:]
time_ref = rotated_vel[ref_point]['time'][16:]

f_ref = f_local_cph[ref_point]
ni_envelope, t_hours = complex_demodulate(u_ref, v_ref, f_ref, fs_cph, lowpass_cph=1 / 40.0)
diurnal_envelope, _ = complex_demodulate(u_ref, v_ref, diurnal_cph, fs_cph, lowpass_cph=1 / 40.0)

ni_amp = np.abs(ni_envelope)
diurnal_amp = np.abs(diurnal_envelope)

# reconstruct the NI-band signal at its true (rotating) frequency, to
# overlay on the raw current and visually confirm the extracted rotation
ni_reconstructed = ni_envelope * np.exp(2j * np.pi * f_ref * t_hours)


ref_point_B = 'B'  # choose the reference mooring for the demodulation check
u_ref_B = rotated_vel[ref_point_B]['u'][16:]
v_ref_B = rotated_vel[ref_point_B]['v'][16:]
time_ref_B = rotated_vel[ref_point_B]['time'][16:]

f_ref_B = f_local_cph[ref_point_B]
ni_envelope_B, t_hours_B = complex_demodulate(u_ref, v_ref, f_ref_B, fs_cph, lowpass_cph=1 / 40.0)
diurnal_envelope_B, _ = complex_demodulate(u_ref, v_ref, diurnal_cph, fs_cph, lowpass_cph=1 / 40.0)

ni_amp_B = np.abs(ni_envelope_B)
diurnal_amp_B = np.abs(diurnal_envelope_B)

# reconstruct the NI-band signal at its true (rotating) frequency, to
# overlay on the raw current and visually confirm the extracted rotation
ni_reconstructed_B = ni_envelope_B * np.exp(2j * np.pi * f_ref_B * t_hours)




###########################################################################
####################-----> Plotting <---###############################
def format_time_axis(ax):
	"""Standardizes time axis formatting."""
	ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
	ax.tick_params(axis='x', labelsize=10, rotation=25)
	ax.set_xlabel(None)



	
fig = plt.figure(figsize=(9, 9)) #width, height
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1,1,1], width_ratios=[1,1])
gs.update(left=0.1, right=0.98,hspace=0.15, wspace=0.2, top=0.98, bottom=0.05)

############------> Loc, z and virtual moorings

# Create a custom colormap with mint cream, beige, purple, and blue
colors = [
	#(0.96, 1.0, 0.98),   # Very shallow (mint cream, almost white)
	(0.9, 0.8, 0.7),     # Shallow (beige)
	(0.7, 1.0, 0.7),     # Mid-deep (pastel green) 
	(0.5, 0.4, 0.8),     # Mid-depth (purple)
	(1.0, 0.3, 0.5),     # Mid-depth (pink or vibrant red)
	(1.0, 0.65, 0.0),    # Shallow (vibrant orange/gold)
	(0.2, 0.2, 0.7),     # Deeper (blue)
	(0.1, 0.1, 0.4),     # Deepest (dark blue)
]


A = [380, 1233]
B = [276, 896]
C = [333, 557]
D = [453, 323]


cmap = LinearSegmentedColormap.from_list('bathymetry_cmap', colors, N=500)
vmin = -5
vmax = 2500
#norm = LogNorm(vmin=vmin, vmax=vmax)
norm = cm.colors.Normalize(vmin=vmin,vmax=vmax)
levels=np.linspace(vmin,vmax,200)

bar_title = "m"
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.set_aspect('auto')
ax.set_ylim(bottom =-33,top = -21)
ax.set_xlim(left = -52,right = -39.2)

### - contourf
norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
ax.contourf(lon_rho, lat_rho, h,  levels=levels, cmap=cmap, norm=norm, extend='max', zorder=0)

ax.text(0.7, 0.98, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
		

#--> virtual moorings
ax.scatter(lon_rho[A[1],A[0]], lat_rho[A[1],A[0]], zorder=5, s=40, marker='*', color='k', label='A')
ax.scatter(lon_rho[B[1],B[0]], lat_rho[B[1],B[0]], zorder=5, s=40, marker='v', color='blue', label='B')
ax.scatter(lon_rho[C[1],C[0]], lat_rho[C[1],C[0]], zorder=5, s=40, marker='s', color='orange', label='C')
ax.scatter(lon_rho[D[1],D[0]], lat_rho[D[1],D[0]], zorder=5, s=40, marker='^', color='red', label='D')

legend = ax.legend(loc=4, fontsize='x-small', markerscale=0.7)

# perks
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax.patch.set_edgecolor('black')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)

gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 8, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 8, 'color': 'dimgrey'}

## isobaths
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]

levels1 = np.asarray(levels_1)
levels2 = np.asarray(levels_2)
levels3 = np.asarray(levels_3)
levels4 = np.asarray(levels_4)

c1 = ax.contour(lon_rho, lat_rho, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = ax.contour(lon_rho, lat_rho, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = ax.contour(lon_rho, lat_rho, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = ax.contour(lon_rho, lat_rho, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
#ax.hlines(y=-25.8, xmin=-49.2, xmax=-42, linewidth=1, color='purple',zorder = 3)
#ax.text(-45.3, -25.9, 'Santos Bifurcation', fontsize=7, ha='left', va='top', 
	   #bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Use proxy artists to create legend entries
legend_lines = [Line2D([0], [0], linestyle='dotted', linewidth=1, color='brown'),
				Line2D([0], [0], linestyle='dotted', linewidth=1, color='grey'),
				Line2D([0], [0], linestyle='dashed', linewidth=1, color='k'),
				Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')]

labels = ['50 m', '200 m', '1000 m', '2000 m']

# Updated legend part using fig.legend
#fig.legend(legend_lines, labels, loc='upper right', title='Isobaths', fontsize='x-small', title_fontsize='x-small')

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='x-small',
	title_fontsize='x-small',
	loc='center',
	bbox_to_anchor=(0.14, 0.85)
)


### -- Miniglobe
extent = [-49, -39.3, -29.5, -22]
center = [-45, -15]
box_color = 'red'
box_alpha = 1
box_edge_width = 1

sub_ax = inset_axes(ax, width="100%", height="100%", loc="lower center", bbox_to_anchor=(0.75, 0.25, 0.25, 0.25),
					bbox_transform=ax.transAxes,
					axes_class=cartopy.mpl.geoaxes.GeoAxes,
					axes_kwargs=dict(map_projection=ccrs.Orthographic(center[0], center[1])))
# Make a nice border around the inset axes.

# Add land and ocean features to the miniglobe
land_feature = NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=COLORS['land'])
ocean_feature = NaturalEarthFeature('physical', 'ocean', '110m', edgecolor='face', facecolor=COLORS['water'])

sub_ax.add_feature(land_feature)
sub_ax.add_feature(ocean_feature)

effect = Stroke(linewidth=.1, foreground='black', alpha=0.5)
sub_ax.patch.set_path_effects([effect])
sub_ax.coastlines(linewidth=0.00000001, edgecolor='k', alpha=.8)
extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='lightgray',
					  edgecolor=box_color, linewidth=box_edge_width, alpha=box_alpha)

# colorbar
cbar_1 = inset_axes(ax, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
norm_1 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm_1, extend='max', orientation='horizontal')
cb1.set_label(bar_title, size='x-small')
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small')  # Set colorbar label size

CF = [-23.018, -41.97]
CS = [-22.015, -40.963]
CM = [-28.60, -48.809]
SSI = [-23.8310, -45.4089]

# CS
#ax.scatter(CS[1], CS[0],marker='D',zorder = 3, color='k', s=20)
#ax.text(CS[1] - 0.3, CS[0], 'CST', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# CF
ax.scatter(CF[1], CF[0],marker='D',zorder = 3, color='k', s=20)
ax.text(CF[1] - 0.3, CF[0] + 0.1, 'CF', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# CM
ax.scatter(CM[1], CM[0],marker='D',zorder = 3, color='k', s=20)
ax.text(CM[1] -0.25, CM[0] + 0.3, 'CSM', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# SSI
ax.scatter(SSI[1], SSI[0],marker='.',zorder = 3, color='k', s=20)
ax.text(SSI[1] -0.25, SSI[0] + 0.3, 'SSI', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')


#---> grid_subgrid
ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.set_aspect('auto')

ax2.scatter(lon_c_flat[coarse_ocean], lat_c_flat[coarse_ocean],s=0.001, color='blue', alpha=0.5, transform=ccrs.PlateCarree())
ax2.scatter(lon_f_flat, lat_f_flat, s=0.001, color='indianred', alpha=0.5, transform=ccrs.PlateCarree(),zorder = 2)

ax2.scatter(lon_c_flat[coarse_land], lat_c_flat[coarse_land],s=0.001, color='gray', alpha=1, transform=ccrs.PlateCarree(),zorder = 2)
#ax.scatter(lon_f_flat[finer_land], lat_f_flat[finer_land], s=0.001, color='whitesmoke', alpha=1, transform=ccrs.PlateCarree(),zorder = 2)

gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 8, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 8, 'color': 'dimgrey'}
# Custom legend
legend_patches = [
	mpatches.Patch(color='blue', label='Coarser'),
	mpatches.Patch(color='indianred', label='Finer'),

]
ax2.text(0.7, 0.98, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')


###########---> Plotting the hurricane track
type_colors = {
	'TD': {'color': 'yellowgreen', 'marker': 'o', 'label': 'Tropical Depression'},
	'TS': {'color': 'magenta', 'marker': 'o', 'label': 'Tropical Storm'},
	'HU': {'color': 'gold', 'marker': 'o', 'label': 'Hurricane'}, # Catarina reached HU status
	'SS': {'color': 'red', 'marker': 'o', 'label': 'Subtropical Storm'},
	'EX': {'color': 'grey', 'marker': 's', 'label': 'Extratropical'}
	# Add other types if needed (e.g., LO, wave, DB, etc.)
}

# --- 2. Plot the Storm Track (Line) ---
# Plotting the line connects the track points sequentially
# Ensure zorder is high so it plots over the bathymetry (zorder=0) and land (zorder=1).
ax2.plot(storm.lon, storm.lat, 
		color='k', 
		linestyle='--', 
		linewidth=0.8, 
		alpha=0.6,
		transform=ccrs.PlateCarree(), 
		zorder=6,
		label='Catarina Track')


# --- 3. Plot Scatter Points with Color by Type ---

# Get the unique storm types present in your storm data
unique_types = np.unique(storm.type)

# Loop through each unique type to plot points and create a legend entry
for storm_type in unique_types:
	if storm_type in type_colors:
		style = type_colors[storm_type]
		
		# Create a boolean mask to select points matching the current type
		mask = (storm.type == storm_type)
		
		# Plot the points
		ax2.scatter(storm.lon[mask], storm.lat[mask], 
				   c=style['color'], 
				   marker=style['marker'], 
				   s=30, # size of the marker
				   edgecolors='k', # black outline for contrast
				   linewidths=0.5,
				   transform=ccrs.PlateCarree(), 
				   zorder=7, 
				   label=style['label'])

# --- 4. Plot Initial and Final Points (Optional, for emphasis) ---

# Optional: Add a large marker for the starting point
ax2.scatter(storm.lon[0], storm.lat[0], 
		   marker='^', 
		   s=60, 
		   color='green', 
		   edgecolor='k', 
		   linewidth=1,
		   transform=ccrs.PlateCarree(), 
		   zorder=8, 
		   label='Start')
		   
# Optional: Add a large marker for the final point
ax2.scatter(storm.lon[-1], storm.lat[-1], 
		   marker='v', 
		   s=60, 
		   color='brown', 
		   edgecolor='k', 
		   linewidth=1,
		   transform=ccrs.PlateCarree(), 
		   zorder=8, 
		   label='End')



ax2.legend(loc='upper left', fontsize=6,  title_fontsize=6)

###---> vamx
pcx = inset_axes(ax2, width="50%", height="30%", loc='lower left',
				 bbox_to_anchor=(0.12, 0.05, 0.96, 0.95), 
				 bbox_transform=ax2.transAxes)

# Give it a solid background so map lines don't bleed through
pcx.set_facecolor('white')
pcx.patch.set_alpha(0.9) 

# Inverted data: Vmax on X-axis, Time on Y-axis
pcx.plot(time_cat, vmax_cat, label='Track', linestyle=':', color='green', linewidth=1.2)
pcx.plot(time_cat,vmax_blend, label='Blend', linestyle='--', color='orange', linewidth=1.2)

# Formatting the inset
pcx.set_ylim(10, 45)
pcx.set_ylabel(r'V$_{max}$ (m s$^{-1}$)', fontsize=7)
pcx.tick_params(axis='both', labelsize=7)

# Handle time formatting on the Y-axis instead of X
pcx.xaxis.set_major_locator(mdates.DayLocator(interval=3))
pcx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

pcx.grid(True, linestyle=':', alpha=0.5)
pcx.legend(loc=2, fontsize=6, frameon=False)



####################################################
#####-----> Stick plot<---##########
point_labels = ['A', 'B', 'C', 'D']
offsets = [0.8, 0.6, 0.4, 0.2]  # Tighten offsets for stress magnitudes
ax4 = plt.subplot(gs[1, 1])
ax4.text(0.1, 0.98, '(d)', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

# Use the time from the xarray object
time = rotated_stress['A']['time']

for i, name in enumerate(['A', 'B', 'C', 'D']):
	data = rotated_stress[name]
	offset = offsets[i]
	
	# Quiver for wind stress
	q = ax4.quiver(data['time'], np.full(len(data['time']), offset),
				   data['tau_x'], data['tau_y'],
				   color='black', 
				   alpha=0.8,
				   units='y',     
				   scale=3,      
				   headlength=0,   
				   headaxislength=0, 
				   width=0.008)

# 3. Update Quiver Legend for Stress
ax4.quiverkey(q, X=0.8, Y=0.96, U=0.1 ,
			  label=r'0.1 $Pa$', labelpos='E', 
			  coordinates='axes', fontproperties={'weight': 'bold'})

# Formatting
ax4.set_yticks(offsets)
ax4.set_yticklabels(point_labels)
ax4.set_ylim(0, 1.0)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=15)
ax4.set_ylabel(None)

h_start = np.datetime64("2004-03-25")
h_end = np.datetime64("2004-03-30") 
ax4.axvspan(h_start, h_end, color='red', alpha=0.15, zorder=0)
ax4.text(h_start, 1, 'Hurricane',color='red', fontsize=8, fontweight='bold', ha='left',va='bottom') 
plt.tight_layout()


# Formatting
ax4.set_yticks(offsets)
ax4.set_yticklabels(point_labels)
ax4.set_ylim(0, 1.0)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.xticks(rotation=15)
ax4.set_ylabel(None)

h_start = np.datetime64("2004-03-25")
h_end = np.datetime64("2004-03-30") 
ax4.axvspan(h_start, h_end, color='red', alpha=0.15, zorder=0)
ax4.text(h_start, 1, 'Hurricane',color='red', fontsize=8, fontweight='bold', ha='left',va='bottom') 


####################################################
#####-----> EKE <---##########
south_top = 420
central_top = 900


vmin = 0
vmax = 6
cmap = plt.cm.gist_ncar_r
norm = cm.colors.Normalize(vmin=vmin,vmax=vmax)
levels=np.linspace(vmin,vmax,200)

bar_title = 'kJ m$^{-2}$'
ax_eke = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax_eke.set_aspect('auto')
ax_eke.set_ylim(bottom =-33,top = -21)
ax_eke.set_xlim(left = -52,right = -39.2)

### - contourf
norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
ax_eke.contourf(lon_eke[5:-5,:-5], lat_eke[5:-5,:-5],(eke_j[5:-5,:-5]/1000),  levels=levels, cmap=cmap, norm=norm, extend='max', zorder=0)

ax_eke.text(0.7, 0.98, '(c)', transform=ax_eke.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
		
ax_eke.coastlines()
ax_eke.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax_eke.patch.set_edgecolor('black')
gl = ax_eke.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)

gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 8, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 8, 'color': 'dimgrey'}

## isobaths
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]

levels1 = np.asarray(levels_1)
levels2 = np.asarray(levels_2)
levels3 = np.asarray(levels_3)
levels4 = np.asarray(levels_4)

c1 = ax_eke.contour(lon_rho, lat_rho, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = ax_eke.contour(lon_rho, lat_rho, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = ax_eke.contour(lon_rho, lat_rho, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = ax_eke.contour(lon_rho, lat_rho, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)


# colorbar
cbar_1 = inset_axes(ax_eke, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
norm_1 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm_1, extend='max', orientation='horizontal')
cb1.set_label(bar_title, size='x-small')
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small')  # Set colorbar label size



ax_eke.scatter(lon_rho[central_top,40:], lat_rho[central_top,40:],c='green',s=0.05,marker='.')
ax_eke.scatter(lon_rho[south_top,40:], lat_rho[south_top,40:],c='k',s=0.05,marker='.')
ax_eke.text(lon_rho[central_top, 608]+0.2, lat_rho[central_top, 608]-1, 'North', color='green',zorder=5, fontsize='x-small', verticalalignment='bottom')
ax_eke.text(lon_rho[south_top, 608]+0.2, lat_rho[south_top, 608]-1, 'South', color='k', zorder=5,fontsize='x-small', verticalalignment='bottom')


####################################################
#####-----> GAMMA  <---##########


point_colors = {'A': 'black', 'B': 'blue', 'C': 'orange', 'D': 'red'}

ax_co = plt.subplot(gs[2, 0])
ax_co.text(0.01, 0.1, '(e)', transform=ax_co.transAxes, fontsize=10,
		  fontweight='bold', va='top', ha='left')

cospec = {}  # kept for the INPUT cross-check below

for name in ['A', 'B', 'C', 'D']:
	tau = tau_clean[name]
	vel = rotated_vel[name]
	color = point_colors[name]

	freq, Co = vector_cospectrum(tau['tau_x'], tau['tau_y'], vel['u'], vel['v'], fs_cph)
	cospec[name] = dict(freq=freq, Co=Co)

	pos = Co > 0
	neg = ~pos

	# crosses = positive (wind -> ocean); dots = negative (ocean -> wind)
	ax_co.scatter(freq[pos], Co[pos], marker='+', s=14, linewidths=0.7,
				color=color, alpha=0.75, zorder=3)
	ax_co.scatter(freq[neg], np.abs(Co[neg]), marker='.', s=10,
				color=color, alpha=0.55, zorder=2)

	# per-mooring f (dashed) and 2f (dotted), colour-matched to the data
	f_val = f_local_cph[name]
	ax_co.axvline(f_val, color=color, linestyle='--', linewidth=1.0, zorder=1)
	ax_co.axvline(2 * f_val, color=color, linestyle=':', linewidth=1.0, zorder=1)



freq_all = np.concatenate([cospec[n]['freq'] for n in ['A', 'B', 'C', 'D']])
Co_all = np.concatenate([cospec[n]['Co'] for n in ['A', 'B', 'C', 'D']])

f_lo = min(f_local_cph.values())
anchor_band = (freq_all >= 0.8 * f_lo) & (freq_all <= 1.2 * f_lo)
if not np.any(anchor_band):
	anchor_band = freq_all > 0  # fallback if the band is empty

sub_freq, sub_Co = freq_all[anchor_band], np.abs(Co_all[anchor_band])
idx_max = np.argmax(sub_Co)
f_anchor, A_anchor = sub_freq[idx_max], sub_Co[idx_max]

print(f"Slope-line anchor: f = {f_anchor:.4f} cph, |Gamma| = {A_anchor:.3e} Pa m/s")

freq_pos = freq_all[freq_all > 0]
freq_ref = np.logspace(np.log10(freq_pos.min()), np.log10(0.95 * nyquist_cph), 50)
ax_co.plot(freq_ref, A_anchor * (freq_ref / f_anchor) ** (-6), 'k--', lw=1, zorder=1)
ax_co.plot(freq_ref, A_anchor * (freq_ref / f_anchor) ** (-4), 'k:', lw=1, zorder=1)

ax_co.set_xscale('log')
ax_co.set_yscale('log')
ax_co.set_xlim(1e-2, 0.95 * nyquist_cph)   
ax_co.set_ylim(1e-5, 1e0)   

ax_co.set_xlabel('Frequency [cph]')
#ax_co.set_ylabel(r'$\left[\mathrm{Pa}\,\dfrac{\mathrm{m}}{\mathrm{s}}\right]$')
ax_co.set_ylabel(r'Pa m s$^{-1}$', fontsize=9)


legend_elems = [
	Line2D([0], [0], marker='+', color='k', linestyle='None', markersize=7,
		   label='positive'),
	Line2D([0], [0], marker='.', color='k', linestyle='None', markersize=7,
		   label='negative'),
	Line2D([0], [0], color='k', lw=1, linestyle='--', label=r'$\omega^{-6}$'),
	Line2D([0], [0], color='k', lw=1, linestyle=':', label=r'$\omega^{-4}$'),
] + [mpatches.Patch(color=point_colors[n], label=f'{n}') for n in ['A', 'B', 'C', 'D']]
ax_co.legend(handles=legend_elems, loc='upper right', fontsize=6.5, ncol=2, frameon=True)

########################---> Demodulation
gs_nested = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 1], hspace=0.15)

axr = plt.subplot(gs_nested[0, 0])
axe = plt.subplot(gs_nested[1, 0], sharex=axr)

# --- Top Plot (axr) ---
# Note: Fixed the transform typo from ax_co to axr for the panel label
axr.text(0.05, 0.1, '(f)', transform=axr.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

axr.plot(time_ref[2:-2], u_ref[2:-2], color='grey', lw=0.6, label='$u$ @ D')
axr.plot(time_ref[2:-2], ni_reconstructed[2:-2].real, color='crimson', lw=1.2, label='NI signal')
axr.axvspan(h_start, h_end, color='red', alpha=0.12, zorder=0)
axr.set_ylabel('m s$^{-1}$')
axr.legend(loc='upper right', fontsize=8)

# Hide x-axis tick labels for the top plot since it shares the axis with the bottom
plt.setp(axr.get_xticklabels(), visible=False)


# --- Bottom Plot (axe) ---
axe.text(0.05, 0.1, '(g)', transform=axe.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

axe.plot(time_ref_B[2:-2], ni_amp_B[2:-2], color='crimson', lw=1.4, label='NI $|A_f|$')
axe.plot(time_ref_B[2:-2], diurnal_amp_B[2:-2], color='steelblue', lw=1.2, ls='--', label='diurnal $|A_{1/24h}|$')
axe.axvspan(h_start, h_end, color='red', alpha=0.12, zorder=0)
axe.set_ylabel('m s$^{-1}$')
axe.legend(loc='upper right', fontsize=8)
axe.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
axe.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig('fig1_loc.png', dpi = 300)
