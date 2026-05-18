
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



A =  [376,1239]
B =  [241,957]
C =  [299,570]
D =  [463,225]

#################################################################################
############----> MLD <------####################################
path_pot ='/data1/roms_dd_waves/analysis_outs/NIW/pot_energy/'

ml = xr.open_mfdataset(path_pot + 'mld_slice_*.nc').isel(ocean_time=slice(96,None)).mld


ml_A = ml.isel(eta_rho=A[1],xi_rho=A[0]).compute()
ml_B = ml.isel(eta_rho=B[1],xi_rho=B[0]).compute()
ml_C = ml.isel(eta_rho=C[1],xi_rho=C[0]).compute()
ml_D = ml.isel(eta_rho=D[1],xi_rho=D[0]).compute()


#################################################################################
############----> dz <------####################################
dzs = '/data1/roms_dd_waves/analysis_outs/dz/niws/'
dz = xr.open_mfdataset(dzs + f'dz_*.nc')

z_A = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=A[1],xi_rho=A[0]).z_rho.values
z_B = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=B[1],xi_rho=B[0]).z_rho.values
z_C = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=C[1],xi_rho=C[0]).z_rho.values
z_D = xr.open_mfdataset(dzs + f'dz_*.nc').isel(eta_rho=D[1],xi_rho=D[0]).z_rho.values


#################################################################################
############----> Shear <------####################################

sh_p = '/data1/roms_dd_waves/analysis_outs/NIW/N2/shear/comps/'

shear = xr.open_mfdataset(sh_p + 'comp_vert_shear_slice_*.nc').isel(ocean_time=slice(96,None),s_w=slice(1,-1))

Uz_A = shear.dudz.isel(eta_rho=A[1],xi_rho=A[0]).compute()
Uz_B = shear.dudz.isel(eta_rho=B[1],xi_rho=B[0]).compute()
Uz_C = shear.dudz.isel(eta_rho=C[1],xi_rho=C[0]).compute()
Uz_D = shear.dudz.isel(eta_rho=D[1],xi_rho=D[0]).compute()


Vz_A = shear.dvdz.isel(eta_rho=A[1],xi_rho=A[0]).compute()
Vz_B = shear.dvdz.isel(eta_rho=B[1],xi_rho=B[0]).compute()
Vz_C = shear.dvdz.isel(eta_rho=C[1],xi_rho=C[0]).compute()
Vz_D = shear.dvdz.isel(eta_rho=D[1],xi_rho=D[0]).compute()


#################################################################################
############----> PSD <------####################################

dt_seconds = (Uz_A.ocean_time[1] - Uz_A.ocean_time[0]).values / np.timedelta64(1, 's')
fs_cph = 3600 / dt_seconds  # Cycles per hour
janela = ('blackman')
noverlap = 20

# 2. Define a wrapper function for Scipy Welch
def calc_shear_psd(da_u, da_v, fs):
	"""
	Calculates the Total Shear PSD (PSD_u + PSD_v).
	Input: DataArrays for Uz and Vz with dimensions (ocean_time, s_w)
	"""
	# Define window length. For 670 points, nperseg=128 or 256 is a good balance.
	nperseg = 128 
	
	# Calculate PSD for U component
	f, Pxx_u = signal.welch(da_u, fs=fs,window =janela,detrend='linear', average='median', nperseg=nperseg, noverlap=noverlap, axis=0)
	
	# Calculate PSD for V component
	_, Pxx_v = signal.welch(da_v, fs=fs,window =janela,detrend='linear', average='median', nperseg=nperseg, noverlap=noverlap, axis=0)
	
	# Sum them to get Total Shear Variance (Magnitude)
	Pxx_total = Pxx_u + Pxx_v
	
	# Create new DataArray
	# Dimensions are now (frequency, s_w)
	psd_da = xr.DataArray(
		Pxx_total,
		coords={'freq': f, 's_w': da_u.s_w},
		dims=('freq', 's_w'),
		name='Shear_PSD',
		attrs={'units': '(1/s^2) / cph', 'long_name': 'Vertical Shear PSD'}
	)
	return psd_da




psd_A = calc_shear_psd(Uz_A.fillna(0), Vz_A.fillna(0), fs_cph)
psd_plot_A = psd_A.isel(freq=slice(1, None))
f_A = gsw.f(ml_A.lat_rho).values #cycles per second
f_A_cph = (f_A*3600)/(2 * np.pi)

psd_B = calc_shear_psd(Uz_B.fillna(0), Vz_B.fillna(0), fs_cph)
psd_plot_B = psd_B.isel(freq=slice(1, None))

f_B = gsw.f(ml_B.lat_rho).values #cycles per second
f_B_cph = (f_B*3600)/(2 * np.pi)


psd_C = calc_shear_psd(Uz_C.fillna(0), Vz_C.fillna(0), fs_cph)
psd_plot_C = psd_C.isel(freq=slice(1, None))

f_C = gsw.f(ml_C.lat_rho).values #cycles per second
f_C_cph = (f_C*3600)/(2 * np.pi)

psd_D = calc_shear_psd(Uz_D.fillna(0), Vz_D.fillna(0), fs_cph)
psd_plot_D = psd_D.isel(freq=slice(1, None))

f_D = gsw.f(ml_D.lat_rho).values #cycles per second
f_D_cph = (f_D*3600)/(2 * np.pi)

M2 = 2*np.pi/12.42
#################################################################################
############----> Zonal shear ni <------####################################


# ---- Sampling frequency (hourly sampling) ----
dt_hours = 1.0
fs = 1 / dt_hours # sampling frequency in Hz

# ---- Bandpass filter parameters ----#
def filter_da_bandpass(da, T1, T2, fs, time_dim='ocean_time', order=4):
	"""
	Apply a Butterworth bandpass filter along the time dimension of a 4D DataArray.

	Parameters:
	-----------
	da : xr.DataArray
		The input 4D DataArray with dimensions (time, s_rho, eta_rho, xi_rho)
	T1, T2 : float
		Period limits in hours (e.g., 28h and 44h)
	time_dim : str
		The name of the time dimension
	order : int
		The order of the Butterworth filter

	Returns:
	--------
	xr.DataArray
		The filtered 4D DataArray, same dims as input
	"""
	# Get time step in hours
	time_vals = da[time_dim].values
	dt_hours = (time_vals[1] - time_vals[0]) / np.timedelta64(1, 'h')
	#fs = 1 / dt_hours  # sampling frequency in cycles per hour

	# Bandpass frequency range in cph
	f_low = 1 / T2  # Higher period = lower frequency
	f_high = 1 / T1  # Lower period = higher frequency

	# Normalised frequencies
	nyq = 0.5 * fs
	low = f_low / nyq
	high = f_high / nyq

	# Butterworth bandpass
	b, a = signal.butter(order, [low, high], btype='band')

	# Reshape to (time, -1) for filtering
	original_shape = da.shape
	reshaped = da.data.reshape((original_shape[0], -1))

	# Apply zero-phase filter along time axis
	filtered = signal.filtfilt(b, a, reshaped, axis=0, padlen=3*max(len(b), len(a)))

	# Restore shape and return as DataArray
	da_filtered = xr.DataArray(
		filtered.reshape(original_shape),
		dims=da.dims,
		coords=da.coords,
		attrs=da.attrs
	)

	return da_filtered


###################
T_A = (2*np.pi/np.absolute(f_A))/3600 #inertial period in hours
T_A_1 = 0.8 * T_A  # Hours
T_A_2 = 1.5 * T_A  # Hours
Uz_A_p = filter_da_bandpass(Uz_A, T_A_1, T_A_2, fs).compute()
Vz_A_p = filter_da_bandpass(Vz_A, T_A_1, T_A_2, fs).compute()


T_B = (2*np.pi/np.absolute(f_B))/3600 #inertial period in hours
T_B_1 = 0.8 * T_B  # Hours
T_B_2 = 1.5 * T_B # Hours
Uz_B_p = filter_da_bandpass(Uz_B, T_B_1, T_B_2, fs).compute()
Vz_B_p = filter_da_bandpass(Vz_B, T_B_1, T_B_2, fs).compute()


T_C = (2*np.pi/np.absolute(f_C))/3600 #inertial period in hours
T_C_1 = 0.8 * T_C  # Hours
T_C_2 = 1.5 * T_C  # Hours
Uz_C_p = filter_da_bandpass(Uz_C, T_C_1, T_C_2, fs).compute()
Vz_C_p = filter_da_bandpass(Vz_C, T_C_1, T_C_2, fs).compute()

T_D = (2*np.pi/np.absolute(f_D))/3600 #inertial period in hours
T_D_1 = 0.8 * T_D  # Hours
T_D_2 = 1.5 * T_D  # Hours
Uz_D_p = filter_da_bandpass(Uz_D, T_D_1, T_D_2, fs).compute()
Vz_D_p = filter_da_bandpass(Vz_D, T_D_1, T_D_2, fs).compute()

time = Uz_A_p.ocean_time



fig = plt.figure(figsize=(9, 5))  #width, height
gs = gridspec.GridSpec(nrows=2, ncols=5, width_ratios=[24.5,24.5,24.5,24.5,2], height_ratios=[1,1])
gs.update(left=0.05, right=0.96, wspace=0.3, hspace=0.2, top=0.99, bottom=0.07)

##############----> PSD <--################
bar_title = r'PSD s$^{-2}$.cph$^{-1}$'
psd_cb = plt.subplot(gs[0, 4])

vmin_psd = 1e-9
vmax_psd = 1e-3
cmap_psd = cmo.cm.thermal

norm_psd =colors.LogNorm(vmin=vmin_psd, vmax=vmax_psd)
cb_psd = mpl.colorbar.ColorbarBase(psd_cb, cmap=cmap_psd, norm=norm_psd, extend='both', orientation='vertical')
cb_psd.set_label(bar_title, size='x-small')
psd_cb.yaxis.set_ticks_position('left')
psd_cb.tick_params(axis='y', labelsize='x-small', rotation=25)

##############----> Shear <--################
bar_title =r'$v_{z} \times 10^{-3}$ s$^{-1}$'
s_cb = plt.subplot(gs[1, 4])

vmin_s = -6
vmax_s = 6
#cmap_s = cmo.cm.curl
cmap_s = plt.cm.RdBu

norm_s = mpl.colors.Normalize(vmin=vmin_s, vmax=vmax_s)

cb_s = mpl.colorbar.ColorbarBase(s_cb, cmap=cmap_s, norm=norm_s, extend='both', orientation='vertical')
cb_s.set_label(bar_title, size='x-small')
s_cb.yaxis.set_ticks_position('left')
s_cb.tick_params(axis='y', labelsize='x-small', rotation=25)

##---> A
#psd
plt_psd_A = plt.subplot(gs[0, 0])
plt_psd_A.text(0.02, 0.05, '(a)', transform=plt_psd_A.transAxes, color='k',fontsize=9, fontweight='bold')
plt_psd_A.set_ylim(-200, 0) 
plt_psd_A.set_xscale('log')
plt_psd_A.pcolormesh(psd_plot_A.freq, z_A, psd_plot_A.T, cmap=cmap_psd,norm=norm_psd)
plt_psd_A.axvline(x=abs(f_A_cph), color='cyan', linestyle='-', label='f')
plt_psd_A.axvline(x=0.8*abs(f_A_cph), color='fuchsia', linestyle=':', label='0.8f')
plt_psd_A.axvline(x=1.5*abs(f_A_cph), color='lime', linestyle='--', label='1.5f')
plt_psd_A.legend(fontsize='x-small')
plt_psd_A.set_xlabel('cph',size='x-small')
plt_psd_A.set_ylabel('m',fontsize='x-small')
plt_psd_A.tick_params(axis='both', labelsize='xx-small')
plt_psd_A.tick_params(axis='x', labelsize='xx-small')
plt_psd_A.tick_params(axis='y', labelsize='xx-small')

#shear
s_A = plt.subplot(gs[1, 0])
s_A.text(0.02, 0.05, '(e)', transform=s_A.transAxes, color='k',fontsize=9, fontweight='bold')
#s_A.fill_between(time, 0, ml_A*-1, color='white', zorder=3)
s_A.plot(time,ml_A*-1,c='grey',lw=1.2,zorder=4,label='mld')
s_A.set_ylim(-200, 0) 
s_A.pcolormesh(time, z_A, Vz_A_p.T*1000, cmap=cmap_s,norm=norm_s)
s_A.xaxis.set_major_locator(mdates.DayLocator(interval=5))
s_A.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
s_A.legend(fontsize='x-small',loc=4)
s_A.set_xlabel(None)
s_A.set_ylabel('m',fontsize='x-small')
s_A.tick_params(axis='both', labelsize='xx-small')
s_A.tick_params(axis='x', labelsize='xx-small', rotation=25)
s_A.tick_params(axis='y', labelsize='xx-small')


##---> B
#psd
plt_psd_B = plt.subplot(gs[0, 1])
plt_psd_B.text(0.02, 0.05, '(b)', transform=plt_psd_B.transAxes, color='k',fontsize=9, fontweight='bold')
plt_psd_B.set_ylim(-200, 0) 
plt_psd_B.set_xscale('log')
plt_psd_B.pcolormesh(psd_plot_B.freq, z_B, psd_plot_B.T, cmap=cmap_psd,norm=norm_psd)
plt_psd_B.axvline(x=abs(f_B_cph), color='cyan', linestyle='-', label='f')
plt_psd_B.axvline(x=0.8*abs(f_B_cph), color='fuchsia', linestyle=':', label='0.8f')
plt_psd_B.axvline(x=1.5*abs(f_B_cph), color='lime', linestyle='--', label='1.5f')
plt_psd_B.set_xlabel('cph',size='x-small')
plt_psd_B.set_ylabel(None)
plt_psd_B.tick_params(axis='both', labelsize='xx-small')
plt_psd_B.tick_params(axis='x', labelsize='xx-small')
plt_psd_B.tick_params(axis='y', labelsize='xx-small')

#shear
s_B = plt.subplot(gs[1, 1])
s_B.text(0.02, 0.05, '(f)', transform=s_B.transAxes, color='k',fontsize=9, fontweight='bold')
#s_B.fill_between(time, 0, ml_B*-1, color='white', zorder=3)
s_B.plot(time,ml_B*-1,c='grey',lw=1.2,zorder=4,label='mld')
s_B.set_ylim(-200, 0) 
s_B.pcolormesh(time, z_B, Vz_B_p.T*1000, cmap=cmap_s,norm=norm_s)
s_B.xaxis.set_major_locator(mdates.DayLocator(interval=5))
s_B.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
s_B.set_xlabel(None)
s_B.set_ylabel(None)
s_B.tick_params(axis='both', labelsize='xx-small')
s_B.tick_params(axis='x', labelsize='xx-small', rotation=25)
s_B.tick_params(axis='y', labelsize='xx-small')

##---> C
#psd
plt_psd_C = plt.subplot(gs[0, 2])
plt_psd_C.text(0.02, 0.05, '(c)', transform=plt_psd_C.transAxes, color='k',fontsize=9, fontweight='bold')
plt_psd_C.set_ylim(-200, 0) 
plt_psd_C.set_xscale('log')
plt_psd_C.pcolormesh(psd_plot_C.freq, z_C, psd_plot_C.T, cmap=cmap_psd,norm=norm_psd)
plt_psd_C.axvline(x=abs(f_C_cph), color='cyan', linestyle='-', label='f')
plt_psd_C.axvline(x=0.8*abs(f_C_cph), color='fuchsia', linestyle=':', label='0.8f')
plt_psd_C.axvline(x=1.5*abs(f_C_cph), color='lime', linestyle='--', label='1.5f')
plt_psd_C.set_xlabel('cph',size='x-small')
plt_psd_C.set_ylabel(None)
plt_psd_C.tick_params(axis='both', labelsize='xx-small')
plt_psd_C.tick_params(axis='x', labelsize='xx-small')
plt_psd_C.tick_params(axis='y', labelsize='xx-small')

#shear
s_C = plt.subplot(gs[1, 2])
s_C.text(0.02, 0.05, '(g)', transform=s_C.transAxes, color='k',fontsize=9, fontweight='bold')
#s_C.fill_Cetween(time, 0, ml_C*-1, color='white', zorder=3)
s_C.plot(time,ml_C*-1,c='grey',lw=1.2,zorder=4,label='mld')
s_C.set_ylim(-200, 0) 
s_C.pcolormesh(time, z_C, Vz_C_p.T*1000, cmap=cmap_s,norm=norm_s)
s_C.xaxis.set_major_locator(mdates.DayLocator(interval=5))
s_C.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
s_C.set_xlabel(None)
s_C.set_ylabel(None)
s_C.tick_params(axis='both', labelsize='xx-small')
s_C.tick_params(axis='x', labelsize='xx-small', rotation=25)
s_C.tick_params(axis='y', labelsize='xx-small')


##---> D
#psd
plt_psd_D = plt.subplot(gs[0, 3])
plt_psd_D.text(0.02, 0.05, '(d)', transform=plt_psd_D.transAxes, color='k',fontsize=9, fontweight='bold')
plt_psd_D.set_ylim(-200, 0) 
plt_psd_D.set_xscale('log')
plt_psd_D.pcolormesh(psd_plot_D.freq, z_D, psd_plot_D.T, cmap=cmap_psd,norm=norm_psd)
plt_psd_D.axvline(x=abs(f_D_cph), color='cyan', linestyle='-', label='f')
plt_psd_D.axvline(x=0.8*abs(f_D_cph), color='fuchsia', linestyle=':', label='0.8f')
plt_psd_D.axvline(x=1.5*abs(f_D_cph), color='lime', linestyle='--', label='1.5f')
plt_psd_D.set_xlabel('cph',size='x-small')
plt_psd_D.set_ylabel(None)
plt_psd_D.tick_params(axis='both', labelsize='xx-small')
plt_psd_D.tick_params(axis='x', labelsize='xx-small')
plt_psd_D.tick_params(axis='y', labelsize='xx-small')

#shear
s_D = plt.subplot(gs[1, 3])
s_D.text(0.02, 0.05, '(h)', transform=s_D.transAxes, color='k',fontsize=9, fontweight='bold')
#s_D.fill_Detween(time, 0, ml_D*-1, color='white', zorder=3)
s_D.plot(time,ml_D*-1,c='grey',lw=1.2,zorder=4,label='mld')
s_D.set_ylim(-200, 0) 
s_D.pcolormesh(time, z_D, Vz_D_p.T*1000, cmap=cmap_s,norm=norm_s)
s_D.xaxis.set_major_locator(mdates.DayLocator(interval=5))
s_D.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
s_D.set_xlabel(None)
s_D.set_ylabel(None)
s_D.tick_params(axis='both', labelsize='xx-small')
s_D.tick_params(axis='x', labelsize='xx-small', rotation=25)
s_D.tick_params(axis='y', labelsize='xx-small')


plt.savefig('shear_u.png',dpi = 300)