
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'


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


# ---- Loop through the files ----#

filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
# Open the dataset with Dask, chunked by 'ocean_time' for efficiency
# Open datasets
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})	

ds, xgrid = xroms.roms_dataset(ds1)

f = ds.f.compute()
f_mean = float(np.abs(f).where(np.abs(f) > 0).mean().values)
T_mean = (2*np.pi/f_mean)/3600 #inertial period in hours
# ---- Define frequency limits ----
f1 = 0.8 * f_mean  # [Hz]
f2 = 1.5 * f_mean  # [Hz]
f1_cph = f1 * 3600  # in cycles per hour
f2_cph = f2 * 3600  # in cycles per hour
T1 = 0.8 * T_mean  # Hours
T2 = 1.4 * T_mean  # Hours

# Define eta and xi slices
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)


for i in range(0,len(etas)):
	place =  True
	print(f"Slicing array for i = {i}")	
	print('working u')
	u = xroms.to_rho(ds.u, xgrid).isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis).compute().transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
	u_file = u.to_dataset(name='u')		
	u_file.to_netcdf(out_path+f'u_slice_{i}.nc')
	#print('filtering u')
	#u_prime = filter_da_bandpass(u, T1, T2, fs).compute().transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')#.isel(ocean_time=slice(0, None, 2))
	#u_prime_file = u_prime.to_dataset(name='u_prime')		
	#u_prime_file.to_netcdf(out_path +f'u_prime_{i}.nc')
	#del u, u_file , u_prime, u_prime_file

	print('working v')
	v = xroms.to_rho(ds.v, xgrid).isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis).compute().transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
	v_file = v.to_dataset(name='v')		
	v_file.to_netcdf(out_path+f'v_slice_{i}.nc')
	print('filtering v')
	#v_prime = filter_da_bandpass(v, T1, T2, fs).compute().transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')#.isel(ocean_time=slice(0, None, 2))
	#v_prime_file = v_prime.to_dataset(name='v_prime')		
	#v_prime_file.to_netcdf(out_path+f'v_prime_{i}.nc')
	#del v, v_file , v_prime, v_prime_file

	print('working w')
	w = xroms.to_s_rho(ds.w, xgrid).isel(eta_rho=slice(etas[i][0],etas[i][-1]),xi_rho=xis).compute().transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
	w_file = w.to_dataset(name='w')		
	w_file.to_netcdf(out_path+f'w_slice_{i}.nc')
	#print('filtering w')
	#w_prime = filter_da_bandpass(w, T1, T2, fs).compute().transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')#.isel(ocean_time=slice(0, None, 2))
	#w_prime_file = w_prime.to_dataset(name='w_prime')		
	#w_prime_file.to_netcdf(out_path+f'w_prime_{i}.nc')
	#del w, w_file,w_prime, w_prime_file	

	print("Jumping to new slice\n\tEnding saving process")
	print("Deleting variables")	
	target_variable = 'place'
	local_vars = locals()

	# Find the index of the target variable
	index_of_target = list(local_vars.keys()).index(target_variable)
	# Delete variables defined later than the target variable
	variables_to_delete = list(local_vars.keys())[index_of_target:]
	for var in variables_to_delete:
		del locals()[var]

del ds1,ds,eta,etas,xis,filename, f, f_mean,f1,f2,f1_cph,f2_cph, T1, T2, T_mean
gc.collect()
print("All slices processed and saved.")





