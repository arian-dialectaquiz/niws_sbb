import os
import sys
import gc
import numpy as np
import xarray as xr
import xroms
from scipy import signal

filename = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_1km_NIWS_V2.nc'    
out_path = '/data1/roms_dd_waves/analysis_outs/NIWS/wind_input_co/'
speeds = '/data1/roms_dd_waves/analysis_outs/NIWS/velocities/'

os.makedirs(out_path, exist_ok=True)

# ---- Bandpass filter parameters ----#
def filter_da_bandpass(da, T1, T2, fs, time_dim='ocean_time', order=4):
	"""
	Apply a Butterworth bandpass filter along the time dimension of a 4D DataArray.
	"""
	time_vals = da[time_dim].values
	dt_hours = (time_vals[1] - time_vals[0]) / np.timedelta64(1, 'h')

	f_low = 1 / T2  
	f_high = 1 / T1  

	nyq = 0.5 * fs
	low = f_low / nyq
	high = f_high / nyq

	b, a = signal.butter(order, [low, high], btype='band')

	original_shape = da.shape
	reshaped = da.data.reshape((original_shape[0], -1))

	filtered = signal.filtfilt(b, a, reshaped, axis=0, padlen=3*max(len(b), len(a)))

	da_filtered = xr.DataArray(
		filtered.reshape(original_shape),
		dims=da.dims,
		coords=da.coords,
		attrs=da.attrs
	)
	return da_filtered

# ---- Sampling frequency ----
dt_hours = 3.0
fs = 1 / dt_hours 
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})    
ds, xgrid = xroms.roms_dataset(ds1)

T1 = 25
T2 = 37
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)

# --- Loop over slices ---
for i in range(len(etas)):
	print(f"Starting calculations for etas slice: {i}")
	
	# 1. Compute WU
	print('fetching & filtering Tu')
	tu = xroms.to_rho(ds.sustr, xgrid).isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis).compute().transpose('ocean_time', 'eta_rho', 'xi_rho')
	tu_prime = filter_da_bandpass(tu, T1, T2, fs).compute().transpose('ocean_time', 'eta_rho', 'xi_rho')
	
	print('u input')
	u = xr.open_dataset(speeds + f'u_prime_{i}.nc', chunks={'ocean_time': 'auto'}).u_prime.isel(s_rho=-1)
	WU = tu_prime * u
	
	# 2. Compute WV
	print('fetching & filtering Tv')
	tv = xroms.to_rho(ds.svstr, xgrid).isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis).compute().transpose('ocean_time', 'eta_rho', 'xi_rho')
	tv_prime = filter_da_bandpass(tv, T1, T2, fs).compute().transpose('ocean_time', 'eta_rho', 'xi_rho')
	
	print('v input')
	# Note: Fixed potential typo below (.u_prime changed to .v_prime assuming standard naming)
	v_ds = xr.open_dataset(speeds + f'v_prime_{i}.nc', chunks={'ocean_time': 'auto'})
	v = v_ds.v_prime.isel(s_rho=-1) if 'v_prime' in v_ds else v_ds.u_prime.isel(s_rho=-1)
	
	WV = tv_prime * v
	
	# 3. Merge into Total Wind Input Vector
	print('Merging components...')
	W_total = WU + WV
	W_total.name = 'wind_work'
	W_total.attrs = {
		'long_name': 'Total Wind energy input to NIW',
		'units': 'W/m^2'
	}
	
	# Save the merged slice to disk
	out_file = os.path.join(out_path, f'Wind_Input_Slice_{i}.nc')
	W_total.to_netcdf(out_file)
	print(f"Saved slice {i} to {out_file}\n")
	
	# Clean up memory explicitly and safely
	del tu, tu_prime, u, WU, tv, tv_prime, v, v_ds, WV, W_total
	gc.collect()

# Free initial large datasets
del ds1, ds
gc.collect()
