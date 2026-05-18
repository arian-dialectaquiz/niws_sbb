
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/theta/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'
fs = 1 / 1  

filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
# Define eta and xi slices

# Define eta and xi slices
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	
ds, xgrid = xroms.roms_dataset(ds1)
f = ds.f.compute()
f_mean = float(np.abs(f).where(np.abs(f) > 0).mean().values)
T_mean = (2*np.pi/f_mean)/3600 #inertial period in hours
T1 = 0.8 * T_mean  # Hours
T2 = 1.5 * T_mean  # Hours

for i in range(1, len(etas)):
	print(f"\nSlicing eta index {i}")
	print("Loading dz...")
	dz = xr.open_dataset(dzs + f'dz_{i}.nc').dz
	theta = ds.temp.isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis)
	chunk_size = 50
	for j in range(0, len(theta.ocean_time), chunk_size):
		print(f"Processing chunk {j} to {j+chunk_size}")
		theta_chunk = theta[j:j+chunk_size]
		# Save theta chunk
		theta_ds = theta_chunk.to_dataset(name='theta').drop_vars('z_rho') 
		theta_file = out_path + f'theta_eta{i}_pos{j}.nc'
		theta_ds.to_netcdf(theta_file)
		# Compute dT/dz
		dtheta_chunk = theta_chunk.diff(dim='s_rho').compute()
		dT_dz_chunk = (dtheta_chunk / dz).transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
		dT_dz_ds = dT_dz_chunk.to_dataset(name='dT_dz')
		dT_dz_file = out_path + f'dT_dz_eta{i}_pos{j}.nc'
		dT_dz_ds.to_netcdf(dT_dz_file)
		del theta_chunk, dtheta_chunk, dT_dz_chunk, theta_ds, dT_dz_ds
		gc.collect()
	# === Merge theta and dT_dz time chunks into full time series ===
	for var in ['theta', 'dT_dz']:
		files = [out_path + f'{var}_eta{i}_pos{j}.nc' for j in range(0, len(theta.ocean_time), chunk_size)]
		datasets = [xr.open_dataset(f) for f in files]
		ds_merged = xr.concat(datasets, dim='ocean_time')
		ds_merged = ds_merged
		merged_file = out_path + f'{var}_slice_{i}.nc'
		ds_merged.to_netcdf(merged_file)
		# Cleanup
		for f in files:
			os.remove(f)
		del datasets, ds_merged
		gc.collect()

	# === Apply band-pass to full theta time series ===
	# === Subsample merged theta to 6h and overwrite ===
	print(f"Loading merged theta for eta slice {i}")
	theta_full = xr.open_dataset(out_path + f'theta_slice_{i}.nc').theta
	# Subsample to 6h
	theta_subsampled = theta_full.compute()
	#theta_subsampled = theta_subsampled.drop_vars('z_rho') 
	print("Applying band-pass filter to full time series...")
	theta_prime = filter_da_bandpass(theta_full, T1, T2, fs).compute().transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
	#theta_prime = theta_prime.drop_vars('z_rho') 
	print("Saving band-pass filtered theta...")
	theta_prime.to_dataset(name='theta_prime').to_netcdf(out_path + f'theta_prime_slice_{i}.nc')		
	del theta_prime
	gc.collect()
	#removing 3h theta full
	theta_file_3h = out_path + f'theta_slice_{i}.nc'
	if os.path.exists(theta_file_3h):
		os.remove(theta_file_3h)		
	# Overwrite the original file with the 6h version
	theta_subsampled.to_dataset(name='theta').to_netcdf(out_path + f'theta_slice_{i}.nc')	
	del theta_full, theta_subsampled, theta_file_3h
	gc.collect()
	print(f"Completed eta slice {i} ")
del ds, ds1, xgrid
gc.collect()

