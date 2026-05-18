
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'

import sys
sys.path.append(p1_dir)
import gc
import dask
import xarray as xr
import numpy as np
import pandas as pd
from utils_roms_p1 import *
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/pressure_b/'

filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})

eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)


for i in range(len(etas)):
	print(f"\n[ETA {i}] Processing eta slice...")
	for k in range(0, len(ds1.ocean_time)):  
		print(f"    - Time step {k}")
		# Load current time step
		ds_step = ds1.isel(ocean_time=k)
		# Convert to ROMS
		ds_roms, xgrid = xroms.roms_dataset(ds_step)
		# Compute buoyancy
		rho = xroms.potential_density(ds_roms.temp, ds_roms.salt)
		b = xroms.buoyancy(rho, rho0=1025.0)
		b_s = xroms.to_s_rho(b, xgrid).isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis)
		b_s = b_s.compute()
		# Save buoyancy for this time step
		temp_file = out_path + f'time_pos_{k}.nc'
		b_s.to_dataset(name='buoyancy').drop_vars('z_rho').to_netcdf(temp_file)
		# Clean memory
		del ds_step, ds_roms, xgrid, rho, b, b_s
		gc.collect()
	# === Merge all time step files for current eta slice ===

	time_files = [out_path + f'time_pos_{k}.nc' for k in range(0, len(ds1.ocean_time))]
	datasets = [xr.open_dataset(f) for f in time_files]
	# Assign proper ocean_time coordinate from ds1
	for idx, ds_tmp in enumerate(datasets):
		ds_tmp = ds_tmp.assign_coords(ocean_time=[ds1.ocean_time[idx].values]) 
		datasets[idx] = ds_tmp
	# Merge and save
	ds_merged = xr.concat(datasets, dim='ocean_time')
	merged_file = out_path + f'b_slice_{i}.nc'
	ds_merged.to_netcdf(merged_file)
	# Remove temporary files
	for f in time_files:
		os.remove(f)
	del datasets, ds_merged
	gc.collect()
	print(f"[ETA {i}] Saved to {merged_file}")
	
del ds1
gc.collect()
print(f"\n>>> Completed all eta slices for part {t}")
