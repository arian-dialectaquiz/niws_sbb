# --- Setup & imports ---
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *  # provides xroms
import xarray as xr
import numpy as np
import dask
import pandas as pd
import gc
import os, glob
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/vorticity/'
os.makedirs(out_path, exist_ok=True)

# --- Loop over ROMS parts ---
filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'

# Open with Dask (chunk along time); keep ds1 lightweight
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
# Define eta/xi slicing layout
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)

# Process each eta-slice block
for i in range(len(etas)):
	print(f'>>> Starting f_eff for eta block {i}  (eta={etas[i][0]}:{etas[i][-1]}, xi=40:470)')
	# --- Per-time processing to limit memory ---
	time_files = []
	for k in range(int(ds1.ocean_time.size)):
		print(f'  - time step k={k}')
		ds_merged = ds1.isel(ocean_time=k)
		# Build ROMS accessors on the single-time dataset
		ds, xgrid = xroms.roms_dataset(ds_merged)
		# Relative vorticity on s-grid -> to rho -> back to s_rho (consistent with your original)
		zeta_s   = xroms.relative_vorticity(ds.u, ds.v, xgrid)
		zeta_rho = xroms.to_rho(zeta_s, xgrid)
		zeta     = xroms.to_s_rho(zeta_rho, xgrid)
		# Coriolis parameter (2D on rho grid, no time)
		f2d = ds.f
		# Slice both zeta and f to the spatial window
		zeta_blk = zeta.isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis)
		f_blk    = f2d.isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis)
		# Compute effective Coriolis: broadcast f over s_rho dimension
		# f_eff = f / (f + zeta)
		f_eff = f_blk + (zeta_blk/2)
		# Persist a *single time step* to disk to keep RAM low
		tmp_time_file = os.path.join(out_path, f'f_eff_time_{k}_blk{i}.nc')
		(f_eff.to_dataset(name='f_eff')
				 .to_netcdf(tmp_time_file, encoding={'f_eff': {'zlib': True, 'complevel': 4}}))
		time_files.append(tmp_time_file)
		# Cleanup
		del ds_merged, ds, xgrid, zeta_s, zeta_rho, zeta, zeta_blk, f2d, f_blk, f_eff
		gc.collect()
	# --- Merge all time-slice files for this eta block ---
	datasets = [xr.open_dataset(fp) for fp in time_files]
	# Attach the correct ocean_time value to each file before concat
	# (use the exact timestamps from ds1)
	for idx, dsi in enumerate(datasets):
		dsi = dsi.assign_coords(ocean_time=[ds1.ocean_time[idx].values])
		datasets[idx] = dsi
	ds_merged_time = xr.concat(datasets, dim='ocean_time')
	# Save merged 3-hourly file for this block
	eta_filename = os.path.join(out_path, f'f_eff_slice_{i}.nc')
	ds_merged_time.to_netcdf(eta_filename, encoding={'f_eff': {'zlib': True, 'complevel': 4}})
	# Close and remove per-time temporary files
	for dsi in datasets:
		dsi.close()
	for fp in time_files:
		try:
			os.remove(fp)
		except FileNotFoundError:
			pass
	# Cleanup
	del ds_merged_time, datasets, time_files
	gc.collect()
# Done with this ROMS part
del ds1, filename
gc.collect()
print(f'>>> [t={t}] All eta blocks processed and saved.')
