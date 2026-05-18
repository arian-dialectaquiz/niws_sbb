##
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
import os # Added os for os.remove
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt
import numpy as np
import xarray as xr

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/N2/shear/comps/'
filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'

ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})

# Define eta and xi slices
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)
# Define your chunk size
chunk_size = 1
total_times = len(ds1.ocean_time)


# Process each eta slice
for i in range(len(etas)):
	# Loop over time
	for k in range(0, total_times, chunk_size): 
		end_index = k + chunk_size
		print(f'Processing time steps {k} to {min(end_index, total_times)} for eta range {i}')
	
		# Select a slice (range) of time steps instead of a single integer
		ds_merged = ds1.isel(ocean_time=slice(k, end_index))
		# Convert to ROMS dataset and calculate derivatives
		ds, xgrid = xroms.roms_dataset(ds_merged)
		
		# Calculate gradients (on their respective grids)
		dudz = xroms.dudz(ds.u, xgrid)
		dvdz = xroms.dudz(ds.v, xgrid) # Assuming xroms.dudz handles ds.v correctly as d/dz
		
		# Calculate total shear magnitude
		#vert_shear_ini = xroms.vertical_shear(dudz, dvdz, xgrid)
		
		# Interpolate everything to s_rho (rho grid) so they can be saved together
		#vert_shear = xroms.to_s_rho(vert_shear_ini, xgrid)
		dudz_rho   = xroms.to_rho(dudz, xgrid)
		dvdz_rho   = xroms.to_rho(dvdz, xgrid)

		# Slice the data
		#vert_shear_eta = vert_shear.isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis)
		dudz_eta       = dudz_rho.isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis)
		dvdz_eta       = dvdz_rho.isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis)

		# Compute (load into memory)
		#vert_shear_eta_comp = vert_shear_eta.compute()
		dudz_eta_comp       = dudz_eta.compute()
		dvdz_eta_comp       = dvdz_eta.compute()

		# Combine into a single Dataset
		ds_out = xr.Dataset({
			#'shear': vert_shear_eta_comp,
			'dudz': dudz_eta_comp,
			'dvdz': dvdz_eta_comp
		})

		time_filename = out_path + f'time_pos_{k}.nc'
		ds_out.to_netcdf(time_filename)
		
		# Clear memory
		del ds_merged, ds, xgrid
		del dudz, dvdz, dudz_rho, dvdz_rho, dudz_eta, dvdz_eta
		del dudz_eta_comp, dvdz_eta_comp, ds_out
		gc.collect()

	# --- Merge Logic ---
	# Merge all time slice files into a single file for the eta slice
	time_files = [out_path + f'time_pos_{k}.nc' for k in range(0, len(ds1.ocean_time))]
	
	# Open datasets (Datasets now contain shear, dudz, and dvdz)
	datasets = [xr.open_dataset(file) for file in time_files]
	
	# Add the ocean_time coordinate to each dataset
	for idx, dataset in enumerate(datasets):
		dataset = dataset.assign_coords(ocean_time=[ds1.ocean_time[idx].values])
		datasets[idx] = dataset
		
	# Concatenate datasets along the ocean_time dimension
	ds_merged_time = xr.concat(datasets, dim='ocean_time')
	
	# Save the final merged dataset for this eta slice
	eta_filename = out_path + f'comp_vert_shear_slice_{i}.nc'
	ds_merged_time.to_netcdf(eta_filename)
	
	# Remove individual time slice files
	# Close datasets first to ensure file handles are released (good practice)
	ds_merged_time.close()
	for d in datasets:
		d.close()
		
	for time_file in time_files:
		if os.path.exists(time_file):
			os.remove(time_file)
			
	# Clear memory for the next eta slice
	del ds_merged_time, datasets
	gc.collect()
	
	print(f"Completed processing and saved vert_shear_slice_{i}.nc for eta range {i}.\n")

gc.collect()
print("All eta slices processed and saved.")