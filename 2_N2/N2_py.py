
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/N2/'


filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'

ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
	
	# Define eta and xi slices
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)


# Process each eta slice
for i in range(0,len(etas)):
	# Loop over time
	for k in range(0,len(ds1.ocean_time)):  # Processing every 20th time point
		print(f'Processing time step {k} for eta range {i}')	
		# Merge datasets for the current time step
		ds_merged = ds1.isel(ocean_time=k)			
		# Convert to ROMS dataset and calculate density
		ds, xgrid = xroms.roms_dataset(ds_merged)
		rho_ini = xroms.density(ds.temp, ds.salt)
		N2_ini = xroms.N2(rho_ini, xgrid)			
		# Calculate N2 for the eta slice
		N2_s_a = xroms.to_s_rho(N2_ini, xgrid).isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis)
		N2_s = N2_s_a.compute()	
		# Save the N2 slice for this time step
		time_filename = out_path + f'time_pos_{k}.nc'
		N2_s.to_dataset(name='N2').to_netcdf(time_filename)	
		# Clear memory for the next time step
		del ds_merged, ds, xgrid, rho_ini, N2_ini, N2_s_a, N2_s
		gc.collect()	
	# Merge all time slice files into a single file for the eta slice
	time_files = [out_path + f'time_pos_{k}.nc' for k in range(0,len(ds1.ocean_time))]
	datasets = [xr.open_dataset(file) for file in time_files]
	# Add the ocean_time coordinate to each dataset
	for idx, dataset in enumerate(datasets):
		dataset = dataset.assign_coords(ocean_time=[ds1.ocean_time[idx].values])
		datasets[idx] = dataset	
	# Concatenate datasets along the ocean_time dimension
	ds_merged_time = xr.concat(datasets, dim='ocean_time')	
	# Save the final merged dataset for this eta slice
	eta_filename = out_path + f'N2_slice_{i}.nc'
	ds_merged_time.to_netcdf(eta_filename)	
	# Remove individual time slice files
	for time_file in time_files:
		os.remove(time_file)	
	# Clear memory for the next eta slice
	del ds_merged_time
	gc.collect()	
	print(f"Completed processing and saved N2_slice_{i}.nc for eta range {i}.\n")	
	del ds1,filename
else:
	pass

gc.collect()
print("All eta slices processed and saved.")





# Match both "dT_dz_slice_i_t.nc" and "dT_dz_slice_t_i.nc"
patterns = [
	os.path.join(out_path, 'N2_slice_0.nc'),
]

files = []
for pat in patterns:
	files.extend(glob.glob(pat))
files = sorted(set(files))

print(f"Found {len(files)} files.")

for fpath in files:
	print(f"\nProcessing: {os.path.basename(fpath)}")
	ds = xr.open_dataset(fpath, chunks={'ocean_time': 'auto'})

	# Only keep the target variable to save memory/disk
	varname = 'N2'

	ds = ds[[varname]]
	
	# Write to a temp file, then replace original (safe overwrite)
	tmp = fpath + '.tmp'
	enc = {varname: {"zlib": True, "complevel": 4}}  # light compression
	ds.to_netcdf(tmp, encoding=enc)

	ds.close()
	# Remove original (3h) and replace with 6h-averaged file
	os.remove(fpath)
	os.rename(tmp, fpath)

	print("  - Saved file and removed original.")