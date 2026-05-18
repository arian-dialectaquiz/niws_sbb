
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt

out_path = '/data1/roms_dd_waves/analysis_outs/NIW/pot_energy/'
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
pressure = out_path


rho = 1025

eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)	



filename = f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc'	
# Open the dataset with Dask, chunked by 'ocean_time' for efficiency	
# Open the dataset with Dask, chunked by 'ocean_time' for efficiency
# Open datasets
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})	
# Define eta and xi slices
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None)		

# Process each eta slice
for i in range(0,len(etas)):
	print(f'Starting calculations for eta range {i}')	
	# Loop over time
	for k in range(0,len(ds1.ocean_time)):  # Processing every 20th time point
		print(f'Processing time step {k} for eta range {i}')	
		# Merge datasets for the current time step
		ds_merged = ds1.isel(ocean_time=k)			
		# Convert to ROMS dataset and calculate density
		ds, xgrid = xroms.roms_dataset(ds_merged)
		#rho_ini = xroms.density(ds.temp, ds.salt)
		mld_ini = ds.xroms.mld(thresh=0.03)			
		# Calculate N2 for the eta slice
		mld_s_a = mld_ini.isel(eta_rho=slice(etas[i][0], etas[i][-1]), xi_rho=xis)
		mld_s = mld_s_a.compute()	
		# Save the N2 slice for this time step
		time_filename = out_path + f'time_pos_{k}.nc'
		mld_s.to_dataset(name='mld').to_netcdf(time_filename)	
		# Clear memory for the next time step
		del ds_merged, ds, xgrid, mld_ini, mld_s_a, mld_s
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
	eta_filename = out_path + f'mld_slice_{i}.nc'
	ds_merged_time.to_netcdf(eta_filename)	

	# Remove individual time slice files
	for time_file in time_files:
		os.remove(time_file)	
	# Clear memory for the next eta slice
	del ds_merged_time
	gc.collect()	
	print(f"Completed processing and saved mld_slice_{i}.nc for eta range {i}.\n")	
del ds1,filename
gc.collect()
print("All eta slices processed and saved.")





# Match both "dT_dz_slice_i_t.nc" and "dT_dz_slice_t_i.nc"
patterns = [
	os.path.join(out_path, 'mld_slice_*.nc'),
]

files = []
for pat in patterns:
	files.extend(glob.glob(pat))
files = sorted(set(files))

print(f"Found {len(files)} files.")

for fpath in files:
	print(f"\nProcessing: {os.path.basename(fpath)}")
	ds = xr.open_dataset(fpath, chunks={'ocean_time': 'auto'})

	ds = ds[[varname]]

	# Build a 6h ocean_time coordinate as the mean of each 2-step window
	new_time = ds['ocean_time'].mean()
	ds = ds.assign_coords(ocean_time=new_time)

	# Write to a temp file, then replace original (safe overwrite)
	tmp = fpath + '.tmp'
	enc = {varname: {"zlib": True, "complevel": 4}}  # light compression
	ds.to_netcdf(tmp, encoding=enc)

	ds.close()
	# Remove original (3h) and replace with 6h-averaged file
	os.remove(fpath)
	os.rename(tmp, fpath)

