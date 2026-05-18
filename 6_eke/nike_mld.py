
import sys
# Assuming the necessary imports (numpy, xarray) are handled within utils_roms_p1 or implicitly used by the code

p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
sys.path.append(p1_dir)
from utils_roms_p1 import * # Assumes this includes numpy (np) and xarray (xr)
import dask
import pandas as pd
import gc
# from scipy.signal import butter, filtfilt # Retaining the original imports

# --- Path and Constant Definitions ---
out_path = '/data1/roms_dd_waves/analysis_outs/NIW/eke/'
path_pot ='/data1/roms_dd_waves/analysis_outs/NIW/pot_energy/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'

# Define slices for eta and xi
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None) 
rho_0 = 1025  # Reference density

# Set Dask configuration if needed
dask.config.set({"distributed.worker.memory.target": 0.6})
# --- Helper Function for Robust Chunking ---
def safe_chunk(da, time_chunk='auto'):
	"""
	Chunks a DataArray explicitly. Sets the 'ocean_time' dimension to the specified chunk size
	and all other dimensions to -1 (a single chunk) to avoid Dask auto-chunking errors 
	on dimensions that are already sliced to a small size.
	"""
	chunks = {}
	
	# Chunk time dimension
	if 'ocean_time' in da.dims:
		chunks['ocean_time'] = time_chunk
	
	# Chunk all other existing dimensions to their full size (single chunk, size -1)
	for dim, size in da.sizes.items():
		if dim != 'ocean_time':
			chunks[dim] = -1 
			
	# Apply chunking
	if chunks:
		return da.chunk(chunks)
	return da




# --- Main Loop ---
for i in range(0,len(etas)):
	print(f"\nProcessing eta slice {i}")
	
	ke = (xr.open_dataset(out_path + f'NIKE_slice_{i}.nc').nike)
	ke = safe_chunk(ke, time_chunk='auto')

	ml = xr.open_dataset(path_pot + f'mld_slice_{i}.nc').mld

	dz = xr.open_dataset(dzs + f'dz_{i}.nc').isel(s_rho=slice(0,-1))

	dez = (-dz['z_rho']).rename('dez').compute()

	diff = xr.apply_ufunc(np.abs, dez - ml)
	diff = diff.where(np.isfinite(diff), np.inf)

	k_indices_loaded = diff.argmin(dim='s_rho').compute().drop_vars('ocean_time', errors='ignore')

	nike_mld = ke.isel(s_rho=k_indices_loaded).compute()

	nike_mld.to_dataset(name='nike_mld').to_netcdf(out_path + f'nike_mld_{i}.nc')


	# Final cleanup
	del ke, ml,dz,dez,diff,k_indices_loaded, nike_mld
	gc.collect()

print("All eta slices processed and saved.")


