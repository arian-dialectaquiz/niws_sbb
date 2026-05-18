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
speeds = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'

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
for i in range(1,len(etas)):
	print(f"\nProcessing eta slice {i}")
	
	# 1. FIX: Corrected Slicing Logic for eta_rho
	# The slice stop must be one index past the last desired index (etas[i][-1] + 1)
	eta_slice = slice(etas[i][0], etas[i][-1])
	
	# --- 2. Load Data Lazily with Early Slicing and Safe Chunking ---
	print(f"Loading velocity components for indices {eta_start} to {eta_end-1}...")
	
	u_prime = (xr.open_dataset(speeds + f'u_prime_{i}.nc')
			   .u_prime
			   )
	u_prime = safe_chunk(u_prime, time_chunk='auto')
	
	v_prime = (xr.open_dataset(speeds + f'v_prime_{i}.nc')
			   .v_prime
			   )
	v_prime = safe_chunk(v_prime, time_chunk='auto')
	
	# --- 3. Calculate EKE (Corrected and Lazy) ---
	print("Calculating Eddy Kinetic Energy (EKE) lazily...")
	
	# EKE = (1/2) * rho_0 * (u'^2 + v'^2)
	u_sq = u_prime**2
	v_sq = v_prime**2
	
	EKE = 0.5 * rho_0 * (u_sq + v_sq)
	
	# Clear intermediate components
	del u_prime, v_prime, u_sq, v_sq
	gc.collect()

	# --- 4. Compute Time Derivative (Lazy) ---
	print("Calculating EKE time tendency...")
	
	# Calculate time difference in hours (need to compute this scalar array)
	dt_hours = (EKE.ocean_time.diff("ocean_time") / np.timedelta64(1, "h")).astype(float).compute()
	
	# Time derivative (dEKE/dt)
	dEKE = EKE.diff("ocean_time") 
	
	# Division is lazy; Xarray/Dask handles the dimension reduction from .diff()
	EKE_tendency = dEKE / dt_hours
	
	# Clear the EKE variable which is now implicitly held by EKE_tendency
	del EKE, dEKE 
	gc.collect()

	# --- 5. Final Compute, Save, and Cleanup ---
	print("Computing result and saving to NetCDF...")
	
	# Trigger the computation only once at the end
	EKE_tendency_computed = EKE_tendency.compute()
	
	# Save the result
	EKE_tendency_computed.to_dataset(name='eke_trend').to_netcdf(out_path + f'eke_trend_slice_{i}.nc')
	
	# Final cleanup
	del EKE_tendency, EKE_tendency_computed
	gc.collect()

print("All eta slices processed and saved.")