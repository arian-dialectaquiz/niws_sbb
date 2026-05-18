###
import sys
import os
import glob # Needed for file globbing in the final merge step

p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
sys.path.append(p1_dir)
from utils_roms_p1 import * # Assumes this includes numpy (np) and xarray (xr)
import dask
import pandas as pd
import gc

# --- Path and Constant Definitions ---
speed = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'
out_path = '/data1/roms_dd_waves/analysis_outs/NIW/EP_flux/'
temp_path = '/data1/roms_dd_waves/analysis_outs/NIW/EP_flux/temp/'

# Define slices for eta and xi
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68))
xis = slice(40, None) 

rho_0 = 1025  # Reference density
dl = 1000     # Mean lateral distance

# Set Dask configuration
dask.config.set({"distributed.worker.memory.target": 0.5})

# Ensure the temporary directory exists
os.makedirs(temp_path, exist_ok=True)

# --- Helper Function for Lazy Gradient Calculation ---
def compute_gradients(da_full, dzii):
	"""
	Computes differences: d(da_full)/dx, d(da_full)/dy, d(da_full)/dz lazily.
	"""
	d_dx = da_full.diff(dim='xi_rho') / dl
	d_dy = da_full.diff(dim='eta_rho') / dl
	
	d_dz_diff = da_full.diff(dim='s_rho')
	dz_sliced = dzii.isel(s_rho=slice(None, -1))
	d_dz = d_dz_diff / dz_sliced
	
	return d_dx, d_dy, d_dz

# --- Main Loop (Processing Eta Slices) ---
for i in range(len(etas)):
	print(f"\nProcessing eta slice {i}")
	
	# --- 1. Load Full Eta Slice Data Lazily ---
	print("Loading all required arrays for full time series (lazily)...")
	
	# Load dz, prime, and full velocity components
	dzii = xr.open_dataset(dzs + f'dz_{i}.nc').dz
	u_prime = xr.open_dataset(speed + f'u_prime_{i}.nc').u_prime.isel(ocean_time=slice(None, None, 2))
	v_prime = xr.open_dataset(speed + f'v_prime_{i}.nc').v_prime.isel(ocean_time=slice(None, None, 2))
	w_prime = xr.open_dataset(speed + f'w_prime_{i}.nc').w_prime.isel(ocean_time=slice(None, None, 2))
	
	u_full = xr.open_dataset(speed + f'u_slice_{i}.nc').u
	v_full = xr.open_dataset(speed + f'v_slice_{i}.nc').v
	w_full = xr.open_dataset(speed + f'w_slice_{i}.nc').w

	# Get the time coordinate for loop iteration
	ocean_time = u_prime['ocean_time']
	num_time_steps = len(ocean_time)
	
	# --- 2. Compute Full Velocity Gradients (Lazy) ---
	print("Calculating full velocity gradients lazily...")
	
	du_dx_full, du_dy_full, du_dz_full = compute_gradients(u_full, dzii)
	dv_dx_full, dv_dy_full, dv_dz_full = compute_gradients(v_full, dzii)
	dw_dx_full, dw_dy_full, dw_dz_full = compute_gradients(w_full, dzii)
	
	# Clear full velocity data
	del u_full, v_full, w_full
	gc.collect()

	# --- 3. Align Prime Velocities (Lazy) ---
	print("Aligning prime velocities lazily...")
	
	xi_diff_slice = slice(None, -1)
	eta_diff_slice = slice(None, -1)
	s_diff_slice = slice(None, -1)
	
	# Apply isel once on prime variables
	u_align_xi = u_prime.isel(xi_rho=xi_diff_slice)
	v_align_xi = v_prime.isel(xi_rho=xi_diff_slice)
	w_align_xi = w_prime.isel(xi_rho=xi_diff_slice)
	
	u_align_eta = u_prime.isel(eta_rho=eta_diff_slice)
	v_align_eta = v_prime.isel(eta_rho=eta_diff_slice)
	w_align_eta = w_prime.isel(eta_rho=eta_diff_slice)
	
	u_align_s = u_prime.isel(s_rho=s_diff_slice)
	v_align_s = v_prime.isel(s_rho=s_diff_slice)
	w_align_s = w_prime.isel(s_rho=s_diff_slice)
	
	# Clear original prime velocity data
	del u_prime, v_prime, w_prime
	gc.collect()

	# --- 4. Define Lazy Pi Tensor Components ---
	print("Defining $\Pi$ tensor operations lazily...")
	
	# Calculate Pi components
	Pi_xx = (u_align_xi * u_align_xi) * du_dx_full
	Pi_xy = (u_align_eta * v_align_eta) * du_dy_full
	Pi_xz = (u_align_s * w_align_s) * du_dz_full
	
	Pi_yx = (v_align_xi * u_align_xi) * dv_dx_full
	Pi_yy = (v_align_eta * v_align_eta) * dv_dy_full
	Pi_yz = (v_align_s * w_align_s) * dv_dz_full
	
	Pi_zx = (w_align_xi * u_align_xi) * dw_dx_full
	Pi_zy = (w_align_eta * v_align_eta) * dw_dy_full
	Pi_zz = (w_align_s * w_align_s) * dw_dz_full
	
	# Sum Pi Tensor (Still lazy)
	Pi_lazy = Pi_xx + Pi_xy + Pi_yx + Pi_yy + Pi_zx + Pi_zy + Pi_xz + Pi_yz + Pi_zz
	
	# Final Integration Factor (Still lazy)
	dz_for_Pi = dzii.isel(s_rho=slice(None, -1))
	PI_dz_lazy = -rho_0 * Pi_lazy * dz_for_Pi
	
	# Vertical sum (Still lazy)
	PI_zsum_lazy = PI_dz_lazy.sum(dim='s_rho', skipna=False)

	# Clean up intermediate components of the large graph
	del (Pi_xx, Pi_xy, Pi_xz, Pi_yx, Pi_yy, Pi_yz, Pi_zx, Pi_zy, Pi_zz, Pi_lazy, 
		 PI_dz_lazy, dzii, dz_for_Pi, du_dx_full, du_dy_full, du_dz_full, 
		 dv_dx_full, dv_dy_full, dv_dz_full, dw_dx_full, dw_dy_full, dw_dz_full,
		 u_align_xi, u_align_eta, u_align_s, v_align_xi, v_align_eta, v_align_s, 
		 w_align_xi, w_align_eta, w_align_s)
	gc.collect()


	# --- 5. Loop Over Time, Compute, and Save Individually ---
	
	time_files = []
	print(f"Computing and saving {num_time_steps} time steps one by one...")

	for k in range(num_time_steps):
		print(f'\tProcessing time step {k} / {num_time_steps-1}') 
		
		# Select one time step and trigger the computation
		pi_t_slice = PI_zsum_lazy.isel(ocean_time=k).compute()
		
		# Save the result for this time step to the temporary directory
		time_filename = temp_path + f'pi_pos_{k}_slice_{i}.nc'
		pi_t_slice.to_dataset(name='mean_flow_wave').to_netcdf(time_filename)
		time_files.append(time_filename)
		
		del pi_t_slice
		gc.collect()


	# --- 6. Merge Time Slice Files and Final Save ---
	
	print(f"\nMerging {len(time_files)} time slices into final file...")
	
	# Open all saved time slices
	datasets = [xr.open_dataset(file) for file in time_files]

	# Assign the correct ocean_time coordinate
	for idx, dataset in enumerate(datasets):
		# Assign the correct time coordinate from the original full list
		dataset = dataset.assign_coords(ocean_time=[ocean_time.values[idx]]) 
		datasets[idx] = dataset
		
	# Concatenate datasets along the ocean_time dimension
	ds_merged_time = xr.concat(datasets, dim='ocean_time')
	
	# Save the final merged dataset for this eta slice
	eta_filename = out_path + f'mean_flow_wave_slice_{i}.nc'
	ds_merged_time.to_netcdf(eta_filename)
	
	# --- 7. Clean up Temporary Files ---
	print("Removing individual time slice files...")
	for time_file in time_files:
		try:
			os.remove(time_file)
		except OSError as e:
			print(f"Error removing temporary file {time_file}: {e}")
			
	# Clear memory for the next eta slice
	del ds_merged_time, ocean_time
	gc.collect()
	print(f"Completed processing and saved {eta_filename}.\n")

print("All eta slices processed and saved.")