#

import sys
import os
import glob # Needed for file globbing in the final merge step

p1_dir = '/home/arian/dd_waves/m2_internal_tide_gen_prop/'
sys.path.append(p1_dir)
from utils_roms_p1 import * # Assumes this includes numpy (np) and xarray (xr)
import dask
import pandas as pd
import gc

# --- Path and Constant Definitions ---
speed = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/'
out_path = '/data1/roms_dd_waves/analysis_outs/NIW/coarse/'
temp_path =  '/data1/roms_dd_waves/analysis_outs/NIW/coarse/temp/'
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
def compute_gradients(da, dzii):
	"""
	Computes gradients for a DataArray (u_prime, v_prime, or w_prime) lazily.
	"""
	d_dx = da.diff(dim='xi_rho') / dl
	d_dy = da.diff(dim='eta_rho') / dl
	
	d_dz_diff = da.diff(dim='s_rho')
	dz_sliced = dzii.isel(s_rho=slice(None, -1))
	d_dz = d_dz_diff / dz_sliced
	
	return d_dx, d_dy, d_dz

# --- Main Loop (Processing Eta Slices) ---
for i in range(len(etas)):
	print(f"\nProcessing eta slice {i}")
	
	# --- 1. Load Full Eta Slice Data Lazily ---
	print("Loading dz and wave velocity arrays for full time series (lazily)...")
	
	# Load dz and prime velocity components
	dzii = xr.open_dataset(dzs + f'dz_{i}.nc').dz
	u_prime = xr.open_dataset(speed + f'u_prime_{i}.nc').u_prime.isel(ocean_time=slice(None, None, 3))
	v_prime = xr.open_dataset(speed + f'v_prime_{i}.nc').v_prime.isel(ocean_time=slice(None, None, 3))
	w_prime = xr.open_dataset(speed + f'w_prime_{i}.nc').w_prime.isel(ocean_time=slice(None, None, 3))

	# Get the time coordinate from the prime data
	ocean_time_coords = u_prime['ocean_time']
	
	# --- 2. Compute Wave Velocity Gradients (Lazy) ---
	print("Calculating wave velocity gradients lazily...")
	
	# Gradients for u_prime
	du_dx, du_dy, du_dz = compute_gradients(u_prime, dzii)
	# Gradients for v_prime
	dv_dx, dv_dy, dv_dz = compute_gradients(v_prime, dzii)
	# Gradients for w_prime
	dw_dx, dw_dy, dw_dz = compute_gradients(w_prime, dzii)
	
	# --- 3. Align Prime Velocities (Lazy) ---
	print("Aligning prime velocities to gradient grid using slice indices...")
	
	xi_diff_slice = slice(None, -1)
	eta_diff_slice = slice(None, -1)
	s_diff_slice = slice(None, -1)
	
	# Apply isel once on prime variables for alignment
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

	# --- 4. Define Lazy Pi Tensor Components (Triple Correlation) ---
	print("Defining $\Pi'$ tensor operations (triple correlation) lazily...")
	
	# $\Pi_{xx} = u' \cdot u' \cdot \frac{\partial u'}{\partial x}$
	Pi_xx = (u_align_xi * u_align_xi) * du_dx
	
	# $\Pi_{xy} = u' \cdot v' \cdot \frac{\partial u'}{\partial y}$
	Pi_xy = (u_align_eta * v_align_eta) * du_dy
	
	# $\Pi_{xz} = u' \cdot w' \cdot \frac{\partial u'}{\partial z}$
	Pi_xz = (u_align_s * w_align_s) * du_dz
	
	# $\Pi_{yx} = v' \cdot u' \cdot \frac{\partial v'}{\partial x}$
	Pi_yx = (v_align_xi * u_align_xi) * dv_dx
	
	# $\Pi_{yy} = v' \cdot v' \cdot \frac{\partial v'}{\partial y}$
	Pi_yy = (v_align_eta * v_align_eta) * dv_dy
	
	# $\Pi_{yz} = v' \cdot w' \cdot \frac{\partial v'}{\partial z}$
	Pi_yz = (v_align_s * w_align_s) * dv_dz
	
	# $\Pi_{zx} = w' \cdot u' \cdot \frac{\partial w'}{\partial x}$
	Pi_zx = (w_align_xi * u_align_xi) * dw_dx
	
	# $\Pi_{zy} = w' \cdot v' \cdot \frac{\partial w'}{\partial y}$
	Pi_zy = (w_align_eta * v_align_eta) * dw_dy
	
	# $\Pi_{zz} = w' \cdot w' \cdot \frac{\partial w'}{\partial z}$
	Pi_zz = (w_align_s * w_align_s) * dw_dz
	
	# Sum and Integrate Vertically (Still lazy)
	Pi_lazy = Pi_xx + Pi_xy + Pi_yx + Pi_yy + Pi_zx + Pi_zy + Pi_xz + Pi_yz + Pi_zz
	
	dz_for_Pi = dzii.isel(s_rho=slice(None, -1))
	PI_dz_lazy = -rho_0 * Pi_lazy * dz_for_Pi
	PI_zsum_lazy = PI_dz_lazy.sum(dim='s_rho', skipna=False)

	# Get the correct time dimension size for looping
	num_time_steps = PI_zsum_lazy.sizes['ocean_time']
	
	# Clean up intermediate components of the large graph
	del (du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz,
		 u_align_xi, u_align_eta, u_align_s, v_align_xi, v_align_eta, v_align_s, 
		 w_align_xi, w_align_eta, w_align_s, Pi_xx, Pi_xy, Pi_xz, Pi_yx, Pi_yy, Pi_yz, 
		 Pi_zx, Pi_zy, Pi_zz, Pi_lazy, PI_dz_lazy, dzii, dz_for_Pi)
	gc.collect()


	# --- 5. Loop Over Time, Compute, and Save Individually (Low Memory) ---
	
	time_files = []
	print(f"Computing and saving {num_time_steps} time steps one by one...")

	for k in range(num_time_steps):
		print(f'\tProcessing time step {k} / {num_time_steps - 1}') 
		
		# Select one time step and trigger the computation
		pi_t_slice = PI_zsum_lazy.isel(ocean_time=k).compute()
		
		# Save the result for this time step to the temporary directory
		time_filename = temp_path + f'pi_wave_pos_{k}_slice_{i}.nc'
		pi_t_slice.to_dataset(name='PI_wave_zsum').to_netcdf(time_filename)
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
		# We assume ocean_time_coords[idx] is the correct time for the k=idx slice
		dataset = dataset.assign_coords(ocean_time=[ocean_time_coords.values[idx]]) 
		datasets[idx] = dataset
		
	# Concatenate datasets along the ocean_time dimension
	ds_merged_time = xr.concat(datasets, dim='ocean_time')
	
	# Save the final merged dataset for this eta slice
	eta_filename = out_path + f'PI_wave_zsum_slice_{i}.nc'
	ds_merged_time.to_netcdf(eta_filename)
	
	# --- 7. Clean up Temporary Files ---
	print("Removing individual time slice files...")
	for time_file in time_files:
		try:
			os.remove(time_file)
		except OSError as e:
			print(f"Error removing temporary file {time_file}: {e}")
			
	# Clear memory for the next eta slice
	del ds_merged_time, ocean_time_coords
	gc.collect()
	print(f"Completed processing and saved {eta_filename}.\n")

print("All eta slices processed and saved.")