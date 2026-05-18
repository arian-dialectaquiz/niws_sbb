##

import sys
import os
import gc
import numpy as np
import xarray as xr
import dask
from scipy import signal

# --- Path Configuration ---
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
sys.path.append(p1_dir)
speed = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
dzs = '/data1/roms_dd_waves/analysis_outs/dz/niws/'

# Output Paths
out_path = '/data1/roms_dd_waves/analysis_outs/NIW/EP_flux_decomposed/'
temp_path = '/data1/roms_dd_waves/analysis_outs/NIW/EP_flux_decomposed/temp/'

# Ensure directories exist
os.makedirs(out_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

# --- Parameters ---
rho_0 = 1025
dl = 1000  # Grid spacing approx (m)
dt_hours = 1.0 # Sampling rate

# --- Define Target Sections (A, B, D) ---
targets = {
	'A': 1239, 
	'B': 957,  
	'D': 225   
}
target_eta_indices = list(targets.values())

# Define Processing Batches (Eta Slices)
eta = np.arange(0, 1360, 1)
etas = np.reshape(eta, (20, 68)) # Processing in chunks of 68 rows

# Dask Config
dask.config.set({"distributed.worker.memory.target": 0.5})

# --- Helper Functions ---

def filter_lowpass(da, T_cutoff, dt_hours, order=4):
	"""
	Low-Pass Butterworth filter. 
	MUST run on the full time-series to be mathematically correct.
	"""
	fs = 1 / dt_hours
	f_cutoff = 1 / T_cutoff 
	nyq = 0.5 * fs
	cutoff_norm = f_cutoff / nyq
	
	if cutoff_norm >= 1:
		return da 

	b, a = signal.butter(order, cutoff_norm, btype='low')

	# Apply filter along time axis (axis=0)
	vals = da.values
	original_shape = vals.shape
	reshaped = vals.reshape((original_shape[0], -1))
	
	filtered_data = signal.filtfilt(b, a, reshaped, axis=0)
	
	return xr.DataArray(
		filtered_data.reshape(original_shape),
		dims=da.dims,
		coords=da.coords
	)

def compute_gradients_step(da_mean_t, dzii):
	""" 
	Computes gradients of the MEAN flow for a SINGLE time step.
	Input da_mean_t is 3D (z, eta, xi).
	"""
	# d/dx and d/dy
	d_dx = da_mean_t.diff(dim='xi_rho') / dl
	d_dy = da_mean_t.diff(dim='eta_rho') / dl
	
	# d/dz
	d_dz_diff = da_mean_t.diff(dim='s_rho')
	
	# Handle dz slicing: dzii might be 3D (z,y,x) or 4D (t,z,y,x)
	# If dzii is static (3D), we just slice s_rho.
	# If dzii varies with time, ensure you passed the slice for time k.
	dz_sliced = dzii.isel(s_rho=slice(None, -1))
	
	d_dz = d_dz_diff / dz_sliced
	
	return d_dx, d_dy, d_dz

# --- Main Loop ---

for i in range(16,len(etas)):
	current_indices = etas[i]
	start_eta = current_indices[0]
	end_eta = current_indices[-1]
	
	print(f"\nProcessing Batch {i}: Eta {start_eta} to {end_eta}")

	found_targets = [t for t in target_eta_indices if t in current_indices]
	
	# --- 1. Load Data ---
	# Load Geometric metrics
	ds_dz = xr.open_dataset(dzs + f'dz_{i}.nc')
	dzii = ds_dz.dz # Keep as dask array initially if possible

	# Load Velocities (Full raw needed for filtering)
	print("  Loading Raw Data...")
	u_full_raw = xr.open_dataset(speed + f'u_slice_{i}.nc').u
	v_full_raw = xr.open_dataset(speed + f'v_slice_{i}.nc').v
	w_full_raw = xr.open_dataset(speed + f'w_slice_{i}.nc').w
	
	# Open Primes (Lazy Load - DO NOT .values() yet)
	ds_u_prime = xr.open_dataset(speed + f'u_prime_{i}.nc')
	ds_v_prime = xr.open_dataset(speed + f'v_prime_{i}.nc')
	ds_w_prime = xr.open_dataset(speed + f'w_prime_{i}.nc')
	
	ocean_time = ds_u_prime.ocean_time

	# --- 2. Compute Mean Flow (Global Filtering) ---
	# Filtering must happen on the whole time series to maintain phase correctness.
	print("  Filtering for Mean Flow (>40h)...")
	u_mean = filter_lowpass(u_full_raw, 40, 1.0).sel(ocean_time=ocean_time)
	v_mean = filter_lowpass(v_full_raw, 40, 1.0).sel(ocean_time=ocean_time)
	w_mean = filter_lowpass(w_full_raw, 40, 1.0).sel(ocean_time=ocean_time)

	# Clean raw inputs immediately
	del u_full_raw, v_full_raw, w_full_raw
	gc.collect()

	# --- Prepare Target Indices ---
	target_rel_indices = []
	target_names = []
	for t_global in found_targets:
		rel_idx = t_global - start_eta
		# Check bounds (accounting for diff reducing size by 1)
		if rel_idx < (len(current_indices) - 1): 
			 target_rel_indices.append(rel_idx)
			 name = [k for k, v in targets.items() if v == t_global][0]
			 target_names.append(name)

	# --- 3. Time Step Loop (Gradients & Production) ---
	print(f"  Calculating Gradients & Production step-by-step...")
	temp_files = []

	for k in range(len(ocean_time)):
		# progress indicator every 10 steps
		if k % 10 == 0: print(f"    Time step {k}/{len(ocean_time)}", end='\r')

		# A. Slice Mean Flow for this timestep
		# We compute() here to get numpy array for the math operations
		u_m_t = u_mean.isel(ocean_time=k).compute()
		v_m_t = v_mean.isel(ocean_time=k).compute()
		# w_mean isn't strictly used in the production formula provided, 
		# but if needed, slice it here.

		# B. Compute Gradients (Instantaneous Mean Gradients)
		# Note: We pass the full dzii. If dz changes in time, select isel(ocean_time=k)
		dU_dx, dU_dy, dU_dz = compute_gradients_step(u_m_t, dzii)
		dV_dx, dV_dy, dV_dz = compute_gradients_step(v_m_t, dzii)

		# C. Load Primes for this timestep ONLY
		# Isel on the dataset pointer loads only this step into memory
		u_p = ds_u_prime.u_prime.isel(ocean_time=k).compute()
		v_p = ds_v_prime.v_prime.isel(ocean_time=k).compute()
		w_p = ds_w_prime.w_prime.isel(ocean_time=k).compute()

		# Align Primes with Gradients (Slice off last index because diff reduced size)
		u_p = u_p.isel(xi_rho=slice(None,-1), eta_rho=slice(None,-1), s_rho=slice(None,-1))
		v_p = v_p.isel(xi_rho=slice(None,-1), eta_rho=slice(None,-1), s_rho=slice(None,-1))
		w_p = w_p.isel(xi_rho=slice(None,-1), eta_rho=slice(None,-1), s_rho=slice(None,-1))

		# D. Calculate Reynolds Stresses
		uu, uv, vv = u_p*u_p, u_p*v_p, v_p*v_p
		uw, vw = u_p*w_p, v_p*w_p

		# E. Calculate Production Terms
		# Horizontal (Deformation)
		term_hor = (uu * dU_dx) + (uv * (dU_dy + dV_dx)) + (vv * dV_dy)
		P_hor_3d = -rho_0 * term_hor
		
		# Vertical (Shear)
		term_vert = (uw * dU_dz) + (vw * dV_dz)
		P_vert_3d = -rho_0 * term_vert

		# F. Integration for Maps
		# Handle dz slicing for integration
		dz_iso = dzii.isel(s_rho=slice(None,-1))
		# If dz has time dim: dz_iso = dz_iso.isel(ocean_time=k)
		
		P_hor_int = (P_hor_3d * dz_iso).sum(dim='s_rho')
		P_vert_int = (P_vert_3d * dz_iso).sum(dim='s_rho')

		# G. Save Output for this Time Step
		ds_out = xr.Dataset()
		
		# 1. Save 2D Maps
		ds_out['P_hor_int'] = P_hor_int
		ds_out['P_vert_int'] = P_vert_int
		
		# 2. Save 3D Transects (Only if targets present)
		if target_rel_indices:
			# Extract specific rows
			p_vert_slice = P_vert_3d.isel(eta_rho=target_rel_indices)
			v_mean_slice = v_m_t.isel(eta_rho=target_rel_indices) 
			# Note: v_m_t might need slicing -1 on x/z if not done already
			# Since gradients reduced u_m_t size, we should ensure v_mean_slice matches P_vert_slice coordinates
			# A safe bet is slicing v_m_t similarly to u_p before saving:
			v_mean_slice = v_mean_slice.isel(xi_rho=slice(None,-1), s_rho=slice(None,-1))

			ds_out['P_vert_section'] = p_vert_slice
			ds_out['v_mean_section'] = v_mean_slice
			
			ds_out.attrs['contained_sections'] = ','.join(target_names)
			ds_out.attrs['section_indices'] = ','.join(map(str, found_targets))
		# 3. Expand Dimensions to include Time
		# This adds 'ocean_time' as a dimension of size 1 and assigns the correct coordinate value.
		current_time_val = ocean_time.values[k]
		ds_out = ds_out.expand_dims(dim={'ocean_time': [current_time_val]})
		
		# Save Temp File
		fname = temp_path + f'prod_{k}_slice_{i}.nc'
		ds_out.to_netcdf(fname)
		temp_files.append(fname)

		# Cleanup Memory for this step
		del P_hor_3d, P_vert_3d, P_hor_int, P_vert_int
		del uu, uv, vv, uw, vw, u_p, v_p, w_p
		del dU_dx, dU_dy, dU_dz, dV_dx, dV_dy, dV_dz
		del ds_out
		gc.collect()

	# --- 4. Merge Temp Files ---
	print(f"\n  Merging {len(temp_files)} time steps...")
	# Open with dask to avoid loading everything
	ds_merged = xr.open_mfdataset(temp_files, concat_dim='ocean_time', combine='nested')
	ds_merged = ds_merged.assign_coords(ocean_time=ocean_time)
	
	final_name = out_path + f'production_terms_slice_{i}.nc'
	ds_merged.to_netcdf(final_name)
	
	# Cleanup Temp Files & Batch Memory
	ds_merged.close()
	for f in temp_files: 
		if os.path.exists(f): os.remove(f)
		
	del u_mean, v_mean, w_mean, ds_u_prime, ds_v_prime, ds_w_prime
	gc.collect()

print("\nProcessing Complete.")
