import sys
import os
import numpy as np
import xarray as xr
import gc
import xeofs as xe

# =============================================================================
# 1. SETUP
# =============================================================================

p1_dir = '/home/arian/dd_waves/m2_internal_tide_gen_prop/'
sys.path.append(p1_dir)

# Directories
velocity_dir = '/data1/roms_dd_waves/analysis_outs/NIW/speed/'
out_path = '/data1/roms_dd_waves/analysis_outs/NIW/speed/modes/' 

os.makedirs(out_path, exist_ok=True)

# Loop Range (Spatial Slices)
range_start = 0
range_end = 20  # loops 0 to 19

# Constants
N_MODES = 3
TIME_CHUNK_SIZE = 766 

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def decompose_and_calculate_ke(da_u, da_v, n_modes=3):
	"""
	Performs Vertical EOF on U and V separately, reconstructs them,
	and calculates Kinetic Energy per mode.
	"""
	
	# --- Internal Helper for EOF ---
	def _perform_eof(da, var_name):
		# 1. Stack dimensions: (ocean_time, eta_rho, xi_rho) -> 'sample'
		stack_dims = ('ocean_time', 'eta_rho', 'xi_rho')
		
		# Ensure we are working with a DataArray
		if isinstance(da, xr.Dataset): 
			# Fallback if a dataset is passed, tries to find the variable
			vn = var_name if var_name in da else f'{var_name}_prime'
			da = da[vn]

		da_stacked = da.stack(sample=stack_dims)
		
		# 2. Handle NaNs (Land Mask)
		da_valid = da_stacked.dropna(dim='sample')
		
		if da_valid.shape[0] == 0:
			return None, None

		# 3. Fit EOF Model (Vertical EOF)
		# s_rho is the feature dimension
		model = xe.single.EOF(n_modes=n_modes, standardize=False, use_coslat=False)
		model.fit(da_valid, dim='sample')
		
		components = model.components() # Shape: (mode, s_rho)
		scores = model.transform(da_valid) # Shape: (sample, mode)
		
		reconstructed_modes = {}
		
		# 4. Reconstruct per mode
		for m in range(1, n_modes + 1):
			try:
				# Select mode (handles both integer and labeled indexing)
				comp_m = components.sel(mode=m)
				score_m = scores.sel(mode=m)
			except:
				comp_m = components.isel(mode=m-1)
				score_m = scores.isel(mode=m-1)
			
			# Reconstruct: Score * Component
			recon_stacked = score_m * comp_m
			
			# Unstack back to 4D (Time, z, y, x)
			recon_4d = (recon_stacked
						.unstack('sample')
						.transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
						.drop_vars('mode', errors='ignore'))
			
			reconstructed_modes[m] = recon_4d.astype(np.float32)
			
		return reconstructed_modes

	# --- Execute EOF ---
	u_modes_dict = _perform_eof(da_u, 'u')
	v_modes_dict = _perform_eof(da_v, 'v')
	
	if u_modes_dict is None or v_modes_dict is None:
		return None

	data_vars = {}
	
	# --- Organize Output and Calculate KE ---
	for m in range(1, n_modes + 1):
		u_rec = u_modes_dict[m]
		v_rec = v_modes_dict[m]
		
		# 1. Save Velocities
		data_vars[f'u_prime_mode{m}'] = u_rec
		data_vars[f'v_prime_mode{m}'] = v_rec
		
		# 2. Calculate KE for this mode
		# KE = 0.5 * (u^2 + v^2)
		ke_4d = 0.5 * (u_rec**2 + v_rec**2)
		
		# 3. Depth Summed KE (Sum along s_rho)
		# Note: True integral requires multiplying by Hz (layer thickness). 
		# This is a summation over sigma levels.
		ke_depth_sum = ke_4d.sum(dim='s_rho')
		
		data_vars[f'ke_mode{m}'] = ke_4d.astype(np.float32)
		data_vars[f'ke_sum_mode{m}'] = ke_depth_sum.astype(np.float32)

	return xr.Dataset(data_vars)
# =============================================================================
# 3. MAIN LOOP
# =============================================================================

for idx in range(range_start, range_end):
	print(f'\n=== Processing EOF Slice {idx} ===')
	
	# 1. Define Paths
	file_u_prime = f"{velocity_dir}u_prime_{idx}.nc"
	file_v_prime = f"{velocity_dir}v_prime_{idx}.nc"
	out_file = f"{out_path}velocity_eof_modes_slice_{idx:02d}.nc"
	
	# Reset/Remove output file if it exists
	if os.path.exists(out_file):
		os.remove(out_file)
	
	# Check inputs
	if not all(os.path.exists(f) for f in [file_u_prime, file_v_prime]):
		print(f"   ! Missing files for index {idx}. Skipping.")
		continue
		
	# 2. Open Datasets
	with xr.open_dataset(file_u_prime, chunks={}) as ds_uprime, \
		 xr.open_dataset(file_v_prime, chunks={}) as ds_vprime:

		total_times = len(ds_uprime.ocean_time)
		print(f"   > Total Time Steps found: {total_times}")

		# 3. Time Chunk Loop
		for t_start in range(0, total_times, TIME_CHUNK_SIZE):
			t_end = min(t_start + TIME_CHUNK_SIZE, total_times)
			
			print(f"     > Processing time chunk: {t_start} to {t_end}...")
			
			try:
				vn_up = 'u_prime' if 'u_prime' in ds_uprime else 'u'
				vn_vp = 'v_prime' if 'v_prime' in ds_vprime else 'v'
				
				# Load chunk
				u_prime_chunk = ds_uprime[vn_up].isel(ocean_time=slice(t_start, t_end)).compute()
				v_prime_chunk = ds_vprime[vn_vp].isel(ocean_time=slice(t_start, t_end)).compute()
				
				# --- EOF Decomposition ---
				ds_modes = decompose_and_calculate_ke(u_prime_chunk, v_prime_chunk, n_modes=N_MODES)
				
				if ds_modes is None:
					print("       ! Chunk contained only NaNs or failed.")
					continue

				# --- CRITICAL FIX: Clear Encoding ---
				# This ensures xarray doesn't try to enforce previous chunk sizes or fixed dimensions
				ds_modes['ocean_time'].encoding = {}
				if 'ocean_time' in ds_modes.coords:
					 ds_modes.coords['ocean_time'].encoding = {}
				ds_modes.encoding = {}

				# --- Save ---
				if t_start == 0:
					# Initialize file with UNLIMITED dimension
					encoding = {v: {'zlib': True, 'complevel': 1, 'dtype': 'float32'} for v in ds_modes.data_vars}
					ds_modes.to_netcdf(
						out_file, 
						mode='w', 
						unlimited_dims=['ocean_time'], 
						encoding=encoding
					)
				else:
					# Append
					ds_modes.to_netcdf(
						out_file, 
						mode='a', 
						unlimited_dims=['ocean_time']
					)
				
				# Cleanup
				del u_prime_chunk, v_prime_chunk, ds_modes
				gc.collect()
				
			except Exception as e:
				print(f"      ! Error in chunk {t_start}-{t_end}: {e}")
				# Raising error here to stop loop if something is truly broken
				# raise e 
				continue

	print(f"   [Done] Saved {out_file}")
	gc.collect()