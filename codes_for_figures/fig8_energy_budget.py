

import numpy as np
from scipy.signal import welch, csd
import matplotlib.pyplot as plt
p1_dir = '/home/arian/dd_waves/wind_driven_iw/'
import sys
sys.path.append(p1_dir)
from utils_roms_p1 import *
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt,windows
from cartopy.feature import NaturalEarthFeature, COLORS
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from scipy import signal
from gsw import *
import time

date_ranges_normal = [
	("2004-03-31", "2004-04-09")
	
]

date_ranges_cf = [
	("2004-03-19", "2004-03-24"),
	("2004-04-09", "2004-04-11")
	
]


date_ranges_hurricane = [
	("2004-03-23", "2004-03-30")
	
]


def create_date_mask(dates_array, date_ranges):
	"""
	Creates a single boolean mask for all time steps that fall sshthin
	any of the specified date ranges.
	"""
	# Initialize a mask of False for all time steps
	combined_mask = np.zeros_like(dates_array, dtype=bool)

	for start_date, end_date in date_ranges:
		# Convert strings to Pandas Timestamps for comparison
		start_ts = pd.to_datetime(start_date)
		end_ts = pd.to_datetime(end_date)

		# Create a mask for the current range: (start <= date <= end)
		range_mask = (dates_array >= start_ts) & (dates_array <= end_ts)

		# Update the combined mask using an 'OR' operation
		combined_mask = combined_mask | range_mask

	return combined_mask



def get_4d_budget_scalar(data_array, time_mask, eta_slice, xi_slice, normalizer):
	"""
	Slices, aggregates, and normalizes the EKE tendency for a given region/time.
	Note: The result is the AVERAGE depth-integrated EKE tendency per unit AREA (W/m^2)
	"""
	# 1. Slice in time immediately (reduces the largest dimension)
	sliced_time = data_array.isel(ocean_time=time_mask)
	
	# 2. Slice in space
	sliced_space = sliced_time.isel(eta_rho=slice(*eta_slice), xi_rho=slice(*xi_slice))
	
	# 3. Aggregation (Still lazy Dask computation)
	#   a) Sum over depth (s_rho) -> Result is EKE_trend * depth (units: J/m^2/s or W/m^2)
	#   b) Take the mean over time
	#   c) Sum over the horizontal dimensions (eta_rho, xi_rho) -> Result is Total EKE Trend (W)
	total_power = sliced_space.mean(dim='ocean_time').sum()

	# 4. Final Calculation and Normalization (Trigger computation here)
	# Total Power (W) / N_X (unitless) * dA (m²)
	# Your original normalization was total_power * dA / N_X. 
	# To keep the original structure, we perform:
	result = (total_power * dA / normalizer).compute()
	
	return result.item()	

############################################################################################################
#########################-----> Defining the control volumes <--------######################################

#The results will be presented in W/m² once we normalized by the amount of grid points
dv_S_eta, dv_S_xi = [40,430], [300,600]

dv_C_eta, dv_C_xi = [450,700],[115,350]

dv_N_eta, dv_N_xi = [800,1200],[430,600]

dx = 1000
dA = dx*dx
# Calculate number of grid cells (N) and Total Area (A) for each region

# South Region (S)
N_S = (dv_S_eta[1] - dv_S_eta[0]) * (dv_S_xi[1] - dv_S_xi[0])
A_S = N_S * dA

# Central Region (C)
N_C = (dv_C_eta[1] - dv_C_eta[0]) * (dv_C_xi[1] - dv_C_xi[0])
A_C = N_C * dA

# North Region (N)
N_N = (dv_N_eta[1] - dv_N_eta[0]) * (dv_N_xi[1] - dv_N_xi[0])
A_N = N_N * dA


print(f"N_S: {N_S}, N_C: {N_C}, N_N: {N_N}")

#N_S: 117000, N_C: 58750, N_N: 68000
############################################################################################################
#########################-----> Term VII-->Potential energy <--------####################################
#
pot_ex = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/pot_energy/pot_exchange_slice_*.nc').pot_ex
dates = pot_ex.ocean_time.to_index()
#
mask_normal = create_date_mask(dates, date_ranges_normal)
mask_cf = create_date_mask(dates, date_ranges_cf)
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)

results = {}
# --- Normal Scenario ---
print("--- Starting Normal Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(pot_ex, mask_normal, dv_S_eta, dv_S_xi, N_S)
results['pot_ex_normal_S'] = temp_scalar
del temp_scalar
gc.collect() 

# North
temp_scalar = get_4d_budget_scalar(pot_ex, mask_normal, dv_N_eta, dv_N_xi, N_N)
results['pot_ex_normal_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(pot_ex, mask_normal, dv_C_eta, dv_C_xi, N_C)
results['pot_ex_normal_C'] = temp_scalar
del temp_scalar
gc.collect() 
print('Done for normal')


# --- CF Scenario ---
print("--- Starting CF Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(pot_ex, mask_cf, dv_S_eta, dv_S_xi, N_S)
results['pot_ex_cf_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(pot_ex, mask_cf, dv_N_eta, dv_N_xi, N_N)
results['pot_ex_cf_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(pot_ex, mask_cf, dv_C_eta, dv_C_xi, N_C)
results['pot_ex_cf_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for cf')
# --- Hurricane Scenario ---
print("--- Starting Hurricane Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(pot_ex, mask_hurricane, dv_S_eta, dv_S_xi, N_S)
results['pot_ex_hurricane_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(pot_ex, mask_hurricane, dv_N_eta, dv_N_xi, N_N)
results['pot_ex_hurricane_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(pot_ex, mask_hurricane, dv_C_eta, dv_C_xi, N_C)
results['pot_ex_hurricane_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for hurricane')

df_pot_ex = pd.DataFrame.from_dict(
	results, 
	orient='index', 
	columns=['pot_ex']
)

df_pot_ex['Scenario'] = df_pot_ex.index.str.split('_').str[2]
df_pot_ex['Region'] = df_pot_ex.index.str.split('_').str[3]
df_pot_ex.reset_index(drop=True, inplace=True)
df_pot_ex = df_pot_ex[['Scenario', 'Region', 'pot_ex']]

df_pot_ex.to_csv('pot_ex.csv', index=False)
"""

############################################################################################################
#########################-----> Term VI--> Wave wave <--------####################################
#
ww = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/coarse/PI_wave_zsum_slice_*.nc').PI_wave_zsum
dates = ww.ocean_time.to_index()
#
mask_normal = create_date_mask(dates, date_ranges_normal)
mask_cf = create_date_mask(dates, date_ranges_cf)
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)

results = {}
# --- Normal Scenario ---
print("--- Starting Normal Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(ww, mask_normal, dv_S_eta, dv_S_xi, N_S)
results['ww_normal_S'] = temp_scalar
del temp_scalar
gc.collect() 

# North
temp_scalar = get_4d_budget_scalar(ww, mask_normal, dv_N_eta, dv_N_xi, N_N)
results['ww_normal_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(ww, mask_normal, dv_C_eta, dv_C_xi, N_C)
results['ww_normal_C'] = temp_scalar
del temp_scalar
gc.collect() 
print('Done for normal')


# --- CF Scenario ---
print("--- Starting CF Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(ww, mask_cf, dv_S_eta, dv_S_xi, N_S)
results['ww_cf_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(ww, mask_cf, dv_N_eta, dv_N_xi, N_N)
results['ww_cf_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(ww, mask_cf, dv_C_eta, dv_C_xi, N_C)
results['ww_cf_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for cf')
# --- Hurricane Scenario ---
print("--- Starting Hurricane Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(ww, mask_hurricane, dv_S_eta, dv_S_xi, N_S)
results['ww_hurricane_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(ww, mask_hurricane, dv_N_eta, dv_N_xi, N_N)
results['ww_hurricane_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(ww, mask_hurricane, dv_C_eta, dv_C_xi, N_C)
results['ww_hurricane_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for hurricane')

df_ww = pd.DataFrame.from_dict(
	results, 
	orient='index', 
	columns=['wave_wave']
)

df_ww['Scenario'] = df_ww.index.str.split('_').str[2]
df_ww['Region'] = df_ww.index.str.split('_').str[3]
df_ww.reset_index(drop=True, inplace=True)
df_ww = df_ww[['Scenario', 'Region', 'wave_wave']]

df_ww.to_csv('wave_wave.csv', index=False)



############################################################################################################
#########################-----> Term VIII--> Viscous dissipation <--------####################################
#
visc = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/viscous/visc_loss_*.nc').viscous
dates = visc.ocean_time.to_index()
#
mask_normal = create_date_mask(dates, date_ranges_normal)
mask_cf = create_date_mask(dates, date_ranges_cf)
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)

results = {}
# --- Normal Scenario ---
print("--- Starting Normal Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(visc, mask_normal, dv_S_eta, dv_S_xi, N_S)
results['visc_normal_S'] = temp_scalar
del temp_scalar
gc.collect() 
#'visc_normal_S': 1.8448142956280775

# North
temp_scalar = get_4d_budget_scalar(visc, mask_normal, dv_N_eta, dv_N_xi, N_N)
results['visc_normal_N'] = temp_scalar
del temp_scalar
gc.collect()
#'visc_normal_N': 3.1250924000560145

# Center
temp_scalar = get_4d_budget_scalar(visc, mask_normal, dv_C_eta, dv_C_xi, N_C)
results['visc_normal_C'] = temp_scalar
del temp_scalar
gc.collect() 
print('Done for normal')
#'visc_normal_C': 0.851357880773520

# --- CF Scenario ---
print("--- Starting CF Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(visc, mask_cf, dv_S_eta, dv_S_xi, N_S)
results['visc_cf_S'] = temp_scalar
del temp_scalar
gc.collect()
#'visc_cf_S': 1.909535338608111

# North
temp_scalar = get_4d_budget_scalar(visc, mask_cf, dv_N_eta, dv_N_xi, N_N)
results['visc_cf_N'] = temp_scalar
del temp_scalar
gc.collect()
#'visc_cf_N': 2.1278024946660503

# Center
temp_scalar = get_4d_budget_scalar(visc, mask_cf, dv_C_eta, dv_C_xi, N_C)
results['visc_cf_C'] = temp_scalar
del temp_scalar
gc.collect()
#'visc_cf_C': 3.131279777256488
print('Done for cf')


# --- Hurricane Scenario ---
print("--- Starting Hurricane Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(visc, mask_hurricane, dv_S_eta, dv_S_xi, N_S)
results['visc_hurricane_S'] = temp_scalar
del temp_scalar
gc.collect()
#'visc_hurricane_S': 2.452730062897135

# North
temp_scalar = get_4d_budget_scalar(visc, mask_hurricane, dv_N_eta, dv_N_xi, N_N)
results['visc_hurricane_N'] = temp_scalar
del temp_scalar
gc.collect()
#'visc_hurricane_N': 1.298143190735637

# Center
temp_scalar = get_4d_budget_scalar(visc, mask_hurricane, dv_C_eta, dv_C_xi, N_C)
results['visc_hurricane_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for hurricane')
#'visc_hurricane_C': 2.3097196790685834

df_visc = pd.DataFrame.from_dict(
	results, 
	orient='index', 
	columns=['viscous']
)

df_visc['Scenario'] = df_visc.index.str.split('_').str[2]
df_visc['Region'] = df_visc.index.str.split('_').str[3]
df_visc.reset_index(drop=True, inplace=True)
df_visc = df_visc[['Scenario', 'Region', 'viscous']]

df_visc.to_csv('viscous.csv', index=False)

############################################################################################################
#########################-----> Term V--> Wave mean flow <--------####################################
#
wv_mf = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/EP_flux/mean_flow_wave_slice_*.nc').mean_flow_wave
dates = wv_mf.ocean_time.to_index()
#
mask_normal = create_date_mask(dates, date_ranges_normal)
mask_cf = create_date_mask(dates, date_ranges_cf)
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)

results = {}
# --- Normal Scenario ---
print("--- Starting Normal Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(wv_mf, mask_normal, dv_S_eta, dv_S_xi, N_S)
results['wv_mf_normal_S'] = temp_scalar
del temp_scalar
gc.collect() 

# North
temp_scalar = get_4d_budget_scalar(wv_mf, mask_normal, dv_N_eta, dv_N_xi, N_N)
results['wv_mf_normal_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(wv_mf, mask_normal, dv_C_eta, dv_C_xi, N_C)
results['wv_mf_normal_C'] = temp_scalar
del temp_scalar
gc.collect() 
print('Done for normal')


# --- CF Scenario ---
print("--- Starting CF Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(wv_mf, mask_cf, dv_S_eta, dv_S_xi, N_S)
results['wv_mf_cf_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(wv_mf, mask_cf, dv_N_eta, dv_N_xi, N_N)
results['wv_mf_cf_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(wv_mf, mask_cf, dv_C_eta, dv_C_xi, N_C)
results['wv_mf_cf_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for cf')
# --- Hurricane Scenario ---
print("--- Starting Hurricane Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(wv_mf, mask_hurricane, dv_S_eta, dv_S_xi, N_S)
results['wv_mf_hurricane_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(wv_mf, mask_hurricane, dv_N_eta, dv_N_xi, N_N)
results['wv_mf_hurricane_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(wv_mf, mask_hurricane, dv_C_eta, dv_C_xi, N_C)
results['wv_mf_hurricane_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for hurricane')

df_wv_mf = pd.DataFrame.from_dict(
	results, 
	orient='index', 
	columns=['wave_mean_flow']
)

df_wv_mf['Scenario'] = df_wv_mf.index.str.split('_').str[2]
df_wv_mf['Region'] = df_wv_mf.index.str.split('_').str[3]
df_wv_mf.reset_index(drop=True, inplace=True)
df_wv_mf = df_wv_mf[['Scenario', 'Region', 'wave_mean_flow']]

df_wv_mf.to_csv('wave_mean_flow.csv', index=False)


############################################################################################################
#########################-----> Term IV--> pressure work <--------####################################
#
p_work = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/pressure_b/pressure_work_slice_*.nc').pw
dates = p_work.ocean_time.to_index()
#
mask_normal = create_date_mask(dates, date_ranges_normal)
mask_cf = create_date_mask(dates, date_ranges_cf)
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)

results = {}
# --- Normal Scenario ---
print("--- Starting Normal Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(p_work, mask_normal, dv_S_eta, dv_S_xi, N_S)
results['p_work_normal_S'] = temp_scalar
del temp_scalar
gc.collect() 

# North
temp_scalar = get_4d_budget_scalar(p_work, mask_normal, dv_N_eta, dv_N_xi, N_N)
results['p_work_normal_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(p_work, mask_normal, dv_C_eta, dv_C_xi, N_C)
results['p_work_normal_C'] = temp_scalar
del temp_scalar
gc.collect() 
print('Done for normal')


# --- CF Scenario ---
print("--- Starting CF Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(p_work, mask_cf, dv_S_eta, dv_S_xi, N_S)
results['p_work_cf_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(p_work, mask_cf, dv_N_eta, dv_N_xi, N_N)
results['p_work_cf_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(p_work, mask_cf, dv_C_eta, dv_C_xi, N_C)
results['p_work_cf_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for cf')
# --- Hurricane Scenario ---
print("--- Starting Hurricane Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(p_work, mask_hurricane, dv_S_eta, dv_S_xi, N_S)
results['p_work_hurricane_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(p_work, mask_hurricane, dv_N_eta, dv_N_xi, N_N)
results['p_work_hurricane_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(p_work, mask_hurricane, dv_C_eta, dv_C_xi, N_C)
results['p_work_hurricane_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for hurricane')

df_p_work = pd.DataFrame.from_dict(
	results, 
	orient='index', 
	columns=['presusre_work_divergence']
)

df_p_work['Scenario'] = df_p_work.index.str.split('_').str[2]
df_p_work['Region'] = df_p_work.index.str.split('_').str[3]
df_p_work.reset_index(drop=True, inplace=True)
df_p_work = df_p_work[['Scenario', 'Region', 'presusre_work_divergence']]

df_p_work.to_csv('pressure_work_divergence.csv', index=False)
############################################################################################################
#########################-----> Term I--> effective wind input <--------####################################

wp_normal = xr.open_dataset('wp_mld_normal.nc').wp.compute()
wp_S_normal =  np.absolute(wp_normal.isel(eta_rho=slice(*dv_S_eta),xi_rho=slice(*dv_S_xi))).sum()*dA/N_S
#1.06756919
wp_N_normal = np.absolute(wp_normal.isel(eta_rho=slice(*dv_N_eta),xi_rho=slice(*dv_N_xi))).sum()*dA/N_N
#0.59710058
wp_C_normal = np.absolute(wp_normal.isel(eta_rho=slice(*dv_C_eta),xi_rho=slice(*dv_C_xi))).sum()*dA/N_C
#1.03702278

wp_cf = xr.open_dataset('wp_mld_cf.nc').wp.compute()
wp_S_cf =  np.absolute(wp_cf.isel(eta_rho=slice(*dv_S_eta),xi_rho=slice(*dv_S_xi))).sum()*dA/N_S
#1.48980909
wp_N_cf = np.absolute(wp_cf.isel(eta_rho=slice(*dv_N_eta),xi_rho=slice(*dv_N_xi))).sum()*dA/N_S
#0.43292402
wp_C_cf = np.absolute(wp_cf.isel(eta_rho=slice(*dv_C_eta),xi_rho=slice(*dv_C_xi))).sum()*dA/N_S
#1.14055266


wp_hurricane = xr.open_dataset('wp_mld_hurricane.nc').wp.compute()
wp_S_hurricane =  np.absolute(wp_hurricane.isel(eta_rho=slice(*dv_S_eta),xi_rho=slice(*dv_S_xi))).sum()*dA/N_S
#1.22495378
wp_N_hurricane = np.absolute(wp_hurricane.isel(eta_rho=slice(*dv_N_eta),xi_rho=slice(*dv_N_xi))).sum()*dA/N_S
#0.30640612
wp_C_hurricane = np.absolute(wp_hurricane.isel(eta_rho=slice(*dv_C_eta),xi_rho=slice(*dv_C_xi))).sum()*dA/N_S
#0.94891155





############################################################################################################
#########################-----> Term III--> eke advection <--------####################################
#

du_ke = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/eke/eke_adv_slice_*.nc').eke_adv
dates = du_ke.ocean_time.to_index()

mask_normal = create_date_mask(dates, date_ranges_normal)
mask_cf = create_date_mask(dates, date_ranges_cf)
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)

results = {}
# --- Normal Scenario ---
print("--- Starting Normal Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(du_ke, mask_normal, dv_S_eta, dv_S_xi, N_S)
results['du_ke_normal_S'] = temp_scalar
del temp_scalar
gc.collect() 

# North
temp_scalar = get_4d_budget_scalar(du_ke, mask_normal, dv_N_eta, dv_N_xi, N_N)
results['du_ke_normal_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(du_ke, mask_normal, dv_C_eta, dv_C_xi, N_C)
results['du_ke_normal_C'] = temp_scalar
del temp_scalar
gc.collect() 
print('Done for normal')


# --- CF Scenario ---
print("--- Starting CF Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(du_ke, mask_cf, dv_S_eta, dv_S_xi, N_S)
results['du_ke_cf_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(du_ke, mask_cf, dv_N_eta, dv_N_xi, N_N)
results['du_ke_cf_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(du_ke, mask_cf, dv_C_eta, dv_C_xi, N_C)
results['du_ke_cf_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for cf')
# --- Hurricane Scenario ---
print("--- Starting Hurricane Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(du_ke, mask_hurricane, dv_S_eta, dv_S_xi, N_S)
results['du_ke_hurricane_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(du_ke, mask_hurricane, dv_N_eta, dv_N_xi, N_N)
results['du_ke_hurricane_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(du_ke, mask_hurricane, dv_C_eta, dv_C_xi, N_C)
results['du_ke_hurricane_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for hurricane')

df_du_ke = pd.DataFrame.from_dict(
	results, 
	orient='index', 
	columns=['EKE_Advection_Mean_Flux']
)
df_du_ke['Scenario'] = df_du_ke.index.str.split('_').str[2]
df_du_ke['Region'] = df_du_ke.index.str.split('_').str[3]
df_du_ke.reset_index(drop=True, inplace=True)
df_du_ke = df_du_ke[['Scenario', 'Region', 'EKE_Advection_Mean_Flux']]

df_du_ke.to_csv('eke_advection_mean_flux.csv', index=False)



############################################################################################################
#########################-----> Term II--> eke tendency <--------####################################
#
dt_ke = xr.open_mfdataset('/data1/roms_dd_waves/analysis_outs/NIW/eke/eke_trend_slice_*.nc').eke_trend
dates = dt_ke.ocean_time.to_index()
#
mask_normal = create_date_mask(dates, date_ranges_normal)
mask_cf = create_date_mask(dates, date_ranges_cf)
mask_hurricane = create_date_mask(dates, date_ranges_hurricane)

results = {}
# --- Normal Scenario ---
print("--- Starting Normal Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(dt_ke, mask_normal, dv_S_eta, dv_S_xi, N_S)
results['dt_ke_normal_S'] = temp_scalar
del temp_scalar
gc.collect() 

# North
temp_scalar = get_4d_budget_scalar(dt_ke, mask_normal, dv_N_eta, dv_N_xi, N_N)
results['dt_ke_normal_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(dt_ke, mask_normal, dv_C_eta, dv_C_xi, N_C)
results['dt_ke_normal_C'] = temp_scalar
del temp_scalar
gc.collect() 
print('Done for normal')


# --- CF Scenario ---
print("--- Starting CF Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(dt_ke, mask_cf, dv_S_eta, dv_S_xi, N_S)
results['dt_ke_cf_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(dt_ke, mask_cf, dv_N_eta, dv_N_xi, N_N)
results['dt_ke_cf_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(dt_ke, mask_cf, dv_C_eta, dv_C_xi, N_C)
results['dt_ke_cf_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for cf')
# --- Hurricane Scenario ---
print("--- Starting Hurricane Scenario ---")

# South
temp_scalar = get_4d_budget_scalar(dt_ke, mask_hurricane, dv_S_eta, dv_S_xi, N_S)
results['dt_ke_hurricane_S'] = temp_scalar
del temp_scalar
gc.collect()

# North
temp_scalar = get_4d_budget_scalar(dt_ke, mask_hurricane, dv_N_eta, dv_N_xi, N_N)
results['dt_ke_hurricane_N'] = temp_scalar
del temp_scalar
gc.collect()

# Center
temp_scalar = get_4d_budget_scalar(dt_ke, mask_hurricane, dv_C_eta, dv_C_xi, N_C)
results['dt_ke_hurricane_C'] = temp_scalar
del temp_scalar
gc.collect()
print('Done for hurricane')

df_dt_ke = pd.DataFrame.from_dict(
	results, 
	orient='index', 
	columns=['EKE_Trend']
)

df_dt_ke['Scenario'] = df_dt_ke.index.str.split('_').str[2]
df_dt_ke['Region'] = df_dt_ke.index.str.split('_').str[3]
df_dt_ke.reset_index(drop=True, inplace=True)
df_dt_ke = df_dt_ke[['Scenario', 'Region', 'EKE_Trend']]

df_dt_ke.to_csv('eke_trend.csv', index=False)
"""