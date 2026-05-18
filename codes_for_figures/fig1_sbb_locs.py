

import sys
sys.path.append('/home/arian/dd_waves/in_situ_data/')

from utils_ts import *
from cartopy.feature import NaturalEarthFeature, COLORS
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import tropycal.tracks as tracks
import cmocean as cmo
from matplotlib.legend_handler import HandlerPatch
#summer = xr.open_dataset('/data1/currents/climatology_sbb/' + 'climatology_uv_mask_DJF.v9.1h.nc')
##
#lat = summer.lat #dlat = 0.125
#lon = summer.lon #dlon = 0.13000000000000256
#
##depth
#gebco = xr.open_dataset('/data1/currents/climatology_sbb/' + 'gebco.nc')
#gebco = gebco.sel(lat=slice(np.nanmin(lat),np.nanmax(lat)))
#gebco = gebco.sel(lon=slice(np.nanmin(lon),np.nanmax(lon)))
#gebco = gebco.interp(lat=lat,lon=lon) #regrid to climatology map
#
#topo = gebco.elevation
#mask = topo < 0
#topo = np.ma.masked_where(~mask, topo)


coarse = xr.open_dataset('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/3km/inputs/deproas_spongebob_grd_cropped_3km.nc')
finer = xr.open_dataset('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/inputs/paper_2_1km_closed_cropped_smooth_sponge.nc')



# Extract mask_rho
f_mask_rho = finer.mask_rho.compute()
c_mask_rho = coarse.mask_rho.compute()


# Get coordinates
lon_c, lat_c = coarse.lon_rho, coarse.lat_rho
lon_f, lat_f = finer.lon_rho, finer.lat_rho

# Extract mask and coordinates
c_mask_rho = coarse.mask_rho.values
f_mask_rho = finer.mask_rho.values
lon_c, lat_c = coarse.lon_rho.values, coarse.lat_rho.values
lon_f, lat_f = finer.lon_rho.values, finer.lat_rho.values

# Flatten arrays
lon_c_flat, lat_c_flat, c_mask_flat = lon_c.ravel(), lat_c.ravel(), c_mask_rho.ravel()
lon_f_flat, lat_f_flat, f_mask_flat = lon_f.ravel(), lat_f.ravel(), f_mask_rho.ravel()

# Select ocean and land points
coarse_ocean = c_mask_flat == 1
finer_ocean = f_mask_flat == 1
coarse_land = c_mask_flat == 0
finer_land = f_mask_flat == 0


avg_1km = xr.open_dataset('/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/avg_paper_2_1km.nc')


h = avg_1km.h.compute()
lat_rho = avg_1km.lat_rho
lon_rho = avg_1km.lon_rho


############_------> Cold Fronts <----#########
# --- 1. Define Locations and their Y-axis Order ---
# These are your specific locations from Rio Grande (RS) to Campos (RJ)
# The order here determines the order on the Y-axis.
locations = [
	"Porto Alegre (50.07° W, 30.29° S)",
	"Torres (49.35° W, 29.35° S)",
	"Florianopolis (48.06° W, 27.64° S)",
	"Paranagua (48.24° W, 25.59° S)",
	"Iguape (47.44° W, 24.73° S)",
	"Santos (46.33° W, 24.07° S)",
	"Ubatuba (44.96° W, 23.56° S)",
	"Rio de Janeiro (43.11° W, 22.99° S)",
	"Cabo Frio (41.94° W, 22.89° S)",
	"Campos (40.84° W, 21.74° S)"
]

# Create a mapping from location name to a numerical y-value for plotting
# This ensures consistent spacing and order.
# Create a mapping from the full location string to a numerical y-value (index)
location_to_y = {loc: i for i, loc in enumerate(locations)}
y_to_location = {i: loc for i, loc in enumerate(locations)}

# Separate the names and coordinates for the two Y-axes
left_y_labels = [loc.split('(')[0].strip() for loc in locations]
right_y_labels = [loc.split('(')[1].replace(')', '').strip() for loc in locations]

# --- 2. Create Example Cold Front Data (YOU WILL REPLACE THIS WITH YOUR REAL DATA) ---
# Each dictionary represents a cold front's passage:
# 'front_id': A unique identifier for the front.
# 'passages': A list of (date_str, location_name) tuples for this front.
#             The dates are in 'YYYY-MM-DD' format.

cold_front_data = [
	{
		'front_id': 1,
		'passages': [
			('2004-03-19', "Florianopolis (48.06° W, 27.64° S)"), 
			('2004-03-20', "Paranagua (48.24° W, 25.59° S)"),
			('2004-03-20', "Iguape (47.44° W, 24.73° S)"),
			('2004-03-20', "Santos (46.33° W, 24.07° S)"),
			('2004-03-21',"Ubatuba (44.96° W, 23.56° S)"),
			('2004-03-22', "Ubatuba (44.96° W, 23.56° S)"),
			('2004-03-23', "Rio de Janeiro (43.11° W, 22.99° S)"),
			('2004-03-23', "Cabo Frio (41.94° W, 22.89° S)"),
			('2004-03-24', "Campos (40.84° W, 21.74° S)"),
		]
	}

]

# --- 3. Prepare Data for Plotting ---
all_fronts_data = []
for front in cold_front_data:
	front_id = front['front_id']
	# Select color based on front_id for distinct lines
	color = 'tab:blue' if front_id == 1 else 'tab:orange'
	for date_str, location_name in front['passages']:
		if location_name in location_to_y:
			all_fronts_data.append({
				'date': pd.to_datetime(date_str),
				'location_y': location_to_y[location_name],
				'front_id': front_id,
				'color': color
			})

df = pd.DataFrame(all_fronts_data)








ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True)


# Create a custom colormap with mint cream, beige, purple, and blue
colors = [
	#(0.96, 1.0, 0.98),   # Very shallow (mint cream, almost white)
	(0.9, 0.8, 0.7),     # Shallow (beige)
	(0.7, 1.0, 0.7),     # Mid-deep (pastel green) 
	(0.5, 0.4, 0.8),     # Mid-depth (purple)
	(1.0, 0.3, 0.5),     # Mid-depth (pink or vibrant red)
	(1.0, 0.65, 0.0),    # Shallow (vibrant orange/gold)
	(0.2, 0.2, 0.7),     # Deeper (blue)
	(0.1, 0.1, 0.4),     # Deepest (dark blue)
]
plt.close('all')
fig = plt.figure(figsize=(8, 5))
gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1,1], height_ratios=[70,30])
gs.update(left=0.13, right=0.99, wspace=0.2, hspace=0.05, top=0.99, bottom=0.1)
#cmap = plt.get_cmap('Blues')

cmap = LinearSegmentedColormap.from_list('bathymetry_cmap', colors, N=500)
vmin = -5
vmax = 2500
#norm = LogNorm(vmin=vmin, vmax=vmax)
norm = cm.colors.Normalize(vmin=vmin,vmax=vmax)
levels=np.linspace(vmin,vmax,200)

bar_title = "m"
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.set_ylim(bottom =-33,top = -21)
ax.set_xlim(left = -52,right = -39.2)

### - contourf
norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
ax.contourf(lon_rho, lat_rho, h,  levels=levels, cmap=cmap, norm=norm, extend='max', zorder=0)
#--> deproas moorings
ax.scatter(-41.73357500, -23.73347222, zorder=5, s=30, marker='*', color='firebrick', label='CF3')
ax.scatter(-42.571 , -23.7246666667, zorder=5, s=30, marker='+', color='purple', label='BG3')
ax.scatter(-44.3681666667, -24.392, zorder=5, s=30, marker='s', color='orange', label='UB3')
#--> canyons scatter
ax.scatter(-44.9, -25.3, zorder=5, s=200, marker='|', color='k', label='Canyons')
ax.scatter(-45.5, -25.9, zorder=5, s=200, marker='|', color='k')
ax.scatter(-46.9, -27.5, zorder=5, s=200, marker='_', color='k')
ax.text(0.7, 0.98, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
		

legend = ax.legend(loc=4, fontsize='xx-small', markerscale=0.7)

# perks
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax.patch.set_edgecolor('black')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)

gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}

## isobaths
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]

levels1 = np.asarray(levels_1)
levels2 = np.asarray(levels_2)
levels3 = np.asarray(levels_3)
levels4 = np.asarray(levels_4)

c1 = ax.contour(lon_rho, lat_rho, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = ax.contour(lon_rho, lat_rho, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = ax.contour(lon_rho, lat_rho, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = ax.contour(lon_rho, lat_rho, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
#ax.hlines(y=-25.8, xmin=-49.2, xmax=-42, linewidth=1, color='purple',zorder = 3)
#ax.text(-45.3, -25.9, 'Santos Bifurcation', fontsize=7, ha='left', va='top', 
	   #bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Use proxy artists to create legend entries
legend_lines = [Line2D([0], [0], linestyle='dotted', linewidth=1, color='brown'),
				Line2D([0], [0], linestyle='dotted', linewidth=1, color='grey'),
				Line2D([0], [0], linestyle='dashed', linewidth=1, color='k'),
				Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')]

labels = ['50 m', '200 m', '1000 m', '2000 m']

# Updated legend part using fig.legend
#fig.legend(legend_lines, labels, loc='upper right', title='Isobaths', fontsize='xx-small', title_fontsize='xx-small')

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='xx-small',
	title_fontsize='xx-small',
	loc='center',
	bbox_to_anchor=(0.18, 0.82)
)


### -- Miniglobe
extent = [-49, -39.3, -29.5, -22]
center = [-45, -15]
box_color = 'red'
box_alpha = 1
box_edge_width = 1

sub_ax = inset_axes(ax, width="100%", height="100%", loc="lower center", bbox_to_anchor=(0.75, 0.25, 0.25, 0.25),
					bbox_transform=ax.transAxes,
					axes_class=cartopy.mpl.geoaxes.GeoAxes,
					axes_kwargs=dict(map_projection=ccrs.Orthographic(center[0], center[1])))
# Make a nice border around the inset axes.

# Add land and ocean features to the miniglobe
land_feature = NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=COLORS['land'])
ocean_feature = NaturalEarthFeature('physical', 'ocean', '110m', edgecolor='face', facecolor=COLORS['water'])

sub_ax.add_feature(land_feature)
sub_ax.add_feature(ocean_feature)

effect = Stroke(linewidth=.1, foreground='black', alpha=0.5)
sub_ax.patch.set_path_effects([effect])
sub_ax.coastlines(linewidth=0.00000001, edgecolor='k', alpha=.8)
extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='lightgray',
					  edgecolor=box_color, linewidth=box_edge_width, alpha=box_alpha)

# colorbar
cbar_1 = inset_axes(ax, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
norm_1 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm_1, extend='max', orientation='horizontal')
cb1.set_label(bar_title, size='xx-small')
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='xx-small')  # Set colorbar label size

CF = [-23.018, -41.97]
CS = [-22.015, -40.963]
CM = [-28.60, -48.809]

# CS
#ax.scatter(CS[1], CS[0],marker='D',zorder = 3, color='k', s=20)
#ax.text(CS[1] - 0.3, CS[0], 'CST', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# CF
ax.scatter(CF[1], CF[0],marker='D',zorder = 3, color='k', s=20)
ax.text(CF[1] - 0.3, CF[0] + 0.1, 'CF', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# CM
ax.scatter(CM[1], CM[0],marker='D',zorder = 3, color='k', s=20)
ax.text(CM[1] -0.25, CM[0] + 0.3, 'CSM', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')


#---> grid_subgrid
ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.scatter(lon_c_flat[coarse_ocean], lat_c_flat[coarse_ocean],s=0.001, color='blue', alpha=0.5, transform=ccrs.PlateCarree())
ax2.scatter(lon_f_flat, lat_f_flat, s=0.001, color='indianred', alpha=0.5, transform=ccrs.PlateCarree(),zorder = 2)

ax2.scatter(lon_c_flat[coarse_land], lat_c_flat[coarse_land],s=0.001, color='gray', alpha=1, transform=ccrs.PlateCarree(),zorder = 2)
#ax.scatter(lon_f_flat[finer_land], lat_f_flat[finer_land], s=0.001, color='whitesmoke', alpha=1, transform=ccrs.PlateCarree(),zorder = 2)

gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}
# Custom legend
legend_patches = [
	mpatches.Patch(color='blue', label='Coarser'),
	mpatches.Patch(color='indianred', label='Finer'),

]
ax2.text(0.05, 0.1, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
###########---> Plotting the hurricane track
storm = ibtracs.get_storm(('catarina',2004))


type_colors = {
	'TD': {'color': 'yellowgreen', 'marker': 'o', 'label': 'Tropical Depression'},
	'TS': {'color': 'magenta', 'marker': 'o', 'label': 'Tropical Storm'},
	'HU': {'color': 'gold', 'marker': 'o', 'label': 'Hurricane'}, # Catarina reached HU status
	'SS': {'color': 'red', 'marker': 'o', 'label': 'Subtropical Storm'},
	'EX': {'color': 'grey', 'marker': 's', 'label': 'Extratropical'}
	# Add other types if needed (e.g., LO, wave, DB, etc.)
}

# --- 2. Plot the Storm Track (Line) ---
# Plotting the line connects the track points sequentially
# Ensure zorder is high so it plots over the bathymetry (zorder=0) and land (zorder=1).
ax2.plot(storm.lon, storm.lat, 
		color='k', 
		linestyle='--', 
		linewidth=0.8, 
		alpha=0.6,
		transform=ccrs.PlateCarree(), 
		zorder=6,
		label='Catarina Track')


# --- 3. Plot Scatter Points with Color by Type ---

# Get the unique storm types present in your storm data
unique_types = np.unique(storm.type)

# Loop through each unique type to plot points and create a legend entry
for storm_type in unique_types:
	if storm_type in type_colors:
		style = type_colors[storm_type]
		
		# Create a boolean mask to select points matching the current type
		mask = (storm.type == storm_type)
		
		# Plot the points
		ax2.scatter(storm.lon[mask], storm.lat[mask], 
				   c=style['color'], 
				   marker=style['marker'], 
				   s=30, # size of the marker
				   edgecolors='k', # black outline for contrast
				   linewidths=0.5,
				   transform=ccrs.PlateCarree(), 
				   zorder=7, 
				   label=style['label'])

# --- 4. Plot Initial and Final Points (Optional, for emphasis) ---

# Optional: Add a large marker for the starting point
ax2.scatter(storm.lon[0], storm.lat[0], 
		   marker='^', 
		   s=60, 
		   color='green', 
		   edgecolor='k', 
		   linewidth=1,
		   transform=ccrs.PlateCarree(), 
		   zorder=8, 
		   label='Start')
		   
# Optional: Add a large marker for the final point
ax2.scatter(storm.lon[-1], storm.lat[-1], 
		   marker='v', 
		   s=60, 
		   color='brown', 
		   edgecolor='k', 
		   linewidth=1,
		   transform=ccrs.PlateCarree(), 
		   zorder=8, 
		   label='End')



ax2.legend(loc='upper left', fontsize=6,  title_fontsize=6)
# --- 6. Custom Legend 2: Hurricane Track Categories ---

###########################################################
###--- Cold fronts
###--- Cold fronts
ax3 = plt.subplot(gs[1, :])
ax3.text(0.05, 0.97, '(c)', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Setting the title with a smaller font size

start_date = datetime(2004, 3, 18)
end_date = datetime(2004, 4, 14)

# Temporal bounds (X-axis)
hurricane_start_date = datetime(2004, 3, 24)
hurricane_end_date = datetime(2004, 3, 30)

loc_florianopolis_y = location_to_y["Florianopolis (48.06° W, 27.64° S)"]

# Adjust Y-bounds for shading to cover the full tick space:
# We shade from 0.5 below the starting point to 0.5 above the ending point.
# Since Y-axis is numbered from bottom (RS) to top (Campos), 
# Florianopolis (SC) is numerically HIGHER than Porto Alegre (RS).
y_min_shade = -0.5 
# y_max_shade: End just after the Florianopolis line (index 3 + 0.5)
y_max_shade = loc_florianopolis_y + 0.01

# --- 5. Plot Front Tracks (Main Axis) ---
for front_id in df['front_id'].unique():
	front_df = df[df['front_id'] == front_id].sort_values('date')
	color = front_df['color'].iloc[0] # Get the defined color
	
	# Plot the line connecting the passages
	ax3.plot(front_df['date'], front_df['location_y'],
			 marker='o', linestyle='-', markersize=4,
			 color=color, linewidth=1.5, zorder=2,
			 label=f'Front {front_id}')
	
	# Plot the front ID number at the end
	if not front_df.empty:
		last_passage = front_df.iloc[-1]
		ax3.text(last_passage['date'] + pd.Timedelta(hours=6),
				 last_passage['location_y'],
				 str(front_id),
				 fontsize='xx-small', color='black', ha='left', va='center', zorder=3)

# --- 5. Customizing X-axis (Dates) ---
ax3.set_xlim(start_date, end_date)
# Format the dates
date_format = mdates.DateFormatter('%d.%m') # Day.Month format
ax3.xaxis.set_major_formatter(date_format)
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=4)) # Increased interval for cleaner look
ax3.xaxis.set_minor_locator(mdates.DayLocator()) # Minor ticks for fine grid
# Setting tick label sizes to small
ax3.tick_params(axis='x', size=5, labelsize='x-small') # Reduced size/labelsize
ax3.tick_params(axis='y', size=5, labelsize='x-small') # Reduced size/labelsize
# --- 8. Customizing LEFT Y-axis (Location Names) ---
ax3.set_yticks(list(location_to_y.values()))
ax3.set_yticklabels(right_y_labels, fontsize='xx-small', color='dimgrey') # Use coordinates
ax3.set_ylabel(None) # Remove right y-axis label
ax3.tick_params(axis='y', size=5)


# --- 9. Creating and Customizing RIGHT Y-axis (Coordinates) ---
#ax4 = ax3.twinx() # Create a twin Axes sharing the X-axis
#ax4.set_ylim(ax3.get_ylim()) # Ensure Y-limits match exactly
#ax4.set_yticks(ax3.get_yticks()) # Use the same Y-tick positions
#ax4.set_yticklabels(right_y_labels, fontsize='xx-small', color='dimgrey') # Use coordinates
#ax4.set_ylabel(None) # Remove right y-axis label
#ax4.tick_params(axis='y', size=5)
# --- 7. Add Gridlines ---
ax3.grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgrey')
ax3.grid(True, which='minor', linestyle=':', linewidth=0.3, color='lightgrey', alpha=0.7)


# --- 9. Labels and Title ---
# Setting axis label sizes to small
ax3.set_xlabel(f"{start_date.strftime('%d.%m')} - {end_date.strftime('%d.%m')} (2004)", fontsize='small')
ax3.set_ylabel(None)

# Add a legend for the fronts (optional, if you want different colors per front)
#ax3.legend(title="Cold Front ID", loc='upper left', fontsize='xx-small', title_fontsize='xx-small', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig("sbb_locs_v4.png", dpi=300)