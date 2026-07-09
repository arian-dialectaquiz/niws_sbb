
	
fig = plt.figure(figsize=(7, 7)) #width, height
gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1,1], width_ratios=[1,1])
gs.update(left=0.08, right=0.98,hspace=0.15, wspace=0.15, top=0.98, bottom=0.08)

############------> Loc, z and deproas moorings

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


A = [380, 1233]
B = [276, 896]
C = [333, 557]
D = [453, 323]


cmap = LinearSegmentedColormap.from_list('bathymetry_cmap', colors, N=500)
vmin = -5
vmax = 2500
#norm = LogNorm(vmin=vmin, vmax=vmax)
norm = cm.colors.Normalize(vmin=vmin,vmax=vmax)
levels=np.linspace(vmin,vmax,200)

bar_title = "m"
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.set_aspect('auto')
ax.set_ylim(bottom =-33,top = -21)
ax.set_xlim(left = -52,right = -39.2)

### - contourf
norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
ax.contourf(lon_rho, lat_rho, h,  levels=levels, cmap=cmap, norm=norm, extend='max', zorder=0)

ax.text(0.7, 0.98, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
		

#--> virtual moorings
ax.scatter(lon_rho[A[1],A[0]], lat_rho[A[1],A[0]], zorder=5, s=40, marker='*', color='k', label='A')
ax.scatter(lon_rho[B[1],B[0]], lat_rho[B[1],B[0]], zorder=5, s=40, marker='v', color='blue', label='B')
ax.scatter(lon_rho[C[1],C[0]], lat_rho[C[1],C[0]], zorder=5, s=40, marker='s', color='orange', label='C')
ax.scatter(lon_rho[D[1],D[0]], lat_rho[D[1],D[0]], zorder=5, s=40, marker='^', color='red', label='D')

legend = ax.legend(loc=4, fontsize='x-small', markerscale=0.7)

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
gl.xlabel_style = {'size': 8, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 8, 'color': 'dimgrey'}

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
#fig.legend(legend_lines, labels, loc='upper right', title='Isobaths', fontsize='x-small', title_fontsize='x-small')

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='x-small',
	title_fontsize='x-small',
	loc='center',
	bbox_to_anchor=(0.13, 0.85)
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
cb1.set_label(bar_title, size='x-small')
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small')  # Set colorbar label size

CF = [-23.018, -41.97]
CS = [-22.015, -40.963]
CM = [-28.60, -48.809]
SSI = [-23.8310, -45.4089]

# CS
#ax.scatter(CS[1], CS[0],marker='D',zorder = 3, color='k', s=20)
#ax.text(CS[1] - 0.3, CS[0], 'CST', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# CF
ax.scatter(CF[1], CF[0],marker='D',zorder = 3, color='k', s=20)
ax.text(CF[1] - 0.3, CF[0] + 0.1, 'CF', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# CM
ax.scatter(CM[1], CM[0],marker='D',zorder = 3, color='k', s=20)
ax.text(CM[1] -0.25, CM[0] + 0.3, 'CSM', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# SSI
ax.scatter(SSI[1], SSI[0],marker='.',zorder = 3, color='k', s=20)
ax.text(SSI[1] -0.25, SSI[0] + 0.3, 'SSI', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')


#---> grid_subgrid
ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.set_aspect('auto')

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
gl.xlabel_style = {'size': 8, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 8, 'color': 'dimgrey'}
# Custom legend
legend_patches = [
	mpatches.Patch(color='blue', label='Coarser'),
	mpatches.Patch(color='indianred', label='Finer'),

]
ax2.text(0.7, 0.98, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')


###########---> Plotting the hurricane track
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

####################_---> Vmax plot

pcx = plt.subplot(gs[1, 0])

pcx.text(0.01, 0.95, '(c)', transform=pcx.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
#pcx.set_ylim(bottom=-30, top=30)

pcx.plot(time_cat,vmax_cat, label='Track', linestyle=':', color='green')
pcx.plot(time_cat, vmax_blend, label='Blend', linestyle='--', color='orange')

pcx.legend(loc=4,fontsize=9)
pcx.set_ylim(10,45)
format_time_axis(pcx)
pcx.xaxis.set_major_locator(mdates.DayLocator(interval=2))
pcx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
pcx.set_ylabel(r'V(max) m s$^{-1}$',fontsize = 10)



####################################################
#####-----> Stick plot<---##########
point_labels = ['A', 'B', 'C', 'D']
offsets = [0.8, 0.6, 0.4, 0.2]  # Tighten offsets for stress magnitudes
ax4 = plt.subplot(gs[1, 1])
ax4.text(0.1, 0.98, '(d)', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

# Use the time from the xarray object
time = rotated_stress['A']['time']

for i, name in enumerate(['A', 'B', 'C', 'D']):
	data = rotated_stress[name]
	offset = offsets[i]
	
	# Quiver for wind stress
	q = ax4.quiver(data['time'], np.full(len(data['time']), offset),
				   data['tau_x'], data['tau_y'],
				   color='black', 
				   alpha=0.8,
				   units='y',     
				   scale=3,      
				   headlength=0,   
				   headaxislength=0, 
				   width=0.008)

# 3. Update Quiver Legend for Stress
ax4.quiverkey(q, X=0.8, Y=0.96, U=0.1 ,
			  label=r'0.1 $Pa$', labelpos='E', 
			  coordinates='axes', fontproperties={'weight': 'bold'})

# Formatting
ax4.set_yticks(offsets)
ax4.set_yticklabels(point_labels)
ax4.set_ylim(0, 1.0)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=15)
ax4.set_ylabel(None)

h_start = np.datetime64("2004-03-25")
h_end = np.datetime64("2004-03-30") 
ax4.axvspan(h_start, h_end, color='red', alpha=0.15, zorder=0)
ax4.text(h_start, 1, 'Hurricane',color='red', fontsize=8, fontweight='bold', ha='left',va='bottom') 
plt.tight_layout()


# Formatting
ax4.set_yticks(offsets)
ax4.set_yticklabels(point_labels)
ax4.set_ylim(0, 1.0)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=15)
ax4.set_ylabel(None)

h_start = np.datetime64("2004-03-25")
h_end = np.datetime64("2004-03-30") 
ax4.axvspan(h_start, h_end, color='red', alpha=0.15, zorder=0)
ax4.text(h_start, 1, 'Hurricane',color='red', fontsize=8, fontweight='bold', ha='left',va='bottom') 


plt.tight_layout()
plt.savefig('fig1_loc.png', dpi = 300)