


##################-----> Plotting <------###################################
colors_c = ['Orange', 'cyan', 'Lime', 'grey']  
labels_c = ['A', 'B', 'C','D']


name = f"fig_5_vort_ke_crosses_v2.png"  # Create a unique filename

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(nrows=3, ncols=3, width_ratios=[1,1,1], height_ratios=[1,1,1])
gs.update(left=0.05, right=0.97, wspace=0.20, hspace=0.1, top=0.96, bottom=0.07)
######----> Normal
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.text(0.03, 0.08, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax.set_title('Normal', loc='left', fontsize=10)
# Define colormap and normalization
cmap = cmo.cm.balance

#cmap = plt.cm.gist_ncar_r
vmin = -0.15
vmax = 0.15
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
#bar_title = r'W.m$^{-2}$'
bar_title = ''

ax.set_ylim(bottom=-32.6, top=-20)
ax.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax.contourf(lon_rho, lat_rho, vort_normal, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)


# Depth contours
levels = [50, 200, 1000, 2000]
styles = ['dotted', 'dotted', 'dashed', 'solid']
colors = ['gray', 'silver', 'dimgrey', 'gray']
for l, s, c in zip(levels, styles, colors):
	ax.contour(lon_rho, lat_rho, h, levels=[l], zorder=3, colors=c, linestyles=s, linewidths=1)

# --- NEW LEGEND LOCATION (Axis Coordinates) ---
legend_lines = [Line2D([0], [0], linestyle=s, linewidth=1, color=c) for s, c in zip(styles, colors)]
labels_iso = ['50 m', '200 m', '1000 m', '2000 m']

# This places the legend inside the axes at the lower right
ax.legend(
	legend_lines,
	labels_iso,
	title='Isobaths',
	fontsize='xx-small',
	title_fontsize='xx-small',
	loc='lower right',           # Automatically anchors to bottom-right of AXIS
	framealpha=0.9,              # Opaque box so lines don't clash with map
	edgecolor='gray'
)
# ----------------------------------------------
# Coastlines and gridlines
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax.patch.set_edgecolor('black')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


# Colorbar
cbar_1 = inset_axes(ax, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)

######----> Cold Front
ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.text(0.03, 0.08, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax2.set_title('Cold Front', loc='left', fontsize=10)
# Define colormap and normalization
cmap = cmo.cm.balance

#cmap = plt.cm.gist_ncar_r
vmin = -0.15
vmax = 0.15
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
#bar_title = r'W.m$^{-2}$'
bar_title = ''

ax2.set_ylim(bottom=-32.6, top=-20)
ax2.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax2.contourf(lon_rho, lat_rho, vort_cf, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)
colors_c = ['Orange', 'cyan', 'Lime', 'grey']  
labels_c = ['A', 'B', 'C', 'D']
#ax2.scatter(lon_rho[A[1],A[0]], lat_rho[A[1],A[0]], zorder=5, s=20, marker='*', color=colors_c[0], label='A')
#ax2.scatter(lon_rho[B[1],B[0]], lat_rho[B[1],B[0]], zorder=5, s=20, marker='+', color=colors_c[1], label='B')
#ax2.scatter(lon_rho[C[1],C[0]], lat_rho[C[1],C[0]], zorder=5, s=20, marker='s', color=colors_c[2], label='C')
#ax2.scatter(lon_rho[D[1],D[0]], lat_rho[D[1],D[0]], zorder=5, s=20, marker='D', color=colors_c[3], label='D')
#
#legend = ax2.legend(loc=4, fontsize='xx-small', markerscale=0.7)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax2.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax2.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax2.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax2.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

# Coastlines and gridlines
ax2.coastlines()
ax2.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax2.patch.set_edgecolor('black')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


# Colorbar
cbar_1 = inset_axes(ax2, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)



######----> Cold Front
ax3 = plt.subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax3.text(0.03, 0.08, '(c)', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
ax3.set_title('Hurricane', loc='left', fontsize=10)
# Define colormap and normalization
cmap = cmo.cm.balance

#cmap = plt.cm.gist_ncar_r
vmin = -0.15
vmax = 0.15
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
#bar_title = r'W.m$^{-2}$'
bar_title = ''

ax3.set_ylim(bottom=-32.6, top=-20)
ax3.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax3.contourf(lon_rho, lat_rho, vort_hurricane, levels=200, cmap=cmap, norm=norm, extend='both', zorder=0)


ax3.scatter(lon_rho[A[1],:], lat_rho[A[1],:],c=colors_c[0],s=0.05,marker='.')
ax3.scatter(lon_rho[B[1],:], lat_rho[B[1],:],c=colors_c[1],s=0.05,marker='.')
ax3.scatter(lon_rho[C[1],:], lat_rho[C[1],:],c=colors_c[2],s=0.05,marker='.')
ax3.scatter(lon_rho[D[1],:], lat_rho[D[1],:],c=colors_c[3],s=0.05,marker='.')

ax3.text(lon_rho[A[1], 609]+0.2, lat_rho[A[1], 609]-1, 'A', color=colors_c[0],zorder=5, fontsize='x-small', verticalalignment='bottom')
ax3.text(lon_rho[[B[1]], 609]+0.2, lat_rho[[B[1]], 609]-1, 'B', color=colors_c[1], zorder=5,fontsize='x-small', verticalalignment='bottom')
ax3.text(lon_rho[C[1], 609]+0.2, lat_rho[C[1], 609]-1, 'C', color=colors_c[2], zorder=5,fontsize='x-small', verticalalignment='bottom')
ax3.text(lon_rho[D[1], 609]+0.2, lat_rho[D[1], 609]-1, 'D', color=colors_c[3], zorder=5,fontsize='x-small', verticalalignment='bottom')

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax3.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax3.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax3.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax3.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

# Coastlines and gridlines
ax3.coastlines()
ax3.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax3.patch.set_edgecolor('black')
gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}


# Colorbar
cbar_1 = inset_axes(ax3, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)



######----> Normal
ax4 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax4.text(0.03, 0.08, '(d)', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization
#cmap = plt.cm.ocean_r
cmap = plt.cm.gist_ncar_r
vmin = 0
vmax = 3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'kJ.m$^{-2}$'

ax4.set_ylim(bottom=-32.6, top=-20)
ax4.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax4.contourf(lon_rho, lat_rho, nike_normal/1000, levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax4.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax4.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax4.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax4.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

# Coastlines and gridlines
ax4.coastlines()
ax4.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax4.patch.set_edgecolor('black')
gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}
# Colorbar
cbar_1 = inset_axes(ax4, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)


######----> cf
ax5 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax5.text(0.03, 0.08, '(e)', transform=ax5.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization
#cmap = plt.cm.ocean_r
cmap = plt.cm.gist_ncar_r
vmin = 0
vmax = 3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'kJ.m$^{-2}$'

ax5.set_ylim(bottom=-32.6, top=-20)
ax5.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax5.contourf(lon_rho, lat_rho, nike_cf/1000, levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax5.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax5.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax5.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax5.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']


# Coastlines and gridlines
ax5.coastlines()
ax5.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax5.patch.set_edgecolor('black')
gl = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}
# Colorbar
cbar_1 = inset_axes(ax5, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)

######----> hurricane
ax6 = plt.subplot(gs[1, 2], projection=ccrs.PlateCarree())
ax6.text(0.03, 0.08, '(f)', transform=ax6.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
# Define colormap and normalization
#cmap = plt.cm.ocean_r
cmap = plt.cm.gist_ncar_r
vmin = 0
vmax = 3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# Define other parameters
bar_title = r'kJ.m$^{-2}$'

ax6.set_ylim(bottom=-32.6, top=-20)
ax6.set_xlim(left=-52, right=-39.2)
# Contourf with normalization
ax6.contourf(lon_rho, lat_rho, nike_hurricane/1000, levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)

# Depth contour levels
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]
# Contour lines for isobaths
ax6.contour(lon_rho, lat_rho, h, levels=levels_1, zorder=3, colors='gray', linestyles='dotted', linewidths=1)
ax6.contour(lon_rho, lat_rho, h, levels=levels_2, zorder=3, colors='silver', linestyles='dotted', linewidths=1)
ax6.contour(lon_rho, lat_rho, h, levels=levels_3, zorder=3, colors='dimgrey', linestyles='dashed', linewidths=1)
ax6.contour(lon_rho, lat_rho, h, levels=levels_4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
# Legend for isobaths
legend_lines = [
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='gray'),
	Line2D([0], [0], linestyle='dotted', linewidth=1, color='silver'),
	Line2D([0], [0], linestyle='dashed', linewidth=1, color='dimgrey'),
	Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')
]
labels = ['50 m', '200 m', '1000 m', '2000 m']

# Coastlines and gridlines
ax6.coastlines()
ax6.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax6.patch.set_edgecolor('black')
gl = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 6, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 6, 'color': 'dimgrey'}
# Colorbar
cbar_1 = inset_axes(ax6, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(fr'{bar_title} ', size='x-small', labelpad=2)
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small', rotation=25)

# ----------- a)Normal
axg = plt.subplot(gs[2, 0])
axg.text(0.02, 0.9, '(g)', transform=axg.transAxes, fontsize=10, fontweight='bold')
colors_c = ['Orange', 'cyan', 'Lime', 'grey']  
labels_c = ['A', 'B', 'C', 'D']

# Define distinct line styles for each dataset
# 'solid', 'dotted', 'dashed', 'dashdot' are the standard matplotlib styles
styles_c = ['solid', 'dotted', 'dashed', 'dashdot']

data_list = [nike_normal_a, nike_normal_b, nike_normal_c, nike_normal_d]
data_H = [-1*Ha, -1*Hb, -1*Hc, -1*Hd] 

# --- PLOTTING BARS (Vertical Lines) ---
for i in range(4):
	axg.vlines(x=data_H[i], 
			   ymin=0, 
			   ymax=np.array(data_list[i]) / 1000, 
			   colors=colors_c[i], 
			   linestyles=styles_c[i],  # <--- Apply the style here
			   linewidth=1.2,           # Slightly thicker to make 'dotted' visible
			   alpha=0.8,               # Higher opacity to see the texture
			   label=labels_c[i],
			   zorder=3)

# --- AXIS FORMATTING ---
axg.set_ylabel(r'kJ m$^{-2}$', labelpad=5, fontsize=7)
axg.set_xlabel('Depth (m)', labelpad=5, fontsize=7)

axg.grid(True, alpha=0.3, linestyle=':', zorder=0)
axg.set_xlim(0, 2700)
axg.set_ylim(0, 3.3)

axg.tick_params(axis='both', labelsize=6)

# --- CUSTOM LEGEND WITH STYLES ---
from matplotlib.lines import Line2D

# We create custom lines that match both Color AND Linestyle
custom_lines = [
	Line2D([0], [0], color=colors_c[0], linestyle=styles_c[0], lw=1.5),
	Line2D([0], [0], color=colors_c[1], linestyle=styles_c[1], lw=1.5),
	Line2D([0], [0], color=colors_c[2], linestyle=styles_c[2], lw=1.5),
	Line2D([0], [0], color=colors_c[3], linestyle=styles_c[3], lw=1.5)
]

axg.legend(custom_lines, labels_c, loc='upper right', fontsize=6, framealpha=0.9, edgecolor='gray')

import matplotlib.ticker as ticker
axg.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))


# ----------- a)CF
axh = plt.subplot(gs[2, 1])
axh.text(0.02, 0.9, '(h)', transform=axh.transAxes, fontsize=10, fontweight='bold')
colors_c = ['Orange', 'cyan', 'Lime', 'grey']  
labels_c = ['A', 'B', 'C', 'D']

# Define distinct line styles for each dataset
# 'solid', 'dotted', 'dashed', 'dashdot' are the standard matplotlib styles
styles_c = ['solid', 'dotted', 'dashed', 'dashdot']

data_list = [nike_cf_a, nike_cf_b, nike_cf_c, nike_cf_d]
data_H = [-1*Ha, -1*Hb, -1*Hc, -1*Hd] 

# --- PLOTTING BARS (Vertical Lines) ---
for i in range(4):
	axh.vlines(x=data_H[i], 
			   ymin=0, 
			   ymax=np.array(data_list[i]) / 1000, 
			   colors=colors_c[i], 
			   linestyles=styles_c[i],  # <--- Apply the style here
			   linewidth=1.2,           # Slightly thicker to make 'dotted' visible
			   alpha=0.8,               # Higher opacity to see the texture
			   label=labels_c[i],
			   zorder=3)

# --- AXIS FORMATTING ---
axh.set_ylabel(r'kJ m$^{-2}$', labelpad=5, fontsize=7)
axh.set_xlabel('Depth (m)', labelpad=5, fontsize=7)

axh.grid(True, alpha=0.3, linestyle=':', zorder=0)
axh.set_xlim(0, 2700)
axh.set_ylim(0, 3.3)

axh.tick_params(axis='both', labelsize=6)

# --- CUSTOM LEGEND WITH STYLES ---
from matplotlib.lines import Line2D

# We create custom lines that match both Color AND Linestyle
custom_lines = [
	Line2D([0], [0], color=colors_c[0], linestyle=styles_c[0], lw=1.5),
	Line2D([0], [0], color=colors_c[1], linestyle=styles_c[1], lw=1.5),
	Line2D([0], [0], color=colors_c[2], linestyle=styles_c[2], lw=1.5),
	Line2D([0], [0], color=colors_c[3], linestyle=styles_c[3], lw=1.5)
]

#axh.legend(custom_lines, labels_c, loc='upper right', fontsize=6, framealpha=0.9, edgecolor='gray')

import matplotlib.ticker as ticker
axh.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))


# ----------- a)hurricane
axi = plt.subplot(gs[2, 2])
axi.text(0.02, 0.9, '(i)', transform=axi.transAxes, fontsize=10, fontweight='bold')
colors_c = ['Orange', 'cyan', 'Lime', 'grey']  
labels_c = ['A', 'B', 'C', 'D']

# Define distinct line styles for each dataset
# 'solid', 'dotted', 'dashed', 'dashdot' are the standard matplotlib styles
styles_c = ['solid', 'dotted', 'dashed', 'dashdot']

data_list = [nike_hurricane_a, nike_hurricane_b, nike_hurricane_c, nike_hurricane_d]
data_H = [-1*Ha, -1*Hb, -1*Hc, -1*Hd] 

# --- PLOTTING BARS (Vertical Lines) ---
for i in range(4):
    axi.vlines(x=data_H[i], 
               ymin=0, 
               ymax=np.array(data_list[i]) / 1000, 
               colors=colors_c[i], 
               linestyles=styles_c[i],  # <--- Apply the style here
               linewidth=1.2,           # Slightly thicker to make 'dotted' visible
               alpha=0.8,               # Higher opacity to see the texture
               label=labels_c[i],
               zorder=3)

# --- AXIS FORMATTING ---
axi.set_ylabel(r'kJ m$^{-2}$', labelpad=5, fontsize=7)
axi.set_xlabel('Depth (m)', labelpad=5, fontsize=7)

axi.grid(True, alpha=0.3, linestyle=':', zorder=0)
axi.set_xlim(0, 2700)
axi.set_ylim(0, 3.3)

axi.tick_params(axis='both', labelsize=6)

# --- CUSTOM LEGEND WITH STYLES ---
from matplotlib.lines import Line2D

# We create custom lines that match both Color AND Linestyle
custom_lines = [
    Line2D([0], [0], color=colors_c[0], linestyle=styles_c[0], lw=1.5),
    Line2D([0], [0], color=colors_c[1], linestyle=styles_c[1], lw=1.5),
    Line2D([0], [0], color=colors_c[2], linestyle=styles_c[2], lw=1.5),
    Line2D([0], [0], color=colors_c[3], linestyle=styles_c[3], lw=1.5)
]

#axi.legend(custom_lines, labels_c, loc='upper right', fontsize=6, framealpha=0.9, edgecolor='gray')

import matplotlib.ticker as ticker
axi.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

plt.savefig(name, dpi=300)



