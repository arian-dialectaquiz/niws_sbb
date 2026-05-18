import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------
# 1. DEFINE FLUX DATA 
# ---------------------------------------------------------
# 'net': Target Net Advection (matches your budget)
# 'v_down_in':  Flow ENTERING from North (Blue)
# 'v_down_out': Flow EXITING to South (Blue)
# 'v_up_in':    Flow ENTERING from South (Red)
# 'v_up_out':   Flow EXITING to North (Red)
# 'slope':      Export to East (Right Arrow) [Positive value]
# 'shelf':      Export to West (Left Arrow)  [Positive value]
# 'slope_in':   Import from East (Optional for Hurricane N)

flux_data = {
	'normal': {
		# Normal: Weak Southward flow driven by internal wind input
		'N': {'net': -0.42, 'v_down_in': 0.00, 'v_up_out': 0.00, 'v_down_out': 0.30, 'v_up_in': 0.05, 'slope': 0.10, 'shelf': 0.07}, 
		'C': {'net': -0.51, 'v_down_in': 0.30, 'v_up_out': 0.05, 'v_down_out': 0.60, 'v_up_in': 0.05, 'slope': 0.11, 'shelf': 0.10},
		'S': {'net': -0.60, 'v_down_in': 0.60, 'v_up_out': 0.05, 'v_down_out': 0.95, 'v_up_in': 0.00, 'slope': 0.15, 'shelf': 0.05},
		# Check S: In(0.6) - Out(0.05+0.95+0.15+0.05) = 0.6 - 1.2 = -0.60 (OK)
	},
	'cf': {
		# Cold Front: Stronger Southward flow, high lateral losses
		'N': {'net': -0.73, 'v_down_in': 0.00, 'v_up_out': 0.00, 'v_down_out': 0.50, 'v_up_in': 0.10, 'slope': 0.20, 'shelf': 0.13}, 
		'C': {'net': -0.89, 'v_down_in': 0.50, 'v_up_out': 0.10, 'v_down_out': 0.90, 'v_up_in': 0.15, 'slope': 0.30, 'shelf': 0.24},
		'S': {'net': -1.05, 'v_down_in': 0.90, 'v_up_out': 0.15, 'v_down_out': 1.40, 'v_up_in': 0.00, 'slope': 0.30, 'shelf': 0.10},
		# Check S: In(0.9) - Out(0.15+1.40+0.30+0.10) = 0.9 - 1.95 = -1.05 (OK)
	},
	'hurricane': {
		# Hurricane: Massive South->North Reversal.
		# N gains from C and Slope (Slope In logic added)
		'N': {'net': 1.20, 'v_down_in': 0.00, 'v_up_out': 0.30, 'v_down_out': 0.00, 'v_up_in': 1.40, 'slope': 0.00, 'shelf': 0.10, 'slope_in': 0.20}, 
		'C': {'net': -0.10, 'v_down_in': 0.00, 'v_up_out': 1.40, 'v_down_out': 0.10, 'v_up_in': 2.00, 'slope': 0.40, 'shelf': 0.20},
		'S': {'net': -2.50, 'v_down_in': 0.10, 'v_up_out': 2.00, 'v_down_out': 0.30, 'v_up_in': 0.00, 'slope': 0.20, 'shelf': 0.10},
		# Check S: In(0.1) - Out(2.0+0.3+0.2+0.1) = 0.1 - 2.6 = -2.50 (OK)
	}
}

# ---------------------------------------------------------
# 2. PLOTTING FUNCTIONS
# ---------------------------------------------------------
def draw_arrow(ax, x, y, dx, dy, val, color='k'):
	"""
	Draws arrow and places text BESIDE/ABOVE it.
	"""
	if val <= 0.02: return 
	
	# Scale width slightly by value
	width = 0.02 + (val * 0.012)
	
	# Draw Arrow
	ax.arrow(x, y, dx, dy, width=width, head_width=0.12, head_length=0.12, 
			 fc=color, ec=color, length_includes_head=True, zorder=5)
	
	# Text Placement Logic
	mid_x = x + dx/2
	mid_y = y + dy/2
	
	offset_x, offset_y = 0, 0
	ha, va = 'center', 'center'
	
	if dy < -0.1:   # Pointing Down (Blue) -> Text Left
		offset_x = -0.22
		ha = 'right'
	elif dy > 0.1:  # Pointing Up (Red) -> Text Right
		offset_x = 0.22
		ha = 'left'
	elif dx > 0:    # Pointing Right (Slope) -> Text Above
		offset_y = 0.12
		va = 'bottom'
	elif dx < 0:    # Pointing Left (Shelf) -> Text Above
		offset_y = 0.12
		va = 'bottom'

	ax.text(mid_x + offset_x, mid_y + offset_y, f"{val:.2f}", 
			ha=ha, va=va, fontsize=9, fontweight='normal', color='black')

def plot_scenario(ax, scen_data, title):
	"""Plots one scenario"""
	
	# -- Geometry --
	box_w = 1.2
	box_h = 1.0
	gap = 0.8  
	cx = 2.0   # Center X
	
	y_N = 4.0
	y_C = y_N - box_h - gap
	y_S = y_C - box_h - gap
	
	positions = {'N': y_N, 'C': y_C, 'S': y_S}
	
	ax.set_xlim(0, 4)
	ax.set_ylim(0, 5.5)
	ax.axis('off')
	ax.set_title(title, fontsize=10, fontweight='demibold', pad=15)
	
	# -- Draw Boxes --
	for region, y in positions.items():
		rect = patches.Rectangle((cx - box_w/2, y), box_w, box_h, 
								 linewidth=1.5, edgecolor='black', facecolor='none', linestyle='-')
		ax.add_patch(rect)
		
		# Labels
		net = scen_data[region]['net']
		sign = "+" if net > 0 else ""
		ax.text(cx, y + box_h*0.65, region, ha='center', va='center', fontsize=8, fontweight='bold', color='black')
		ax.text(cx, y + box_h*0.35, f"Net: {sign}{net:.2f}", ha='center', va='center', fontsize=8, color='black')

	# -- Draw Fluxes --
	
	# 1. NORTH BOX
	d = scen_data['N']
	# Outer Boundary (Top) - Only Outflow allowed per request
	draw_arrow(ax, cx+0.3, 5.2, 0, 0.4, d['v_up_out'], 'red')  
	
	# Exchange N <-> C
	draw_arrow(ax, cx-0.3, y_N, 0, -gap, d['v_down_out'], 'blue') 
	draw_arrow(ax, cx+0.3, y_N-gap, 0, gap, d['v_up_in'], 'red')
	
	# Lateral
	# Slope (Right)
	draw_arrow(ax, cx+box_w/2, y_N+box_h/2, 0.5, 0, d['slope'], 'k')
	# Shelf (Left)
	draw_arrow(ax, cx-box_w/2, y_N+box_h/2, -0.5, 0, d['shelf'], 'k')
	
	# Special Case: Slope In for Hurricane N
	if 'slope_in' in d and d['slope_in'] > 0:
		draw_arrow(ax, cx+box_w/2+0.6, y_N+box_h/2, -0.6, 0, d['slope_in'], 'k') # Arrow IN from right

	# 2. COASTAL BOX
	d = scen_data['C']
	# Exchange C <-> S
	draw_arrow(ax, cx-0.3, y_C, 0, -gap, d['v_down_out'], 'blue')
	draw_arrow(ax, cx+0.3, y_C-gap, 0, gap, d['v_up_in'], 'red')
	# Lateral
	draw_arrow(ax, cx+box_w/2, y_C+box_h/2, 0.5, 0, d['slope'], 'k')
	draw_arrow(ax, cx-box_w/2, y_C+box_h/2, -0.5, 0, d['shelf'], 'k')

	# 3. SOUTH BOX
	d = scen_data['S']
	# Outer Boundary (Bottom) - Only Outflow allowed per request
	draw_arrow(ax, cx-0.3, y_S, 0, -0.4, d['v_down_out'], 'blue')
	
	# Lateral
	draw_arrow(ax, cx+box_w/2, y_S+box_h/2, 0.5, 0, d['slope'], 'k')
	draw_arrow(ax, cx-box_w/2, y_S+box_h/2, -0.5, 0, d['shelf'], 'k')
	
	# Add labels for Shelf/Slope text
	if title == 'Normal Winds': # Just once
		ax.text(3.5, y_N+box_h/2, "Slope", fontsize=8, color='gray', ha='left', va='center')
		ax.text(0.5, y_N+box_h/2, "Shelf", fontsize=8, color='gray', ha='right', va='center')

# ---------------------------------------------------------
# 3. GENERATE PLOT
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(8, 5))

scen_keys = ['normal', 'cf', 'hurricane']
titles = ['Normal Winds', 'Cold Front', 'Hurricane']

for i, key in enumerate(scen_keys):
	plot_scenario(axes[i], flux_data[key], titles[i])

plt.tight_layout()
plt.savefig('advection_flux_final.png', dpi=300)
