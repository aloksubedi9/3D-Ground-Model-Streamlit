import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
from matplotlib import cm

st.title("Interactive 3D Ground Surface with Borehole Logs")

# ===== Update these paths if your CSV files have different names/locations =====
surface_csv_path = r'CB1.csv'                  # Surface points
borehole_csv_path = r'boreholes.csv'           # Borehole locations
bh_details_csv_path = r'BHdetails2.csv'         # Borehole stratigraphy

# ===== 1. Read CSVs =====
df_surface = pd.read_csv(surface_csv_path)
df_boreholes = pd.read_csv(borehole_csv_path)
df_bh_details = pd.read_csv(bh_details_csv_path)

# Column checks
for df, cols, name in [
    (df_surface, ['Easting', 'Northing', 'Elevation'], "Surface CSV"),
    (df_boreholes, ['BH ID', 'Easting', 'Northing'], "Boreholes CSV"),
    (df_bh_details, ['BH', 'FROM', 'TO', 'SOIL TYPE'], "BH Details CSV")
]:
    if not all(col in df.columns for col in cols):
        st.error(f"{name} must contain columns: {cols}")
        st.stop()

# Extract data
easting = df_surface['Easting'].values
northing = df_surface['Northing'].values
elevation = df_surface['Elevation'].values

bh_ids = df_boreholes['BH ID'].values
bh_easting = df_boreholes['Easting'].values
bh_northing = df_boreholes['Northing'].values

# ===== Get unique soil types actually present =====
present_soil_types = df_bh_details['SOIL TYPE'].dropna().unique().tolist()

# ===== Predefined colors =====
predefined_soil_color_map = {
    'SM-ML': 'rgb(255,255,0)', 'SC': 'rgb(210,180,140)', 'CI': 'rgb(0,128,0)',
    'SM/SM-ML': 'rgb(255,215,0)', 'ROCK': 'rgb(139,69,19)', 'SM': 'rgb(255,255,224)',
    'GP-GM': 'rgb(128,128,128)', 'CH': 'rgb(0,100,0)', 'ML-SM': 'rgb(255,192,203)',
    'CL': 'rgb(135,206,250)', 'GM': 'rgb(169,169,169)', 'GC': 'rgb(112,128,144)',
    'ML': 'rgb(255,228,181)', 'SP': 'rgb(255,222,173)', 'CHAR': 'rgb(47,79,79)'
}

# Build color map only for present types
soil_color_map = {}
for soil_type in present_soil_types:
    soil_color_map[soil_type] = predefined_soil_color_map.get(soil_type, 'rgb(0,0,0)')
    if soil_type not in predefined_soil_color_map:
        st.warning(f"Soil type '{soil_type}' not predefined – using black.")

# ===== Grid & exaggeration settings =====
h_exaggeration = 2
ve_factor = 1.0
grid_resolution = 100

# Scaled coordinates
easting_scaled = easting * h_exaggeration
northing_scaled = northing * h_exaggeration
bh_easting_scaled = bh_easting * h_exaggeration
bh_northing_scaled = bh_northing * h_exaggeration

# Grid
x_min, x_max = easting_scaled.min(), easting_scaled.max()
y_min, y_max = northing_scaled.min(), northing_scaled.max()
buffer_x = (x_max - x_min) * 0.05
buffer_y = (y_max - y_min) * 0.05
grid_x, grid_y = np.mgrid[x_min-buffer_x:x_max+buffer_x:grid_resolution*1j,
                          y_min-buffer_y:y_max+buffer_y:grid_resolution*1j]

grid_z = griddata((easting_scaled, northing_scaled), elevation, (grid_x, grid_y),
                  method='linear', fill_value=np.nanmean(elevation))

mean_z = np.nanmean(grid_z)
grid_z = (grid_z - mean_z) * ve_factor + mean_z

# Borehole surface elevations
bh_z = griddata((easting_scaled, northing_scaled), elevation,
                (bh_easting_scaled, bh_northing_scaled), method='linear')
bh_z = (bh_z - mean_z) * ve_factor + mean_z

bh_coord_map = dict(zip(df_boreholes['BH ID'],
                        zip(bh_easting_scaled, bh_northing_scaled, bh_z)))

# Convex hull mask
points = np.vstack((easting_scaled, northing_scaled)).T
hull = ConvexHull(points)

def point_in_hull(p, h, tol=1e-12):
    return all(np.dot(eq[:-1], p) + eq[-1] <= tol for eq in h.equations)

mask = np.array([[point_in_hull([grid_x[i,j], grid_y[i,j]], hull)
                  for j in range(grid_x.shape[1])] for i in range(grid_x.shape[0])])
grid_z_valid = np.ma.masked_where(~mask, grid_z)

# Colorscale
cmap = cm.turbo
steps = 20
colorscale = [[i/(steps-1), f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})']
              for i, c in enumerate(cmap(np.linspace(0, 1, steps)))]

# Z limits
max_depth = df_bh_details['TO'].max() * ve_factor
z_min = min(np.nanmin(grid_z), np.nanmin(bh_z) - max_depth * 1.1)
z_max = np.nanmax(grid_z) * 1.1

# Figure
fig = go.Figure()

fig.add_trace(go.Surface(x=grid_x, y=grid_y, z=grid_z_valid,
                         colorscale=colorscale, showscale=True,
                         colorbar=dict(title='Elevation (m)', len=0.5, x=1.1),
                         name='Ground Surface'))

fig.add_trace(go.Scatter3d(x=bh_easting_scaled, y=bh_northing_scaled, z=bh_z + 0.25,
                           mode='markers', marker=dict(size=12, color='red'),
                           name='Boreholes'))

# BH labels
for i, bh_id in enumerate(bh_ids):
    fig.add_trace(go.Scatter3d(x=[bh_easting_scaled[i]], y=[bh_northing_scaled[i]], z=[bh_z[i]],
                               mode='text', text=[bh_id],
                               textfont=dict(size=26, color='black'),
                               showlegend=False))

# Legend entries (only present soils)
for soil_type, color in soil_color_map.items():
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None],
                               mode='lines', line=dict(color=color, width=20),
                               name=soil_type, showlegend=True))

# Borehole logs
for bh_id, group in df_bh_details.groupby('BH'):
    if bh_id not in bh_coord_map:
        continue
    x, y, surface_z = bh_coord_map[bh_id]
    for _, row in group.iterrows():
        z_from = surface_z - row['FROM'] * ve_factor
        z_to = surface_z - row['TO'] * ve_factor
        fig.add_trace(go.Scatter3d(x=[x,x], y=[y,y], z=[z_from, z_to],
                                   mode='lines',
                                   line=dict(color=soil_color_map.get(row['SOIL TYPE'], 'rgb(0,0,0)'), width=20),
                                   showlegend=False))

fig.update_layout(title=f"3D Ground Model (H×{h_exaggeration}, V×{ve_factor})",
                  scene=dict(xaxis_title='Easting', yaxis_title='Northing',
                             zaxis_title='Elevation', zaxis=dict(range=[z_min, z_max]),
                             aspectratio=dict(x=h_exaggeration, y=h_exaggeration, z=1)),
                  legend=dict(x=0, y=0.5, title='Soil Types & Boreholes'))

st.plotly_chart(fig, use_container_width=True)
fig.write_html("borelog_model.html", include_plotlyjs="cdn")
st.success("Model generated & saved as borelog_model.html")
