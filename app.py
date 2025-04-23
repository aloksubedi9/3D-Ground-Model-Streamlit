import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import streamlit as st
import plotly.graph_objects as go

# File uploader for input.xlsx
uploaded_file = st.file_uploader("Upload your input.xlsx file", type=["xlsx"])
if uploaded_file is not None:
    # Load data from the uploaded file
    try:
        df = pd.read_excel(uploaded_file, sheet_name='Sheet 1')
        df = df[df['Include in 3D'].str.strip().str.lower() == 'yes'].reset_index(drop=True)
        st.success("Loaded input.xlsx from uploaded file.")
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        st.stop()
else:
    # Try to load from the local directory as a fallback
    try:
        df = pd.read_excel('input.xlsx', sheet_name='Sheet 1')
        df = df[df['Include in 3D'].str.strip().str.lower() == 'yes'].reset_index(drop=True)
        st.success("Loaded input.xlsx from local directory.")
    except FileNotFoundError:
        st.error("No 'input.xlsx' file found in the directory. Please upload the file using the uploader above.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading the local input.xlsx file: {e}")
        st.stop()

# Scale factor for vertical exaggeration
z_scale = 5

# Get unique layer types to assign colors
all_layers = pd.concat([
    df[['Layer1 Type']].rename(columns={'Layer1 Type': 'Type'}),
    df[['Layer2 Type']].rename(columns={'Layer2 Type': 'Type'}),
    df[['Layer3 Type']].rename(columns={'Layer3 Type': 'Type'})
])
unique_types = all_layers['Type'].dropna().unique()

# Assign color map (Plotly accepts hex colors)
color_map = {
    layer: color for layer, color in zip(
        unique_types,
        ['#8B0000', '#00008B', '#FFA500', '#800080', '#708090', '#006400', '#00BFFF']  # extend if needed
    )
}

# Prepare borehole data
coords = df[['Easting', 'Northing']].values
ground_levels = df['Ground Level'].values

# Apply vertical exaggeration for plotting
ground_levels_exag = ground_levels * z_scale
x_vals, y_vals = coords[:, 0], coords[:, 1]

# Initialize layers dict for surface building by type
layers_data_by_type = {'Ground Level': []}
for layer_type in unique_types:
    layers_data_by_type[layer_type] = []

# Fill layers_data_by_type dict with exaggerated Z values based on layer type
for i in range(len(df)):
    x, y = x_vals[i], y_vals[i]
    gl = df.at[i, 'Ground Level'] * z_scale
    l1 = df.at[i, 'Layer1 Depth'] * z_scale if pd.notna(df.at[i, 'Layer1 Depth']) else np.nan
    l2 = df.at[i, 'Layer2 Depth'] * z_scale if pd.notna(df.at[i, 'Layer2 Depth']) else np.nan
    l3 = df.at[i, 'Layer3 Depth'] * z_scale if pd.notna(df.at[i, 'Layer3 Depth']) else np.nan

    # Add ground level
    layers_data_by_type['Ground Level'].append((x, y, gl))

    # Add depths based on layer type
    if pd.notna(df.at[i, 'Layer1 Depth']) and pd.notna(df.at[i, 'Layer1 Type']):
        layer_type = df.at[i, 'Layer1 Type']
        layers_data_by_type[layer_type].append((x, y, l1))

    if pd.notna(df.at[i, 'Layer2 Depth']) and pd.notna(df.at[i, 'Layer2 Type']):
        layer_type = df.at[i, 'Layer2 Type']
        layers_data_by_type[layer_type].append((x, y, l2))

    if pd.notna(df.at[i, 'Layer3 Depth']) and pd.notna(df.at[i, 'Layer3 Type']):
        layer_type = df.at[i, 'Layer3 Type']
        layers_data_by_type[layer_type].append((x, y, l3))

# Create interpolation grid for surfaces
grid_x, grid_y = np.meshgrid(
    np.linspace(x_vals.min(), x_vals.max(), 60),
    np.linspace(y_vals.min(), y_vals.max(), 60)
)

def interp_layer(layer_pts):
    pts = np.array(layer_pts)
    return griddata(pts[:, :2], pts[:, 2], (grid_x, grid_y), method='cubic')

# Interpolated surfaces based on layer type
surfaces = {}
for layer_type, pts in layers_data_by_type.items():
    if len(pts) > 3:  # Need at least 4 points for interpolation
        z_grid = interp_layer(pts)
        z_grid = np.nan_to_num(z_grid, nan=np.nanmean(z_grid))
        surfaces[layer_type] = z_grid

# Function to plot 3D visualization using Plotly
def plot_3d_visualization(view_mode):
    try:
        # Initialize the Plotly figure
        fig = go.Figure()

        # Track plotted layer types to avoid duplicate legend entries
        plotted_layer_types = set()

        # Plot surfaces (if applicable)
        if view_mode in [2, 3]:  # Modes 2 and 3 include surfaces
            for layer_type, z_grid in surfaces.items():
                if layer_type == 'Ground Level':
                    fig.add_trace(go.Surface(
                        x=grid_x, y=grid_y, z=z_grid,
                        colorscale=[[0, '#228B22'], [1, '#228B22']],
                        name='Ground Surface',
                        showscale=False
                    ))
                else:
                    color = color_map.get(layer_type, 'grey')
                    fig.add_trace(go.Surface(
                        x=grid_x, y=grid_y, z=z_grid,
                        colorscale=[[0, color], [1, color]],
                        name=f"{layer_type} Surface",
                        showscale=False
                    ))

        # Plot boreholes (if applicable)
        if view_mode in [1, 2]:  # Modes 1 and 2 include borelogs
            for i in range(len(df)):
                x, y = x_vals[i], y_vals[i]
                z0 = df.at[i, 'Ground Level'] * z_scale
                BHID = df.at[i, 'BHID']
                actual_elev = df.at[i, 'Ground Level']  # Actual (non-exaggerated) elevation

                points = [(x, y, z0)]
                colors = []

                for layer_idx in [1, 2, 3]:
                    depth_col = f'Layer{layer_idx} Depth'
                    type_col = f'Layer{layer_idx} Type'
                    if pd.notna(df.at[i, depth_col]):
                        z = df.at[i, depth_col] * z_scale
                        points.append((x, y, z))
                        layer_type = df.at[i, type_col]
                        colors.append(color_map.get(layer_type, 'grey'))

                # Draw segments with labels for the legend
                for j in range(len(points) - 1):
                    x_seg, y_seg, z_seg = zip(points[j], points[j + 1])
                    layer_type = df.at[i, f'Layer{j + 1} Type']
                    label = layer_type if layer_type not in plotted_layer_types else None
                    if label:
                        plotted_layer_types.add(layer_type)
                    fig.add_trace(go.Scatter3d(
                        x=x_seg, y=y_seg, z=z_seg,
                        mode='lines',
                        line=dict(color=colors[j], width=5),
                        name=label or layer_type,
                        showlegend=bool(label)
                    ))

                # Add borehole ID with actual elevation as a marker
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z0],
                    mode='markers+text',
                    text=[f"{BHID}<br>Elev: {actual_elev:.2f}"],
                    textposition="top center",
                    marker=dict(size=5, color='black'),
                    showlegend=False
                ))

        # Update layout for better visualization with maximized dimensions
        fig.update_layout(
            title='3D Ground Model with Borehole Stratigraphy',
            scene=dict(
                xaxis_title='',  # Remove X-axis label
                yaxis_title='',  # Remove Y-axis label
                zaxis_title='',  # Remove Z-axis label
                xaxis_showticklabels=False,  # Optional: hide tick labels if desired
                yaxis_showticklabels=False,
                zaxis_showticklabels=False,
                bgcolor='black'
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=True,
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.5)"
            ),
            # Enable fullscreen mode automatically
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5)  # Adjust camera for better initial view
            )
        )

        # Add a fullscreen button to the modebar and trigger it automatically
        fig.update_layout(
            modebar_add=["togglefullscreen"],
            modebar_activecolor="#00BFFF"
        )

        return fig

    except Exception as e:
        st.error(f"Error generating 3D visualization: {e}")
        return None

# Function to plot 2D cross-section with improved styling
def plot_2d_cross_section(selected_bhids):
    # Validate input
    if len(selected_bhids) < 2:
        st.error("Please select at least 2 boreholes for the cross-section.")
        return None

    # Filter the dataframe for selected boreholes
    section_df = df[df['BHID'].isin(selected_bhids)].copy()

    # Sort the boreholes based on the order of selection to maintain user-specified order
    section_df['BHID_order'] = section_df['BHID'].apply(lambda x: selected_bhids.index(x))
    section_df = section_df.sort_values('BHID_order')

    # Calculate cumulative distance along the section
    eastings = section_df['Easting'].values
    northings = section_df['Northing'].values
    distances = [0]  # Start at 0
    for i in range(1, len(eastings)):
        dx = eastings[i] - eastings[i-1]
        dy = northings[i] - northings[i-1]
        distance = np.sqrt(dx**2 + dy**2)
        distances.append(distances[-1] + distance)

    # Extract elevation data for the cross-section (exaggerated for plotting)
    ground_levels = section_df['Ground Level'].values * z_scale
    layer1_depths = section_df['Layer1 Depth'].values * z_scale
    layer2_depths = section_df['Layer2 Depth'].values * z_scale
    layer3_depths = section_df['Layer3 Depth'].values * z_scale
    actual_ground_levels = section_df['Ground Level'].values

    # Create the 2D cross-section plot with improved styling
    plt.style.use('seaborn')  # Use a modern style for better aesthetics
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

    # Track plotted layer types for the legend
    plotted_layer_types = set()
    legend_patches = []

    # Plot each layer as a filled polygon with improved colors and styling
    for i in range(len(section_df)):
        x = distances[i]
        gl = ground_levels[i]
        l1 = layer1_depths[i] if pd.notna(layer1_depths[i]) else gl
        l2 = layer2_depths[i] if pd.notna(layer2_depths[i]) else l1
        l3 = layer3_depths[i] if pd.notna(layer3_depths[i]) else l2

        # Determine the deepest depth for the borehole representation
        deepest_depth = gl
        if pd.notna(layer1_depths[i]):
            deepest_depth = l1
        if pd.notna(layer2_depths[i]):
            deepest_depth = l2
        if pd.notna(layer3_depths[i]):
            deepest_depth = l3

        # Represent borehole as a rectangle with better styling
        rect_width = 1.5  # Slightly wider rectangles
        rect = Rectangle(
            (x - rect_width/2, deepest_depth),
            rect_width,
            gl - deepest_depth,
            facecolor='#D3D3D3',  # Light gray for boreholes
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(rect)

        # Ground to Layer 1
        if pd.notna(layer1_depths[i]):
            layer_type = section_df.iloc[i]['Layer1 Type']
            color = color_map.get(layer_type, 'grey')
            if i < len(section_df) - 1:
                next_x = distances[i + 1]
                next_gl = ground_levels[i + 1]
                next_l1 = layer1_depths[i + 1] if pd.notna(layer1_depths[i + 1]) else next_gl
                polygon = Polygon([
                    (x, gl), (x, l1), (next_x, next_l1), (next_x, next_gl)
                ], facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
                ax.add_patch(polygon)
                if layer_type not in plotted_layer_types:
                    plotted_layer_types.add(layer_type)
                    legend_patches.append(Polygon([[0, 0]], facecolor=color, edgecolor='black', alpha=0.7, label=layer_type))

        # Layer 1 to Layer 2
        if pd.notna(layer2_depths[i]):
            layer_type = section_df.iloc[i]['Layer2 Type']
            color = color_map.get(layer_type, 'grey')
            if i < len(section_df) - 1:
                next_x = distances[i + 1]
                next_l1 = layer1_depths[i + 1] if pd.notna(layer1_depths[i + 1]) else ground_levels[i + 1]
                next_l2 = layer2_depths[i + 1] if pd.notna(layer2_depths[i + 1]) else next_l1
                polygon = Polygon([
                    (x, l1), (x, l2), (next_x, next_l2), (next_x, next_l1)
                ], facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
                ax.add_patch(polygon)
                if layer_type not in plotted_layer_types:
                    plotted_layer_types.add(layer_type)
                    legend_patches.append(Polygon([[0, 0]], facecolor=color, edgecolor='black', alpha=0.7, label=layer_type))

        # Layer 2 to Layer 3
        if pd.notna(layer3_depths[i]):
            layer_type = section_df.iloc[i]['Layer3 Type']
            color = color_map.get(layer_type, 'grey')
            if i < len(section_df) - 1:
                next_x = distances[i + 1]
                next_l2 = layer2_depths[i + 1] if pd.notna(layer2_depths[i + 1]) else layer1_depths[i + 1]
                next_l3 = layer3_depths[i + 1] if pd.notna(layer3_depths[i + 1]) else next_l2
                polygon = Polygon([
                    (x, l2), (x, l3), (next_x, next_l3), (next_x, next_l2)
                ], facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
                ax.add_patch(polygon)
                if layer_type not in plotted_layer_types:
                    plotted_layer_types.add(layer_type)
                    legend_patches.append(Polygon([[0, 0]], facecolor=color, edgecolor='black', alpha=0.7, label=layer_type))

    # Plot the ground surface line with improved styling
    ax.plot(distances, ground_levels, 'k-', label='Ground Surface', linewidth=2)

    # Add borehole labels with actual elevation with better styling
    for i, (x, gl) in enumerate(zip(distances, ground_levels)):
        bhid = section_df.iloc[i]['BHID']
        actual_elev = section_df.iloc[i]['Ground Level']
        ax.text(x, gl + 5, f"{bhid}\nElev: {actual_elev:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    # Set labels and title with improved styling
    ax.set_xlabel('Distance Along Section (units)', fontsize=12, fontweight='bold')
    ax.set_title('2D Cross-Section of Selected Boreholes', fontsize=14, fontweight='bold', pad=20)

    # Show Y-axis with depth labels (exaggerated depths)
    ax.set_ylabel('Depth (exaggerated)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()  # Invert Y-axis to show depth increasing downward
    ax.grid(True, linestyle='--', alpha=0.7)  # Add gridlines for better readability

    # Customize the legend
    ax.legend(handles=[plt.Line2D([0], [0], color='black', linewidth=2, label='Ground Surface')] + legend_patches, 
              loc='upper right', fontsize=10, frameon=True, edgecolor='black')

    # Customize the plot background and spines
    ax.set_facecolor('#F5F5F5')  # Light gray background
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # Display the plot in Streamlit
    plt.tight_layout()
    return fig

# Streamlit App
st.title("3D Ground Model Visualization")

# Option Buttons for 3D Visualizations
st.header("3D Visualizations")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("1) Only Borelogs"):
        fig = plot_3d_visualization(1)
        if fig:
            # Display the chart and trigger fullscreen automatically
            chart = st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
                <script>
                document.querySelector('iframe').contentWindow.document.querySelector('.modebar-btn[data-title="Toggle Fullscreen"]').click();
                </script>
                """,
                unsafe_allow_html=True
            )

with col2:
    if st.button("2) Borelogs with Surfaces"):
        fig = plot_3d_visualization(2)
        if fig:
            chart = st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
                <script>
                document.querySelector('iframe').contentWindow.document.querySelector('.modebar-btn[data-title="Toggle Fullscreen"]').click();
                </script>
                """,
                unsafe_allow_html=True
            )

with col3:
    if st.button("3) Only Surfaces"):
        fig = plot_3d_visualization(3)
        if fig:
            chart = st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                """
                <script>
                document.querySelector('iframe').contentWindow.document.querySelector('.modebar-btn[data-title="Toggle Fullscreen"]').click();
                </script>
                """,
                unsafe_allow_html=True
            )

# 2D Cross-Section with Checklist
st.header("2D Cross-Section")
st.subheader("Select Boreholes for 2D Cross-Section (at least 2)")

# Checklist for BHIDs
available_bhids = df['BHID'].tolist()
selected_bhids = st.multiselect("Select Boreholes", available_bhids)

if st.button("Generate 2D Cross-Section"):
    if len(selected_bhids) >= 2:
        fig = plot_2d_cross_section(selected_bhids)
        if fig:
            st.pyplot(fig)
    else:
        st.error("Please select at least 2 boreholes for the cross-section.")