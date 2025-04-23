import streamlit as st
import pyvista as pv
from pyvista import examples
from tempfile import NamedTemporaryFile
import streamlit.components.v1 as components
import plotly.graph_objects as go

# Example PyVista plot
def create_pyvista_plot():
    plotter = pv.Plotter(off_screen=True)
    mesh = examples.download_st_helens().warp_by_scalar()
    plotter.add_mesh(mesh, cmap='terrain')
    return plotter

# Save PyVista plot to HTML
def save_pyvista_to_html(plotter):
    tmp_file = NamedTemporaryFile(delete=False, suffix='.html')
    plotter.export_html(tmp_file.name)
    return tmp_file.name

# Plotly fallback example
def create_plotly_plot():
    # Dummy 3D surface
    import numpy as np
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)
    z = np.sin(np.sqrt(x**2 + y**2))

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='Plotly Surface', autosize=True,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Streamlit UI
st.title("3D Ground Model Viewer")

view_option = st.radio("Select View Mode:", ("PyVista (HTML iframe)", "Plotly Fallback"))

if view_option == "PyVista (HTML iframe)":
    with st.spinner("Rendering PyVista..."):
        plotter = create_pyvista_plot()
        html_path = save_pyvista_to_html(plotter)
        with open(html_path, 'r') as f:
            components.html(f.read(), height=600)
else:
    st.plotly_chart(create_plotly_plot(), use_container_width=True)
