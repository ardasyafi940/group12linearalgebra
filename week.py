import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Matrix Transformations Visualizer", layout="wide")

# Title
st.title("🔢 Matrix Transformations Visualizer")
st.markdown("**Interactive 2D Linear Algebra Tool** - Visualize $T(\\mathbf{x}) = A\\mathbf{x}$")

# Session state initialization
if 'a11' not in st.session_state:
    st.session_state.a11, st.session_state.a12 = 1.0, 0.0
    st.session_state.a21, st.session_state.a22 = 0.0, 1.0

# Sidebar Controls
st.sidebar.header("🎛️ Matrix Controls")

# Custom Matrix Input
st.sidebar.subheader("📝 Custom Matrix")
col1, col2 = st.sidebar.columns(2)
st.session_state.a11 = col1.number_input("a₁₁", value=st.session_state.a11, step=0.1, format="%.2f")
st.session_state.a12 = col2.number_input("a₁₂", value=st.session_state.a12, step=0.1, format="%.2f")
st.session_state.a21 = col1.number_input("a₂₁", value=st.session_state.a21, step=0.1, format="%.2f")
st.session_state.a22 = col2.number_input("a₂₂", value=st.session_state.a22, step=0.1, format="%.2f")

# Matrix A
A = np.array([[st.session_state.a11, st.session_state.a12],
              [st.session_state.a21, st.session_state.a22]])
det_A = np.linalg.det(A)
trace_A = np.trace(A)

# Display matrix
st.sidebar.markdown(f"""
<div style='text-align: center; font-size: 18px;'>
$$
A = \\begin{{pmatrix}}
{st.session_state.a11:.2f} & {st.session_state.a12:.2f} \\\\
{st.session_state.a21:.2f} & {st.session_state.a22:.2f}
\\end{{pmatrix}}
$$
</div>
""")

st.sidebar.metric("🔢 Determinant", f"{det_A:.3f}")
st.sidebar.metric("📏 Trace", f"{trace_A:.3f}")

# Preset Transformations
st.sidebar.subheader("✨ Quick Transforms")
col_ref1, col_ref2 = st.sidebar.columns(2)
with col_ref1:
    if st.button("↕️ X-Reflection", use_container_width=True):
        st.session_state.a11, st.session_state.a12 = 1.0, 0.0
        st.session_state.a21, st.session_state.a22 = 0.0, -1.0
        st.rerun()
    if st.button("↔️ Y-Reflection", use_container_width=True):
        st.session_state.a11, st.session_state.a12 = -1.0, 0.0
        st.session_state.a21, st.session_state.a22 = 0.0, 1.0
        st.rerun()

with col_ref2:
    if st.button("⚫ Origin Reflection", use_container_width=True):
        st.session_state.a11, st.session_state.a12 = -1.0, 0.0
        st.session_state.a21, st.session_state.a22 = 0.0, -1.0
        st.rerun()
    if st.button("🔄 Reset Identity", use_container_width=True):
        st.session_state.a11, st.session_state.a12 = 1.0, 0.0
        st.session_state.a21, st.session_state.a22 = 0.0, 1.0
        st.rerun()

# Enhanced Features
st.sidebar.subheader("🧪 Test Features")
test_x = st.sidebar.number_input("Test Vector x", value=0.5, step=0.1)
test_y = st.sidebar.number_input("Test Vector y", value=0.5, step=0.1)
animate_t = st.sidebar.slider("Animation", 0.0, 1.0, 1.0, 0.01)

# Animation matrix
A_animated = (1-animate_t) * np.eye(2) + animate_t * A

# Generate geometry data
def get_geometry():
    # Unit square
    square = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])
    
    # Basis vectors
    basis_x = np.array([[0,0], [1,0]])
    basis_y = np.array([[0,0], [0,1]])
    
    # Grid points
    x_grid = np.linspace(-0.5, 1.5, 9)
    y_grid = np.linspace(-0.5, 1.5, 9)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([Xg.ravel(), Yg.ravel()])
    
    # Test vector
    test_vec = np.array([test_x, test_y])
    
    return square, basis_x, basis_y, grid_points, test_vec

square, basis_x, basis_y, grid_points, test_vec = get_geometry()

# Transform everything
square_t = (A_animated @ square.T).T
basis_x_t = (A_animated @ basis_x.T).T
basis_y_t = (A_animated @ basis_y.T).T
grid_t = (A_animated @ grid_points.T).T
test_t = A_animated @ test_vec

# Main Visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("📐 Original Space")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    # Unit square
    ax1.plot(square[:,0], square[:,1], 'k-', linewidth=4, label='Unit Square')
    ax1.fill(square[:,0], square[:,1], alpha=0.1, color='black')
    
    # Basis vectors
    ax1.arrow(0, 0, 1, 0, head_width=0.08, head_length=0.1, fc='blue', ec='blue', linewidth=3, label='e₁')
    ax1.arrow(0, 0, 0, 1, head_width=0.08, head_length=0.1, fc='green', ec='green', linewidth=3, label='e₂')
    
    # Grid
    ax1.scatter(grid_points[:,0], grid_points[:,1], c='lightgray', s=30, alpha=0.6)
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_xlim(-0.7, 1.7)
    ax1.set_ylim(-0.7, 1.7)
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.set_title("Original Unit Square + Basis Vectors")
    st.pyplot(fig1)

with col2:
    st.subheader("✨ Transformed Space")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Transformed square
    ax2.plot(square_t[:,0], square_t[:,1], 'r-', linewidth=4, label='T(Unit Square)')
    ax2.fill(square_t[:,0], square_t[:,1], alpha=0.2, color='red')
    
    # Transformed basis
    ax2.arrow(0, 0, basis_x_t[1,0], basis_x_t[1,1], head_width=0.1, head_length=0.12, 
              fc='blue', ec='blue', linewidth=3, label='T(e₁)')
    ax2.arrow(0, 0, basis_y_t[1,0], basis_y_t[1,1], head_width=0.1, head_length=0.12, 
              fc='green', ec='green', linewidth=3, label='T(e₂)')
    
    # Transformed grid
    ax2.scatter(grid_t[:,0], grid_t[:,1], c='orange', s=25, alpha=0.7, label='T(Grid)')
    
    # Test vector
    ax2.arrow(0, 0, test_t[0], test_t[1], head_width=0.12, head_length=0.15, 
              fc='lime', ec='lime', linewidth=4, label=f'T([{test_x:.1f},{test_y:.1f}])')
    
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.set_title(f"Transformed (det(A) = {det_A:.3f})")
    st.pyplot(fig2)

# Bottom metrics - FIXED LaTeX
st.markdown("---")
col_metrics1, col_metrics2, col_math = st.columns([1,1,1])

with col_metrics1:
    st.metric("📐 Area Scaling", f"{abs(det_A):.3f}")
    
with col_metrics2:
    st.metric("🎯 Test Vector", 
              f"({test_x:.1f}, {test_y:.1f})", 
              f"({test_t[0]:.2f}, {test_t[1]:.2f})")

with col_math:
    st.latex(r"T(\mathbf{x}) = A\mathbf{x}, \quad \det(A) = ad-bc")
    st.latex(r"\text{Area scaling} = |\det(A)|")

# Export
if st.sidebar.button("📥 Export Data"):
    df = pd.DataFrame({
        'x_original': grid_points[:,0],
        'y_original': grid_points[:,1],
        'x_transformed': grid_t[:,0],
        'y_transformed': grid_t[:,1]
    })
    csv = df.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, "matrix_transformations.csv", "text/csv")

st.markdown("---")
st.markdown("*✨ Fully functional - Perfect LaTeX rendering ✨*")
