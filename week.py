import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------- CONFIG --------------
st.set_page_config(page_title="Matrix Transformations Visualizer",
                   layout="wide")

st.title("🔢 Matrix Transformations Visualizer")
st.markdown(
    "**Interactive 2D Linear Algebra Tool** – visualize "
    r"$T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ on a unit square, grid, and vectors."
)

# -------------- SESSION STATE --------------
if "A" not in st.session_state:
    st.session_state.A = np.eye(2, dtype=float)   # linear part
if "b" not in st.session_state:
    st.session_state.b = np.array([0.0, 0.0], dtype=float)   # translation

# -------------- SIDEBAR: MATRIX A --------------
st.sidebar.header("🎛️ Linear Part (Matrix A)")

col1, col2 = st.sidebar.columns(2)
a11 = col1.number_input("a₁₁", value=float(st.session_state.A[0, 0]),
                        step=0.1, format="%.2f")
a12 = col2.number_input("a₁₂", value=float(st.session_state.A[0, 1]),
                        step=0.1, format="%.2f")
a21 = col1.number_input("a₂₁", value=float(st.session_state.A[1, 0]),
                        step=0.1, format="%.2f")
a22 = col2.number_input("a₂₂", value=float(st.session_state.A[1, 1]),
                        step=0.1, format="%.2f")

A = np.array([[a11, a12],
              [a21, a22]], dtype=float)
st.session_state.A = A.copy()

det_A = float(np.linalg.det(A))
trace_A = float(np.trace(A))

st.sidebar.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.latex(
    rf"A = \begin{{pmatrix}}"
    rf"{A[0,0]:.2f} & {A[0,1]:.2f} \\"
    rf"{A[1,0]:.2f} & {A[1,1]:.2f}"
    r"\end{pmatrix}"
)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.metric("🔢 det(A)", f"{det_A:.3f}")
st.sidebar.metric("📏 trace(A)", f"{trace_A:.3f}")

# -------------- SIDEBAR: TRANSLATION b --------------
st.sidebar.header("📍 Translation (Vector b)")

tx = st.sidebar.number_input("b₁ (translate x)", value=float(st.session_state.b[0]),
                             step=0.1, format="%.2f")
ty = st.sidebar.number_input("b₂ (translate y)", value=float(st.session_state.b[1]),
                             step=0.1, format="%.2f")
b = np.array([tx, ty], dtype=float)
st.session_state.b = b.copy()

st.sidebar.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.latex(
    st.latex(r"A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}")
)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# -------------- PRESET TRANSFORMATIONS --------------
st.sidebar.header("✨ Preset Transformations")

mode = st.sidebar.radio(
    "Choose group:",
    [
        "Basic: Scaling / Rotation / Shearing",
        "Reflections"
    ]
)

# BASIC: Scaling / Rotation / Shearing
if mode == "Basic: Scaling / Rotation / Shearing":
    st.sidebar.markdown("**Scaling**")
    s_x = st.sidebar.slider("Scale X (sₓ)", 0.1, 3.0, 1.0, 0.1)
    s_y = st.sidebar.slider("Scale Y (sᵧ)", 0.1, 3.0, 1.0, 0.1)

    st.sidebar.markdown("**Rotation**")
    angle_deg = st.sidebar.slider("Angle θ (degrees)", -180.0, 180.0, 0.0, 1.0)
    theta = np.radians(angle_deg)

    st.sidebar.markdown("**Shearing**")
    sh_x = st.sidebar.slider("Shear X (kₓ)", -2.0, 2.0, 0.0, 0.1)
    sh_y = st.sidebar.slider("Shear Y (kᵧ)", -2.0, 2.0, 0.0, 0.1)

    if st.sidebar.button("Apply Scale + Rotate + Shear"):
        S = np.array([[s_x, 0.0],
                      [0.0, s_y]], dtype=float)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=float)
        H = np.array([[1.0, sh_x],
                      [sh_y, 1.0]], dtype=float)
        A_basic = H @ R @ S
        st.session_state.A = A_basic
        A = A_basic.copy()

# REFLECTIONS
if mode == "Reflections":
    st.sidebar.markdown("**Reflection matrices**")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.sidebar.button("X-axis"):
            st.session_state.A = np.array([[1.0, 0.0],
                                           [0.0, -1.0]], dtype=float)
        if st.sidebar.button("Y-axis"):
            st.session_state.A = np.array([[-1.0, 0.0],
                                           [0.0, 1.0]], dtype=float)
    with c2:
        if st.sidebar.button("Origin"):
            st.session_state.A = np.array([[-1.0, 0.0],
                                           [0.0, -1.0]], dtype=float)
        if st.sidebar.button("Line y = x"):
            st.session_state.A = np.array([[0.0, 1.0],
                                           [1.0, 0.0]], dtype=float)

# update A setelah kemungkinan preset
A = st.session_state.A.copy()
det_A = float(np.linalg.det(A))

# -------------- TEST VECTOR & ANIMATION --------------
st.sidebar.header("🧪 Test & Animation")

test_x = st.sidebar.number_input("Test vector x", value=0.5, step=0.1, format="%.2f")
test_y = st.sidebar.number_input("Test vector y", value=0.5, step=0.1, format="%.2f")
t = st.sidebar.slider("Animation t (0 = I, 1 = A, b)", 0.0, 1.0, 1.0, 0.01)

A_t = (1.0 - t) * np.eye(2) + t * A          # animasi linear part
b_t = t * b                                  # animasi translasi

test_vec = np.array([test_x, test_y], dtype=float)
test_t = A_t @ test_vec + b_t

# -------------- GEOMETRY (UNIT SQUARE, GRID, BASIS) --------------
def make_geometry():
    square = np.array([[0, 0],
                       [1, 0],
                       [1, 1],
                       [0, 1],
                       [0, 0]], dtype=float)
    e1 = np.array([[0, 0],
                   [1, 0]], dtype=float)
    e2 = np.array([[0, 0],
                   [0, 1]], dtype=float)
    xs = np.linspace(-0.5, 1.5, 9)
    ys = np.linspace(-0.5, 1.5, 9)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    return square, e1, e2, grid

square, e1, e2, grid = make_geometry()

# T(x) = A_t x + b_t
square_t = (A_t @ square.T).T + b_t
e1_t = (A_t @ e1.T).T + b_t
e2_t = (A_t @ e2.T).T + b_t
grid_t = (A_t @ grid.T).T + b_t

# -------------- PLOTS --------------
colL, colR = st.columns(2)

with colL:
    st.subheader("📐 Original Space")
    fig1, ax1 = plt.subplots(figsize=(5, 5))

    ax1.plot(square[:, 0], square[:, 1], "k-", linewidth=3, label="Unit square")
    ax1.fill(square[:, 0], square[:, 1], color="lightgray", alpha=0.5)

    ax1.arrow(0, 0, 1, 0, head_width=0.05, color="blue", linewidth=3, label="e₁")
    ax1.arrow(0, 0, 0, 1, head_width=0.05, color="green", linewidth=3, label="e₂")

    ax1.scatter(grid[:, 0], grid[:, 1], color="gray", s=20, alpha=0.6)

    ax1.set_aspect("equal", "box")
    ax1.set_xlim(-0.7, 1.7)
    ax1.set_ylim(-0.7, 1.7)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_title("Original unit square & basis")
    ax1.legend(loc="upper right")
    st.pyplot(fig1)

with colR:
    st.subheader("✨ Transformed Space  (T(x) = A x + b)")
    fig2, ax2 = plt.subplots(figsize=(5, 5))

    ax2.plot(square_t[:, 0], square_t[:, 1], "r-", linewidth=3,
             label="T(unit square)")
    ax2.fill(square_t[:, 0], square_t[:, 1], color="red", alpha=0.25)

    ax2.arrow(b_t[0], b_t[1],
              e1_t[1, 0] - b_t[0], e1_t[1, 1] - b_t[1],
              head_width=0.07, color="blue", linewidth=3, label="T(e₁)")
    ax2.arrow(b_t[0], b_t[1],
              e2_t[1, 0] - b_t[0], e2_t[1, 1] - b_t[1],
              head_width=0.07, color="green", linewidth=3, label="T(e₂)")

    ax2.scatter(grid_t[:, 0], grid_t[:, 1], color="orange", s=18, alpha=0.8,
                label="T(grid)")

    ax2.arrow(b_t[0], b_t[1],
              test_t[0] - b_t[0], test_t[1] - b_t[1],
              head_width=0.09, color="lime", linewidth=3,
              label=f"T([{test_x:.1f}, {test_y:.1f}])")

    ax2.set_aspect("equal", "box")
    ax2.set_xlim(-4.0, 4.0)
    ax2.set_ylim(-4.0, 4.0)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_title(f"Transformed (det(A) = {det_A:.3f})")
    ax2.legend(loc="upper right")
    st.pyplot(fig2)

# -------------- METRICS & FORMULAS --------------
st.markdown("---")
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("📐 Area scaling = |det(A)|", f"{abs(det_A):.3f}")

with m2:
    st.metric("🎯 Test vector",
              f"({test_x:.2f}, {test_y:.2f})",
              f"T(x) = ({test_t[0]:.2f}, {test_t[1]:.2f})")

with m3:
    st.latex(r"T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}")
    st.latex(r"\det(A) = ad - bc")
    st.latex(r"\text{Area scaling} = |\det(A)|")

# -------------- EXPORT CSV --------------
st.markdown("---")
if st.sidebar.button("📥 Export grid data as CSV"):
    df = pd.DataFrame({
        "x_original": grid[:, 0],
        "y_original": grid[:, 1],
        "x_transformed": grid_t[:, 0],
        "y_transformed": grid_t[:, 1],
    })
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        "Download CSV",
        data=csv,
        file_name="matrix_transformations.csv",
        mime="text/csv",
    )

st.markdown("*✅ Translation, Scaling, Rotation, Shearing, Reflection – ready to run with `streamlit run app.py`*")
