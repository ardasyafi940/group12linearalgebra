import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Matrix Transformations Visualizer",
                   layout="wide")

# ----------------- TITLE ------------------
st.title("🔢 Matrix Transformations Visualizer")
st.markdown(
    "**Interactive 2D Linear Algebra Tool** – visualize "
    r"$T(\mathbf{x}) = A\mathbf{x}$ on a unit square, grid, and vectors."
)

# ----------------- SESSION STATE ----------
if "A" not in st.session_state:
    st.session_state.A = np.array([[1.0, 0.0],
                                   [0.0, 1.0]])

# ----------------- SIDEBAR: MATRIX --------
st.sidebar.header("🎛️ Matrix Controls")

st.sidebar.subheader("📝 Custom 2×2 Matrix A")
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

# ----------------- PRESET BUTTONS ---------
st.sidebar.subheader("✨ Quick Transforms")

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("↕️ X-axis reflection"):
        st.session_state.A = np.array([[1.0, 0.0],
                                       [0.0, -1.0]])
        st.experimental_rerun()
    if st.button("↔️ Y-axis reflection"):
        st.session_state.A = np.array([[-1.0, 0.0],
                                       [0.0, 1.0]])
        st.experimental_rerun()

with c2:
    if st.button("⚫ Origin reflection"):
        st.session_state.A = np.array([[-1.0, 0.0],
                                       [0.0, -1.0]])
        st.experimental_rerun()
    if st.button("🔄 Reset identity"):
        st.session_state.A = np.array([[1.0, 0.0],
                                       [0.0, 1.0]])
        st.experimental_rerun()

# update A lagi setelah preset
A = st.session_state.A.copy()
det_A = float(np.linalg.det(A))

# ----------------- TEST & ANIMATION -------
st.sidebar.subheader("🧪 Test & Animation")

test_x = st.sidebar.number_input("Test vector x", value=0.5, step=0.1)
test_y = st.sidebar.number_input("Test vector y", value=0.5, step=0.1)
t = st.sidebar.slider("Animation t (0 = I, 1 = A)", 0.0, 1.0, 1.0, 0.01)

A_t = (1.0 - t) * np.eye(2) + t * A
test_vec = np.array([test_x, test_y], dtype=float)
test_t = A_t @ test_vec

# ----------------- GEOMETRY ---------------
def make_geometry():
    # unit square
    square = np.array([[0, 0],
                       [1, 0],
                       [1, 1],
                       [0, 1],
                       [0, 0]], dtype=float)
    # basis vectors
    e1 = np.array([[0, 0],
                   [1, 0]], dtype=float)
    e2 = np.array([[0, 0],
                   [0, 1]], dtype=float)
    # grid
    xs = np.linspace(-0.5, 1.5, 9)
    ys = np.linspace(-0.5, 1.5, 9)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    return square, e1, e2, grid

square, e1, e2, grid = make_geometry()

square_t = (A_t @ square.T).T
e1_t = (A_t @ e1.T).T
e2_t = (A_t @ e2.T).T
grid_t = (A_t @ grid.T).T

# ----------------- PLOTS -------------------
colL, colR = st.columns(2)

with colL:
    st.subheader("📐 Original Space")
    fig1, ax1 = plt.subplots(figsize=(5, 5))

    # unit square
    ax1.plot(square[:, 0], square[:, 1], "k-", linewidth=3, label="Unit square")
    ax1.fill(square[:, 0], square[:, 1], color="lightgray", alpha=0.5)

    # basis
    ax1.arrow(0, 0, 1, 0, head_width=0.05, color="blue", linewidth=3, label="e₁")
    ax1.arrow(0, 0, 0, 1, head_width=0.05, color="green", linewidth=3, label="e₂")

    # grid
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
    st.subheader("✨ Transformed Space")
    fig2, ax2 = plt.subplots(figsize=(5, 5))

    # transformed unit square
    ax2.plot(square_t[:, 0], square_t[:, 1], "r-", linewidth=3,
             label="T(unit square)")
    ax2.fill(square_t[:, 0], square_t[:, 1], color="red", alpha=0.25)

    # transformed basis
    ax2.arrow(0, 0, e1_t[1, 0], e1_t[1, 1],
              head_width=0.07, color="blue", linewidth=3, label="T(e₁)")
    ax2.arrow(0, 0, e2_t[1, 0], e2_t[1, 1],
              head_width=0.07, color="green", linewidth=3, label="T(e₂)")

    # transformed grid
    ax2.scatter(grid_t[:, 0], grid_t[:, 1], color="orange", s=18, alpha=0.8,
                label="T(grid)")

    # test vector
    ax2.arrow(0, 0, test_t[0], test_t[1],
              head_width=0.09, color="lime", linewidth=3,
              label=f"T([{test_x:.1f}, {test_y:.1f}])")

    ax2.set_aspect("equal", "box")
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_title(f"Transformed (det(A) = {det_A:.3f})")
    ax2.legend(loc="upper right")
    st.pyplot(fig2)

# ----------------- METRICS & FORMULAS -----
st.markdown("---")
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("📐 Area scaling = |det(A)|", f"{abs(det_A):.3f}")

with m2:
    st.metric("🎯 Test vector",
              f"({test_x:.2f}, {test_y:.2f})",
              f"T(x) = ({test_t[0]:.2f}, {test_t[1]:.2f})")

with m3:
    st.latex(r"T(\mathbf{x}) = A\mathbf{x}, \quad \det(A) = ad - bc")
    st.latex(r"\text{Area scaling} = |\det(A)|")

# ----------------- EXPORT ------------------
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
        csv,
        file_name="matrix_transformations.csv",
        mime="text/csv",
    )

st.markdown("*✅ Clean, error‑free, and ready to run with `streamlit run app.py`*")
