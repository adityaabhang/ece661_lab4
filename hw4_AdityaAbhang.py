#!/usr/bin/env python

##  hw4_skeleton.py

__version__   = '1.0.0'
__author__    = "Aditya Uday ABhang (aabhang@purdue.edu)"
__date__      = '2026-September-23'


__doc__ = '''

    hw4_skeleton.py

    This is the skeleton for Homework 4 of ECE 661, Computer Vision, Fall 2026.

    Rename this file to hw4_<FirstName><LastName>.py before you submit it.

    Three tasks, three lectures, mostly independent of one another:

        Task 1    Sobel and Laplacian-of-Gaussian edge detection, by hand on a
                  small array and then on a real image crop
        Task 2    points, planes and lines in 3D, and the transformation group
                  hierarchy and quadrics, all worked on numbers given directly
                  in the handout
        Task 3    recorded correspondences between two photographs, a metric
                  rectification of the ground plane by the point-to-point
                  method from HW3, and an anamorphic banner exactly like
                  HW2 Task 3

    Plus an optional Bonus problem (+15 points, does not affect the 150-point
    total): a demonstration of why that same rectifying homography cannot be
    pointed at a pixel from a different plane in the scene.

    Rename every RECORDED_* placeholder below with your own pixel readings
    before you run anything in Task 3. Task 2's numbers are given to you in
    the handout and are already filled in; do not change them.

    Everything is inside functions and the calls at the bottom sit behind the
    usual __main__ guard. Keep it that way. No OpenCV, no library homography
    or warp -- see the handout, Section "Additional guidelines".

    Allowed: numpy, matplotlib, scipy, PIL. Fill in every place marked TODO.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from scipy.signal import convolve2d, correlate2d
from matplotlib.colors import ListedColormap 
from matplotlib.patches import Patch


##  --------------------------------------------------------------------------
##  Paths
##  --------------------------------------------------------------------------
DATA_DIR = "."
OUT_DIR  = "output"


def load_image(name, gray=False):
    """Read an image, optionally converted to grayscale float."""
    img = Image.open(os.path.join(DATA_DIR, name))
    return np.array(img.convert("L"), float) if gray else np.array(img.convert("RGB"))


def save_image(arr, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray(np.uint8(arr)).save(os.path.join(OUT_DIR, name), quality=93)

def to_display(arr):
    """Linearly stretch any array so its min maps to 0 and its max to 255."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) * 255


##  ============================================================================
##  Task 1: edge detection
##  ============================================================================

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)

##  a clean vertical step edge
LECTURE_ARRAY = np.array([
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 10, 10, 10, 10, 10],
], float)


def sobel(img):
    """Gradient magnitude and direction (degrees, wrapped into [0, 360)) of
    `img`, by sliding SOBEL_X and SOBEL_Y across the image and summing
    elementwise products at each position, using the masks exactly as given
    (do NOT flip them first. that would be true convolution, not the
    correlation-style sliding-window sum the handout asks for, and it would
    rotate every direction here by 180 degrees).
    Args:
        img: (H, W) float array
    Returns:
        mag: (H, W) float array, gradient magnitude
        ang: (H, W) float array, gradient direction in degrees, in [0, 360)
    """
    ##  TODO: scipy.signal.correlate2d(img, SOBEL_X, mode="same",
    ##  boundary="symm") and the same for SOBEL_Y (boundary="symm" reflects
    ##  at the image edge instead of zero-padding it -- zero-padding invents
    ##  a fake edge at the border). Combine into magnitude and direction.
    img = np.asarray(img, dtype=float)
    grad_x = correlate2d(img, SOBEL_X, mode="same", boundary="symm")
    grad_y = correlate2d(img, SOBEL_Y, mode="same", boundary="symm")
    #print(f"grad_x = {grad_x}")
    #print(f"grad_y = {grad_y}")
    mag = np.hypot(grad_x, grad_y)
    ang = np.arctan2(grad_y, grad_x) * 180 / np.pi % 360
    return mag, ang


def log_kernel(sigma, half=None):
    """The Laplacian-of-Gaussian kernel, sampled directly from
        h(x,y) = -(1/(pi sigma^4)) [1 - (x^2+y^2)/(2 sigma^2)] exp(-(x^2+y^2)/(2 sigma^2))
    `half` is the kernel's half-width in pixels; default 3*sigma, rounded up.
    Subtract the kernel's own mean so it sums to (approximately) zero -- a
    flat region should convolve to 0, not to some nonzero constant.
    """
    ##  TODO
    if half is None:
        half = int(np.ceil(3 * sigma))
    coords = np.arange(-half, half + 1)     #PEP 8 (79 characters per line)
    X, Y = np.meshgrid(coords, coords)
    r2 = X**2 + Y**2                      # squared distance from the center
    scale = -1 / (np.pi * sigma**4)
    h = scale * (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
    h -= h.mean()
    return h

    
def zero_crossings(img):
    """Boolean array: does this pixel have a horizontal or vertical neighbor
    of the opposite sign?"""
    ##  TODO
    img = np.asarray(img, dtype=float)
    zc = np.zeros_like(img, dtype=bool)
    vertical = img[1:, :] * img[:-1, :] < 0     # True where a pixel and the one below it differ in sign
    horizontal = img[:, 1:] * img[:, :-1] < 0   # same, for a pixel and the one to its right
    zc[1:, :] |= vertical      # mark the lower pixel of each pair
    zc[:-1, :] |= vertical     # mark the upper pixel too
    zc[:, 1:] |= horizontal    # mark the right pixel
    zc[:, :-1] |= horizontal   # mark the left pixel too
    return zc


def task1():
    """Task 1: Sobel and LoG, on the lecture array and then on a real crop."""
    ##  TODO
    ##    (a) apply sobel() to LECTURE_ARRAY, report the two 10x10 arrays
    ##    (b) apply sobel() to edge_patch.jpg (grayscale), threshold the
    ##        magnitude, show magnitude / direction / binary edge map
    ##    (c) build a LoG kernel, convolve with the same crop, find zero
    ##        crossings, compare with your Sobel edge map
    #task 1(a)
    mag, ang = sobel(LECTURE_ARRAY)
    np.set_printoptions(linewidth=200)
    print(f"Mag ={mag}")
    print(f"Ang ={ang}")
    #task1(b)
    edge_patch = load_image("edge_patch.jpg", gray=True)
    mag, ang = sobel(edge_patch)
    T = 60
    binary_edges = mag > T
    save_image(to_display(mag), "task1b_mag.jpg")
    save_image(to_display(ang), "task1b_ang.jpg")
    save_image(binary_edges * 255, "task1b_binary_edges.jpg")

        # Evidence for choosing T: magnitude statistics and a histogram
    print(f"mag min {mag.min():.1f}, max {mag.max():.1f}, median {np.median(mag):.1f}")
    plt.hist(mag.ravel(), bins=50)
    plt.yscale("log")
    plt.xlabel("gradient magnitude")
    plt.ylabel("pixel count (log scale)")
    plt.savefig(os.path.join(OUT_DIR, "task1b_hist.jpg"))
    plt.show()

    
    # Compare candidate thresholds: what fraction of pixels count as edges?
    candidates = [30, 60, 100, 150, 250]
    fig, axes = plt.subplots(1, len(candidates), figsize=(15, 3))
    for ax, t in zip(axes, candidates):
        ax.imshow(mag > t, cmap="gray")
        ax.set_title(f"T = {t}")
        ax.axis("off")
        print(f"T = {t}: {100 * np.mean(mag > t):.1f}% of pixels are edges")
    plt.savefig(os.path.join(OUT_DIR, "task1b_thresholds.jpg"))
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mag, cmap="gray")
    axes[0].set_title("Sobel magnitude")
    dir_im = axes[1].imshow(np.where(binary_edges, ang, np.nan),
                            cmap="hsv", vmin=0, vmax=360)
    axes[1].set_title("Direction (degrees), edges only")
    fig.colorbar(dir_im, ax=axes[1], fraction=0.046)
    axes[2].imshow(binary_edges, cmap="gray")
    axes[2].set_title(f"Edge map, T = {T}")
    for ax in axes:
        ax.axis("off")
    plt.savefig(os.path.join(OUT_DIR, "task1b_figure.jpg"), dpi=150)
    plt.show()

    #task1(c)
    sigma = 1.6
    log_k = log_kernel(sigma)
    log_response = convolve2d(edge_patch, log_k, mode="same", boundary="symm")
    zc = zero_crossings(log_response)
    save_image(to_display(log_response), "task1c_log_response.jpg")
    save_image(zc * 255, "task1c_zero_crossings.jpg")

    compare = np.zeros_like(edge_patch, dtype=int)
    compare[binary_edges & zc] = 2  # both
    compare[binary_edges & ~zc] = 1  # Sobel only
    compare[~binary_edges & zc] = 3  # LoG only
    # Colours 
    colors = ["black", "red", "white", "cyan"]
    names = ["neither", "Sobel only", "both", "LoG only"]
    plt.imshow(compare, cmap=ListedColormap(colors), vmin=0, vmax=3)
    handles = [Patch(facecolor=c, edgecolor="gray", label=n)
               for c, n in zip(colors, names)]
    plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.title(f"Sobel (T = {T}) vs LoG zero crossings")
    plt.axis("off")
    plt.savefig(os.path.join(OUT_DIR, "task1c_compare.jpg"),
                dpi=150, bbox_inches="tight")
    plt.show()


##  ============================================================================
##  Task 2: points, planes, lines, and quadrics in 3D
##  ============================================================================
##  All of these are given directly in the handout. Do not change them.

FACADE_A = (1500.0, 750.0, 0.0)
FACADE_B = (4000.0, 750.0, 0.0)
FACADE_C = (4000.0, 750.0, 1150.0)
FACADE_D = (1500.0, 750.0, 1150.0)          # checks your plane, not used to fit it

LINE_E = (3800.0, 500.0, 700.0)             # top of a streetlamp
LINE_F = (3900.0, 350.0, 550.0)             # corner of a parked van's roof

SPHERE_CENTER = (3400.0, 700.0, 500.0)
SPHERE_RADIUS = 60.0

##  Task 2(d): given directly, to 6 decimal places, matching the handout.
##  H_EUCLIDEAN is a genuine rigid motion (R^T R = I, det R = 1, last row
##  (0,0,0,1)). H_PROJECTIVE perturbs that same rotation block with noise
##  and has a nonzero last row, so it satisfies neither constraint.
H_EUCLIDEAN = np.array([
    [-0.040507, -0.999179,  0.000000, 223.330063],
    [-0.002698,  0.000109, -0.999996, 162.542525],
    [ 0.999176, -0.040507, -0.002700, 910.235470],
    [ 0.0,       0.0,       0.0,        1.0],
])
H_PROJECTIVE = np.array([
    [-0.191210, -1.119508,  0.018564, 223.330063],
    [-0.271551, -0.095732, -0.982529, 162.542525],
    [ 0.725102, -0.096783, -0.180126, 910.235470],
    [ 0.000210, -0.000140,  0.000300,   1.0],
])


def to_h(p):
    """A 3D point (X, Y, Z) as a homogeneous 4-vector."""
    return np.array([p[0], p[1], p[2], 1.0])


def unit_plane(pi):
    """Scale a plane so its first three entries have unit length."""
    return pi / np.linalg.norm(pi[:3])


def plane_through_3_points_svd(A, B, C):
    """The plane through 3 points, as the null vector of the 3 x 4 matrix
    whose rows are the points' homogeneous coordinates. Normalize so the
    returned vector's first three entries have unit norm.
    """
    ##  TODO: stack to_h(A), to_h(B), to_h(C); take the null vector by SVD.
    rows = np.vstack([to_h(A), to_h(B), to_h(C)])
    _, _, vt = np.linalg.svd(rows)
    return unit_plane(vt[-1])   # last row of V^T is the null vector
    #raise NotImplementedError


def plane_through_3_points_cofactor(A, B, C):
    """The same plane, by the explicit cofactor formula from Lecture 6 page
    6-2: Pi = (D234, -D134, D124, -D123), where D_ijk is the determinant of
    the 3x3 minor of [A;B;C] (as homogeneous row vectors) formed by columns
    i, j, k (1-indexed in the handout; 0-indexed here).
    """
    ##  TODO
    #raise NotImplementedError
    rows = np.vstack([to_h(A), to_h(B), to_h(C)])
    return np.array([
        np.linalg.det(rows[:, [1, 2, 3]]),     # D234
        -np.linalg.det(rows[:, [0, 2, 3]]),    # -D134
        np.linalg.det(rows[:, [0, 1, 3]]),     # D124
        -np.linalg.det(rows[:, [0, 1, 2]]),    # -D123
    ])

def line_dual_planes(A, B):
    """The line joining A and B, in its dual form: two planes P, Q that both
    contain the line, found as a basis for the 2D null space of the 2 x 4
    matrix W = [A^T; B^T] (by SVD).
    Returns:
        P, Q: two (4,) arrays
    """
    ##  TODO
    #raise NotImplementedError
    W = np.vstack([to_h(A), to_h(B)])
    _, _, vt = np.linalg.svd(W)
    return vt[-2], vt[-1]   # last two rows of V^T


def intersect_three_planes(P, Q, Pi):
    """The point common to three planes, as the null vector of the 3 x 4
    matrix they stack into. Return it with the 4th (homogeneous) entry
    normalized to 1.
    """
    ##  TODO
    #raise NotImplementedError
    rows = np.vstack([P, Q, Pi])
    _, _, vt = np.linalg.svd(rows)
    return vt[-1] / vt[-1][3]   # normalize the 4th entry to 1


def plucker_matrix(A, B):
    """The Plucker matrix of the line through A and B: L = A B^T - B A^T,
    using their homogeneous coordinates."""
    ##  TODO
    #raise NotImplementedError
    A_h = to_h(A).reshape(4, 1)  # column vector
    B_h = to_h(B).reshape(4, 1)  # column vector
    return A_h @ B_h.T - B_h @ A_h.T  # outer products


def build_euclidean_transform(R, C):
    """A 4x4 rigid-body (Euclidean) transform from a 3x3 rotation R and a
    camera center C: X_cam = R (X_world - C), assembled as a 4x4 matrix with
    last row (0, 0, 0, 1)."""
    ##  TODO
    #raise NotImplementedError
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = C
    return H


def build_projective_transform(R, C, last_row):
    """The same rotation and center, perturbed and given a nonzero last row
    `last_row` (a length-3 sequence), so the result is NOT a member of the
    Euclidean/affine subgroup."""
    ##  TODO
    #raise NotImplementedError
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = C
    H[3, :3] = last_row
    return H


def transform_plane(H, pi):
    """Pi' = H^{-T} Pi."""
    ##  TODO
    #raise NotImplementedError
    return np.linalg.inv(H).T @ pi

def transform_quadric(H, Q):
    """Q' = H^{-T} Q H^{-1}."""
    ##  TODO
    #raise NotImplementedError
    H_inv = np.linalg.inv(H)
    return H_inv.T @ Q @ H_inv

def sphere_quadric(center, radius):
    """The 4x4 point-quadric matrix of a sphere of the given radius and
    center: Q = [[I, -O], [-O^T, O.O - r^2]]."""
    ##  TODO
    #raise NotImplementedError
    O = np.array(center, dtype=float)
    Q = np.eye(4)
    Q[:3, 3] = -O
    Q[3, :3] = -O
    Q[3, 3] = O @ O - radius**2
    return Q


def task2():
    """Task 2: the facade plane, the pierced line, the Plucker matrix, and
    the transformation-group / quadric comparison."""
    ##  TODO
    ##    (a) plane_through_3_points_svd and _cofactor on FACADE_A/B/C; check
    ##        against FACADE_D
    ##    (b) line_dual_planes on LINE_E, LINE_F; intersect with the facade
    ##        plane from (a); verify against all three planes
    ##    (c) plucker_matrix on LINE_E, LINE_F; check det == 0 and rank == 2
    ##    (d) H_EUCLIDEAN and H_PROJECTIVE above are given directly, not
    ##        built -- state their DOF and the constraint that distinguishes
    ##        them (R^T R = I, det R = 1, and the last row, not the last row
    ##        alone); transform_plane(H, pi_infinity) for pi_infinity =
    ##        (0,0,0,1) by both; transform_quadric(H, sphere_quadric(
    ##        SPHERE_CENTER, SPHERE_RADIUS)) by both and look at the
    ##        eigenvalues of the upper-left 3x3 block of each result.
    ##        (build_euclidean_transform / build_projective_transform above
    ##        are optional -- e.g. to double-check H_EUCLIDEAN really does
    ##        come from a rotation and center the way the handout says.)
    #"""
    #task 2(a)
    pi_svd = plane_through_3_points_svd(FACADE_A, FACADE_B, FACADE_C)
    pi_cof = plane_through_3_points_cofactor(FACADE_A, FACADE_B, FACADE_C)
    print("SVD      :", pi_svd)
    print("cofactor :", pi_cof)
    print("cofactor, unit :", unit_plane(pi_cof))
    same = (np.allclose(pi_svd, unit_plane(pi_cof))
        or np.allclose(pi_svd, -unit_plane(pi_cof)))
    print("agree up to scale and sign:", same)
    print("pi . D =", to_h(FACADE_D) @ pi_svd)

    #task 2(b)
    
    P, Q = line_dual_planes((0, 0, 0), (1, 1, 1))
    W = np.array([to_h((0, 0, 0)), to_h((1, 1, 1))])
    print("W @ P :", W @ P)
    print("W @ Q :", W @ Q)
    pt = intersect_three_planes(P, Q, np.array([0.0, 0.0, 1.0, -3.0]))
    print("toy pierce point :", pt)
    P, Q = line_dual_planes(LINE_E, LINE_F)
    pt = intersect_three_planes(P, Q, pi_svd)
    print("pierce point :", pt)
    print("P, Q, pi checks :", P @ pt, Q @ pt, pi_svd @ pt)

    d = np.array(LINE_F) - np.array(LINE_E)
    t = (pt[:3] - np.array(LINE_E)) / d
    print("t per coordinate:", t)
    

    #task 2(c)
    L = plucker_matrix(LINE_E, LINE_F)
    print("det:", np.linalg.det(L))
    print("rank:", np.linalg.matrix_rank(L))
    print("singular values:", np.linalg.svd(L)[1])
    #"""

    #task 2(d)
    pi_inf = np.array([0.0, 0.0, 0.0, 1.0])
    pi_e = transform_plane(H_EUCLIDEAN, pi_inf)
    pi_p = transform_plane(H_PROJECTIVE, pi_inf)
    print("H_EUCLIDEAN . pi_infinity  :", pi_e)
    print("H_PROJECTIVE . pi_infinity :", pi_p)




##  ============================================================================
##  Task 3: homography, rectification, and the anamorphic banner
##  ============================================================================
##  Record these yourself from view_a.jpg and view_b.jpg. Same order in both.
RECORDED_BOX_A = [(980, 661), (1018, 672), (749, 673), (751, 662)]     # TODO
RECORDED_BOX_B = [(757, 688), (732, 712), (347, 708), (457, 686)]    # TODO: same 4 corners, view_b
RECORDED_TREE_P_A = (678, 600)                             # TODO
RECORDED_TREE_P_B = (500, 598)                             # TODO
RECORDED_TREE_Q_A = (900, 596)                             # TODO
RECORDED_TREE_Q_B = (882, 598)                             # TODO

##  Global ground-plane coordinates, same order as RECORDED_BOX_A/B above.
##  Do NOT re-origin this to (0,0),(400,0),(400,600),(0,600): the (optional)
##  Bonus problem rebuilds this same rectification from view_b and compares,
##  which only works if both use this same global frame.
BOX_WORLD = [(1000.0, -300.0), (1400.0, -300.0), (1400.0, 300.0), (1000.0, 300.0)]

RECORDED_BANNER_B = [(381, 264), (526, 357), (529, 489), (384, 444)]   # TODO: 4 corners, view_b
BANNER_ASPECT = 2.0

def normalize_points(pts):
    """Shift pts to be centered at the origin and scaled so their average
    distance from the origin is sqrt(2). Returns the normalized points and
    the 3x3 matrix T such that T @ [x, y, 1] gives the normalized point.
    """
    pts = np.asarray(pts, dtype=float)
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0,     -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,     0,      1],
    ])
    ones = np.ones((len(pts), 1))
    hom = np.hstack([pts, ones])
    normalized = (T @ hom.T).T[:, :2]
    return normalized, T


def build_system(src, dst):
    """Two rows per correspondence, h33 = 1. Carried over from HW2/HW3."""
    ##  TODO
    #raise NotImplementedError
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = src[i]
        xp, yp = dst[i]
        A[2 * i]     = [x, y, 1, 0, 0, 0, -x * xp, -y * xp]
        b[2 * i]     = xp
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * yp, -y * yp]
        b[2 * i + 1] = yp
    return A, b


def estimate_homography(src, dst):
    """3 x 3 homography carrying src onto dst, h33 = 1. Carried over."""
    ##  TODO
    #raise NotImplementedError
    src_n, T_src = normalize_points(src)
    dst_n, T_dst = normalize_points(dst)
    A, b = build_system(src_n, dst_n)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    H_n = np.append(h, 1.0).reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_n @ T_src   # undo the normalization
    return H / H[2, 2]          

def apply_homography(H, pts):
    """Send (x, y) points through H, dividing by the third coordinate."""
    ##  TODO
    #raise NotImplementedError
    pts = np.asarray(pts, dtype=float)
    single = pts.ndim == 1
    if single:
        pts = pts[None, :]
    ones = np.ones((len(pts), 1))
    hom = np.hstack([pts, ones])          # (N, 3)
    out = (H @ hom.T).T                   # (N, 3)
    out = out[:, :2] / out[:, 2:3]
    return out[0] if single else out

def warp_rectangle(photo, H, out_w, out_h):
    """Inverse-map `photo` into an out_h x out_w canvas through H, where H
    carries a canvas pixel to the photo pixel it came from. Carried over
    from HW2/HW3's inverse-mapping warp."""
    ##  TODO
    #raise NotImplementedError
    photo = np.asarray(photo)
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    canvas_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    src_pts = apply_homography(H, canvas_pts)          # where each canvas pixel comes from
    src_x = src_pts[:, 0].reshape(out_h, out_w)
    src_y = src_pts[:, 1].reshape(out_h, out_w)

    h_img, w_img = photo.shape[:2]
    valid = (src_x >= 0) & (src_x <= w_img - 1) & (src_y >= 0) & (src_y <= h_img - 1)
    src_x_c = np.clip(np.round(src_x).astype(int), 0, w_img - 1)
    src_y_c = np.clip(np.round(src_y).astype(int), 0, h_img - 1)

    if photo.ndim == 3:
        out = photo[src_y_c, src_x_c, :]
        out[~valid] = 0
    else:
        out = photo[src_y_c, src_x_c]
        out[~valid] = 0
    return out

def rectangle_of_aspect(aspect, width=400):
    """Four corners of a rectangle with the given width:height aspect ratio,
    ordered the same way as a recorded quad (e.g. TL, TR, BR, BL)."""
    height = width / aspect
    return [(0, 0), (width, 0), (width, height), (0, height)]


def task3():
    """Task 3: recorded correspondences, metric rectification, and the
    banner. (The optional Bonus problem -- what happens when a homography
    built for one plane is pointed at a pixel from a different plane --
    is in bonus() below; it is not required for Task 3's 58 points.)"""
    ##  TODO
    ##    (a) nothing to compute -- just make sure RECORDED_BOX_A/B and
    ##        RECORDED_TREE_P/Q_A/B above are filled in and consistently
    ##        ordered between the two views
    ##    (b) estimate_homography(RECORDED_BOX_A, BOX_WORLD); apply to the
    ##        two recorded tree points; report the distance between them
    ##    (c) estimate_homography(rectangle_of_aspect(BANNER_ASPECT),
    ##        RECORDED_BANNER_B); warp_rectangle(); report what it says
    ##  (a) nothing to compute -- correspondences are the RECORDED_* values
    ##  above; just make sure they're filled in and consistently ordered.

    ##  (b) ground-plane rectification
    H_rect = estimate_homography(RECORDED_BOX_A, BOX_WORLD)
    p_world = apply_homography(H_rect, RECORDED_TREE_P_A)
    q_world = apply_homography(H_rect, RECORDED_TREE_Q_A)
    dist_pq = np.linalg.norm(p_world - q_world)
    print("H_rect:\n", H_rect)
    print("Tree P in world coords:", p_world)
    print("Tree Q in world coords:", q_world)
    print("Distance between P and Q (cm):", dist_pq)

    ##  (c) banner rectification
    banner_dst = rectangle_of_aspect(BANNER_ASPECT, width=400)
    H_banner = estimate_homography(banner_dst, RECORDED_BANNER_B)
    view_b_img = load_image("view_b.jpg")
    out_w, out_h = 400, int(400 / BANNER_ASPECT)
    banner_rect = warp_rectangle(view_b_img, H_banner, out_w, out_h)
    save_image(banner_rect, "task3c_banner.jpg")
    plt.figure()
    plt.imshow(np.uint8(banner_rect))
    plt.title("Rectified banner")
    plt.axis("off")
    plt.savefig(os.path.join(OUT_DIR, "task3c_banner_figure.jpg"))
    plt.show()

    return H_rect, H_banner


def bonus():
    """Optional, +15 points, does not affect Task 3: builds a
    second ground-plane homography from view_b and compares it against
    Task 3(b)'s, then applies it to a banner corner to show what goes
    wrong when a homography built for one plane is pointed at a pixel
    from a different plane."""
    ##  TODO
    ##    (i) estimate_homography(RECORDED_BOX_B, BOX_WORLD) -- the SAME
    ##        box and world coordinates as Task 3(b), but from view_b
    ##        instead of view_a. Apply it to the two recorded tree points
    ##        in view_b and compare to Task 3(b)'s results. They need not
    ##        match exactly -- both trees sit well outside the box's own
    ##        footprint, so this is an extrapolated homography and can be
    ##        very sensitive to pixel precision (this is expected, not a
    ##        bug, and there is no required magnitude -- report what you
    ##        actually get). Report the difference between the two P-Q
    ##        distances, and separately how far apart the two views' P's
    ##        are and the two Q's are. Comment on why a point far from the
    ##        calibration box is generally more exposed to this than one
    ##        close to it.
    ##    (ii) apply that same part-(i) homography to one of
    ##        RECORDED_BANNER_B's four points (the banner corner is not on
    ##        the ground plane this homography was built from). Report the
    ##        resulting (X, Y) and explain, using the facade plane from Task
    ##        2(a), why it is not a valid location for that banner corner.
    ##    (iii) answer in text: could one homography have related every
    ##        point in the scene, on the ground and on the facade alike?
    ##  (i) ground-plane homography from view_b instead of view_a
    H_rect_b = estimate_homography(RECORDED_BOX_B, BOX_WORLD)
    p_world_b = apply_homography(H_rect_b, RECORDED_TREE_P_B)
    q_world_b = apply_homography(H_rect_b, RECORDED_TREE_Q_B)
    dist_pq_b = np.linalg.norm(p_world_b - q_world_b)

    H_rect_a = estimate_homography(RECORDED_BOX_A, BOX_WORLD)
    p_world_a = apply_homography(H_rect_a, RECORDED_TREE_P_A)
    q_world_a = apply_homography(H_rect_a, RECORDED_TREE_Q_A)
    dist_pq_a = np.linalg.norm(p_world_a - q_world_a)

    print("P-Q distance from view_a:", dist_pq_a)
    print("P-Q distance from view_b:", dist_pq_b)
    print("Difference:", abs(dist_pq_a - dist_pq_b))
    print("How far apart the two P estimates are:",
          np.linalg.norm(p_world_a - p_world_b))
    print("How far apart the two Q estimates are:",
          np.linalg.norm(q_world_a - q_world_b))

    ##  (ii) apply H_rect_b to a banner corner (which is NOT on the ground plane)
    banner_corner_on_ground = apply_homography(H_rect_b, RECORDED_BANNER_B[0])
    print("Banner corner mapped onto ground plane (X, Y):", banner_corner_on_ground)
    print("This is where the camera ray hits Z=0, not the banner corner's "
          "real 3D location, since the banner corner lies on the facade "
          "plane (Y=750) from Task 2(a), not on the ground (Z=0).")

    return H_rect_a, H_rect_b

##  --------------------------------------------------------------------------
def main():
    task1()
    task2()
    task3()
    bonus()     # comment out if you skip the Bonus problem
    
    A, _ = build_system(RECORDED_BOX_A, BOX_WORLD)
    print("condition number, view_a box:", np.linalg.cond(A))
    A, _ = build_system(RECORDED_BOX_B, BOX_WORLD)
    print("condition number, view_b box:", np.linalg.cond(A))
    src_n, T_src = normalize_points(RECORDED_BOX_A)
    dst_n, T_dst = normalize_points(BOX_WORLD)   # <- this line is missing from your code
    A_n, _ = build_system(src_n, dst_n)
    print("condition number, view_a, normalized:", np.linalg.cond(A_n))
    
if __name__ == '__main__':
    main()
