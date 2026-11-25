#!/usr/bin/env python3
"""
Numerical Solver & PINN Suite for HJR Monograph (First Principles to Neural Reachability)
Uses strictly monotone Osher-Sethian / Godunov upwind schemes (L_inf stable, zero artificial blowup):
  1. 2D Phase-Plane HJI-VI (1D Drone vs Wall + Bounded Pillar BRS vs BRT)
  2. 3D Fixed-Wing / Dubins Air3D HJI-VI (x, y, psi) Twisted Reachable Tube
  3. 4D Planar Drone Arena Avoid HJI-VI + Per-Obstacle Union Under-Approximation
  4. 4D Precision Dock Landing Reach-Avoid HJI-VI (x, z, vx, vz)
  5. 3D FaSTrack Quadrotor-With-Pitch-Lag Tracking Error Bound (e_p, v, theta)
  6. DeepReach SIREN PINN vs Steady-State Collapse + Lin & Bansal Conformal Verification
  7. Cluttered 45m x 45m Multi-Obstacle Navigation Benchmarks

Outputs both `data.json` and `data.js` (zero-CORS file:// compatible).
"""

import base64
import json
import os
import time
import numpy as np
from scipy.stats import beta as beta_dist

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def b64_quant_u8(arr, vmin, vmax):
    q = np.clip(np.round((arr - vmin) / (vmax - vmin) * 255.0), 0, 255).astype(np.uint8)
    return {
        "b64": base64.b64encode(q.tobytes()).decode("ascii"),
        "min": float(vmin),
        "max": float(vmax),
        "shape": list(arr.shape),
    }


def b64_i16(arr, scale=500.0):
    q = np.clip(np.round(arr * scale), -32767, 32767).astype("<i2")
    return {
        "b64": base64.b64encode(q.tobytes()).decode("ascii"),
        "scale": float(scale),
        "shape": list(arr.shape),
    }


def one_sided(V, h, axis, periodic=False, slope_lo=0.0, slope_hi=0.0):
    """
    Monotone one-sided differences Dm = (V_i - V_{i-1})/h, Dp = (V_{i+1} - V_i)/h
    using np.diff + boundary slopes (0.0 = clamped Neumann boundary, strictly L_inf contractive).
    """
    if periodic:
        Dm = (V - np.roll(V, 1, axis=axis)) / h
        Dp = (np.roll(V, -1, axis=axis) - V) / h
        return Dm, Dp
    d = np.diff(V, axis=axis) / h
    pad_lo = [(0, 0)] * V.ndim
    pad_lo[axis] = (1, 0)
    pad_hi = [(0, 0)] * V.ndim
    pad_hi[axis] = (0, 1)
    Dm = np.pad(d, pad_lo, mode="constant", constant_values=slope_lo)
    Dp = np.pad(d, pad_hi, mode="constant", constant_values=slope_hi)
    return Dm, Dp


def upwind_advect(vel, Dm, Dp):
    """Monotone upwind for vel * dV/dx in backward-time PDE dV/dtau = H."""
    return np.where(vel >= 0.0, vel * Dp, vel * Dm)


def godunov_max_abs(Dm, Dp):
    """Osher-Sethian Godunov upwind for +|dV/dx| (maximizer u in [-1, 1])."""
    return np.maximum(np.maximum(Dp, -Dm), 0.0)


def godunov_min_abs(Dm, Dp):
    """Osher-Sethian Godunov upwind for -|dV/dx| (minimizer d in [-1, 1])."""
    return -np.maximum(np.maximum(-Dp, Dm), 0.0)


# =====================================================================
# 1. 1D Drone (2D State: p, v) — Exact vs Osher-Sethian Godunov PDE
# =====================================================================
def run_1d_benchmarks():
    t0 = time.perf_counter()
    p_wall, a_max, d_max = 10.0, 3.0, 0.5
    p_grid = np.linspace(-2.0, 12.0, 141)
    v_grid = np.linspace(-4.0, 8.0, 121)
    dp, dv = p_grid[1] - p_grid[0], v_grid[1] - v_grid[0]
    PP, VV = np.meshgrid(p_grid, v_grid, indexing="ij")
    L = p_wall - PP

    ae = a_max - d_max
    dt = 0.90 / (np.abs(v_grid).max() / dp + ae / dv)
    tau_max = 4.0
    V = L.copy()
    tau = 0.0
    while tau < tau_max - 1e-9:
        step = min(dt, tau_max - tau)
        # l(p) = 10 - p has exact slope -1 along p!
        Dpm, Dpp = one_sided(V, dp, 0, slope_lo=-1.0, slope_hi=-1.0)
        Dvm, Dvp = one_sided(V, dv, 1, slope_lo=0.0, slope_hi=0.0)
        H = upwind_advect(VV, Dpm, Dpp) + ae * godunov_max_abs(Dvm, Dvp)
        V = np.minimum(L, V + step * H)
        tau += step

    vp = np.maximum(VV, 0.0)
    t_stop = np.minimum(vp / ae, tau_max)
    V_true = L - (vp * t_stop - 0.5 * ae * t_stop**2)

    inside = (PP >= 0) & (PP <= 11) & (VV >= -3) & (VV <= 7)
    ut, uh = (V_true < 0) & inside, (V < 0) & inside
    iou = float((ut & uh).sum() / (ut | uh).sum())
    max_err = float(np.abs(V - V_true)[inside].max())
    false_safe = float((ut & ~uh).sum() / inside.sum())
    elapsed = time.perf_counter() - t0
    print(f"[1D PDE] solved in {elapsed:.3f}s | IoU={iou:.4f} | max_err={max_err:.3f}m | false_safe={false_safe:.4f}")
    return {
        "p_wall": p_wall,
        "a_max": a_max,
        "d_max": d_max,
        "grid_shape": [len(p_grid), len(v_grid)],
        "pde_time_s": round(elapsed, 4),
        "iou": round(iou, 4),
        "max_err": round(max_err, 4),
        "false_safe": round(false_safe, 5),
    }


# =====================================================================
# 2. 3D Fixed-Wing / Dubins Air3D HJI-VI (x, y, psi)
# =====================================================================
def solve_dubins_3d():
    """
    Fixed-wing / cruising VTOL drone avoiding a cylindrical obstacle at (0,0) with R=1.6m.
    Constant airspeed v=5.0 m/s (cannot stop!), max turn rate w_max=1.25 rad/s,
    adversarial wind gust ||d|| <= 0.75 m/s.
    """
    t0 = time.perf_counter()
    R_obs = 1.6
    v_air = 5.0
    w_max = 1.25
    d_wind = 0.75

    nx, ny, npsi = 49, 49, 36
    xs = np.linspace(-8.5, 5.5, nx)
    ys = np.linspace(-7.0, 7.0, ny)
    psis = np.linspace(-np.pi, np.pi, npsi, endpoint=False)
    hx, hy, hp = xs[1] - xs[0], ys[1] - ys[0], psis[1] - psis[0]

    X, Y, PSI = np.meshgrid(xs, ys, psis, indexing="ij")
    L = (np.hypot(X, Y) - R_obs).astype(np.float32)

    vx_nom = (v_air * np.cos(PSI)).astype(np.float32)
    vy_nom = (v_air * np.sin(PSI)).astype(np.float32)

    dt = 0.88 / ((v_air + d_wind) / hx + (v_air + d_wind) / hy + w_max / hp)
    tau_max = 2.0
    n_steps = int(np.ceil(tau_max / dt))
    dt = tau_max / n_steps

    V = L.copy()
    for _ in range(n_steps):
        # Far from obstacle, signed distance grows outward at ~1
        Dxm, Dxp = one_sided(V, hx, 0, slope_lo=-1.0, slope_hi=1.0)
        Dym, Dyp = one_sided(V, hy, 1, slope_lo=-1.0, slope_hi=1.0)
        Dpm, Dpp = one_sided(V, hp, 2, periodic=True)

        adv = upwind_advect(vx_nom, Dxm, Dxp) + upwind_advect(vy_nom, Dym, Dyp)
        turn = w_max * godunov_max_abs(Dpm, Dpp)
        gx_min = godunov_min_abs(Dxm, Dxp)
        gy_min = godunov_min_abs(Dym, Dyp)
        wind = -d_wind * np.sqrt(gx_min**2 + gy_min**2)

        V = np.minimum(L, V + dt * (adv + turn + wind))

    elapsed = time.perf_counter() - t0
    brt_frac = float((V < 0).mean())
    obs_frac = float((L < 0).mean())
    print(
        f"[3D Dubins] {V.size} cells, {n_steps} steps in {elapsed:.2f}s | "
        f"obs={100*obs_frac:.1f}% -> BRT={100*brt_frac:.1f}% | range=[{V.min():.2f}, {V.max():.2f}]"
    )
    return {
        "xs": [float(xs[0]), float(xs[-1]), nx],
        "ys": [float(ys[0]), float(ys[-1]), ny],
        "npsi": npsi,
        "R_obs": R_obs,
        "v_air": v_air,
        "w_max": w_max,
        "d_wind": d_wind,
        "r_turn": round(v_air / w_max, 2),
        "tau_max": tau_max,
        "solve_s": round(elapsed, 3),
        "obs_frac": round(obs_frac, 4),
        "brt_frac": round(brt_frac, 4),
        "V": b64_quant_u8(V, -2.0, 5.5),
    }


# =====================================================================
# 3. 4D Planar Drone Arena HJI-VI + Per-Obstacle Union Comparison
# =====================================================================
ARENA = 24.0
PILLARS = np.array([
    [8.0, 9.0, 1.5],
    [15.5, 14.0, 1.5],
    [10.5, 17.5, 1.1],
    [17.5, 6.5, 1.25],
])
A2, D2, VMAX = 3.0, 0.5, 4.0


def l_arena(px, py, pillars=PILLARS, walls=True):
    ds = [px, ARENA - px, py, ARENA - py] if walls else []
    for cx, cy, r in pillars:
        ds.append(np.hypot(px - cx, py - cy) - r)
    return np.min(np.stack(ds, axis=0), axis=0)


def solve_4d_grid(L, xs, vs, tau_max=3.0, a=A2, d=D2):
    hx, hv = xs[1] - xs[0], vs[1] - vs[0]
    ae = a - d
    dt = 0.88 / (2.0 * VMAX / hx + 2.0 * ae / hv)
    n_steps = int(np.ceil(tau_max / dt))
    dt = tau_max / n_steps
    VX = vs[None, None, :, None].astype(np.float32)
    VY = vs[None, None, None, :].astype(np.float32)
    V = L.astype(np.float32).copy()
    for _ in range(n_steps):
        Dxm, Dxp = one_sided(V, hx, 0, slope_lo=1.0, slope_hi=-1.0)
        Dym, Dyp = one_sided(V, hx, 1, slope_lo=1.0, slope_hi=-1.0)
        Dum, Dup = one_sided(V, hv, 2)
        Dwm, Dwp = one_sided(V, hv, 3)

        adv = upwind_advect(VX, Dxm, Dxp) + upwind_advect(VY, Dym, Dyp)
        gu = godunov_max_abs(Dum, Dup)
        gw = godunov_max_abs(Dwm, Dwp)
        H = adv + ae * np.sqrt(gu**2 + gw**2)
        V = np.minimum(L, V + dt * H)
    return V, n_steps


def build_arena_4d():
    t0 = time.perf_counter()
    xs = np.linspace(0.0, ARENA, 61)
    vs = np.linspace(-VMAX, VMAX, 17)
    X4, Y4, _, _ = np.meshgrid(xs, xs, vs, vs, indexing="ij")
    L4 = l_arena(X4, Y4).astype(np.float32)
    V4, n_steps = solve_4d_grid(L4, xs, vs)
    t_joint = time.perf_counter() - t0

    t1 = time.perf_counter()
    V_parts = [solve_4d_grid(l_arena(X4, Y4, pillars=np.zeros((0, 3)), walls=True), xs, vs)[0]]
    for i in range(len(PILLARS)):
        V_parts.append(solve_4d_grid(l_arena(X4, Y4, pillars=PILLARS[i : i + 1], walls=False), xs, vs)[0])
    V_union = np.min(np.stack(V_parts, axis=0), axis=0)
    missed_mask = (V4 < 0) & (V_union >= 0)
    missed_idx = np.argwhere(missed_mask).tolist()
    print(
        f"[4D Arena] {V4.size} cells in {t_joint:.2f}s | joint BRT={100*(V4<0).mean():.2f}% | "
        f"union missed={len(missed_idx)} states ({time.perf_counter()-t1:.1f}s) | range=[{V4.min():.2f}, {V4.max():.2f}]"
    )
    return V4, xs, vs, {
        "xs": [float(xs[0]), float(xs[-1]), len(xs)],
        "vs": [float(vs[0]), float(vs[-1]), len(vs)],
        "arena": ARENA,
        "pillars": PILLARS.tolist(),
        "a_max": A2,
        "d_max": D2,
        "vmax": VMAX,
        "steps": n_steps,
        "solve_s": round(t_joint, 2),
        "joint_brt_frac": round(float((V4 < 0).mean()), 4),
        "union_brt_frac": round(float((V_union < 0).mean()), 4),
        "missed_count": len(missed_idx),
        "missed_idx": missed_idx[:900],
        "V": b64_quant_u8(V4, -2.0, 6.0),
    }


# =====================================================================
# 4. Precision Dock Landing Reach-Avoid (4D: x, z, vx, vz)
# =====================================================================
def solve_dock_reach_avoid():
    t0 = time.perf_counter()
    nx, nz, nvx, nvz = 33, 33, 17, 17
    xs = np.linspace(-6.0, 6.0, nx)
    zs = np.linspace(0.0, 9.6, nz)
    vxs = np.linspace(-3.2, 3.2, nvx)
    vzs = np.linspace(-3.8, 1.8, nvz)
    hx, hz, hvx, hvz = xs[1] - xs[0], zs[1] - zs[0], vxs[1] - vxs[0], vzs[1] - vzs[0]

    X, Z, VX, VZ = np.meshgrid(xs, zs, vxs, vzs, indexing="ij")

    # Safe touchdown box: |x| <= 0.95m, 0 <= z <= 0.65m, |vx| <= 0.95 m/s, -1.4 <= vz <= 0.35 m/s
    r_target = np.minimum.reduce([
        0.95 - np.abs(X),
        0.65 - Z,
        Z - 0.0,
        0.95 - np.abs(VX),
        0.875 - np.abs(VZ + 0.525),
    ]).astype(np.float32)

    # Failure complement l_safe >= 0:
    # Deck strike outside pad, dock coaming walls, hard vertical slam, or arena walls
    deck_outside = np.maximum(Z - 0.30, 0.95 - np.abs(X))
    coaming = np.maximum(Z - 1.25, np.abs(np.abs(X) - 1.45) - 0.38)
    slam = np.maximum(Z - 0.35, VZ - (-1.45))
    arena_bounds = np.minimum(5.7 - np.abs(X), 9.3 - Z)
    l_safe = np.minimum.reduce([deck_outside, coaming, slam, arena_bounds]).astype(np.float32)

    ax_max, az_max = 3.0, 3.6
    dx_max, dz_max = 0.60, 0.50
    aex, aez = ax_max - dx_max, az_max - dz_max

    dt = 0.88 / (3.2 / hx + 3.8 / hz + aex / hvx + aez / hvz)
    tau_max = 3.8
    n_steps = int(np.ceil(tau_max / dt))
    dt = tau_max / n_steps

    V = np.minimum(l_safe, r_target).copy()
    for _ in range(n_steps):
        Dxm, Dxp = one_sided(V, hx, 0)
        Dzm, Dzp = one_sided(V, hz, 1)
        Duxm, Duxp = one_sided(V, hvx, 2)
        Duzm, Duzp = one_sided(V, hvz, 3)

        H = (
            upwind_advect(VX, Dxm, Dxp)
            + upwind_advect(VZ, Dzm, Dzp)
            + aex * godunov_max_abs(Duxm, Duxp)
            + aez * godunov_max_abs(Duzm, Duzp)
        )
        V = np.minimum(l_safe, np.maximum(r_target, V + dt * H))

    elapsed = time.perf_counter() - t0
    viable_frac = float((V >= 0).mean())
    print(
        f"[4D Dock Reach-Avoid] {V.size} cells, {n_steps} steps in {elapsed:.2f}s | "
        f"viable funnel={100*viable_frac:.1f}% | range=[{V.min():.2f}, {V.max():.2f}]"
    )
    return {
        "xs": [float(xs[0]), float(xs[-1]), nx],
        "zs": [float(zs[0]), float(zs[-1]), nz],
        "vxs": [float(vxs[0]), float(vxs[-1]), nvx],
        "vzs": [float(vzs[0]), float(vzs[-1]), nvz],
        "ax_max": ax_max,
        "az_max": az_max,
        "dx_max": dx_max,
        "dz_max": dz_max,
        "solve_s": round(elapsed, 2),
        "viable_frac": round(viable_frac, 4),
        "V": b64_quant_u8(V, -2.0, 1.5),
    }


# =====================================================================
# 5. FaSTrack 3D Relative Subsystem (e_p, v, theta) for Lagged Quadrotor
# =====================================================================
def solve_fastrack_teb():
    """
    FaSTrack relative tracking game (Herbert et al., CDC 2017):
      Tracker (quadrotor): dot(p) = v, dot(v) = g * tan(theta) + d_wind, dot(theta) = omega
      Planner (virtual):   dot(p_hat) = v_hat,  |v_hat| <= v_plan
      Relative state:      (e, v, theta)
    """
    t0 = time.perf_counter()
    g = 9.81
    v_plan = 1.6
    d_wind = 0.4
    w_max = 6.0
    th_lim = 0.45  # ~25.8 deg -> max accel ~4.74 m/s^2

    ne, nv, nth = 55, 45, 31
    es = np.linspace(-2.5, 2.5, ne)
    vs = np.linspace(-4.5, 4.5, nv)
    ths = np.linspace(-th_lim, th_lim, nth)
    he, hv, hth = es[1] - es[0], vs[1] - vs[0], ths[1] - ths[0]

    E, V_vel, TH = np.meshgrid(es, vs, ths, indexing="ij")
    R_ref = 2.0
    L = (R_ref - np.abs(E)).astype(np.float32)

    acc_th = (g * np.tan(TH)).astype(np.float32)
    alpha_e = 4.5 + v_plan
    alpha_v = float(g * np.tan(th_lim) + d_wind)
    dt = 0.86 / (alpha_e / he + alpha_v / hv + w_max / hth)
    tau_max = 2.4
    n_steps = int(np.ceil(tau_max / dt))
    dt = tau_max / n_steps

    V = L.copy()
    for _ in range(n_steps):
        Dem, Dep = one_sided(V, he, 0, slope_lo=1.0, slope_hi=-1.0)
        Dvm, Dvp = one_sided(V, hv, 1)
        Dtm, Dtp = one_sided(V, hth, 2)

        # Tracker pitch rate omega in [-w_max, w_max], constrained inward at +/- th_lim
        ctrl_th = np.where(
            TH >= th_lim - 1e-4,
            w_max * np.maximum(-Dtm, 0.0),
            np.where(TH <= -th_lim + 1e-4, w_max * np.maximum(Dtp, 0.0), w_max * godunov_max_abs(Dtm, Dtp)),
        )

        H = (
            upwind_advect(V_vel, Dem, Dep)
            + v_plan * godunov_min_abs(Dem, Dep)
            + upwind_advect(acc_th, Dvm, Dvp)
            + d_wind * godunov_min_abs(Dvm, Dvp)
            + ctrl_th
        )
        V = np.minimum(L, V + dt * H)

    # Worst-case future tracking error M(e, v, theta) = R_ref - V(e, v, theta)
    M = R_ref - V
    teb_min = float(M.min())
    teb_origin = float(M[ne // 2, nv // 2, nth // 2])
    elapsed = time.perf_counter() - t0
    print(
        f"[3D FaSTrack] {V.size} cells in {elapsed:.2f}s | "
        f"min TEB={teb_min:.2f}m, from hover={teb_origin:.2f}m | M range=[{M.min():.2f}, {M.max():.2f}]"
    )
    return {
        "es": [float(es[0]), float(es[-1]), ne],
        "vs": [float(vs[0]), float(vs[-1]), nv],
        "ths": [float(ths[0]), float(ths[-1]), nth],
        "v_plan": v_plan,
        "d_wind": d_wind,
        "w_max": w_max,
        "th_lim": th_lim,
        "teb_m": round(max(teb_min, teb_origin) + 0.06, 2),
        "solve_s": round(elapsed, 2),
        "M": b64_quant_u8(M, 0.0, 3.5),
    }


# =====================================================================
# 6. PINN Training (DeepReach SIREN vs Steady Ablation) + Conformal Verification
# =====================================================================
class SirenPINN1D:
    def __init__(self, width=40, seed=42, w0=3.5, steady=False):
        rng = np.random.default_rng(seed)
        self.steady = steady
        self.w0 = np.float32(w0)
        in_dim = 2 if steady else 3
        self.W1 = rng.uniform(-1.0 / in_dim, 1.0 / in_dim, (in_dim, width)).astype(np.float32)
        self.b1 = np.zeros(width, dtype=np.float32)
        lim2 = np.sqrt(6.0 / width) / w0
        self.W2 = rng.uniform(-lim2, lim2, (width, width)).astype(np.float32)
        self.b2 = np.zeros(width, dtype=np.float32)
        self.W3 = rng.uniform(-lim2, lim2, (width, 1)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step_idx = 0

    def forward_and_grads(self, p, vel, tau=None):
        w0 = self.w0
        if self.steady:
            z = np.stack([(p - 5.0) / 7.0, (vel - 2.0) / 6.0], axis=1)
        else:
            z = np.stack([(p - 5.0) / 7.0, (vel - 2.0) / 6.0, 2.0 * tau / 4.0 - 1.0], axis=1)

        a1 = w0 * (z @ self.W1 + self.b1)
        h1 = np.sin(a1)
        c1 = w0 * np.cos(a1)

        a2 = w0 * (h1 @ self.W2 + self.b2)
        h2 = np.sin(a2)
        c2 = w0 * np.cos(a2)

        out = (h2 @ self.W3 + self.b3)[:, 0]

        q2 = self.W3[:, 0][None, :] * c2
        q1 = (q2 @ self.W2.T) * c1
        dout_dz = q1 @ self.W1.T

        if self.steady:
            V = 5.0 * out
            Vp = 5.0 * dout_dz[:, 0] / 7.0
            Vv = 5.0 * dout_dz[:, 1] / 6.0
            Vt = np.zeros_like(V)
        else:
            scale = 5.0 * (tau / 4.0)
            V = (10.0 - p) + scale * out
            Vp = -1.0 + scale * (dout_dz[:, 0] / 7.0)
            Vv = scale * (dout_dz[:, 1] / 6.0)
            Vt = (5.0 / 4.0) * out + scale * (dout_dz[:, 2] * 0.5)

        cache = (p, vel, tau, z, a1, h1, c1, a2, h2, c2, out, q2, q1)
        return V, Vp, Vv, Vt, cache

    def train_step(self, p, vel, tau, lr=2.0e-3):
        B = np.float32(len(p))
        w0 = self.w0
        ae = np.float32(2.5)
        V, Vp, Vv, Vt, cache = self.forward_and_grads(p, vel, tau)
        _, _, _, z, a1, h1, c1, a2, h2, c2, out, q2, q1 = cache

        l_val = 10.0 - p
        term1 = l_val - V
        abs_Vv = np.sqrt(Vv**2 + np.float32(1e-7))
        sgn_Vv = Vv / abs_Vv
        H = Vp * vel + ae * abs_Vv
        term2 = H - Vt

        use_t1 = term1 <= term2
        res = np.where(use_t1, term1, term2)
        loss = float(np.mean(res**2))
        dL_dres = (np.float32(2.0) / B) * res

        dL_dV = np.where(use_t1, -dL_dres, np.float32(0.0))
        dL_dVp = np.where(use_t1, np.float32(0.0), dL_dres * vel)
        dL_dVv = np.where(use_t1, np.float32(0.0), dL_dres * ae * sgn_Vv)
        dL_dVt = np.where(use_t1, np.float32(0.0), -dL_dres)

        if self.steady:
            dL_dout = 5.0 * dL_dV
            g_z = np.stack([5.0 * dL_dVp / 7.0, 5.0 * dL_dVv / 6.0], axis=1)
        else:
            scale = 5.0 * (tau / 4.0)
            dL_dout = scale * dL_dV + (5.0 / 4.0) * dL_dVt
            g_z = np.stack([scale * dL_dVp / 7.0, scale * dL_dVv / 6.0, scale * dL_dVt * 0.5], axis=1)

        u1 = g_z @ self.W1
        dq2_from_q1 = (u1 * c1) @ self.W2
        dW2_from_q1 = (u1 * c1).T @ q2
        dpre1_from_c1 = u1 * (q2 @ self.W2.T) * (-w0 * w0 * h1)

        dW3_from_q2 = np.sum(dq2_from_q1 * c2, axis=0)[:, None]
        dpre2_from_c2 = dq2_from_q1 * self.W3[:, 0][None, :] * (-w0 * w0 * h2)

        dW3 = h2.T @ dL_dout[:, None] + dW3_from_q2
        db3 = np.sum(dL_dout, axis=0, keepdims=True)

        dh2_total = dL_dout[:, None] * self.W3[:, 0][None, :]
        dpre2_total = dh2_total * c2 + dpre2_from_c2

        dW2 = h1.T @ dpre2_total + dW2_from_q1
        db2 = np.sum(dpre2_total, axis=0)

        dh1_total = dpre2_total @ self.W2.T
        dpre1_total = dh1_total * c1 + dpre1_from_c1

        dW1 = z.T @ dpre1_total + g_z.T @ q1
        db1 = np.sum(dpre1_total, axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        self.step_idx += 1
        b1_a, b2_a, eps = np.float32(0.9), np.float32(0.999), np.float32(1e-8)
        c_b1 = np.float32(1.0 - 0.9**self.step_idx)
        c_b2 = np.float32(1.0 - 0.999**self.step_idx)
        lr32 = np.float32(lr)
        for i, (param, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1_a * self.m[i] + (np.float32(1.0) - b1_a) * g
            self.v[i] = b2_a * self.v[i] + (np.float32(1.0) - b2_a) * (g * g)
            param -= lr32 * (self.m[i] / c_b1) / (np.sqrt(self.v[i] / c_b2) + eps)

        return loss

    def eval_grid(self, PP, VV, tau_val=4.0):
        p = PP.ravel().astype(np.float32)
        v = VV.ravel().astype(np.float32)
        t = None if self.steady else np.full_like(p, tau_val)
        V, _, Vv, _, _ = self.forward_and_grads(p, v, t)
        return V.reshape(PP.shape), Vv.reshape(PP.shape)


def run_pinn_and_verification():
    t0 = time.perf_counter()
    p_grid = np.linspace(-2.0, 12.0, 71, dtype=np.float32)
    v_grid = np.linspace(-4.0, 8.0, 61, dtype=np.float32)
    PP, VV = np.meshgrid(p_grid, v_grid, indexing="ij")
    vp = np.maximum(VV, 0.0)
    t_stop = np.minimum(vp / 2.5, 4.0)
    V_true = (10.0 - PP) - (vp * t_stop - 0.5 * 2.5 * t_stop**2)

    rng = np.random.default_rng(7)
    net = SirenPINN1D(width=44, seed=11, w0=3.5, steady=False)
    steps = 3000
    snap_steps = {1, 40, 120, 300, 600, 1000, 1500, 2200, 3000}
    snaps = []
    loss_curve = []

    for k in range(1, steps + 1):
        frac = min(1.0, k / (0.52 * steps))
        lr = 2.4e-3 * (0.06 + 0.94 * 0.5 * (1.0 + np.cos(np.pi * k / steps)))
        P = rng.uniform(-2.0, 12.0, 1024).astype(np.float32)
        Vb = rng.uniform(-4.0, 8.0, 1024).astype(np.float32)
        Tb = rng.uniform(0.0, 4.0 * frac, 1024).astype(np.float32)
        loss = net.train_step(P, Vb, Tb, lr=lr)
        if k % 20 == 0 or k == 1:
            loss_curve.append([k, round(loss, 6)])
        if k in snap_steps:
            Vg, _ = net.eval_grid(PP, VV, 4.0)
            snaps.append({"step": k, "tau_cur": round(4.0 * frac, 2), "V": b64_i16(Vg, 500.0)})

    V_final, _ = net.eval_grid(PP, VV, 4.0)

    # Under-trained model for conformal verification lab
    small_net = SirenPINN1D(width=24, seed=21, w0=3.0, steady=False)
    for k in range(1, 301):
        frac = min(1.0, k / 200.0)
        P = rng.uniform(-2.0, 12.0, 1024).astype(np.float32)
        Vb = rng.uniform(-4.0, 8.0, 1024).astype(np.float32)
        Tb = rng.uniform(0.0, 4.0 * frac, 1024).astype(np.float32)
        small_net.train_step(P, Vb, Tb, lr=2.0e-3)
    V_small, _ = small_net.eval_grid(PP, VV, 4.0)

    # Steady-state ablation (no tau, free output -> collapses to constant <= min l = -2)
    steady_net = SirenPINN1D(width=32, seed=31, w0=2.0, steady=True)
    for k in range(1, 2201):
        P = rng.uniform(-2.0, 12.0, 1024).astype(np.float32)
        Vb = rng.uniform(-4.0, 8.0, 1024).astype(np.float32)
        st_loss = steady_net.train_step(P, Vb, None, lr=4.0e-3)
    V_steady, _ = steady_net.eval_grid(PP, VV)

    def rollout_policy(model, p0, v0, dt=0.02, T=4.0):
        p = p0.astype(np.float32).copy()
        v = v0.astype(np.float32).copy()
        min_l = 10.0 - p
        t_arr = np.full_like(p, 4.0)
        for _ in range(int(T / dt)):
            _, _, Vv, _, _ = model.forward_and_grads(p, v, t_arr)
            u = np.float32(3.0) * np.where(Vv >= 0.0, np.float32(1.0), np.float32(-1.0))
            acc = u + np.float32(0.5)
            p = p + v * dt + np.float32(0.5) * acc * (dt * dt)
            v = v + acc * dt
            min_l = np.minimum(min_l, np.float32(10.0) - p)
        return min_l

    def verify_model(model, name, n_cal=12000, n_val=10000, beta=1e-3):
        pts, vals = [], []
        while sum(len(x) for x in pts) < n_cal:
            p = rng.uniform(0.0, 10.8, 2 * n_cal).astype(np.float32)
            v = rng.uniform(-3.0, 7.0, 2 * n_cal).astype(np.float32)
            Vpred, _, _, _, _ = model.forward_and_grads(p, v, np.full_like(p, 4.0))
            keep = Vpred > 0.0
            pts.append(np.stack([p[keep], v[keep]], axis=1))
            vals.append(Vpred[keep])
        cal = np.concatenate(pts, axis=0)[:n_cal]
        V_cal = np.concatenate(vals, axis=0)[:n_cal]
        min_l = rollout_policy(model, cal[:, 0], cal[:, 1])
        viol = min_l < 0.0
        delta = float(V_cal[viol].max() + 0.005) if viol.any() else 0.0

        vpts = []
        while sum(len(x) for x in vpts) < n_val:
            p = rng.uniform(0.0, 10.8, 2 * n_val).astype(np.float32)
            v = rng.uniform(-3.0, 7.0, 2 * n_val).astype(np.float32)
            Vpred, _, _, _, _ = model.forward_and_grads(p, v, np.full_like(p, 4.0))
            keep = Vpred > delta
            vpts.append(np.stack([p[keep], v[keep]], axis=1))
        val = np.concatenate(vpts, axis=0)[:n_val]
        k_val = int((rollout_policy(model, val[:, 0], val[:, 1]) < 0.0).sum())
        eps = float(beta_dist.ppf(1.0 - beta, k_val + 1, n_val - k_val))
        print(f"[Verify {name}] viol={viol.sum()}/{n_cal} -> delta={delta:.3f}m | val={k_val}/{n_val} -> eps<{eps:.2e}")
        return {
            "n_cal": n_cal,
            "n_viol": int(viol.sum()),
            "delta": round(delta, 4),
            "n_val": n_val,
            "k_val": k_val,
            "eps_999": round(eps, 6),
            "viol_pts": cal[viol][:300].astype(float).round(3).tolist(),
            "viol_V": V_cal[viol][:300].astype(float).round(4).tolist(),
        }

    ver_small = verify_model(small_net, "Under-trained SIREN (300 steps)")
    ver_full = verify_model(net, "Full DeepReach SIREN (3000 steps)")

    inside = (PP >= 0) & (PP <= 11) & (VV >= -3) & (VV <= 7)
    iou_full = float((((V_true < 0) & (V_final < 0) & inside).sum()) / (((V_true < 0) | (V_final < 0)) & inside).sum())
    print(
        f"[PINN Suite] finished in {time.perf_counter()-t0:.1f}s | Full SIREN IoU={iou_full:.4f} | "
        f"Steady loss={st_loss:.2e}, V_steady in [{V_steady.min():.2f}, {V_steady.max():.2f}]"
    )

    return {
        "p": [float(p_grid[0]), float(p_grid[-1]), len(p_grid)],
        "v": [float(v_grid[0]), float(v_grid[-1]), len(v_grid)],
        "snaps": snaps,
        "loss": loss_curve,
        "final": b64_i16(V_final, 500.0),
        "small": b64_i16(V_small, 500.0),
        "steady": b64_i16(V_steady, 500.0),
        "steady_loss": float(st_loss),
        "steady_range": [round(float(V_steady.min()), 3), round(float(V_steady.max()), 3)],
        "iou_full": round(iou_full, 4),
        "verify": {"small": ver_small, "full": ver_full},
    }


# =====================================================================
# 7. Forensic Audit of NeuroHJR (arXiv:2512.01897)
# =====================================================================
def audit_neurohjr():
    W_MAP = 45.0
    START = np.array([1.0, 1.0])
    GOAL = np.array([40.0, 40.0])
    RD = 3.0
    rng = np.random.default_rng(1)
    obs = []
    while len(obs) < 10:
        c = START + rng.uniform(0.14, 0.88) * (GOAL - START) + rng.normal(0, 6.5, 2)
        if (
            np.all((c > 4.0) & (c < W_MAP - 4.0))
            and np.linalg.norm(c - START) > 6.5
            and np.linalg.norm(c - GOAL) > 6.5
            and all(np.linalg.norm(c - o) > 2 * RD + 2.8 for o in obs)
        ):
            obs.append(c)
    obs = np.array(obs)

    return {
        "map_size": W_MAP,
        "start": START.tolist(),
        "goal": GOAL.tolist(),
        "r_obs": 2.0,
        "margin": 1.0,
        "r_total": RD,
        "obstacles": obs.round(3).tolist(),
        "paper_tables": {
            "scenarios_time_neuro": [44.9, 77.3, 54.7],
            "scenarios_time_conv": [59.3, 105.4, 70.8],
            "scenarios_len_neuro": [57.24, 100.48, 68.89],
            "scenarios_len_conv": [59.10, 104.24, 70.28],
            "sensor_radii": [3, 5, 7],
            "sensor_time": [80.6, 77.3, 72.2],
            "sensor_len": [103.23, 100.48, 95.36],
        },
    }


def main():
    t_all = time.perf_counter()
    data = {
        "meta": {
            "paper_id": "arXiv:2512.01897v1",
            "paper_title": "NeuroHJR: Hamilton-Jacobi Reachability-based Obstacle Avoidance in Complex Environments with Physics-Informed Neural Networks",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "bench1d": run_1d_benchmarks(),
        "dubins3d": solve_dubins_3d(),
        "dock4d": solve_dock_reach_avoid(),
        "fastrack3d": solve_fastrack_teb(),
        "pinn": run_pinn_and_verification(),
        "neurohjr": audit_neurohjr(),
    }
    _, _, _, arena_meta = build_arena_4d()
    data["arena4d"] = arena_meta

    json_path = os.path.join(OUT_DIR, "data.json")
    js_path = os.path.join(OUT_DIR, "data.js")
    json_str = json.dumps(data, separators=(",", ":"))
    with open(json_path, "w") as f:
        f.write(json_str)
    with open(js_path, "w") as f:
        f.write("/* Precomputed HJR, Dubins 3D, Reach-Avoid Dock, FaSTrack & PINN tensors — file:// safe */\n")
        f.write("window.HJR_DATA = " + json_str + ";\n")

    print(
        f"\nAll solvers finished in {time.perf_counter()-t_all:.1f}s! "
        f"data.json: {os.path.getsize(json_path)/1024:.1f} KB, data.js: {os.path.getsize(js_path)/1024:.1f} KB"
    )


if __name__ == "__main__":
    main()
