/* ==========================================================================
   core.js — Theme Palette, High-DPI 2D/3D Canvas Engine, Marching Squares,
   Tensor Decoders, Multilinear Interpolators, Exact 2D QP, & Navigation
   ========================================================================== */
(function () {
  "use strict";

  const HJR = (window.HJR = window.HJR || {});
  HJR.$ = (sel, root = document) => root.querySelector(sel);
  HJR.$$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  HJR.clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
  HJR.lerp = (a, b, t) => a + (b - a) * t;

  /* ---------- Theme Management & Live Color Tokens ---------- */
  const COL = (HJR.COL = {});
  function parseColorToRgb(str) {
    str = (str || "").trim();
    if (!str) return [128, 128, 128];
    if (str.startsWith("#")) {
      let hex = str.slice(1);
      if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
      const n = parseInt(hex.slice(0, 6), 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    const m = str.match(/[\d.]+/g);
    if (m && m.length >= 3) return [+m[0], +m[1], +m[2]];
    return [128, 128, 128];
  }

  function syncThemeColors() {
    const cs = getComputedStyle(document.documentElement);
    const keys = [
      "bg", "bg-elev", "bg-subtle", "bg-hud", "ink", "ink-2", "muted",
      "line", "line-soft", "accent", "safe", "unsafe", "warn", "violet",
      "c-none", "c-apf", "c-cbf", "c-hjsw", "c-cbvf", "c-gold"
    ];
    for (const k of keys) {
      const val = cs.getPropertyValue("--" + k).trim();
      COL[k] = val;
      COL[k + "_rgb"] = parseColorToRgb(val);
    }
    COL.isDark = document.documentElement.getAttribute("data-theme") === "dark";
  }

  const redrawCallbacks = [];
  HJR.onThemeChange = (fn) => redrawCallbacks.push(fn);

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("hjr-theme", theme); } catch (_) {}
    syncThemeColors();
    const btn = HJR.$("#theme-toggle");
    if (btn) btn.innerHTML = theme === "dark" ? "☀ Light mode" : "☾ Dark mode";
    for (const fn of redrawCallbacks) {
      try { fn(); } catch (err) { console.error(err); }
    }
  }

  const savedTheme = (() => {
    try { return localStorage.getItem("hjr-theme"); } catch (_) { return null; }
  })();
  const initialTheme = savedTheme || "light";
  document.documentElement.setAttribute("data-theme", initialTheme);
  syncThemeColors();

  HJR.rgba = (rgb, a) => `rgba(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0},${a})`;
  HJR.mixRgb = (c1, c2, t) => [
    c1[0] + (c2[0] - c1[0]) * t,
    c1[1] + (c2[1] - c1[1]) * t,
    c1[2] + (c2[2] - c1[2]) * t,
  ];

  /* Diverging safety field colormap: crimson (< 0) <-> crisp white/slate near 0 <-> teal/sky (> 0) */
  HJR.valueColormap = (vmax = 8.0, vmin = -3.0) => {
    const base = COL["bg-hud_rgb"];
    const bad = COL.unsafe_rgb;
    const good = COL.safe_rgb;
    const sky = COL.accent_rgb;
    return (v) => {
      if (v < 0) {
        const s = HJR.clamp(-v / Math.abs(vmin), 0, 1);
        return HJR.mixRgb(base, bad, 0.34 + 0.56 * Math.pow(s, 0.78));
      }
      const s = HJR.clamp(v / vmax, 0, 1);
      const mid = HJR.mixRgb(good, sky, 0.45);
      return HJR.mixRgb(base, mid, 0.06 + 0.44 * Math.pow(s, 0.85));
    };
  };

  /* ---------- Binary Tensor Decoders (Base64 -> Float32Array) ---------- */
  HJR.decodeQuantU8 = (spec) => {
    const bin = atob(spec.b64);
    const n = bin.length;
    const out = new Float32Array(n);
    const scale = (spec.max - spec.min) / 255.0;
    const mn = spec.min;
    for (let i = 0; i < n; i++) out[i] = mn + bin.charCodeAt(i) * scale;
    return out;
  };

  HJR.decodeI16 = (spec) => {
    const bin = atob(spec.b64);
    const n = bin.length >> 1;
    const out = new Float32Array(n);
    const inv = 1.0 / (spec.scale || 500.0);
    for (let i = 0; i < n; i++) {
      let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      if (v >= 32768) v -= 65536;
      out[i] = v * inv;
    }
    return out;
  };

  /* ---------- Multilinear Interpolators (2D, 3D, 4D) ---------- */
  HJR.interp2D = (F, nx, ny, x0, x1, y0, y1, x, y) => {
    const dx = (x1 - x0) / (nx - 1), dy = (y1 - y0) / (ny - 1);
    const tx = HJR.clamp((x - x0) / dx, 0, nx - 1 - 1e-6);
    const ty = HJR.clamp((y - y0) / dy, 0, ny - 1 - 1e-6);
    const i = tx | 0, j = ty | 0, fx = tx - i, fy = ty - j;
    const k = i * ny + j;
    return (1 - fx) * ((1 - fy) * F[k] + fy * F[k + 1]) +
           fx * ((1 - fy) * F[k + ny] + fy * F[k + ny + 1]);
  };

  HJR.interp3D = (F, nx, ny, nz, x0, x1, y0, y1, z0, z1, x, y, z, zPeriodic = false) => {
    const dx = (x1 - x0) / (nx - 1), dy = (y1 - y0) / (ny - 1);
    const tx = HJR.clamp((x - x0) / dx, 0, nx - 1 - 1e-6);
    const ty = HJR.clamp((y - y0) / dy, 0, ny - 1 - 1e-6);
    const i = tx | 0, j = ty | 0, fx = tx - i, fy = ty - j;
    let k0, k1, fz;
    if (zPeriodic) {
      const period = 2 * Math.PI;
      let u = ((z - z0) % period + period) % period;
      const tz = (u / period) * nz;
      k0 = (tz | 0) % nz;
      k1 = (k0 + 1) % nz;
      fz = tz - Math.floor(tz);
    } else {
      const dz = (z1 - z0) / (nz - 1);
      const tz = HJR.clamp((z - z0) / dz, 0, nz - 1 - 1e-6);
      k0 = tz | 0; k1 = k0 + 1; fz = tz - k0;
    }
    const at = (ii, jj, kk) => F[(ii * ny + jj) * nz + kk];
    const c00 = (1 - fz) * at(i, j, k0) + fz * at(i, j, k1);
    const c01 = (1 - fz) * at(i, j + 1, k0) + fz * at(i, j + 1, k1);
    const c10 = (1 - fz) * at(i + 1, j, k0) + fz * at(i + 1, j, k1);
    const c11 = (1 - fz) * at(i + 1, j + 1, k0) + fz * at(i + 1, j + 1, k1);
    return (1 - fx) * ((1 - fy) * c00 + fy * c01) + fx * ((1 - fy) * c10 + fy * c11);
  };

  HJR.interp4D = (F, dims, bounds, pt) => {
    const [n0, n1, n2, n3] = dims;
    const idx = [0, 0, 0, 0], f = [0, 0, 0, 0];
    for (let d = 0; d < 4; d++) {
      const [lo, hi] = bounds[d];
      const h = (hi - lo) / (dims[d] - 1);
      const t = HJR.clamp((pt[d] - lo) / h, 0, dims[d] - 1 - 1e-6);
      idx[d] = t | 0;
      f[d] = t - idx[d];
    }
    let acc = 0;
    for (let c = 0; c < 16; c++) {
      const b0 = c & 1, b1 = (c >> 1) & 1, b2 = (c >> 2) & 1, b3 = (c >> 3) & 1;
      const w = (b0 ? f[0] : 1 - f[0]) * (b1 ? f[1] : 1 - f[1]) * (b2 ? f[2] : 1 - f[2]) * (b3 ? f[3] : 1 - f[3]);
      if (w === 0) continue;
      acc += w * F[(((idx[0] + b0) * n1 + (idx[1] + b1)) * n2 + (idx[2] + b2)) * n3 + (idx[3] + b3)];
    }
    return acc;
  };

  /* ---------- High-DPI 2D Plotter ---------- */
  HJR.createPlot2D = (cv, opts = {}) => {
    const w = +cv.getAttribute("width") || 640;
    const h = +cv.getAttribute("height") || 440;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = w * dpr;
    cv.height = h * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const m = Object.assign({ l: 48, r: 14, t: 14, b: 40 }, opts.margin || {});
    const P = {
      cv, ctx, w, h,
      xlim: opts.xlim || [0, 1],
      ylim: opts.ylim || [0, 1],
      px: m.l, py: m.t,
      pw: w - m.l - m.r,
      ph: h - m.t - m.b,
    };
    P.X = (x) => P.px + ((x - P.xlim[0]) / (P.xlim[1] - P.xlim[0])) * P.pw;
    P.Y = (y) => P.py + (1 - (y - P.ylim[0]) / (P.ylim[1] - P.ylim[0])) * P.ph;
    P.inv = (sx, sy) => [
      P.xlim[0] + ((sx - P.px) / P.pw) * (P.xlim[1] - P.xlim[0]),
      P.ylim[0] + (1 - (sy - P.py) / P.ph) * (P.ylim[1] - P.ylim[0]),
    ];
    P.evtPos = (e) => {
      const r = cv.getBoundingClientRect();
      return [((e.clientX - r.left) / r.width) * P.w, ((e.clientY - r.top) / r.height) * P.h];
    };
    P.clear = (fill) => {
      ctx.fillStyle = fill || COL["bg-hud"];
      ctx.fillRect(0, 0, w, h);
    };
    P.clip = () => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(P.px, P.py, P.pw, P.ph);
      ctx.clip();
    };
    P.axes = (xlabel, ylabel, xticks, yticks, grid = false) => {
      ctx.save();
      const fontUI = getComputedStyle(document.documentElement).getPropertyValue("--sans").trim() || "Inter, sans-serif";
      ctx.font = "11.5px " + fontUI;
      if (grid) {
        ctx.strokeStyle = HJR.rgba(COL.line_rgb, 0.35);
        ctx.lineWidth = 1;
        for (const x of xticks) {
          const sx = P.X(x);
          ctx.beginPath(); ctx.moveTo(sx, P.py); ctx.lineTo(sx, P.py + P.ph); ctx.stroke();
        }
        for (const y of yticks) {
          const sy = P.Y(y);
          ctx.beginPath(); ctx.moveTo(P.px, sy); ctx.lineTo(P.px + P.pw, sy); ctx.stroke();
        }
      }
      ctx.strokeStyle = COL.line;
      ctx.fillStyle = COL.muted;
      ctx.lineWidth = 1;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const x of xticks) {
        const sx = P.X(x);
        ctx.beginPath(); ctx.moveTo(sx, P.py + P.ph); ctx.lineTo(sx, P.py + P.ph + 4); ctx.stroke();
        ctx.fillText(String(x), sx, P.py + P.ph + 6);
      }
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const y of yticks) {
        const sy = P.Y(y);
        ctx.beginPath(); ctx.moveTo(P.px - 4, sy); ctx.lineTo(P.px, sy); ctx.stroke();
        ctx.fillText(String(y), P.px - 6, sy);
      }
      ctx.strokeRect(P.px, P.py, P.pw, P.ph);
      ctx.fillStyle = COL["ink-2"];
      ctx.font = "500 12px " + fontUI;
      if (xlabel) {
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(xlabel, P.px + P.pw / 2, h - 4);
      }
      if (ylabel) {
        ctx.save();
        ctx.translate(13, P.py + P.ph / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(ylabel, 0, 0);
        ctx.restore();
      }
      ctx.restore();
    };
    return P;
  };

  /* Offscreen cached raster renderer for 2D scalar fields F[i*ny + j] */
  HJR.drawScalarField = (P, F, nx, ny, x0, x1, y0, y1, colorFn) => {
    const off = P._off && P._off.width === nx && P._off.height === ny ? P._off : (() => {
      const c = document.createElement("canvas");
      c.width = nx; c.height = ny;
      P._off = c;
      return c;
    })();
    const octx = off.getContext("2d");
    const img = octx.createImageData(nx, ny);
    const data = img.data;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const rgb = colorFn(F[i * ny + j]);
        const idx = ((ny - 1 - j) * nx + i) * 4;
        data[idx] = rgb[0];
        data[idx + 1] = rgb[1];
        data[idx + 2] = rgb[2];
        data[idx + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    const dx = (x1 - x0) / (nx - 1), dy = (y1 - y0) / (ny - 1);
    const sx0 = P.X(x0 - dx / 2), sx1 = P.X(x1 + dx / 2);
    const sy0 = P.Y(y1 + dy / 2), sy1 = P.Y(y0 - dy / 2);
    P.ctx.save();
    P.ctx.imageSmoothingEnabled = true;
    P.ctx.drawImage(off, sx0, sy0, sx1 - sx0, sy1 - sy0);
    P.ctx.restore();
  };

  /* Marching Squares iso-contour segments */
  HJR.marchingSquares = (F, nx, ny, x0, x1, y0, y1, level = 0) => {
    const dx = (x1 - x0) / (nx - 1), dy = (y1 - y0) / (ny - 1);
    const segs = [];
    const at = (i, j) => F[i * ny + j] - level;
    const cut = (a, b) => a / (a - b);
    for (let i = 0; i < nx - 1; i++) {
      const X = x0 + i * dx;
      for (let j = 0; j < ny - 1; j++) {
        const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
        const mask = (a > 0 ? 1 : 0) | (b > 0 ? 2 : 0) | (c > 0 ? 4 : 0) | (d > 0 ? 8 : 0);
        if (mask === 0 || mask === 15) continue;
        const Y = y0 + j * dy;
        const pts = [];
        if ((a > 0) !== (b > 0)) pts.push([X + cut(a, b) * dx, Y]);
        if ((b > 0) !== (c > 0)) pts.push([X + dx, Y + cut(b, c) * dy]);
        if ((c > 0) !== (d > 0)) pts.push([X + (1 - cut(c, d)) * dx, Y + dy]);
        if ((d > 0) !== (a > 0)) pts.push([X, Y + (1 - cut(d, a)) * dy]);
        if (pts.length === 2) segs.push([pts[0], pts[1]]);
        else if (pts.length === 4) { segs.push([pts[0], pts[1]]); segs.push([pts[2], pts[3]]); }
      }
    }
    return segs;
  };

  HJR.strokeSegments = (P, segs, color, width = 2, dash = []) => {
    if (!segs.length) return;
    const c = P.ctx;
    c.save();
    c.strokeStyle = color;
    c.lineWidth = width;
    c.setLineDash(dash);
    c.beginPath();
    for (const [p, q] of segs) {
      c.moveTo(P.X(p[0]), P.Y(p[1]));
      c.lineTo(P.X(q[0]), P.Y(q[1]));
    }
    c.stroke();
    c.restore();
  };

  HJR.drawPolyline = (P, pts, color, width = 2.2, dash = []) => {
    if (!pts || pts.length < 2) return;
    const c = P.ctx;
    c.save();
    c.strokeStyle = color;
    c.lineWidth = width;
    c.lineJoin = "round";
    c.lineCap = "round";
    c.setLineDash(dash);
    c.beginPath();
    c.moveTo(P.X(pts[0][0]), P.Y(pts[0][1]));
    for (let i = 1; i < pts.length; i++) c.lineTo(P.X(pts[i][0]), P.Y(pts[i][1]));
    c.stroke();
    c.restore();
  };

  HJR.drawDot = (P, x, y, r, fill, stroke, lw = 2) => {
    const c = P.ctx;
    c.save();
    c.beginPath();
    c.arc(P.X(x), P.Y(y), r, 0, 2 * Math.PI);
    if (fill) { c.fillStyle = fill; c.fill(); }
    if (stroke) { c.strokeStyle = stroke; c.lineWidth = lw; c.stroke(); }
    c.restore();
  };

  HJR.drawArrowPx = (ctx, x0, y0, x1, y1, color, lw = 2.2, head = 8) => {
    const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
    if (len < 1.5) return;
    const ang = Math.atan2(dy, dx);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - head * Math.cos(ang - 0.42), y1 - head * Math.sin(ang - 0.42));
    ctx.lineTo(x1 - head * Math.cos(ang + 0.42), y1 - head * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  };

  HJR.bindDrag = (cv, getEvtPos, handlers) => {
    let dragging = false;
    cv.addEventListener("pointerdown", (e) => {
      const pos = getEvtPos(e);
      if (handlers.down && handlers.down(pos, e) === false) return;
      dragging = true;
      try { cv.setPointerCapture(e.pointerId); } catch (_) {}
      e.preventDefault();
    });
    cv.addEventListener("pointermove", (e) => {
      const pos = getEvtPos(e);
      if (dragging) handlers.move && handlers.move(pos, e, false);
      else if (e.pointerType === "mouse") handlers.move && handlers.move(pos, e, true);
    });
    const finish = (e) => {
      if (!dragging) return;
      dragging = false;
      handlers.up && handlers.up(getEvtPos(e), e);
    };
    cv.addEventListener("pointerup", finish);
    cv.addEventListener("pointercancel", finish);
  };

  /* ---------- Exact 2-Variable Quadratic Program Solver (Active-Set Enumeration) ----------
     Solves: min ||u - u_nom||^2  s.t. A_i . u <= b_i  for i = 0..m-1.
     Returns { u: [ux, uy], infeasible: bool, activeIdx: int[] } */
  const DISC16 = Array.from({ length: 16 }, (_, k) => [
    Math.cos((2 * Math.PI * k) / 16),
    Math.sin((2 * Math.PI * k) / 16),
  ]);
  HJR.DISC16 = DISC16;

  HJR.solveQP2D = (uNom, A, b) => {
    const m = A.length;
    const cands = [uNom.slice()];
    for (let i = 0; i < m; i++) {
      const [ax, ay] = A[i];
      const viol = ax * uNom[0] + ay * uNom[1] - b[i];
      const n2 = ax * ax + ay * ay + 1e-12;
      if (viol > 0) cands.push([uNom[0] - (viol / n2) * ax, uNom[1] - (viol / n2) * ay]);
    }
    for (let i = 0; i < m; i++) {
      const [a1, a2] = A[i];
      for (let j = i + 1; j < m; j++) {
        const [c1, c2] = A[j];
        const det = a1 * c2 - a2 * c1;
        if (Math.abs(det) < 1e-9) continue;
        cands.push([(b[i] * c2 - b[j] * a2) / det, (a1 * b[j] - c1 * b[i]) / det]);
      }
    }
    let bestU = null, bestCost = Infinity;
    let fallbackU = uNom.slice(), bestSlack = -Infinity;
    for (const u of cands) {
      let minSlack = Infinity;
      for (let i = 0; i < m; i++) {
        const s = b[i] - (A[i][0] * u[0] + A[i][1] * u[1]);
        if (s < minSlack) minSlack = s;
      }
      if (minSlack >= -1e-6) {
        const cost = (u[0] - uNom[0]) ** 2 + (u[1] - uNom[1]) ** 2;
        if (cost < bestCost) { bestCost = cost; bestU = u; }
      }
      if (minSlack > bestSlack) { bestSlack = minSlack; fallbackU = u; }
    }
    return bestU
      ? { u: bestU, infeasible: false, minSlack: 0 }
      : { u: fallbackU, infeasible: true, minSlack: bestSlack };
  };

  /* ---------- Quadrotor Vector Sprite Renderer (Side-View / Top-View) ---------- */
  HJR.drawQuadrotorSide = (ctx, cx, cy, pitchRad, scale = 1.0, bodyColor = COL.ink, thrustColor = COL.accent, phase = 0) => {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(pitchRad);
    const arm = 22 * scale;
    // Thrust vector arrow perpendicular to frame
    ctx.strokeStyle = thrustColor;
    ctx.fillStyle = thrustColor;
    ctx.lineWidth = 2 * scale;
    HJR.drawArrowPx(ctx, 0, 2 * scale, 0, -28 * scale, thrustColor, 2 * scale, 6 * scale);
    // Arms & fuselage
    ctx.strokeStyle = bodyColor;
    ctx.lineWidth = 3 * scale;
    ctx.beginPath();
    ctx.moveTo(-arm, 0);
    ctx.lineTo(arm, 0);
    ctx.stroke();
    ctx.fillStyle = bodyColor;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(-8 * scale, -4 * scale, 16 * scale, 8 * scale, 3 * scale) : ctx.rect(-8 * scale, -4 * scale, 16 * scale, 8 * scale);
    ctx.fill();
    // Spinning rotors
    const rSpread = (7 + 4 * Math.abs(Math.sin(phase))) * scale;
    ctx.strokeStyle = thrustColor;
    ctx.lineWidth = 2 * scale;
    for (const sgn of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(sgn * arm, 0);
      ctx.lineTo(sgn * arm, -5 * scale);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(sgn * arm - rSpread, -5 * scale);
      ctx.lineTo(sgn * arm + rSpread, -5 * scale);
      ctx.stroke();
    }
    ctx.restore();
  };

  /* ---------- DOM Ready: Theme Button, Progress Bar, Scroll-Spy, KaTeX ---------- */
  window.addEventListener("DOMContentLoaded", () => {
    const btn = HJR.$("#theme-toggle");
    if (btn) {
      btn.innerHTML = COL.isDark ? "☀ Light mode" : "☾ Dark mode";
      btn.addEventListener("click", () => applyTheme(COL.isDark ? "light" : "dark"));
    }

    const pbar = HJR.$("#reading-progress");
    const updateScroll = () => {
      if (!pbar) return;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const pct = max > 0 ? HJR.clamp((window.scrollY / max) * 100, 0, 100) : 0;
      pbar.style.width = pct.toFixed(1) + "%";
    };
    window.addEventListener("scroll", updateScroll, { passive: true });
    updateScroll();

    const navLinks = HJR.$$(".toc-nav a");
    const sections = navLinks.map((a) => HJR.$(a.getAttribute("href")));
    if ("IntersectionObserver" in window && sections.length) {
      const obs = new IntersectionObserver(
        (entries) => {
          for (const ent of entries) {
            if (ent.isIntersecting) {
              const idx = sections.indexOf(ent.target);
              navLinks.forEach((a, j) => a.classList.toggle("active", j === idx));
            }
          }
        },
        { rootMargin: "-25% 0px -62% 0px" }
      );
      sections.forEach((s) => s && obs.observe(s));
    }

    HJR.typeset = (el = document.body) => {
      if (window.renderMathInElement && el) {
        window.renderMathInElement(el, {
          delimiters: [
            { left: "\\[", right: "\\]", display: true },
            { left: "\\(", right: "\\)", display: false },
          ],
          throwOnError: false,
        });
      }
    };
    HJR.typeset(document.body);
    window.addEventListener("load", () => HJR.typeset(document.body));
  });
})();
