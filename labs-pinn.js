/* ==========================================================================
   labs-pinn.js — Interactive Labs 06 to 10:
     Lab 06: Live 60fps In-Browser PINN Studio (Exact 2nd-Order Autodiff + Adam)
     Lab 07: Cluttered Navigation Progression (Uncoupled vs Lookahead vs Eikonal HJB)
     Lab 08: Conformal / Scenario Verification Explorer (Lin & Bansal)
     Lab 09: Production Drone Envelope & Architecture Sizer
     Lab 10: 5-Scenario Autonomy Lead Engineering Diagnostic
   ========================================================================== */
(function () {
  "use strict";

  window.addEventListener("DOMContentLoaded", () => {
    const HJR = window.HJR;
    const DATA = window.HJR_DATA || {};
    const { $, $$, clamp, COL, rgba, mixRgb } = HJR;

    /* =====================================================================
       LAB 06: LIVE 60FPS IN-BROWSER PINN TRAINING STUDIO
               Exact analytical 2nd-order PDE residual backprop + Adam in JS!
       ===================================================================== */
    (function initLab06() {
      const cvVal = $("#pinn-val-cv");
      const cvLoss = $("#pinn-loss-cv");
      if (!cvVal || !cvLoss) return;

      const PV = HJR.createPlot2D(cvVal, { xlim: [-2, 12], ylim: [-4, 8], margin: { l: 46, r: 12, t: 20, b: 36 } });
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const WL = 580, HL = 400;
      cvLoss.width = WL * dpr; cvLoss.height = HL * dpr;
      const ctxL = cvLoss.getContext("2d");
      ctxL.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Evaluation grid (71 x 61) matching offline PINN grid
      const NX = 71, NY = 61, X0 = -2, X1 = 12, Y0 = -4, Y1 = 8;
      const dx = (X1 - X0) / (NX - 1), dy = (Y1 - Y0) / (NY - 1);
      const exactGrid = new Float32Array(NX * NY);
      const insideMask = new Uint8Array(NX * NY);
      for (let i = 0; i < NX; i++) {
        const p = X0 + i * dx;
        for (let j = 0; j < NY; j++) {
          const v = Y0 + j * dy;
          const vp = Math.max(v, 0), t = Math.min(vp / 2.5, 4.0);
          exactGrid[i * NY + j] = (10.0 - p) - (vp * t - 0.5 * 2.5 * t * t);
          insideMask[i * NY + j] = p >= 0 && p <= 11 && v >= -3 && v <= 7 ? 1 : 0;
        }
      }

      // Compact 2-hidden-layer MLP [D_in, W, W, 1] with exact 1st & 2nd activation derivatives!
      const W = 26;
      let actType = "siren";     // "siren" | "tanh" | "relu"
      let formType = "deepreach"; // "deepreach" | "steady"
      let useCurr = true;
      let running = false, raf = null;
      let usingOffline = false;

      let prngState = 1337;
      const rand = () => {
        prngState = (Math.imul(prngState, 1664525) + 1013904223) >>> 0;
        return prngState / 4294967296;
      };

      let net = null;
      let lossHist = [];

      function initNet() {
        usingOffline = false;
        prngState = 2026;
        const steady = formType === "steady";
        const D = steady ? 2 : 3;
        const w0 = actType === "siren" ? 3.2 : 1.0;
        const W1 = new Float32Array(D * W), b1 = new Float32Array(W);
        const W2 = new Float32Array(W * W), b2 = new Float32Array(W);
        const W3 = new Float32Array(W), b3 = new Float32Array(1);

        const lim1 = actType === "siren" ? 1.0 / D : Math.sqrt(6 / (D + W));
        const lim2 = actType === "siren" ? Math.sqrt(6 / W) / w0 : Math.sqrt(6 / (2 * W));
        for (let i = 0; i < W1.length; i++) W1[i] = (2 * rand() - 1) * lim1;
        for (let i = 0; i < W2.length; i++) W2[i] = (2 * rand() - 1) * lim2;
        for (let i = 0; i < W3.length; i++) W3[i] = (2 * rand() - 1) * lim2;

        const arrays = [W1, b1, W2, b2, W3, b3];
        net = {
          D, w0, steady, step: 0,
          W1, b1, W2, b2, W3, b3,
          arrays,
          m: arrays.map((a) => new Float32Array(a.length)),
          v: arrays.map((a) => new Float32Array(a.length)),
        };
        lossHist = [];
      }

      // Activation value h(x), 1st deriv h'(x), 2nd deriv h''(x) wrt pre-activation u = z @ W + b
      function actEval(u, w0, kind) {
        if (kind === "siren") {
          const a = w0 * u, s = Math.sin(a), c = Math.cos(a);
          return [s, w0 * c, -w0 * w0 * s];
        }
        if (kind === "tanh") {
          const t = Math.tanh(u), dt = 1 - t * t;
          return [t, dt, -2 * t * dt];
        }
        // ReLU: piecewise linear -> 2nd derivative is ZERO everywhere!
        return u > 0 ? [u, 1.0, 0.0] : [0.0, 0.0, 0.0];
      }

      function evalPoint(p, vel, tau) {
        const { D, w0, steady, W1, b1, W2, b2, W3, b3 } = net;
        const z0 = (p - 5.0) / 7.0, z1 = (vel - 2.0) / 6.0, z2 = 2.0 * (tau / 4.0) - 1.0;
        const z = steady ? [z0, z1] : [z0, z1, z2];

        const h1 = new Float32Array(W), d1 = new Float32Array(W), dd1 = new Float32Array(W);
        for (let j = 0; j < W; j++) {
          let sum = b1[j];
          for (let d = 0; d < D; d++) sum += z[d] * W1[d * W + j];
          const [hv, dv_, ddv] = actEval(sum, w0, actType);
          h1[j] = hv; d1[j] = dv_; dd1[j] = ddv;
        }

        const h2 = new Float32Array(W), d2 = new Float32Array(W), dd2 = new Float32Array(W);
        for (let k = 0; k < W; k++) {
          let sum = b2[k];
          for (let j = 0; j < W; j++) sum += h1[j] * W2[j * W + k];
          const [hv, dv_, ddv] = actEval(sum, w0, actType);
          h2[k] = hv; d2[k] = dv_; dd2[k] = ddv;
        }

        let out = b3[0];
        const q2 = new Float32Array(W);
        for (let k = 0; k < W; k++) {
          out += h2[k] * W3[k];
          q2[k] = W3[k] * d2[k];
        }

        const q1 = new Float32Array(W);
        for (let j = 0; j < W; j++) {
          let s = 0;
          for (let k = 0; k < W; k++) s += q2[k] * W2[j * W + k];
          q1[j] = s * d1[j];
        }

        const dout_dz = new Float32Array(D);
        for (let d = 0; d < D; d++) {
          let s = 0;
          for (let j = 0; j < W; j++) s += q1[j] * W1[d * W + j];
          dout_dz[d] = s;
        }

        let V, Vp, Vv, Vt;
        if (steady) {
          V = 5.0 * out;
          Vp = (5.0 * dout_dz[0]) / 7.0;
          Vv = (5.0 * dout_dz[1]) / 6.0;
          Vt = 0;
        } else {
          const sc = 5.0 * (tau / 4.0);
          V = (10.0 - p) + sc * out;
          Vp = -1.0 + (sc * dout_dz[0]) / 7.0;
          Vv = (sc * dout_dz[1]) / 6.0;
          Vt = 1.25 * out + sc * dout_dz[2] * 0.5;
        }
        return { V, Vp, Vv, Vt, out, z, h1, d1, dd1, h2, d2, dd2, q1, q2 };
      }

      function trainBatch(B = 192) {
        const { D, steady, W1, W2, W3, arrays, m, v } = net;
        net.step++;
        const frac = useCurr ? Math.min(1.0, net.step / 650) : 1.0;
        const grads = arrays.map((a) => new Float32Array(a.length));
        const [gW1, gb1, gW2, gb2, gW3, gb3] = grads;

        let lossSum = 0;
        const ae = 2.5;
        const invB = 2.0 / B;

        for (let b = 0; b < B; b++) {
          const p = -2.0 + 14.0 * rand();
          const vel = -4.0 + 12.0 * rand();
          const tau = 4.0 * frac * rand();
          const ev = evalPoint(p, vel, tau);

          const t1 = (10.0 - p) - ev.V;
          const absVv = Math.sqrt(ev.Vv * ev.Vv + 1e-6);
          const sgnVv = ev.Vv / absVv;
          const H = ev.Vp * vel + ae * absVv;
          const t2 = H - ev.Vt;

          const pick1 = t1 <= t2;
          const res = pick1 ? t1 : t2;
          lossSum += res * res;
          const dRes = invB * res;

          const dV = pick1 ? -dRes : 0;
          const dVp = pick1 ? 0 : dRes * vel;
          const dVv = pick1 ? 0 : dRes * ae * sgnVv;
          const dVt = pick1 ? 0 : -dRes;

          let dOut, gz0, gz1, gz2 = 0;
          if (steady) {
            dOut = 5.0 * dV;
            gz0 = (5.0 * dVp) / 7.0;
            gz1 = (5.0 * dVv) / 6.0;
          } else {
            const sc = 5.0 * (tau / 4.0);
            dOut = sc * dV + 1.25 * dVt;
            gz0 = (sc * dVp) / 7.0;
            gz1 = (sc * dVv) / 6.0;
            gz2 = sc * dVt * 0.5;
          }
          const gz = steady ? [gz0, gz1] : [gz0, gz1, gz2];

          // Backprop through dout_dz = q1 @ W1^T and direct out = h2 @ W3 + b3
          const u1 = new Float32Array(W);
          for (let j = 0; j < W; j++) {
            let s = 0;
            for (let d = 0; d < D; d++) {
              s += gz[d] * W1[d * W + j];
              gW1[d * W + j] += gz[d] * ev.q1[j];
            }
            u1[j] = s;
          }

          const dq2 = new Float32Array(W);
          const dpre1_grad = new Float32Array(W);
          for (let j = 0; j < W; j++) {
            const uc = u1[j] * ev.d1[j];
            let q2w = 0;
            for (let k = 0; k < W; k++) {
              dq2[k] += uc * W2[j * W + k];
              gW2[j * W + k] += uc * ev.q2[k];
              q2w += ev.q2[k] * W2[j * W + k];
            }
            dpre1_grad[j] = u1[j] * q2w * ev.dd1[j];
          }

          const dpre2 = new Float32Array(W);
          gb3[0] += dOut;
          for (let k = 0; k < W; k++) {
            gW3[k] += ev.h2[k] * dOut + dq2[k] * ev.d2[k];
            const dpre2_g = dq2[k] * W3[k] * ev.dd2[k];
            dpre2[k] = dOut * W3[k] * ev.d2[k] + dpre2_g;
            gb2[k] += dpre2[k];
          }

          for (let j = 0; j < W; j++) {
            let dh1 = 0;
            for (let k = 0; k < W; k++) {
              gW2[j * W + k] += ev.h1[j] * dpre2[k];
              dh1 += dpre2[k] * W2[j * W + k];
            }
            const dp1 = dh1 * ev.d1[j] + dpre1_grad[j];
            gb1[j] += dp1;
            for (let d = 0; d < D; d++) gW1[d * W + j] += ev.z[d] * dp1;
          }
        }

        // Adam update
        const lr = formType === "steady" ? 3.5e-3 : 2.6e-3 * Math.max(0.18, 1 - net.step / 3500);
        const b1a = 0.9, b2a = 0.999, eps = 1e-8;
        const cb1 = 1 - Math.pow(b1a, net.step), cb2 = 1 - Math.pow(b2a, net.step);
        for (let a = 0; a < arrays.length; a++) {
          const pArr = arrays[a], gArr = grads[a], mArr = m[a], vArr = v[a];
          for (let i = 0; i < pArr.length; i++) {
            mArr[i] = b1a * mArr[i] + (1 - b1a) * gArr[i];
            vArr[i] = b2a * vArr[i] + (1 - b2a) * gArr[i] * gArr[i];
            pArr[i] -= lr * (mArr[i] / cb1) / (Math.sqrt(vArr[i] / cb2) + eps);
          }
        }

        const meanLoss = lossSum / B;
        if (net.step % 5 === 0 || net.step === 1) lossHist.push([net.step, Math.max(meanLoss, 1e-8)]);
        return meanLoss;
      }

      let offlineField = null;
      function draw() {
        let Vgrid, resGrid = new Float32Array(NX * NY);
        if (usingOffline && DATA.pinn) {
          if (!offlineField) offlineField = HJR.decodeI16(DATA.pinn.final);
          Vgrid = offlineField;
        } else {
          Vgrid = new Float32Array(NX * NY);
          for (let i = 0; i < NX; i++) {
            const p = X0 + i * dx;
            for (let j = 0; j < NY; j++) {
              const v = Y0 + j * dy;
              const ev = evalPoint(p, v, 4.0);
              Vgrid[i * NY + j] = ev.V;
              const t1 = (10.0 - p) - ev.V;
              const t2 = ev.Vp * v + 2.5 * Math.abs(ev.Vv) - ev.Vt;
              resGrid[i * NY + j] = Math.abs(Math.min(t1, t2));
            }
          }
        }

        // Compute BRT IoU & false-safe vs exact
        let inter = 0, uni = 0, fs = 0, oc = 0, tot = 0;
        let vmin = 1e9, vmax = -1e9;
        for (let k = 0; k < Vgrid.length; k++) {
          if (Vgrid[k] < vmin) vmin = Vgrid[k];
          if (Vgrid[k] > vmax) vmax = Vgrid[k];
          if (!insideMask[k]) continue;
          tot++;
          const t = exactGrid[k] < 0, s = Vgrid[k] < 0;
          if (t && s) inter++;
          if (t || s) uni++;
          if (t && !s) fs++;
          if (!t && s) oc++;
        }
        const iou = uni > 0 ? inter / uni : 0;

        /* --- Left Canvas: Learned Value Field vs Exact Zero Contour --- */
        PV.clear();
        HJR.drawScalarField(PV, Vgrid, NX, NY, X0, X1, Y0, Y1, HJR.valueColormap(10, -3.5));
        PV.clip();
        HJR.strokeSegments(PV, HJR.marchingSquares(Vgrid, NX, NY, X0, X1, Y0, Y1, 0), COL.ink, 2.4);
        HJR.strokeSegments(PV, HJR.marchingSquares(exactGrid, NX, NY, X0, X1, Y0, Y1, 0), COL["c-gold"], 2.2, [6, 4]);
        PV.ctx.restore();
        PV.ctx.fillStyle = COL.ink;
        PV.ctx.font = "600 11.5px " + getComputedStyle(document.body).fontFamily;
        PV.ctx.fillText("Solid = PINN V_θ(x, τ=4s) = 0   ·   Dashed Gold = Exact Analytical BRT", PV.px + 6, 14);
        PV.axes("position p [m]", "velocity v [m/s]", [-2, 0, 2, 4, 6, 8, 10, 12], [-4, -2, 0, 2, 4, 6, 8]);

        /* --- Right Canvas: Top = PDE Residual Heatmap, Bottom = Live Log-Loss Curve --- */
        ctxL.fillStyle = COL["bg-hud"];
        ctxL.fillRect(0, 0, WL, HL);

        const PR = {
          ctx: ctxL, px: 48, py: 22, pw: WL - 62, ph: 165,
          xlim: [X0, X1], ylim: [Y0, Y1],
          X: (x) => 48 + ((x - X0) / (X1 - X0)) * (WL - 62),
          Y: (y) => 22 + (1 - (y - Y0) / (Y1 - Y0)) * 165,
        };
        HJR.drawScalarField(PR, resGrid, NX, NY, X0, X1, Y0, Y1, (r) => {
          const t = clamp(r / 1.4, 0, 1);
          return mixRgb(COL["bg-hud_rgb"], COL.unsafe_rgb, 0.05 + 0.88 * Math.pow(t, 0.65));
        });
        ctxL.strokeStyle = COL.line;
        ctxL.strokeRect(PR.px, PR.py, PR.pw, PR.ph);
        ctxL.fillStyle = COL.ink;
        ctxL.font = "600 11.5px " + getComputedStyle(document.body).fontFamily;
        ctxL.fillText(usingOffline ? "Converged Offline SIREN (3,000 steps, width 44)" : "Live Pointwise HJI Residual |min{l − V, H − ∂_τ V}|", PR.px + 4, 15);

        // Loss Curve Box (bottom half)
        const lx0 = 48, ly0 = 216, lw = WL - 62, lh = 148;
        ctxL.strokeStyle = COL.line;
        ctxL.strokeRect(lx0, ly0, lw, lh);
        const ptsSrc = usingOffline && DATA.pinn ? DATA.pinn.loss : lossHist;
        if (ptsSrc.length > 1) {
          const maxStep = Math.max(100, ptsSrc[ptsSrc.length - 1][0]);
          const logVals = ptsSrc.map((q) => Math.log10(Math.max(q[1], 1e-7)));
          const yMin = Math.min(-4.5, ...logVals), yMax = Math.max(0.8, ...logVals);
          ctxL.save();
          ctxL.strokeStyle = COL.accent;
          ctxL.lineWidth = 2;
          ctxL.beginPath();
          ptsSrc.forEach(([st, ls], idx) => {
            const sx = lx0 + (st / maxStep) * lw;
            const sy = ly0 + (1 - (Math.log10(Math.max(ls, 1e-7)) - yMin) / (yMax - yMin)) * lh;
            idx ? ctxL.lineTo(sx, sy) : ctxL.moveTo(sx, sy);
          });
          ctxL.stroke();
          ctxL.restore();
          ctxL.fillStyle = COL.muted;
          ctxL.font = "11px " + getComputedStyle(document.body).fontFamily;
          ctxL.fillText(`Step 0 → ${maxStep} (log10 MSE residual loss, range 10^${yMin.toFixed(1)} to 10^${yMax.toFixed(1)})`, lx0 + 6, ly0 + lh + 18);
        } else {
          ctxL.fillStyle = COL.muted;
          ctxL.font = "12px " + getComputedStyle(document.body).fontFamily;
          ctxL.fillText("Click '▶ Train Live in Browser' to stream 2nd-order autodiff training!", lx0 + 18, ly0 + lh / 2);
        }

        const curLoss = usingOffline && DATA.pinn
          ? DATA.pinn.loss[DATA.pinn.loss.length - 1][1]
          : lossHist.length ? lossHist[lossHist.length - 1][1] : NaN;
        const stepDisp = usingOffline ? 3000 : net.step;
        const tauCur = usingOffline ? 4.0 : (useCurr ? Math.min(4.0, (4.0 * net.step) / 650) : 4.0);

        $("#pinn-readout").innerHTML =
          `<div class="hud-headline"><span class="pill ${iou > 0.92 ? "safe" : formType === "steady" ? "unsafe" : "accent"}">BRT IoU = ${iou.toFixed(3)}</span> ` +
          `<span>Step ${stepDisp.toLocaleString()} · Residual MSE = ${isFinite(curLoss) ? curLoss.toExponential(2) : "—"} · Curriculum τ<sub>max</sub> = ${tauCur.toFixed(2)} s</span></div>` +
          `<div>False-safe (dangerous): <b>${((100 * fs) / tot).toFixed(2)}%</b> · Over-conservative: <b>${((100 * oc) / tot).toFixed(2)}%</b> · Predicted V range: [${vmin.toFixed(2)}, ${vmax.toFixed(2)}] m</div>` +
          (formType === "steady"
            ? `<div style="color:var(--unsafe)"><b>Spurious Steady-State Collapse:</b> Residual loss drops toward 0 while V_θ flattens near ${vmin.toFixed(2)} m ≤ min l(x), marking 100% of the phase plane unsafe!</div>`
            : actType === "relu"
            ? `<div style="color:var(--warn)"><b>ReLU Gradient Pathology:</b> Because ReLU has h''(z) ≡ 0, 2nd-order PDE residual backprop loses curvature signal and leaves jagged kinks!</div>`
            : `<div class="muted">Exact hard boundary V_θ(x,0) ≡ l(x) + sinusoidal derivatives propagate the viscosity front smoothly from τ = 0 → 4 s.</div>`);
      }

      function loop() {
        if (!running) return;
        for (let k = 0; k < 5; k++) trainBatch(160);
        draw();
        if (net.step < 2400) raf = requestAnimationFrame(loop);
        else { running = false; $("#pinn-toggle-btn").textContent = "▶ Train More Steps"; }
      }

      const stopLive = () => {
        running = false;
        cancelAnimationFrame(raf);
        $("#pinn-toggle-btn").textContent = "▶ Train Live in Browser";
      };

      $("#pinn-toggle-btn").addEventListener("click", () => {
        if (usingOffline) initNet();
        running = !running;
        $("#pinn-toggle-btn").textContent = running ? "⏸ Pause Live Training" : "▶ Resume Live Training";
        if (running) raf = requestAnimationFrame(loop);
      });
      $("#pinn-reset-btn").addEventListener("click", () => { stopLive(); initNet(); draw(); });
      $("#pinn-offline-btn").addEventListener("click", () => {
        stopLive();
        formType = "deepreach"; actType = "siren";
        $$("#pinn-act-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.act === "siren"));
        $$("#pinn-form-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.form === "deepreach"));
        usingOffline = true;
        draw();
      });

      $$("#pinn-act-tabs button").forEach((b) => b.addEventListener("click", () => {
        $$("#pinn-act-tabs button").forEach((x) => x.classList.toggle("active", x === b));
        stopLive(); actType = b.dataset.act; initNet(); draw();
      }));
      $$("#pinn-form-tabs button").forEach((b) => b.addEventListener("click", () => {
        $$("#pinn-form-tabs button").forEach((x) => x.classList.toggle("active", x === b));
        stopLive(); formType = b.dataset.form; initNet(); draw();
      }));
      $("#pinn-curr").addEventListener("change", (e) => { useCurr = e.target.checked; stopLive(); initNet(); draw(); });

      HJR.onThemeChange(draw);
      initNet();
      if (DATA.pinn) {
        usingOffline = true;
      } else {
        for (let k = 0; k < 220; k++) trainBatch(128);
      }
      draw();
    })();

    /* =====================================================================
       LAB 07: CLUTTERED NAVIGATION PROGRESSION (FROM LOCAL LOSS TO GLOBAL HJB)
       ===================================================================== */
    (function initLab07() {
      const cv = $("#neuro-map-cv");
      if (!cv || !DATA.neurohjr) return;

      const NH = DATA.neurohjr;
      const W = 45.0, H = 0.25, N = Math.round(W / H) + 1, RD = 3.0;
      const START = [1.0, 1.0], GOAL = [40.0, 40.0];
      const PM = HJR.createPlot2D(cv, { xlim: [0, W], ylim: [0, W], margin: { l: 42, r: 12, t: 12, b: 36 } });

      let obs = NH.obstacles.map((o) => o.slice());
      let rho = 5.0;
      let dragIdx = -1;
      let sol = null;

      function solveMapAndRollouts() {
        const t0 = performance.now();
        const free = new Uint8Array(N * N);
        for (let i = 0; i < N; i++) {
          const x = i * H;
          for (let j = 0; j < N; j++) {
            const y = j * H;
            let ok = 1;
            for (const [cx, cy] of obs) {
              if ((x - cx) ** 2 + (y - cy) ** 2 <= RD * RD) { ok = 0; break; }
            }
            free[i * N + j] = ok;
          }
        }

        // Exact 8-connected Eikonal / Dijkstra with Euclidean step weights (H or sqrt(2)*H under ||u||_2 <= 1, or H under box |u_i| <= 1!)
        // Paper uses box control |ux|, |uy| <= 1 => L_inf speed = 1 => every 8-neighbor step takes H seconds!
        const T = new Float32Array(N * N).fill(Infinity);
        const gi = Math.round(GOAL[0] / H) * N + Math.round(GOAL[1] / H);
        T[gi] = 0;
        const hk = [], hv = [];
        const push = (t, v) => {
          hk.push(t); hv.push(v);
          let i = hk.length - 1;
          while (i > 0) {
            const p = (i - 1) >> 1;
            if (hk[p] <= hk[i]) break;
            [hk[p], hk[i]] = [hk[i], hk[p]];
            [hv[p], hv[i]] = [hv[i], hv[p]];
            i = p;
          }
        };
        const pop = () => {
          const t = hk[0], v = hv[0], lt = hk.pop(), lv = hv.pop();
          if (hk.length) {
            hk[0] = lt; hv[0] = lv;
            let i = 0;
            for (;;) {
              const l = 2 * i + 1, r = l + 1;
              let m = i;
              if (l < hk.length && hk[l] < hk[m]) m = l;
              if (r < hk.length && hk[r] < hk[m]) m = r;
              if (m === i) break;
              [hk[m], hk[i]] = [hk[i], hk[m]];
              [hv[m], hv[i]] = [hv[i], hv[m]];
              i = m;
            }
          }
          return [t, v];
        };
        push(0, gi);
        while (hk.length) {
          const [t, v] = pop();
          if (t > T[v]) continue;
          const i = (v / N) | 0, j = v % N;
          for (let di = -1; di <= 1; di++) {
            for (let dj = -1; dj <= 1; dj++) {
              if (!di && !dj) continue;
              const a = i + di, b = j + dj;
              if (a < 0 || b < 0 || a >= N || b >= N) continue;
              const w = a * N + b;
              if (!free[w] || t + H >= T[w]) continue;
              T[w] = t + H;
              push(t + H, w);
            }
          }
        }
        const solveMs = performance.now() - t0;

        // 1. Optimal Eikonal HJB Descent (-grad T)
        const optPath = [];
        let ci = Math.round(START[0] / H), cj = Math.round(START[1] / H);
        if (isFinite(T[ci * N + cj])) {
          optPath.push([ci * H, cj * H]);
          let guard = 0;
          while (T[ci * N + cj] > 0 && guard++ < 4 * N) {
            let bi = ci, bj = cj, bt = T[ci * N + cj];
            for (let di = -1; di <= 1; di++) {
              for (let dj = -1; dj <= 1; dj++) {
                const a = ci + di, b = cj + dj;
                if (a >= 0 && b >= 0 && a < N && b < N && T[a * N + b] < bt) {
                  bt = T[a * N + b]; bi = a; bj = b;
                }
              }
            }
            if (bi === ci && bj === cj) break;
            ci = bi; cj = bj;
            optPath.push([ci * H, cj * H]);
          }
        }

        // Helper: rollout a policy fn(p) -> [ux, uy] in [-1, 1]^2
        const rollout = (policyFn) => {
          let p = START.slice();
          const pts = [p.slice()], dt = 0.08;
          for (let k = 0; k < 1400; k++) {
            const u = policyFn(p);
            p = [p[0] + clamp(u[0], -1, 1) * dt, p[1] + clamp(u[1], -1, 1) * dt];
            pts.push(p.slice());
            for (const [cx, cy] of obs) {
              if (Math.hypot(p[0] - cx, p[1] - cy) < RD - 0.02) return { pts, out: "COLLISION", t: (k + 1) * dt };
            }
            if (Math.hypot(GOAL[0] - p[0], GOAL[1] - p[1]) < 0.6) return { pts, out: "GOAL", t: (k + 1) * dt };
            if (pts.length > 140 && Math.hypot(p[0] - pts[pts.length - 130][0], p[1] - pts[pts.length - 130][1]) < 0.35) {
              return { pts, out: "STUCK", t: (k + 1) * dt };
            }
          }
          return { pts, out: "TIMEOUT", t: 112 };
        };

        // 2. Uncoupled collocation penalty (d(L_obs)/d(theta) == 0, L_goal = -<u, r_goal> => u = tanh(big * r_goal))
        const rAsWritten = rollout((p) => {
          const dx = GOAL[0] - p[0], dy = GOAL[1] - p[1], n = Math.hypot(dx, dy) + 1e-9;
          return [Math.tanh(4 * dx / n), Math.tanh(4 * dy / n)];
        });

        // 3. Generous 1-Step Lookahead Fix: outside sensor radius rho, straight line; inside rho, minimize 1-step lookahead penalty + goal alignment
        const rLookahead = rollout((p) => {
          const dxG = GOAL[0] - p[0], dyG = GOAL[1] - p[1], nG = Math.hypot(dxG, dyG) + 1e-9;
          const rgx = dxG / nG, rgy = dyG / nG;
          let minD = Infinity;
          for (const [cx, cy] of obs) minD = Math.min(minD, Math.hypot(p[0] - cx, p[1] - cy));
          if (minD > rho) return [clamp(dxG, -1, 1), clamp(dyG, -1, 1)];
          // Optimize u in [-1,1]^2 to minimize 1-step violation + tangent bias - <u, r_goal>
          let bestU = [rgx, rgy], bestJ = Infinity;
          for (let a = 0; a < 24; a++) {
            const ang = (2 * Math.PI * a) / 24;
            const ux = clamp(1.25 * Math.cos(ang), -1, 1), uy = clamp(1.25 * Math.sin(ang), -1, 1);
            const px1 = p[0] + 0.35 * ux, py1 = p[1] + 0.35 * uy;
            let pen = 0;
            for (const [cx, cy] of obs) {
              const d = Math.hypot(px1 - cx, py1 - cy);
              if (d < RD + 0.85) pen += 18 * (RD + 0.85 - d) ** 2;
            }
            const J = pen - (ux * rgx + uy * rgy);
            if (J < bestJ) { bestJ = J; bestU = [ux, uy]; }
          }
          return bestU;
        });

        // 4. Classical APF
        const rApf = rollout((p) => {
          const dxG = GOAL[0] - p[0], dyG = GOAL[1] - p[1], nG = Math.hypot(dxG, dyG) + 1e-9;
          let fx = (2.0 * dxG) / nG, fy = (2.0 * dyG) / nG;
          for (const [cx, cy] of obs) {
            const rx = p[0] - cx, ry = p[1] - cy, rn = Math.hypot(rx, ry) + 1e-9, h = rn - RD;
            if (h < 3.2) {
              const hc = Math.max(h, 0.03), mag = 11 * (1 / hc - 1 / 3.2) / (hc * hc);
              fx += (mag * rx) / rn; fy += (mag * ry) / rn;
            }
          }
          return [fx + 0.03, fy - 0.02];
        });

        sol = { T, solveMs, optPath, rAsWritten, rLookahead, rApf };
      }

      const pathLen = (pts) => {
        let s = 0;
        for (let i = 1; i < pts.length; i++) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
        return s;
      };

      function draw() {
        if (!sol) solveMapAndRollouts();
        const { T, solveMs, optPath, rAsWritten, rLookahead, rApf } = sol;
        PM.clear();

        let tmax = 1;
        for (let i = 0; i < T.length; i++) if (isFinite(T[i]) && T[i] > tmax) tmax = T[i];
        HJR.drawScalarField(PM, T, N, N, 0, W, 0, W, (t) =>
          isFinite(t)
            ? mixRgb(COL["bg-hud_rgb"], COL.accent_rgb, 0.05 + 0.36 * (1 - t / tmax))
            : mixRgb(COL["bg-hud_rgb"], COL.unsafe_rgb, 0.32)
        );

        PM.clip();
        // Wavefront contours every 5 seconds
        const Tfinite = new Float32Array(T.length);
        for (let i = 0; i < T.length; i++) Tfinite[i] = isFinite(T[i]) ? T[i] : 9999;
        for (let lv = 5; lv < tmax; lv += 5) {
          HJR.strokeSegments(PM, HJR.marchingSquares(Tfinite, N, N, 0, W, 0, W, lv), rgba(COL.ink_rgb, 0.16), 1.0);
        }

        const c = PM.ctx;
        for (const [cx, cy] of obs) {
          // Sensor trigger ring rho (subtle) + Safety margin R_d=3m + Core R=2m
          c.strokeStyle = rgba(COL.warn_rgb, 0.25);
          c.lineWidth = 1;
          c.setLineDash([3, 3]);
          c.beginPath(); c.arc(PM.X(cx), PM.Y(cy), PM.X(rho) - PM.X(0), 0, 2 * Math.PI); c.stroke();
          c.setLineDash([]);

          c.fillStyle = rgba(COL.unsafe_rgb, 0.24);
          c.strokeStyle = COL.unsafe;
          c.lineWidth = 1.5;
          c.beginPath(); c.arc(PM.X(cx), PM.Y(cy), PM.X(RD) - PM.X(0), 0, 2 * Math.PI); c.fill(); c.stroke();

          c.fillStyle = COL.ink;
          c.beginPath(); c.arc(PM.X(cx), PM.Y(cy), PM.X(2.0) - PM.X(0), 0, 2 * Math.PI); c.fill();
        }

        HJR.drawPolyline(PM, optPath, COL.safe, 3.2);
        HJR.drawPolyline(PM, rApf.pts, COL["c-apf"], 2.1, [6, 4]);
        HJR.drawPolyline(PM, rLookahead.pts, COL["c-cbf"], 2.3);
        HJR.drawPolyline(PM, rAsWritten.pts, COL.unsafe, 2.5, [3, 3]);

        // Collision X on As-Written
        if (rAsWritten.out === "COLLISION") {
          const ep = rAsWritten.pts[rAsWritten.pts.length - 1];
          const sx = PM.X(ep[0]), sy = PM.Y(ep[1]);
          c.strokeStyle = COL.unsafe; c.lineWidth = 3;
          c.beginPath(); c.moveTo(sx - 7, sy - 7); c.lineTo(sx + 7, sy + 7); c.moveTo(sx + 7, sy - 7); c.lineTo(sx - 7, sy + 7); c.stroke();
        }

        HJR.drawDot(PM, START[0], START[1], 6, COL.safe, COL["bg-elev"], 2);
        c.fillStyle = COL["c-gold"]; c.font = "bold 18px " + getComputedStyle(document.body).fontFamily;
        c.textAlign = "center"; c.textBaseline = "middle";
        c.fillText("★", PM.X(GOAL[0]), PM.Y(GOAL[1]));
        c.restore();
        PM.axes("P_x [m]", "P_y [m]", [0, 10, 20, 30, 40], [0, 10, 20, 30, 40]);

        const tOpt = T[Math.round(START[0] / H) * N + Math.round(START[1] / H)];
        $("#neuro-readout").innerHTML =
          `<div class="hud-headline"><span class="pill safe">EXACT EIKONAL HJB SOLVE: ${solveMs.toFixed(1)} ms</span> <span>(32,761 grid nodes, zero retraining per map)</span></div>` +
          `<div><span style="color:var(--safe)">■</span> <b>Exact Eikonal HJB (−∇T):</b> ${isFinite(tOpt) ? `${tOpt.toFixed(1)} s · ${pathLen(optPath).toFixed(1)} m` : "Blocked"} <span class="pill safe">GLOBAL OPTIMUM</span></div>` +
          `<div><span style="color:var(--c-cbf)">■</span> <b>1-Step Lookahead Policy (ρ=${rho.toFixed(1)}m):</b> <span class="pill ${rLookahead.out === "GOAL" ? "accent" : "warn"}">${rLookahead.out}</span> ${pathLen(rLookahead.pts).toFixed(1)} m (local horizon)</div>` +
          `<div><span style="color:var(--c-apf)">■</span> <b>Classical APF Baseline:</b> <span class="pill ${rApf.out === "GOAL" ? "accent" : "warn"}">${rApf.out}</span> ${pathLen(rApf.pts).toFixed(1)} m</div>` +
          `<div><span style="color:var(--unsafe)">■</span> <b>Goal-Only / Uncoupled L_obs:</b> <span class="pill unsafe">${rAsWritten.out}</span> at ${pathLen(rAsWritten.pts).toFixed(1)} m (∇_θ L_obs = 0)</div>`;
      }

      HJR.bindDrag(cv, PM.evtPos, {
        down: (pos) => {
          const [x, y] = PM.inv(pos[0], pos[1]);
          dragIdx = obs.findIndex(([cx, cy]) => Math.hypot(x - cx, y - cy) < RD + 0.5);
          return dragIdx >= 0 ? true : false;
        },
        move: (pos, _e, hover) => {
          if (hover || dragIdx < 0) return;
          const [x, y] = PM.inv(pos[0], pos[1]);
          obs[dragIdx] = [clamp(x, 3, W - 3), clamp(y, 3, W - 3)];
          solveMapAndRollouts();
          draw();
        },
        up: () => { dragIdx = -1; },
      });

      $("#neuro-rho").addEventListener("input", (e) => {
        rho = +e.target.value;
        $("#neuro-rho-v").textContent = rho.toFixed(1) + " m";
        solveMapAndRollouts();
        draw();
      });

      let randSeed = 51;
      $$("#neuro-map-tabs button").forEach((b) => b.addEventListener("click", () => {
        $$("#neuro-map-tabs button").forEach((x) => x.classList.toggle("active", x === b));
        const kind = b.dataset.nmap;
        if (kind === "paper") obs = NH.obstacles.map((o) => o.slice());
        else if (kind === "utrap") {
          obs = [-35, 5, 45, 85, 125].map((deg) => {
            const r = (deg * Math.PI) / 180;
            return [22 + 7.2 * Math.cos(r), 22 + 7.2 * Math.sin(r)];
          });
        } else {
          let s = randSeed++;
          const r = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
          obs = [];
          let guard = 0;
          while (obs.length < 10 && guard++ < 4000) {
            const u = 0.14 + 0.78 * r();
            const c = [1 + 39 * u + (r() - 0.5) * 18, 1 + 39 * u + (r() - 0.5) * 18];
            if (c[0] > 4 && c[0] < 41 && c[1] > 4 && c[1] < 41 && Math.hypot(c[0] - 1, c[1] - 1) > 6.5 && Math.hypot(c[0] - 40, c[1] - 40) > 6.5 && obs.every((o) => Math.hypot(c[0] - o[0], c[1] - o[1]) > 7.5)) {
              obs.push(c);
            }
          }
        }
        solveMapAndRollouts();
        draw();
      }));

      HJR.onThemeChange(draw);
      solveMapAndRollouts();
      draw();
    })();

    /* =====================================================================
       LAB 08: CONFORMAL / SCENARIO VERIFICATION EXPLORER (LIN & BANSAL)
       ===================================================================== */
    (function initLab08() {
      const cv = $("#ver-cv");
      if (!cv || !DATA.pinn) return;

      const P = HJR.createPlot2D(cv, { xlim: [-2, 12], ylim: [-4, 8], margin: { l: 46, r: 14, t: 16, b: 36 } });
      const R = DATA.pinn;
      const [x0, x1, nx] = R.p, [y0, y1, ny] = R.v;
      const dx = (x1 - x0) / (nx - 1), dy = (y1 - y0) / (ny - 1);

      const Vsmall = HJR.decodeI16(R.small);
      const Vfull = HJR.decodeI16(R.final);
      const Exact = new Float32Array(nx * ny);
      const Mask = new Uint8Array(nx * ny);
      for (let i = 0; i < nx; i++) {
        const p = x0 + i * dx;
        for (let j = 0; j < ny; j++) {
          const v = y0 + j * dy, vp = Math.max(v, 0), t = Math.min(vp / 2.5, 4.0);
          Exact[i * ny + j] = (10 - p) - (vp * t - 1.25 * t * t);
          Mask[i * ny + j] = p >= 0 && p <= 11 && v >= -3 && v <= 7 ? 1 : 0;
        }
      }

      let which = "small";
      function draw() {
        const delta = +$("#ver-delta").value;
        const negLogBeta = +$("#ver-conf").value; // beta = 10^-negLogBeta
        const beta = Math.pow(10, -negLogBeta);
        const confPct = (1 - beta) * 100;
        $("#ver-delta-v").textContent = delta.toFixed(3) + " m";
        $("#ver-conf-v").textContent = confPct.toFixed(negLogBeta >= 4 ? 3 : negLogBeta >= 3 ? 1 : 0) + "%";

        const F = which === "small" ? Vsmall : Vfull;
        const info = R.verify[which];
        const shifted = new Float32Array(F.length);
        let fs = 0, oc = 0, tot = 0;
        for (let k = 0; k < F.length; k++) {
          shifted[k] = F[k] - delta;
          if (!Mask[k]) continue;
          tot++;
          const t = Exact[k] < 0, s = F[k] <= delta;
          if (t && !s) fs++;
          if (!t && s) oc++;
        }

        P.clear();
        HJR.drawScalarField(P, shifted, nx, ny, x0, x1, y0, y1, HJR.valueColormap(10, -3.5));
        P.clip();
        HJR.strokeSegments(P, HJR.marchingSquares(F, nx, ny, x0, x1, y0, y1, delta), COL.ink, 2.4);
        HJR.strokeSegments(P, HJR.marchingSquares(Exact, nx, ny, x0, x1, y0, y1, 0), COL["c-gold"], 2.2, [6, 4]);

        let activeViol = 0;
        info.viol_pts.forEach((pt, idx) => {
          const stillInside = info.viol_V[idx] > delta;
          if (stillInside) activeViol++;
          HJR.drawDot(P, pt[0], pt[1], stillInside ? 3.8 : 2.5, stillInside ? COL.unsafe : rgba(COL.muted_rgb, 0.55), stillInside ? COL["bg-elev"] : null, 1);
        });
        P.ctx.restore();
        P.axes("position p [m]", "velocity v [m/s]", [-2, 0, 2, 4, 6, 8, 10, 12], [-4, -2, 0, 2, 4, 6, 8]);

        // Clopper-Pearson upper bound for k=0 (when delta >= info.delta) on N=10,000 validation rollouts: eps = 1 - beta^(1/N)
        const Nval = info.n_val;
        const certified = delta >= info.delta - 1e-4;
        const epsZero = 1 - Math.pow(beta, 1 / Nval);

        $("#ver-readout").innerHTML =
          `<div class="hud-headline">${
            certified
              ? `<span class="pill safe">CERTIFIED · 0 COUNTEREXAMPLES REMAIN</span>`
              : `<span class="pill unsafe">UNSAFE · ${activeViol} COUNTEREXAMPLES INSIDE {V_θ &gt; δ}</span>`
          }</div>` +
          `<div>Calibration (${info.n_cal.toLocaleString()} rollouts): found <b>${info.n_viol}</b> false-safe crashes → required shift <b>δ* = ${info.delta.toFixed(3)} m</b></div>` +
          `<div>Grid comparison at δ = ${delta.toFixed(3)} m: False-safe cells = <b>${((100 * fs) / tot).toFixed(2)}%</b> · Over-conservative cost = <b>${((100 * oc) / tot).toFixed(2)}%</b></div>` +
          `<div class="muted">${
            certified
              ? `Clopper-Pearson guarantee (${Nval.toLocaleString()} fresh rollouts, 1−β = ${confPct}%): P(crash | V_θ(x) > δ) ≤ ${(100 * epsZero).toFixed(3)}%`
              : `Raise δ to ≥ ${info.delta.toFixed(3)} m so every red counterexample state is excluded from the certified safe set.`
          }</div>`;
      }

      $$("#ver-model-tabs button").forEach((b) => b.addEventListener("click", () => {
        $$("#ver-model-tabs button").forEach((x) => x.classList.toggle("active", x === b));
        which = b.dataset.vmod;
        $("#ver-delta").value = 0;
        draw();
      }));
      $("#ver-delta").addEventListener("input", draw);
      $("#ver-conf").addEventListener("input", draw);
      $("#ver-auto-btn").addEventListener("click", () => { $("#ver-delta").value = R.verify[which].delta; draw(); });
      $("#ver-zero-btn").addEventListener("click", () => { $("#ver-delta").value = 0; draw(); });

      HJR.onThemeChange(draw);
      draw();
    })();

    /* =====================================================================
       LAB 09: PRODUCTION DRONE ENVELOPE & ARCHITECTURE SIZER
       ===================================================================== */
    (function initLab09() {
      const cv = $("#ind-envelope-cv");
      const out = $("#ind-verdict");
      if (!cv || !out) return;

      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const W = 640, H = 195;
      cv.width = W * dpr; cv.height = H * dpr;
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const PRESETS = {
        dock:   { v: 6.0,  tilt: 25, gust: 1.2, lat: 90,  dim: 4, tag: "Precision Dock Landing / Abort" },
        forest: { v: 14.0, tilt: 35, gust: 1.8, lat: 60,  dim: 4, tag: "High-Speed Forest RL Wrapper (RTA)" },
        vtol:   { v: 20.0, tilt: 30, gust: 2.0, lat: 140, dim: 3, tag: "Fixed-Wing VTOL Air3D Deconfliction" },
        slung:  { v: 8.0,  tilt: 28, gust: 1.5, lat: 80,  dim: 9, tag: "Slung-Load Cable-Payload Quadrotor" },
        indoor: { v: 1.0,  tilt: 20, gust: 0.1, lat: 50,  dim: 2, tag: "Slow Indoor Warehouse Inspection" },
      };

      function update() {
        const v = +$("#ind-v").value;
        const tilt = +$("#ind-tilt").value;
        const gust = +$("#ind-gust").value;
        const latMs = +$("#ind-lat").value;
        const dim = +$("#ind-dim").value;

        $("#ind-v-v").textContent = v.toFixed(1) + " m/s";
        $("#ind-tilt-v").textContent = tilt + "°";
        $("#ind-gust-v").textContent = gust.toFixed(1) + " m/s²";
        $("#ind-lat-v").textContent = latMs + " ms";
        $("#ind-dim-v").textContent = dim + "D";

        const g = 9.81;
        const aMax = g * Math.tan((tilt * Math.PI) / 180);
        const ae = aMax - gust;
        const tLat = latMs / 1000;

        // Stopping distance breakdown:
        const dLat = v * tLat + 0.5 * gust * tLat * tLat;
        const vAfterLat = v + gust * tLat;
        const dIdealBrake = (v * v) / (2 * aMax);
        const dTotalBrake = ae > 0.15 ? (vAfterLat * vAfterLat) / (2 * ae) : 99.0;
        const dGustExtra = Math.max(0, dTotalBrake - dIdealBrake);
        const dGridMargin = dim <= 5 ? 0.35 : 0.75;
        const totalM = Math.min(99, dLat + dIdealBrake + dGustExtra + dGridMargin);

        // Memory calculation at 41 pts/axis (int16 = 2 bytes/cell) vs PINN (~180 KB)
        const cells = Math.pow(35, dim);
        const gridMB = (cells * 2) / 1e6;

        // Draw Stacked Breakdown Bar + Memory Gauge
        ctx.fillStyle = COL["bg-hud"];
        ctx.fillRect(0, 0, W, H);
        const font = getComputedStyle(document.body).fontFamily;

        ctx.fillStyle = COL.ink;
        ctx.font = "600 12px " + font;
        ctx.fillText(`Minimum Worst-Case Stopping / Safety Envelope: ${ae > 0.15 ? totalM.toFixed(2) + " m" : "∞ (Gust ≥ Max Tilt Authority!)"}`, 16, 24);

        const bx = 16, by = 36, bw = W - 32, bh = 30;
        const scaleM = Math.max(totalM, 8.0);
        const segs = [
          { label: `Latency (${dLat.toFixed(2)}m)`, val: dLat, col: COL["c-gold"] },
          { label: `Calm Braking (${dIdealBrake.toFixed(2)}m)`, val: dIdealBrake, col: COL.accent },
          { label: `Gust Penalty (+${dGustExtra.toFixed(2)}m)`, val: dGustExtra, col: COL.unsafe },
          { label: `LUT/PINN δ (${dGridMargin.toFixed(2)}m)`, val: dGridMargin, col: COL.violet },
        ];
        let curX = bx;
        ctx.fillStyle = COL["bg-subtle"];
        ctx.fillRect(bx, by, bw, bh);
        for (const sg of segs) {
          const wSeg = clamp((sg.val / scaleM) * bw, 0, bw - (curX - bx));
          ctx.fillStyle = sg.col;
          ctx.fillRect(curX, by, wSeg, bh);
          curX += wSeg;
        }
        ctx.strokeStyle = COL.line;
        ctx.strokeRect(bx, by, bw, bh);

        // Legend for breakdown bar
        let lx = 16;
        ctx.font = "11px " + font;
        for (const sg of segs) {
          ctx.fillStyle = sg.col;
          ctx.fillRect(lx, 76, 9, 9);
          ctx.fillStyle = COL["ink-2"];
          ctx.fillText(sg.label, lx + 13, 84);
          lx += ctx.measureText(sg.label).width + 26;
        }

        // Memory Scaling Comparison (Log10 bar)
        ctx.fillStyle = COL.ink;
        ctx.font = "600 12px " + font;
        const memStr = gridMB < 1 ? `${(gridMB * 1024).toFixed(0)} KB` : gridMB < 1024 ? `${gridMB.toFixed(1)} MB` : gridMB < 1e6 ? `${(gridMB / 1024).toFixed(1)} GB` : `${(gridMB / 1e6).toFixed(0)} TB`;
        ctx.fillText(`${dim}D Grid LUT Memory (35^${dim} int16): ${memStr}   vs   Verified PINN Weights: 180 KB`, 16, 118);

        const logMB = clamp((Math.log10(Math.max(gridMB, 0.01)) + 2) / 11, 0.02, 1.0); // 10 KB (-2) to 100 PB (+9)
        const limitX = bx + ((Math.log10(256) + 2) / 11) * bw; // 256 MB onboard RAM budget line
        ctx.fillStyle = COL["bg-subtle"];
        ctx.fillRect(bx, 128, bw, 22);
        ctx.fillStyle = gridMB <= 256 ? COL.safe : COL.unsafe;
        ctx.fillRect(bx, 128, logMB * bw, 22);
        ctx.strokeStyle = COL["c-gold"];
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(limitX, 122); ctx.lineTo(limitX, 156); ctx.stroke();
        ctx.fillStyle = COL["c-gold"];
        ctx.font = "10.5px " + font;
        ctx.fillText("256 MB Embedded LUT Limit", limitX + 5, 168);

        // Engineering Recommendation Card
        let title, badge, body, alt;
        if (ae <= 0.15) {
          badge = `<span class="pill unsafe">INFEASIBLE AIRFRAME AUTHORITY</span>`;
          title = "Max Tilt Cannot Overcome Stated Wind Gust!";
          body = `At θ<sub>max</sub> = ${tilt}°, horizontal accel is ${aMax.toFixed(2)} m/s² vs ${gust.toFixed(2)} m/s² gust. Increase max tilt limit or restrict dispatch wind envelope before designing any controller.`;
          alt = "Hardware / flight-envelope limit—no software filter can invent thrust.";
        } else if (totalM < 0.7 && v <= 1.5) {
          badge = `<span class="pill warn">SKIP HJR · GEOMETRIC MARGIN SUFFICES</span>`;
          title = "Quasi-Static Regime (BRT ≈ Obstacle)";
          body = `Your worst-case stopping + latency envelope is only <b>${totalM.toFixed(2)} m</b>. Inflate your ESDF / voxel map by ${totalM.toFixed(2)} m and run standard MPC or a simple velocity CBF. Full HJR adds complexity with negligible gain.`;
          alt = "PX4 Collision Prevention 1D sector check as lightweight backup.";
        } else if (dim <= 4) {
          badge = `<span class="pill safe">SHIP TODAY · OFFLINE ${dim}D GRID LUT + CBVF</span>`;
          title = `Precomputed ${dim}D Quantized Table (${memStr}) + Smooth CBVF-QP`;
          body = `Fits comfortably in RAM (${memStr}), evaluates in <b>~2 µs</b> in C++ with zero neural-network verification headaches, and handles the full <b>${totalM.toFixed(2)} m</b> momentum + gust + latency envelope exactly.`;
          alt = "Also pair with ASTM F3269 RTA switch if filtering a learned policy.";
        } else if (dim <= 6) {
          badge = `<span class="pill accent">DECOMPOSE FIRST · FASTRACK OR 3D+3D SPLIT</span>`;
          title = `Raw ${dim}D Grid Costs ${memStr} — Use FaSTrack Decomposition!`;
          body = `Before reaching for PINNs, decouple longitudinal/lateral/vertical relative dynamics (e.g. two 3D FaSTrack tables = <b>~180 KB total</b> instead of ${memStr}!). Inflate planner obstacles by TEB online.`;
          alt = "If dynamics are strongly coupled (e.g. high-bank aerobatics), use DeepReach + Conformal δ.";
        } else {
          badge = `<span class="pill accent">VERIFIED PINN SWEET SPOT (${dim}D)</span>`;
          title = `DeepReach SIREN PINN (180 KB) + Conformal δ + RTA Backup`;
          body = `A ${dim}D grid (${memStr}) is physically impossible. Train a time-dependent SIREN PINN conditioned on payload/wind parameters, calibrate level δ via 20k+ worst-case rollouts (Chapter 8), and deploy strictly as a Run-Time Assurance monitor with a verified emergency fallback.`;
          alt = "Never deploy an uncalibrated PINN directly in primary flight control.";
        }

        out.innerHTML = `
          <div class="card highlight">
            <div>${badge}</div>
            <h3>${title}</h3>
            <p class="small">${body}</p>
            <div class="kpi-strip" style="margin-top:0.35rem">
              <div class="kpi"><div class="kpi-label">Max Accel g·tanθ</div><div class="kpi-val">${aMax.toFixed(1)} m/s²</div></div>
              <div class="kpi"><div class="kpi-label">Net Braking a_e</div><div class="kpi-val">${ae.toFixed(1)} m/s²</div></div>
              <div class="kpi"><div class="kpi-label">Safety Envelope</div><div class="kpi-val">${ae > 0.15 ? totalM.toFixed(1) + " m" : "∞"}</div></div>
            </div>
            <p class="small muted" style="margin-top:0.3rem"><strong>Secondary note:</strong> ${alt}</p>
          </div>
        `;
      }

      $$("#ind-preset-tabs button").forEach((b) => b.addEventListener("click", () => {
        $$("#ind-preset-tabs button").forEach((x) => x.classList.toggle("active", x === b));
        const p = PRESETS[b.dataset.ipre];
        $("#ind-v").value = p.v;
        $("#ind-tilt").value = p.tilt;
        $("#ind-gust").value = p.gust;
        $("#ind-lat").value = p.lat;
        $("#ind-dim").value = p.dim;
        update();
      }));
      ["ind-v", "ind-tilt", "ind-gust", "ind-lat", "ind-dim"].forEach((id) => $("#" + id).addEventListener("input", update));

      HJR.onThemeChange(update);
      update();
    })();

    /* =====================================================================
       LAB 10: 5-SCENARIO AUTONOMY LEAD DIAGNOSTIC QUIZ
       ===================================================================== */
    (function initLab10() {
      const box = $("#quiz-container");
      if (!box) return;

      const QUESTIONS = [
        {
          tag: "Scenario 01 · Autonomous Shipboard Dock Landing",
          q: "Your 4 kg quadrotor lands on a heaving 1.8 m vessel pad in 10 m/s gusts. You need an exact millisecond go-around trigger when safe touchdown becomes impossible.",
          opts: [
            ["Artificial Potential Field around the deck coaming", false, "APF repels you away from the pad you're trying to land on and has zero notion of vertical slam velocity or gust feasibility."],
            ["Offline 4D Reach-Avoid HJR LUT (x, z, vx, vz) + Go-Around Trigger", true, "Spot on! Relative to the pad, 4D Reach-Avoid fits in ~2 MB of RAM, evaluates in 2 µs, and gives the exact V_RA < 0 go-around boundary."],
            ["Per-map steady-state PINN without stage cost", false, "Steady-state HJB without running cost collapses to constants, and 4D fits easily on an exact verifiable grid without neural approximation error."],
          ],
        },
        {
          tag: "Scenario 02 · Vision-RL Policy Flying at 14 m/s in Forests",
          q: "An end-to-end neural policy flies aggressively through trees using depth vision. Management wants a certifiable safety net without making flight sluggish.",
          opts: [
            ["ASTM F3269 Run-Time Assurance using a 4D relative HJR/CBVF filter over local ESDF obstacles", true, "Exactly right! Relative to the nearest LiDAR/ESDF obstacle points, a 4D CBVF/RTA filter lets RL fly freely until V(x) nears ε, then blends/switches to verified braking-and-swerving."],
            ["Tune a high-gain linear HOCBF with fixed k = 3.0", false, "At 14 m/s under hard tilt limits, a fixed linear HOCBF safe set crosses the braking parabola—causing sudden QP infeasibility right before tree trunks!"],
            ["Retrain a 2D map-specific PINN before every forest flight", false, "You don't even have the forest map beforehand, and 2D ignores velocity & tilt limits."],
          ],
        },
        {
          tag: "Scenario 03 · 10D Quadrotor With Suspended 3 m Pendulum Payload",
          q: "A delivery drone carries a heavy medical crate on a 3 m cable (10 coupled states: drone pos/vel/tilt + cable swing angles & rates) near buildings.",
          opts: [
            ["Full 10D uniform finite-difference grid solver", false, "At 35^10 cells, one float16 array requires ~5.5 Terabytes of RAM—the curse of dimensionality!"],
            ["DeepReach time-dependent SIREN PINN + Conformal δ-Verification + Backup Hover Mode", true, "Bullseye! Coupled 10D dynamics cannot be gridded directly; this is the genuine sweet spot for parameter-conditioned DeepReach PINNs paired with Lin & Bansal statistical verification."],
            ["Ignore cable swing and use a 1D braking parabola", false, "When the drone brakes hard, the 3 m cable swings forward past the drone and smashes the payload into the building!"],
          ],
        },
        {
          tag: "Scenario 04 · Fast Global A* Planner + Agile Multirotor in Wind",
          q: "Your onboard A* planner replans 3D paths in 4 ms, but the real quadrotor cuts sharp corners in wind and clips warehouse racks.",
          opts: [
            ["FaSTrack: Offline relative HJR Tracking Error Bound (TEB) + inflate A* obstacles by TEB", true, "Textbook FaSTrack! Solve the relative tracker-vs-planner differential game once offline (~180 KB), inflate racks by TEB meters, and keep your 4 ms A* planner untouched."],
            ["Run full 6D grid HJR online over the entire warehouse at 20 Hz", false, "Unnecessary and far too expensive when relative planner-tracker decomposition solves it offline independent of map layout!"],
            ["Add stronger attractive gain in APF", false, "Stronger attraction makes corner-cutting overshoot even worse."],
          ],
        },
        {
          tag: "Scenario 05 · Slow 0.4 m/s Indoor Inventory Micro-Drone",
          q: "A 250 g ducted micro-drone crawls at 0.4 m/s down calm indoor aisles scanning barcodes (stops within 2 cm).",
          opts: [
            ["Train a custom DeepReach PINN with 50,000 collocation points", false, "Massive overkill! At 0.4 m/s in calm indoor air, stopping distance is 2 cm—the reachable tube is virtually identical to the obstacle."],
            ["Simple geometric ESDF margin + standard velocity controller / basic CBF", true, "Correct! Great engineers know when NOT to use HJR: when momentum and disturbances are negligible, geometric clearance is simpler, cheaper, and completely sufficient."],
            ["3D Dubins fixed-wing reachable horn", false, "A hovering micro-drone can stop in place; it has no minimum stall speed."],
          ],
        },
      ];

      box.innerHTML = QUESTIONS.map((item, qi) => `
        <div class="card" id="qcard-${qi}">
          <span class="tag">${item.tag}</span>
          <h4>${item.q}</h4>
          <div class="quiz-options">
            ${item.opts.map((o, oi) => `<button type="button" class="quiz-opt" data-qi="${qi}" data-oi="${oi}">${o[0]}</button>`).join("")}
          </div>
          <div class="small muted" id="qfb-${qi}" style="min-height:1.2rem"></div>
        </div>
      `).join("");

      $$(".quiz-opt", box).forEach((btn) => btn.addEventListener("click", () => {
        const qi = +btn.dataset.qi, oi = +btn.dataset.oi;
        const chosen = QUESTIONS[qi].opts[oi];
        $$(`[data-qi="${qi}"]`, box).forEach((b, idx) => {
          b.classList.remove("correct", "wrong");
          if (idx === oi) b.classList.add(chosen[1] ? "correct" : "wrong");
          else if (QUESTIONS[qi].opts[idx][1] && !chosen[1]) b.classList.add("correct");
        });
        const fb = $("#qfb-" + qi);
        fb.style.color = chosen[1] ? "var(--safe)" : "var(--unsafe)";
        fb.innerHTML = `<strong>${chosen[1] ? "✓ Spot on:" : "✕ Watch out:"}</strong> ${chosen[2]}`;
      }));
    })();
  });
})();
