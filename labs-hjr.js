/* ==========================================================================
   labs-hjr.js — Interactive Flight-Dynamics Labs 01 to 05:
     Lab 01: Synchronized Physical Quadrotor + 2D Phase-Space BRT (with Latency)
     Lab 02: 3D Level-Set Surface Explorer (Wall, BRS vs BRT Pillar, Viscosity Kink)
     Lab 03: 3D Fixed-Wing / Cruise VTOL Twisted Reachable Tube (Air3D / Dubins)
     Lab 04: 3-Mode Drone Deck (4D Arena + 4D Reach-Avoid Dock + FaSTrack 6D Quad)
     Lab 05: 5-Filter Arena Showdown + Live Control-Space (ux, uy) QP Microscope
   ========================================================================== */
(function () {
  "use strict";

  window.addEventListener("DOMContentLoaded", () => {
    const HJR = window.HJR;
    const DATA = window.HJR_DATA || {};
    const { $, $$, clamp, COL, rgba, mixRgb } = HJR;

    /* =====================================================================
       LAB 01: PHYSICAL QUADROTOR <-> PHASE-SPACE POINT OF NO RETURN
       ===================================================================== */
    (function initLab01() {
      const cvFlight = $("#hero-flight-cv");
      const cvPhase = $("#hero-phase-cv");
      if (!cvFlight || !cvPhase) return;

      const PF = HJR.createPlot2D(cvFlight, { xlim: [-2, 12], ylim: [0, 3], margin: { l: 48, r: 16, t: 10, b: 26 } });
      const PP = HJR.createPlot2D(cvPhase, { xlim: [-2, 12], ylim: [-4, 8], margin: { l: 48, r: 16, t: 12, b: 38 } });

      const PWALL = 10.0;
      const nx = 141, ny = 121, x0 = -2, x1 = 12, y0 = -4, y1 = 8;
      const dp = (x1 - x0) / (nx - 1), dv = (y1 - y0) / (ny - 1);
      const L0 = new Float32Array(nx * ny);
      for (let i = 0; i < nx; i++) {
        const lval = PWALL - (x0 + i * dp);
        for (let j = 0; j < ny; j++) L0[i * ny + j] = lval;
      }

      let amax = 3.0, dmax = 0.5, latMs = 120, tau = 4.0;
      let state = [6.0, 5.0]; // [p, v]
      let snaps = [L0.slice()];
      const DT = 0.02, SNAP_EVERY = 4, NSTEPS = 200;
      let solveTicket = 0;
      let animFlight = null, animTau = null, flightTrace = null, rotorPhase = 0;

      // Exact value including pipeline latency: during [0, t_lat], worst-case coasting has u_prev = 0 (or worst drift +dmax),
      // then full braking -(amax - dmax) over remaining horizon!
      function exactValWithLatency(p, v, horizon, a, d, tLat) {
        const ae = a - d;
        const tc = Math.min(tLat, horizon);
        // Coasting under tailwind gust d during reaction latency tc:
        const p1 = p + v * tc + 0.5 * d * tc * tc;
        const v1 = v + d * tc;
        let minMargin = Math.min(PWALL - p, PWALL - p1);
        const rem = Math.max(0, horizon - tc);
        if (rem <= 1e-6 || v1 <= 0) return minMargin;
        if (ae <= 1e-6) return PWALL - (p1 + v1 * rem);
        const tStop = Math.min(v1 / ae, rem);
        const pPeak = p1 + v1 * tStop - 0.5 * ae * tStop * tStop;
        return Math.min(minMargin, PWALL - pPeak);
      }

      function solveDP() {
        const myTicket = ++solveTicket;
        snaps = [L0.slice()];
        let V = L0.slice(), W = new Float32Array(nx * ny);
        let step = 0;
        const accs = [[-amax - dmax, -amax + dmax], [amax - dmax, amax + dmax]];

        function chunk() {
          if (myTicket !== solveTicket) return;
          const tStart = performance.now();
          while (step < NSTEPS && performance.now() - tStart < 10) {
            for (let i = 0; i < nx; i++) {
              const p = x0 + i * dp;
              for (let j = 0; j < ny; j++) {
                const v = y0 + j * dv;
                let best = -1e9;
                for (let ui = 0; ui < 2; ui++) {
                  let worst = 1e9;
                  for (let di = 0; di < 2; di++) {
                    const ac = accs[ui][di];
                    const pn = p + v * DT + 0.5 * ac * DT * DT;
                    const vn = v + ac * DT;
                    const val = HJR.interp2D(V, nx, ny, x0, x1, y0, y1, pn, vn);
                    if (val < worst) worst = val;
                  }
                  if (worst > best) best = worst;
                }
                const idx = i * ny + j;
                W[idx] = Math.min(L0[idx], best);
              }
            }
            const tmp = V; V = W; W = tmp;
            step++;
            if (step % SNAP_EVERY === 0) snaps.push(V.slice());
          }
          render();
          if (step < NSTEPS) requestAnimationFrame(chunk);
        }
        requestAnimationFrame(chunk);
      }

      function getActiveField() {
        const tLat = latMs / 1000;
        const effTau = Math.max(0, tau - tLat);
        const idx = Math.min(snaps.length - 1, Math.round(effTau / (DT * SNAP_EVERY)));
        const baseV = snaps[idx];
        const tauUsed = idx * DT * SNAP_EVERY + Math.min(tau, tLat);
        if (tLat <= 1e-4 || tau <= 1e-4) return { V: baseV, tauHave: idx * DT * SNAP_EVERY };
        // Pull back baseV along latency coasting flow (p -> p + v*tc + 0.5*d*tc^2, v -> v + d*tc)
        const tc = Math.min(tLat, tau);
        const out = new Float32Array(nx * ny);
        for (let i = 0; i < nx; i++) {
          const p = x0 + i * dp;
          for (let j = 0; j < ny; j++) {
            const v = y0 + j * dv;
            const p1 = p + v * tc + 0.5 * dmax * tc * tc;
            const v1 = v + dmax * tc;
            const vAfter = HJR.interp2D(baseV, nx, ny, x0, x1, y0, y1, p1, v1);
            out[i * ny + j] = Math.min(PWALL - p, PWALL - p1, vAfter);
          }
        }
        return { V: out, tauHave: tauUsed };
      }

      function computeRollout(p0, v0) {
        const tLat = latMs / 1000;
        const ae = amax - dmax;
        const coastPts = [[p0, v0]];
        const brakePts = [];
        let p = p0, v = v0;
        const dtSim = 0.015;
        let t = 0;
        while (t < Math.min(tLat, 3.0) - 1e-6) {
          const step = Math.min(dtSim, tLat - t);
          p += v * step + 0.5 * dmax * step * step;
          v += dmax * step;
          t += step;
          coastPts.push([p, v]);
          if (p > 12.2) break;
        }
        const pLatEnd = p, vLatEnd = v;
        brakePts.push([p, v]);
        while (t < 6.5 && p <= 12.2) {
          const u = v > 0 ? -amax : 0;
          const acc = u + dmax;
          p += v * dtSim + 0.5 * acc * dtSim * dtSim;
          v += acc * dtSim;
          t += dtSim;
          brakePts.push([p, v]);
          if (v <= 0 && ae > 0) break;
        }
        const maxP = Math.max(...coastPts.map((q) => q[0]), ...brakePts.map((q) => q[0]));
        return { coastPts, brakePts, pLatEnd, vLatEnd, maxP, crashed: maxP > PWALL };
      }

      function render() {
        const { V, tauHave } = getActiveField();
        const [p0, v0] = flightTrace ? flightTrace.cur : state;
        const roll = computeRollout(state[0], state[1]);
        const Vnow = HJR.interp2D(V, nx, ny, x0, x1, y0, y1, p0, v0);
        const doomed = Vnow < 0;

        /* --- Top Canvas: Physical Quadrotor Side View --- */
        PF.clear();
        const cf = PF.ctx;
        PF.clip();
        // Danger zone behind wall p >= 10
        cf.fillStyle = rgba(COL.unsafe_rgb, 0.16);
        cf.fillRect(PF.X(PWALL), PF.py, PF.X(12) - PF.X(PWALL), PF.ph);
        // Hazard net / wall stripes
        cf.strokeStyle = COL.unsafe;
        cf.lineWidth = 3;
        cf.beginPath();
        cf.moveTo(PF.X(PWALL), PF.py);
        cf.lineTo(PF.X(PWALL), PF.py + PF.ph);
        cf.stroke();
        for (let yy = PF.py + 4; yy < PF.py + PF.ph; yy += 12) {
          cf.strokeStyle = rgba(COL.unsafe_rgb, 0.55);
          cf.lineWidth = 1.5;
          cf.beginPath();
          cf.moveTo(PF.X(PWALL), yy);
          cf.lineTo(PF.X(PWALL) + 9, yy + 7);
          cf.stroke();
        }

        // Wind gust streamlines
        if (dmax > 0.02) {
          cf.strokeStyle = rgba(COL.warn_rgb, 0.38);
          cf.lineWidth = 1.3;
          for (const wy of [0.65, 1.5, 2.35]) {
            for (const wx of [-0.5, 3.0, 6.5]) {
              const len = 0.4 + dmax * 0.65;
              HJR.drawArrowPx(cf, PF.X(wx), PF.Y(wy), PF.X(wx + len), PF.Y(wy), rgba(COL.warn_rgb, 0.48), 1.4, 5);
            }
          }
        }

        const yDrone = PF.Y(1.55);
        // Latency coasting bar (amber) + Braking bar (cyan/crimson)
        if (state[1] > 0.05) {
          if (latMs > 0) {
            cf.strokeStyle = COL["c-gold"];
            cf.lineWidth = 5;
            cf.beginPath();
            cf.moveTo(PF.X(state[0]), yDrone + 28);
            cf.lineTo(PF.X(Math.min(12, roll.pLatEnd)), yDrone + 28);
            cf.stroke();
          }
          cf.strokeStyle = roll.crashed ? COL.unsafe : COL.safe;
          cf.lineWidth = 5;
          cf.beginPath();
          cf.moveTo(PF.X(Math.min(12, roll.pLatEnd)), yDrone + 28);
          cf.lineTo(PF.X(Math.min(12, roll.maxP)), yDrone + 28);
          cf.stroke();
          // Stopping marker
          HJR.drawDot(PF, Math.min(11.8, roll.maxP), 1.55 - (28 / PF.ph) * 3, 4.5, roll.crashed ? COL.unsafe : COL.safe, COL["bg-elev"], 1.5);
        }

        // Velocity vector from quadrotor
        if (Math.abs(v0) > 0.08) {
          HJR.drawArrowPx(cf, PF.X(p0), yDrone, PF.X(p0 + v0 * 0.32), yDrone, doomed ? COL.unsafe : COL.accent, 2.2, 7);
        }

        // Pitch tilt visualization: during coasting tilt forward if v>0, during braking tilt backward!
        const isCoasting = flightTrace && flightTrace.t < latMs / 1000;
        const pitchVis = flightTrace
          ? (isCoasting ? 0.14 : (v0 > 0.05 ? -Math.atan2(amax, 9.81) * 1.35 : 0))
          : (v0 > 0.2 ? 0.12 : 0);
        HJR.drawQuadrotorSide(
          cf,
          PF.X(p0),
          yDrone,
          pitchVis,
          0.95,
          doomed ? COL.unsafe : COL.ink,
          doomed ? COL.unsafe : COL.safe,
          rotorPhase
        );

        cf.fillStyle = COL.unsafe;
        cf.font = "600 11.5px " + getComputedStyle(document.body).fontFamily;
        cf.fillText("NET / WALL (10 m)", PF.X(PWALL) + 12, PF.py + 16);
        cf.restore();
        PF.axes("position p [m]  (amber bar = latency coast, teal/red bar = max braking)", "", [-2, 0, 2, 4, 6, 8, 10, 12], []);

        /* --- Bottom Canvas: 2D Phase Space (p, v) --- */
        PP.clear();
        HJR.drawScalarField(PP, V, nx, ny, x0, x1, y0, y1, HJR.valueColormap(10, -4));
        PP.clip();
        HJR.strokeSegments(PP, HJR.marchingSquares(V, nx, ny, x0, x1, y0, y1, 0), COL.ink, 2.2);

        if ($("#h-show-exact").checked) {
          const E = new Float32Array(nx * ny);
          const tLat = latMs / 1000;
          for (let i = 0; i < nx; i++) {
            const p = x0 + i * dp;
            for (let j = 0; j < ny; j++) E[i * ny + j] = exactValWithLatency(p, y0 + j * dv, tauHave, amax, dmax, tLat);
          }
          HJR.strokeSegments(PP, HJR.marchingSquares(E, nx, ny, x0, x1, y0, y1, 0), COL["c-gold"], 2.2, [6, 4]);
        }

        const cp = PP.ctx;
        cp.strokeStyle = rgba(COL.ink_rgb, 0.45);
        cp.lineWidth = 1;
        cp.setLineDash([3, 3]);
        cp.beginPath();
        cp.moveTo(PP.X(PWALL), PP.py); cp.lineTo(PP.X(PWALL), PP.py + PP.ph);
        cp.moveTo(PP.px, PP.Y(0)); cp.lineTo(PP.px + PP.pw, PP.Y(0));
        cp.stroke();
        cp.setLineDash([]);

        // Coast + Brake phase curve
        if (roll.coastPts.length > 1) HJR.drawPolyline(PP, roll.coastPts, COL["c-gold"], 2.8);
        HJR.drawPolyline(PP, roll.brakePts, roll.crashed ? COL.unsafe : COL.accent, 2.6);
        cp.restore();

        HJR.drawDot(PP, state[0], state[1], 6.5, doomed ? COL.unsafe : COL.safe, COL["bg-elev"], 2);
        if (flightTrace) {
          HJR.drawDot(PP, p0, v0, 5.5, COL["c-gold"], COL.ink, 1.5);
        }
        PP.axes("position p [m]", "velocity v [m/s]", [-2, 0, 2, 4, 6, 8, 10, 12], [-4, -2, 0, 2, 4, 6, 8]);

        /* --- Telemetry Readout --- */
        const tLat = latMs / 1000;
        const dCoast = Math.max(0, roll.pLatEnd - state[0]);
        const dBrake = Math.max(0, roll.maxP - roll.pLatEnd);
        const have = PWALL - state[0];
        $("#h-readout").innerHTML =
          `<div class="hud-headline">${
            Vnow < 0
              ? '<span class="pill unsafe">DOOMED · INSIDE BRT</span>'
              : '<span class="pill safe">VIABLE · RECOVERABLE</span>'
          } <span>V(x, τ) = ${Vnow >= 0 ? "+" : ""}${Vnow.toFixed(2)} m</span></div>` +
          `<div>State: p = <b>${state[0].toFixed(2)} m</b> (${have.toFixed(2)} m to net), v = <b>${state[1].toFixed(2)} m/s</b> · Net decel (a<sub>max</sub> − d<sub>max</sub>) = <b>${(amax - dmax).toFixed(2)} m/s²</b></div>` +
          `<div>Stopping budget: <b>${dCoast.toFixed(2)} m</b> lost in ${latMs} ms latency + <b>${dBrake.toFixed(2)} m</b> full-tilt braking = <b>${(dCoast + dBrake).toFixed(2)} m</b> needed</div>`;
      }

      const updateStateFromPhase = (pos) => {
        cancelAnimationFrame(animFlight);
        flightTrace = null;
        const [p, v] = PP.inv(pos[0], pos[1]);
        state = [clamp(p, -1.8, 11.6), clamp(v, -3.8, 7.8)];
        render();
      };
      HJR.bindDrag(cvPhase, PP.evtPos, {
        down: (pos) => updateStateFromPhase(pos),
        move: (pos, _e, hover) => { if (!hover) updateStateFromPhase(pos); },
      });
      HJR.bindDrag(cvFlight, PF.evtPos, {
        down: (pos) => { cancelAnimationFrame(animFlight); flightTrace = null; state[0] = clamp(PF.inv(pos[0], pos[1])[0], -1.8, 11.6); render(); },
        move: (pos, _e, hover) => { if (!hover) { state[0] = clamp(PF.inv(pos[0], pos[1])[0], -1.8, 11.6); render(); } },
      });

      let debounce = null;
      const onParamChange = () => {
        amax = +$("#h-amax").value;
        dmax = Math.min(+$("#h-dmax").value, amax - 0.1);
        if (+$("#h-dmax").value > dmax) $("#h-dmax").value = dmax;
        latMs = +$("#h-lat").value;
        tau = +$("#h-tau").value;
        $("#h-amax-v").textContent = amax.toFixed(1) + " m/s²";
        $("#h-dmax-v").textContent = dmax.toFixed(2) + " m/s²";
        $("#h-lat-v").textContent = latMs.toFixed(0) + " ms";
        $("#h-tau-v").textContent = tau.toFixed(2) + " s";
        clearTimeout(debounce);
        debounce = setTimeout(solveDP, 45);
      };
      ["h-amax", "h-dmax"].forEach((id) => $("#" + id).addEventListener("input", onParamChange));
      ["h-lat", "h-tau"].forEach((id) => $("#" + id).addEventListener("input", () => {
        latMs = +$("#h-lat").value;
        tau = +$("#h-tau").value;
        $("#h-lat-v").textContent = latMs.toFixed(0) + " ms";
        $("#h-tau-v").textContent = tau.toFixed(2) + " s";
        render();
      }));
      $("#h-show-exact").addEventListener("change", render);

      $("#h-play-tau").addEventListener("click", () => {
        cancelAnimationFrame(animTau);
        const t0 = performance.now();
        const loop = (now) => {
          tau = Math.min(4.0, ((now - t0) / 1000) * 1.4);
          $("#h-tau").value = tau;
          $("#h-tau-v").textContent = tau.toFixed(2) + " s";
          render();
          if (tau < 4.0) animTau = requestAnimationFrame(loop);
        };
        animTau = requestAnimationFrame(loop);
      });

      $("#h-launch-test").addEventListener("click", () => {
        cancelAnimationFrame(animFlight);
        let p = state[0], v = state[1], t = 0;
        let prev = performance.now();
        flightTrace = { cur: [p, v], t: 0 };
        const stepAnim = (now) => {
          const dt = Math.min(0.04, (now - prev) / 1000);
          prev = now;
          const tLat = latMs / 1000;
          const u = t < tLat ? 0 : (v > 0 ? -amax : 0);
          const acc = u + dmax;
          p += v * dt + 0.5 * acc * dt * dt;
          v += acc * dt;
          t += dt;
          rotorPhase += dt * 32;
          flightTrace = { cur: [p, v], t };
          render();
          if (p <= 11.8 && t < 5.0 && (t < tLat || v > 0.02)) {
            animFlight = requestAnimationFrame(stepAnim);
          } else {
            setTimeout(() => { flightTrace = null; render(); }, 900);
          }
        };
        animFlight = requestAnimationFrame(stepAnim);
      });

      HJR.onThemeChange(render);
      solveDP();
    })();

    /* =====================================================================
       LAB 02: INTERACTIVE 3D LEVEL-SET SURFACE EXPLORER (BRS vs BRT & KINKS)
       ===================================================================== */
    (function initLab02() {
      const cv3 = $("#surf3d-cv");
      const cv2 = $("#surf2d-cv");
      if (!cv3 || !cv2) return;

      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const W3 = 600, H3 = 410;
      cv3.width = W3 * dpr; cv3.height = H3 * dpr;
      const ctx3 = cv3.getContext("2d");
      ctx3.setTransform(dpr, 0, 0, dpr, 0, 0);

      const P2 = HJR.createPlot2D(cv2, { xlim: [-1, 11], ylim: [-4, 7], margin: { l: 48, r: 14, t: 16, b: 38 } });

      let mode = "wall";   // "wall" | "pillar" | "kink"
      let tubeMode = "brt"; // "brt" | "brs"
      let tau = 2.2, azDeg = -42, elDeg = 28;

      const NX = 49, NY = 43;
      let cache = null;

      function solveSurfaceGrid() {
        let x0 = -1, x1 = 11, y0 = -4, y1 = 7;
        if (mode === "kink") { x0 = -7; x1 = 3; y0 = -5; y1 = 5; }
        const dx = (x1 - x0) / (NX - 1), dy = (y1 - y0) / (NY - 1);
        const L = new Float32Array(NX * NY);

        for (let i = 0; i < NX; i++) {
          const x = x0 + i * dx;
          for (let j = 0; j < NY; j++) {
            const y = y0 + j * dy;
            if (mode === "wall") L[i * NY + j] = 9.5 - x;
            else if (mode === "pillar") L[i * NY + j] = Math.abs(x - 5.2) - 0.9;
            else L[i * NY + j] = Math.hypot(x, y) - 1.5; // circular obstacle at (0,0)
          }
        }

        const DT = 0.025, NSTEPS = 128; // up to 3.2s
        const snapsBRT = [L.slice()], snapsBRS = [L.slice()];
        let Vbrt = L.slice(), Vbrs = L.slice();
        let Wbrt = new Float32Array(NX * NY), Wbrs = new Float32Array(NX * NY);

        for (let s = 1; s <= NSTEPS; s++) {
          for (let i = 0; i < NX; i++) {
            const x = x0 + i * dx;
            for (let j = 0; j < NY; j++) {
              const y = y0 + j * dy;
              let bestT = -1e9, bestS = -1e9;
              if (mode === "kink") {
                // Drift dx/dt = +2.6 m/s toward obstacle at (0,0), lateral swerve dy/dt = u in [-1.6, 1.6]
                const vx = 2.6;
                for (const uy of [-1.6, 0, 1.6]) {
                  const xn = x + vx * DT, yn = y + uy * DT;
                  const vt = HJR.interp2D(Vbrt, NX, NY, x0, x1, y0, y1, xn, yn);
                  const vs = HJR.interp2D(Vbrs, NX, NY, x0, x1, y0, y1, xn, yn);
                  if (vt > bestT) bestT = vt;
                  if (vs > bestS) bestS = vs;
                }
              } else {
                // Double integrator: dx/dt = y (velocity), dy/dt = u + d, ae = 2.5
                for (const u of [-2.5, 2.5]) {
                  const xn = x + y * DT + 0.5 * u * DT * DT;
                  const yn = y + u * DT;
                  const vt = HJR.interp2D(Vbrt, NX, NY, x0, x1, y0, y1, xn, yn);
                  const vs = HJR.interp2D(Vbrs, NX, NY, x0, x1, y0, y1, xn, yn);
                  if (vt > bestT) bestT = vt;
                  if (vs > bestS) bestS = vs;
                }
              }
              const idx = i * NY + j;
              Wbrt[idx] = Math.min(L[idx], bestT);
              Wbrs[idx] = bestS;
            }
          }
          [Vbrt, Wbrt] = [Wbrt, Vbrt];
          [Vbrs, Wbrs] = [Wbrs, Vbrs];
          if (s % 2 === 0) {
            snapsBRT.push(Vbrt.slice());
            snapsBRS.push(Vbrs.slice());
          }
        }
        cache = { x0, x1, y0, y1, dx, dy, L, snapsBRT, snapsBRS };
      }

      function draw() {
        if (!cache) solveSurfaceGrid();
        const { x0, x1, y0, y1, dx, dy, snapsBRT, snapsBRS } = cache;
        const sIdx = clamp(Math.round(tau / 0.05), 0, snapsBRT.length - 1);
        const F = tubeMode === "brt" ? snapsBRT[sIdx] : snapsBRS[sIdx];
        const Fref = snapsBRT[sIdx];

        /* --- Left Canvas: 3D Perspective Surface + z=0 Waterline Plane --- */
        ctx3.fillStyle = COL["bg-hud"];
        ctx3.fillRect(0, 0, W3, H3);

        const az = (azDeg * Math.PI) / 180;
        const el = (elDeg * Math.PI) / 180;
        const ca = Math.cos(az), sa = Math.sin(az);
        const ce = Math.cos(el), se = Math.sin(el);

        const xm = 0.5 * (x0 + x1), ym = 0.5 * (y0 + y1);
        const xSpan = x1 - x0, ySpan = y1 - y0;
        const zScale = mode === "wall" ? 0.13 : 0.28;
        const scalePx = 235;

        const proj = (x, y, z) => {
          const nx = (x - xm) / (0.55 * xSpan);
          const ny = (y - ym) / (0.55 * ySpan);
          const nz = clamp(z, -4.5, 6.5) * zScale;
          const rx = nx * ca - ny * sa;
          const ry = nx * sa + ny * ca;
          const sx = W3 * 0.5 + rx * scalePx * 0.72;
          const sy = H3 * 0.56 - (nz * ce - ry * se * 0.58) * scalePx * 0.62;
          const depth = ry * ce + nz * se;
          return [sx, sy, depth];
        };

        // Draw z=0 translucent sea-level reference plane corners first
        const planeCorners = [
          proj(x0, y0, 0), proj(x1, y0, 0), proj(x1, y1, 0), proj(x0, y1, 0)
        ];
        ctx3.save();
        ctx3.fillStyle = rgba(COL.accent_rgb, COL.isDark ? 0.10 : 0.08);
        ctx3.strokeStyle = rgba(COL.accent_rgb, 0.55);
        ctx3.lineWidth = 1.4;
        ctx3.beginPath();
        planeCorners.forEach((p, k) => (k ? ctx3.lineTo(p[0], p[1]) : ctx3.moveTo(p[0], p[1])));
        ctx3.closePath();
        ctx3.fill();
        ctx3.stroke();
        ctx3.restore();

        // Painter's algorithm sorted quads (strided 2x2 for crisp 60fps wireframe+shaded mesh)
        const quads = [];
        const stepI = 1, stepJ = 1;
        for (let i = 0; i < NX - stepI; i += stepI) {
          for (let j = 0; j < NY - stepJ; j += stepJ) {
            const xa = x0 + i * dx, xb = x0 + (i + stepI) * dx;
            const ya = y0 + j * dy, yb = y0 + (j + stepJ) * dy;
            const z00 = F[i * NY + j];
            const z10 = F[(i + stepI) * NY + j];
            const z11 = F[(i + stepI) * NY + (j + stepJ)];
            const z01 = F[i * NY + (j + stepJ)];
            const p00 = proj(xa, ya, z00), p10 = proj(xb, ya, z10);
            const p11 = proj(xb, yb, z11), p01 = proj(xa, yb, z01);
            const zAvg = 0.25 * (z00 + z10 + z11 + z01);
            const depth = 0.25 * (p00[2] + p10[2] + p11[2] + p01[2]);
            const dzdx = (z10 - z00) / dx, dzdy = (z01 - z00) / dy;
            quads.push({ p00, p10, p11, p01, zAvg, depth, dzdx, dzdy, crossZero: (z00 < 0) !== (z11 < 0) || (z10 < 0) !== (z01 < 0) });
          }
        }
        quads.sort((a, b) => b.depth - a.depth);

        const light = [-0.45, -0.35, 0.82];
        for (const q of quads) {
          const nrm = Math.hypot(-q.dzdx * 0.3, -q.dzdy * 0.3, 1);
          const shade = clamp(((-q.dzdx * 0.3 * light[0] - q.dzdy * 0.3 * light[1] + light[2]) / nrm) * 0.55 + 0.45, 0.25, 1.0);
          let baseRgb = q.zAvg < 0
            ? mixRgb(COL["bg-hud_rgb"], COL.unsafe_rgb, 0.62 + 0.32 * clamp(-q.zAvg / 2.5, 0, 1))
            : mixRgb(COL["bg-hud_rgb"], COL.safe_rgb, 0.25 + 0.55 * clamp(q.zAvg / 4.5, 0, 1));
          const lit = baseRgb.map((c) => clamp(c * (0.68 + 0.48 * shade), 0, 255));
          ctx3.fillStyle = rgba(lit, q.zAvg < 0 ? 0.90 : 0.78);
          ctx3.strokeStyle = q.crossZero ? COL["c-gold"] : rgba(COL.ink_rgb, COL.isDark ? 0.14 : 0.10);
          ctx3.lineWidth = q.crossZero ? 1.8 : 0.55;
          ctx3.beginPath();
          ctx3.moveTo(q.p00[0], q.p00[1]);
          ctx3.lineTo(q.p10[0], q.p10[1]);
          ctx3.lineTo(q.p11[0], q.p11[1]);
          ctx3.lineTo(q.p01[0], q.p01[1]);
          ctx3.closePath();
          ctx3.fill();
          ctx3.stroke();
        }

        // Re-stroke z=0 plane front border & label
        ctx3.fillStyle = COL.ink;
        ctx3.font = "600 12px " + getComputedStyle(document.body).fontFamily;
        ctx3.fillText("3D Value Surface z = V(x, τ)  ·  Gold seam = z = 0 waterline", 14, 22);
        ctx3.fillStyle = COL.muted;
        ctx3.font = "11px " + getComputedStyle(document.body).fontFamily;
        ctx3.fillText("Crimson trough (z < 0) = Doomed states  ·  Drag anywhere to orbit 3D", 14, H3 - 12);

        /* --- Right Canvas: Synchronized 2D Zero-Sublevel Set & Characteristics --- */
        P2.xlim = [x0, x1];
        P2.ylim = [y0, y1];
        P2.clear();
        HJR.drawScalarField(P2, F, NX, NY, x0, x1, y0, y1, HJR.valueColormap(mode === "wall" ? 8 : 3.5, -2.5));
        P2.clip();
        HJR.strokeSegments(P2, HJR.marchingSquares(F, NX, NY, x0, x1, y0, y1, 0), COL["c-gold"], 2.5);
        if (tubeMode === "brs") {
          // Overlay dashed BRT boundary so user sees what BRS missed!
          HJR.strokeSegments(P2, HJR.marchingSquares(Fref, NX, NY, x0, x1, y0, y1, 0), COL.ink, 1.8, [5, 4]);
        }

        // Pedagogical overlays per mode
        if (mode === "pillar") {
          const c = P2.ctx;
          c.strokeStyle = COL.unsafe;
          c.lineWidth = 1.5;
          c.setLineDash([4, 3]);
          c.strokeRect(P2.X(4.3), P2.py, P2.X(6.1) - P2.X(4.3), P2.ph);
          c.setLineDash([]);
          // Show fast punch-through trajectory from (1.0, 5.8)
          const pts = [];
          let p = 1.0, v = 5.8;
          for (let t = 0; t <= Math.max(0.05, tau); t += 0.03) {
            pts.push([p, v]);
            p += v * 0.03 - 0.5 * 2.5 * 0.03 * 0.03;
            v -= 2.5 * 0.03;
          }
          HJR.drawPolyline(P2, pts, COL.accent, 2.5);
          HJR.drawDot(P2, 1.0, 5.8, 5.5, COL.accent, COL["bg-elev"], 1.8);
        } else if (mode === "kink") {
          // Highlight the viscosity kink ridge along y = 0, x < -1.5
          HJR.drawPolyline(P2, [[-7, 0], [-1.5, 0]], COL["c-cbf"], 2.5, [5, 3]);
        }
        P2.ctx.restore();

        const xl = mode === "kink" ? "relative x [m]" : "position p [m]";
        const yl = mode === "kink" ? "lateral y [m]" : "velocity v [m/s]";
        const xt = mode === "kink" ? [-6, -4, -2, 0, 2] : [0, 2, 4, 6, 8, 10];
        const yt = mode === "kink" ? [-4, -2, 0, 2, 4] : [-4, -2, 0, 2, 4, 6];
        P2.axes(xl, yl, xt, yt);

        // Explanation banner
        let msg = "";
        if (mode === "wall") {
          msg = `<div class="hud-headline"><span class="pill accent">WALL BRAKING</span> Horizon τ = ${tau.toFixed(2)} s</div>` +
            `<div>At τ = 0, z = l(p) = 9.5 − p is a flat tilted plane. As τ grows, states with v &gt; 0 slide down into the crimson z &lt; 0 trough, carving the parabolic braking boundary!</div>`;
        } else if (mode === "pillar") {
          msg = `<div class="hud-headline"><span class="pill ${tubeMode === "brs" ? "unsafe" : "safe"}">${tubeMode.toUpperCase()} ACTIVE</span> Fast Punch-Through Test at (p=1.0 m, v=5.8 m/s)</div>` +
            (tubeMode === "brt"
              ? `<div><b>BRT (Ratchet ON):</b> Notice the cyan trajectory smashes through the pillar [4.3, 6.1] m mid-flight before exiting on the right. Because min{l − V, …} = 0 pins V ≤ l, the entire punch-through corridor stays crimson (unsafe)! <i>Now click "BRS (Set)" above!</i></div>`
              : `<div><b>BRS TRAP EXPOSED:</b> Without the min{l − V, …} floor, the unsafe blob drifted past (1.0, 5.8)—marking that state <b>SAFE</b> at τ = ${tau.toFixed(2)} s simply because the drone already exited the far side of the concrete pillar! Dashed black line = true BRT.</div>`);
        } else {
          msg = `<div class="hud-headline"><span class="pill warn">VISCOSITY KINK</span> Non-Differentiable Value Ridge along y = 0</div>` +
            `<div>Directly upwind of the cylinder (dashed violet line y = 0), swerving left (+u<sub>y</sub>) and swerving right (−u<sub>y</sub>) tie. Orbit the 3D surface to see the sharp inverted-V crease where ∇V jumps discontinuously—why grid solvers need upwind monotonicity and smooth PINNs struggle at switching seams!</div>`;
        }
        $("#surf-readout").innerHTML = msg;
      }

      // 3D Orbit Drag
      let dragStart = null;
      HJR.bindDrag(cv3, (e) => [e.clientX, e.clientY], {
        down: (pos) => { dragStart = { x: pos[0], y: pos[1], az: azDeg, el: elDeg }; },
        move: (pos, _e, hover) => {
          if (hover || !dragStart) return;
          azDeg = clamp(dragStart.az + (pos[0] - dragStart.x) * 0.55, -160, 160);
          elDeg = clamp(dragStart.el + (pos[1] - dragStart.y) * 0.35, 10, 68);
          $("#surf-az").value = azDeg; $("#surf-az-v").textContent = Math.round(azDeg) + "°";
          $("#surf-el").value = elDeg; $("#surf-el-v").textContent = Math.round(elDeg) + "°";
          draw();
        },
        up: () => { dragStart = null; },
      });

      $$("#surf-mode-tabs button").forEach((b) => b.addEventListener("click", () => {
        $$("#surf-mode-tabs button").forEach((x) => x.classList.toggle("active", x === b));
        mode = b.dataset.surfMode;
        solveSurfaceGrid();
        draw();
      }));
      $$("#surf-tube-tabs button").forEach((b) => b.addEventListener("click", () => {
        $$("#surf-tube-tabs button").forEach((x) => x.classList.toggle("active", x === b));
        tubeMode = b.dataset.tube;
        draw();
      }));
      $("#surf-tau").addEventListener("input", (e) => { tau = +e.target.value; $("#surf-tau-v").textContent = tau.toFixed(2) + " s"; draw(); });
      $("#surf-az").addEventListener("input", (e) => { azDeg = +e.target.value; $("#surf-az-v").textContent = Math.round(azDeg) + "°"; draw(); });
      $("#surf-el").addEventListener("input", (e) => { elDeg = +e.target.value; $("#surf-el-v").textContent = Math.round(elDeg) + "°"; draw(); });

      HJR.onThemeChange(draw);
      solveSurfaceGrid();
      draw();
    })();

    /* =====================================================================
       LAB 03: 3D FIXED-WING / DUBINS AIR3D TWISTED REACHABLE TUBE
       ===================================================================== */
    (function initLab03() {
      const cv3 = $("#dubins3d-cv");
      const cv2 = $("#dubins2d-cv");
      if (!cv3 || !cv2 || !DATA.dubins3d) return;

      const D = DATA.dubins3d;
      const V3 = HJR.decodeQuantU8(D.V);
      const [x0, x1, nx] = D.xs;
      const [y0, y1, ny] = D.ys;
      const npsi = D.npsi;

      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const W3 = 580, H3 = 420;
      cv3.width = W3 * dpr; cv3.height = H3 * dpr;
      const ctx3 = cv3.getContext("2d");
      ctx3.setTransform(dpr, 0, 0, dpr, 0, 0);

      const P2 = HJR.createPlot2D(cv2, { xlim: [x0, x1], ylim: [y0, y1], margin: { l: 44, r: 12, t: 14, b: 36 } });

      let psiDeg = 0, rotDeg = -36;
      let probePt = [-5.5, 0.6]; // aircraft test position on 2D slice

      // Pre-extract zero-level-set contours for all 36 heading slices for fast 3D helical tube rendering!
      const sliceContours = [];
      for (let k = 0; k < npsi; k++) {
        const S = new Float32Array(nx * ny);
        for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) S[i * ny + j] = V3[(i * ny + j) * npsi + k];
        const psi = -Math.PI + (2 * Math.PI * k) / npsi;
        sliceContours.push({ psi, k, segs: HJR.marchingSquares(S, nx, ny, x0, x1, y0, y1, 0) });
      }

      function valDubins(x, y, psi) {
        return HJR.interp3D(V3, nx, ny, npsi, x0, x1, y0, y1, -Math.PI, Math.PI, x, y, psi, true);
      }

      function draw() {
        const psiRad = (psiDeg * Math.PI) / 180;

        /* --- Left: 3D Helical Tube over (x, y, psi) --- */
        ctx3.fillStyle = COL["bg-hud"];
        ctx3.fillRect(0, 0, W3, H3);

        const az = (rotDeg * Math.PI) / 180, el = 0.46;
        const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
        const xm = 0.5 * (x0 + x1), ym = 0;
        const proj = (x, y, psi) => {
          const nx_ = (x - xm) / 7.2, ny_ = (y - ym) / 7.2, nz_ = psi / Math.PI;
          const rx = nx_ * ca - ny_ * sa;
          const ry = nx_ * sa + ny_ * ca;
          return [
            W3 * 0.5 + rx * 175,
            H3 * 0.53 - (nz_ * ce * 125 - ry * se * 135),
            ry * ce + nz_ * se
          ];
        };

        // Draw 3D bounding box base & central cylindrical obstacle spine (x=0, y=0)
        ctx3.strokeStyle = rgba(COL.line_rgb, 0.45);
        ctx3.lineWidth = 1;
        for (const zPsi of [-Math.PI, psiRad, Math.PI]) {
          const box = [proj(x0, y0, zPsi), proj(x1, y0, zPsi), proj(x1, y1, zPsi), proj(x0, y1, zPsi)];
          ctx3.strokeStyle = zPsi === psiRad ? rgba(COL["c-gold_rgb"], 0.7) : rgba(COL.line_rgb, 0.35);
          ctx3.lineWidth = zPsi === psiRad ? 1.6 : 1;
          ctx3.beginPath();
          box.forEach((p, idx) => (idx ? ctx3.lineTo(p[0], p[1]) : ctx3.moveTo(p[0], p[1])));
          ctx3.closePath();
          ctx3.stroke();
        }

        // Stacked 3D helical contours from bottom (-pi) to top (+pi)
        for (let k = 0; k < npsi; k++) {
          const sc = sliceContours[k];
          const distPsi = Math.abs(Math.atan2(Math.sin(sc.psi - psiRad), Math.cos(sc.psi - psiRad)));
          const isActive = distPsi < (Math.PI / npsi) * 1.1;
          const tHue = (k / npsi);
          const rgb = isActive ? COL["c-gold_rgb"] : mixRgb(COL.unsafe_rgb, COL.violet_rgb, 0.35 * Math.sin(tHue * Math.PI));
          ctx3.strokeStyle = isActive ? COL["c-gold"] : rgba(rgb, 0.46);
          ctx3.lineWidth = isActive ? 2.8 : 1.25;
          ctx3.beginPath();
          for (const [p, q] of sc.segs) {
            const pA = proj(p[0], p[1], sc.psi);
            const pB = proj(q[0], q[1], sc.psi);
            ctx3.moveTo(pA[0], pA[1]);
            ctx3.lineTo(pB[0], pB[1]);
          }
          ctx3.stroke();
        }

        // Central obstacle cylinder wireframe at r = R_obs
        ctx3.strokeStyle = rgba(COL.ink_rgb, 0.55);
        ctx3.lineWidth = 1.2;
        for (const zPsi of [-Math.PI, 0, Math.PI]) {
          ctx3.beginPath();
          for (let a = 0; a <= 32; a++) {
            const th = (2 * Math.PI * a) / 32;
            const p = proj(D.R_obs * Math.cos(th), D.R_obs * Math.sin(th), zPsi);
            a ? ctx3.lineTo(p[0], p[1]) : ctx3.moveTo(p[0], p[1]);
          }
          ctx3.stroke();
        }

        ctx3.fillStyle = COL.ink;
        ctx3.font = "600 12px " + getComputedStyle(document.body).fontFamily;
        ctx3.fillText("3D Twisted Reachable Tube in (x, y, ψ ∈ [−π, π])", 14, 22);
        ctx3.fillStyle = COL.muted;
        ctx3.font = "11px " + getComputedStyle(document.body).fontFamily;
        ctx3.fillText("Vertical axis = heading ψ  ·  Gold slice = active heading on right", 14, H3 - 12);

        /* --- Right: 2D Slice at Selected Heading psi + Optimal Bank-Turn Rollout --- */
        const S2 = new Float32Array(nx * ny);
        for (let i = 0; i < nx; i++) {
          const x = x0 + (i / (nx - 1)) * (x1 - x0);
          for (let j = 0; j < ny; j++) {
            const y = y0 + (j / (ny - 1)) * (y1 - y0);
            S2[i * ny + j] = valDubins(x, y, psiRad);
          }
        }
        P2.clear();
        HJR.drawScalarField(P2, S2, nx, ny, x0, x1, y0, y1, HJR.valueColormap(4.5, -2.0));
        P2.clip();
        HJR.strokeSegments(P2, HJR.marchingSquares(S2, nx, ny, x0, x1, y0, y1, 0), COL["c-gold"], 2.4);

        // Draw obstacle cylinder at (0,0)
        const c2 = P2.ctx;
        c2.fillStyle = rgba(COL.ink_rgb, 0.85);
        c2.beginPath();
        c2.arc(P2.X(0), P2.Y(0), P2.X(D.R_obs) - P2.X(0), 0, 2 * Math.PI);
        c2.fill();

        // Simulate optimal bank-turn evasion from probePt at initial heading psiRad!
        const traj = [[probePt[0], probePt[1]]];
        let x = probePt[0], y = probePt[1], psi = psiRad;
        let minClear = Math.hypot(x, y) - D.R_obs;
        const dt = 0.025;
        for (let s = 0; s < 90; s++) {
          const vpPlus = valDubins(x, y, psi + 0.08);
          const vpMinus = valDubins(x, y, psi - 0.08);
          const wOpt = vpPlus >= vpMinus ? D.w_max : -D.w_max;
          // Worst-case wind pushes toward obstacle center (minimizes V)
          const vxG = valDubins(x + 0.12, y, psi) - valDubins(x - 0.12, y, psi);
          const vyG = valDubins(x, y + 0.12, psi) - valDubins(x, y - 0.12, psi);
          const gn = Math.hypot(vxG, vyG) + 1e-9;
          const dxW = -D.d_wind * (vxG / gn), dyW = -D.d_wind * (vyG / gn);
          x += (D.v_air * Math.cos(psi) + dxW) * dt;
          y += (D.v_air * Math.sin(psi) + dyW) * dt;
          psi += wOpt * dt;
          traj.push([x, y]);
          minClear = Math.min(minClear, Math.hypot(x, y) - D.R_obs);
          if (minClear < 0 || x < x0 - 0.5 || x > x1 + 0.5 || y < y0 - 0.5 || y > y1 + 0.5) break;
        }
        const vProbe = valDubins(probePt[0], probePt[1], psiRad);
        HJR.drawPolyline(P2, traj, vProbe < 0 ? COL.unsafe : COL.accent, 2.5);

        // Draw fixed-wing delta nose at probePt
        const px = P2.X(probePt[0]), py = P2.Y(probePt[1]);
        c2.save();
        c2.translate(px, py);
        c2.rotate(-psiRad);
        c2.fillStyle = vProbe < 0 ? COL.unsafe : COL.safe;
        c2.strokeStyle = COL["bg-elev"];
        c2.lineWidth = 1.5;
        c2.beginPath();
        c2.moveTo(11, 0);
        c2.lineTo(-8, -6.5);
        c2.lineTo(-4, 0);
        c2.lineTo(-8, 6.5);
        c2.closePath();
        c2.fill();
        c2.stroke();
        c2.restore();

        c2.restore();
        P2.axes("x [m]", "y [m]", [-8, -6, -4, -2, 0, 2, 4], [-6, -4, -2, 0, 2, 4, 6]);

        $("#dub-readout").innerHTML =
          `<div class="hud-headline">${
            vProbe < 0
              ? '<span class="pill unsafe">UNAVOIDABLE COLLISION</span>'
              : '<span class="pill safe">BANK-TURN ESCAPE VIABLE</span>'
          } <span>V(x, y, ψ) = ${vProbe >= 0 ? "+" : ""}${vProbe.toFixed(2)} m · Min turn radius R<sub>min</sub> = v / ω<sub>max</sub> = ${D.r_turn} m</span></div>` +
          `<div>Aircraft at (${probePt[0].toFixed(1)}, ${probePt[1].toFixed(1)}) m, heading ψ = ${psiDeg}° at constant v = ${D.v_air.toFixed(1)} m/s against ${D.d_wind} m/s wind · Closest approach on optimal turn: <b>${minClear.toFixed(2)} m</b></div>` +
          `<div class="muted">Notice why HJR is indispensable here: the obstacle covers ${
            (100 * D.obs_frac).toFixed(1)
          }% of (x,y), yet the true 3D doomed tube fills ${(100 * D.brt_frac).toFixed(1)}% of state space!</div>`;
      }

      let drag3 = null;
      HJR.bindDrag(cv3, (e) => [e.clientX, e.clientY], {
        down: (pos) => { drag3 = { x: pos[0], rot: rotDeg }; },
        move: (pos, _e, hover) => {
          if (hover || !drag3) return;
          rotDeg = clamp(drag3.rot + (pos[0] - drag3.x) * 0.6, -180, 180);
          $("#dub-rot").value = rotDeg; $("#dub-rot-v").textContent = Math.round(rotDeg) + "°";
          draw();
        },
        up: () => { drag3 = null; },
      });

      const setProbe = (pos) => {
        const [x, y] = P2.inv(pos[0], pos[1]);
        probePt = [clamp(x, x0 + 0.2, x1 - 0.2), clamp(y, y0 + 0.2, y1 - 0.2)];
        draw();
      };
      HJR.bindDrag(cv2, P2.evtPos, {
        down: setProbe,
        move: (pos, _e, hover) => { if (!hover) setProbe(pos); },
      });

      const dirName = (d) => {
        const a = ((d % 360) + 360) % 360;
        if (a < 22 || a >= 338) return "East";
        if (a < 68) return "NE";
        if (a < 112) return "North";
        if (a < 158) return "NW";
        if (a < 202) return "West";
        if (a < 248) return "SW";
        if (a < 292) return "South";
        return "SE";
      };
      $("#dub-psi").addEventListener("input", (e) => {
        psiDeg = +e.target.value;
        $("#dub-psi-v").textContent = `${psiDeg}° (${dirName(psiDeg)})`;
        draw();
      });
      $("#dub-rot").addEventListener("input", (e) => {
        rotDeg = +e.target.value;
        $("#dub-rot-v").textContent = Math.round(rotDeg) + "°";
        draw();
      });

      HJR.onThemeChange(draw);
      draw();
    })();

    /* =====================================================================
       LAB 04: 3-IN-1 DRONE PLANNING & CONTROL DECK
               Tab A: 4D Cluttered Arena + Union Under-Approximation
               Tab B: 4D Precision Dock Landing Reach-Avoid Funnel
               Tab C: FaSTrack Planner + 6D Pitch-Lagged Quadrotor
       ===================================================================== */
    (function initLab04() {
      const cvMain = $("#drone-main-cv");
      const cvPad = $("#drone-pad-cv");
      if (!cvMain || !cvPad || !DATA.arena4d || !DATA.dock4d || !DATA.fastrack3d) return;

      const A4 = DATA.arena4d;
      const V_ARENA = HJR.decodeQuantU8(A4.V);
      const DOCK = DATA.dock4d;
      const V_DOCK = HJR.decodeQuantU8(DOCK.V);
      const FT = DATA.fastrack3d;
      const M_FT = HJR.decodeQuantU8(FT.M);

      const PM = HJR.createPlot2D(cvMain, { xlim: [0, 24], ylim: [0, 24], margin: { l: 44, r: 12, t: 12, b: 36 } });
      const PQ = HJR.createPlot2D(cvPad, { xlim: [-4.3, 4.3], ylim: [-4.3, 4.3], margin: { l: 34, r: 10, t: 18, b: 28 } });

      let tab = "arena"; // "arena" | "dock" | "fastrack"
      let animId = null;

      // State per tab
      let arenaVel = [3.2, 1.2], showUnionMiss = true, hoverArena = null;
      let dockVel = [0.6, -1.6], dockSim = null;
      let ftUseBubble = true, ftSim = null, ftWaypoints = [[3.5, 4.0], [11.5, 6.5], [13.0, 14.5], [19.5, 19.5]];

      function evalArena(px, py, vx, vy) {
        return HJR.interp4D(
          V_ARENA,
          [A4.xs[2], A4.xs[2], A4.vs[2], A4.vs[2]],
          [[A4.xs[0], A4.xs[1]], [A4.xs[0], A4.xs[1]], [A4.vs[0], A4.vs[1]], [A4.vs[0], A4.vs[1]]],
          [px, py, vx, vy]
        );
      }

      function evalDock(x, z, vx, vz) {
        return HJR.interp4D(
          V_DOCK,
          [DOCK.xs[2], DOCK.zs[2], DOCK.vxs[2], DOCK.vzs[2]],
          [[DOCK.xs[0], DOCK.xs[1]], [DOCK.zs[0], DOCK.zs[1]], [DOCK.vxs[0], DOCK.vxs[1]], [DOCK.vzs[0], DOCK.vzs[1]]],
          [x, z, vx, vz]
        );
      }

      function buildControls() {
        const box = $("#drone-dyn-controls");
        const prompt = $("#drone-prompt");
        if (tab === "arena") {
          prompt.innerHTML = `<strong>4D Avoid Slice:</strong> Drag the velocity joystick below to slice \\(V(p_x, p_y, v_x, v_y)\\). Amber squares mark multi-obstacle trap states where escaping one pillar throws you into another!`;
          box.innerHTML = `
            <div class="btn-row">
              <button class="btn" type="button" data-av="0,0">Hover (0,0)</button>
              <button class="btn" type="button" data-av="3.5,0">3.5 m/s East</button>
              <button class="btn" type="button" data-av="2.8,2.8">4.0 m/s NE</button>
              <button class="btn" type="button" data-av="0,-3.8">3.8 m/s South</button>
            </div>
            <label class="chk"><input id="chk-union-miss" type="checkbox" ${showUnionMiss ? "checked" : ""}/>Highlight states missed by per-obstacle union (${A4.missed_count} cells)</label>
          `;
          $$("[data-av]", box).forEach((b) => b.addEventListener("click", () => {
            arenaVel = b.dataset.av.split(",").map(Number);
            render();
          }));
          $("#chk-union-miss", box).addEventListener("change", (e) => { showUnionMiss = e.target.checked; render(); });
        } else if (tab === "dock") {
          prompt.innerHTML = `<strong>4D Reach-Avoid Landing Funnel:</strong> Teal region = guaranteed safe touchdown onto the charging pad under gusts. <strong>Click anywhere in the sky</strong> to drop a quadrotor and watch precision touchdown or automatic HJR Go-Around!`;
          box.innerHTML = `
            <div class="btn-row">
              <button class="btn primary" type="button" id="dock-drop-ok">▶ Drop Inside Funnel</button>
              <button class="btn danger" type="button" id="dock-drop-abort">▶ Drop in Crosswind Gust (Trigger Go-Around!)</button>
            </div>
          `;
          $("#dock-drop-ok", box).addEventListener("click", () => startDockSim(-1.6, 7.2, 0.5, -1.4));
          $("#dock-drop-abort", box).addEventListener("click", () => startDockSim(-3.8, 4.2, 2.4, -2.5));
        } else {
          prompt.innerHTML = `<strong>FaSTrack Planner-Tracker Game:</strong> A geometric planner turns razor-sharp corners around pillars at 2.0 m/s while a <strong>6D pitch-lagged quadrotor</strong> tracks it in gusts. Toggle the HJR Tracking Error Bound (TEB = ${FT.teb_m} m) ON vs OFF!`;
          box.innerHTML = `
            <div class="btn-row">
              <button class="btn ${ftUseBubble ? "primary" : ""}" type="button" id="ft-on">✓ FaSTrack Bubble + HJR Tracker</button>
              <button class="btn ${!ftUseBubble ? "danger active" : ""}" type="button" id="ft-off">✕ Unbuffered Planner + PD (Overshoots!)</button>
            </div>
            <button class="btn" type="button" id="ft-replay">↺ Re-Fly Mission</button>
          `;
          $("#ft-on", box).addEventListener("click", () => { ftUseBubble = true; buildControls(); startFaSTrackSim(); });
          $("#ft-off", box).addEventListener("click", () => { ftUseBubble = false; buildControls(); startFaSTrackSim(); });
          $("#ft-replay", box).addEventListener("click", () => startFaSTrackSim());
        }
        if (HJR.typeset) HJR.typeset(prompt);
      }

      function startDockSim(x0, z0, vx0, vz0) {
        cancelAnimationFrame(animId);
        dockVel = [vx0, vz0];
        dockSim = { x: x0, z: z0, vx: vx0, vz: vz0, mode: "approach", trail: [[x0, z0]], t: 0 };
        const loop = () => {
          if (tab !== "dock" || !dockSim) return;
          const dt = 0.025;
          for (let k = 0; k < 2; k++) {
            const s = dockSim;
            if (s.mode === "landed" || s.mode === "aborted_safe") break;
            const V = evalDock(s.x, s.z, s.vx, s.vz);
            if (s.mode === "approach" && V < 0.04) s.mode = "go_around";

            let ax, az;
            if (s.mode === "go_around") {
              // Verified abort maneuver: level lateral speed and climb away from deck!
              ax = clamp(-1.8 * s.vx, -DOCK.ax_max, DOCK.ax_max);
              az = DOCK.az_max;
              if (s.z > 7.8 && s.vz > 1.0) s.mode = "aborted_safe";
            } else {
              // Blend nominal pad guidance with Reach-Avoid optimal gradient near funnel wall
              const axNom = clamp(-1.4 * s.x - 1.9 * s.vx, -DOCK.ax_max, DOCK.ax_max);
              const vzDes = clamp(-0.55 - 0.35 * s.z, -1.6, -0.35);
              const azNom = clamp(2.2 * (vzDes - s.vz), -DOCK.az_max, DOCK.az_max);
              if (V < 0.35) {
                const gvx = evalDock(s.x, s.z, s.vx + 0.25, s.vz) - evalDock(s.x, s.z, s.vx - 0.25, s.vz);
                const gvz = evalDock(s.x, s.z, s.vx, s.vz + 0.25) - evalDock(s.x, s.z, s.vx, s.vz - 0.25);
                ax = Math.abs(gvx) > 1e-4 ? DOCK.ax_max * Math.sign(gvx) : axNom;
                az = Math.abs(gvz) > 1e-4 ? DOCK.az_max * Math.sign(gvz) : azNom;
              } else {
                ax = axNom; az = azNom;
              }
            }
            // Crosswind gust
            const gx = 0.48 * Math.cos(1.7 * s.t), gz = -0.32;
            s.x += s.vx * dt + 0.5 * (ax + gx) * dt * dt;
            s.z = Math.max(0.08, s.z + s.vz * dt + 0.5 * (az + gz) * dt * dt);
            s.vx = clamp(s.vx + (ax + gx) * dt, -3.2, 3.2);
            s.vz = clamp(s.vz + (az + gz) * dt, -3.8, 2.2);
            s.t += dt;
            s.trail.push([s.x, s.z]);
            if (s.mode === "approach" && s.z <= 0.48 && Math.abs(s.x) <= 0.92 && Math.abs(s.vx) <= 0.95 && s.vz >= -1.4) {
              s.mode = "landed";
            }
          }
          render();
          if (dockSim && (dockSim.mode === "approach" || dockSim.mode === "go_around") && dockSim.t < 10) {
            animId = requestAnimationFrame(loop);
          }
        };
        animId = requestAnimationFrame(loop);
      }

      function startFaSTrackSim() {
        cancelAnimationFrame(animId);
        // Waypoint path: buffered path skirts pillars by (R + TEB), unbuffered cuts close to pillars!
        const teb = FT.teb_m;
        const wps = ftUseBubble
          ? [[3.0, 3.5], [5.2, 11.8], [12.4, 11.4], [13.2, 19.6], [20.5, 20.5]]
          : [[3.0, 3.5], [6.5, 10.2], [13.8, 12.8], [12.0, 17.0], [20.5, 20.5]];
        ftWaypoints = wps;
        ftSim = {
          phat: wps[0].slice(), wpIdx: 1,
          p: wps[0].slice(), v: [0, 0], th: [0, 0],
          trailQuad: [wps[0].slice()], trailPlan: [wps[0].slice()],
          errTrail: [[0, 0]], crashed: false, done: false, maxErr: 0, t: 0
        };
        const g = 9.81;
        const loop = () => {
          if (tab !== "fastrack" || !ftSim) return;
          const dt = 0.02;
          for (let step = 0; step < 2; step++) {
            const s = ftSim;
            if (s.crashed || s.done) break;
            // Virtual planner moves at v_plan = 2.0 m/s along waypoints
            let vhat = [0, 0];
            if (s.wpIdx < wps.length) {
              const tgt = wps[s.wpIdx];
              const dx = tgt[0] - s.phat[0], dy = tgt[1] - s.phat[1], d = Math.hypot(dx, dy);
              if (d < 0.15) s.wpIdx++;
              else { vhat = [(dx / d) * FT.v_plan, (dy / d) * FT.v_plan]; }
            }
            s.phat[0] += vhat[0] * dt;
            s.phat[1] += vhat[1] * dt;

            // Relative error e = p - phat
            const ex = s.p[0] - s.phat[0], ey = s.p[1] - s.phat[1];
            const axes = [[ex, s.v[0], s.th[0], vhat[0]], [ey, s.v[1], s.th[1], vhat[1]]];
            const omegas = [0, 0];
            for (let a = 0; a < 2; a++) {
              const [e, vel, th, vh] = axes[a];
              if (ftUseBubble) {
                // HJR Optimal Pitch-Rate Tracker minimizing worst-case M(e, v, th)
                const mPlus = HJR.interp3D(M_FT, FT.es[2], FT.vs[2], FT.ths[2], FT.es[0], FT.es[1], FT.vs[0], FT.vs[1], FT.ths[0], FT.ths[1], e, vel, clamp(th + 0.05, -FT.th_lim, FT.th_lim));
                const mMinus = HJR.interp3D(M_FT, FT.es[2], FT.vs[2], FT.ths[2], FT.es[0], FT.es[1], FT.vs[0], FT.vs[1], FT.ths[0], FT.ths[1], e, vel, clamp(th - 0.05, -FT.th_lim, FT.th_lim));
                const errMag = Math.abs(e);
                const thDes = clamp((-2.6 * e - 1.9 * (vel - vh)) / g, -FT.th_lim, FT.th_lim);
                omegas[a] = errMag > 0.45
                  ? (mPlus < mMinus ? FT.w_max : -FT.w_max)
                  : clamp(7.5 * (thDes - th), -FT.w_max, FT.w_max);
              } else {
                // Standard sluggish PD attitude tracker
                const thDes = clamp((-1.3 * e - 1.1 * (vel - vh)) / g, -FT.th_lim, FT.th_lim);
                omegas[a] = clamp(2.8 * (thDes - th), -FT.w_max, FT.w_max);
              }
            }

            // Turbulent wind gusts
            const wx = FT.d_wind * Math.cos(1.4 * s.t + 0.6);
            const wy = FT.d_wind * Math.sin(1.9 * s.t);

            for (let a = 0; a < 2; a++) {
              s.th[a] = clamp(s.th[a] + omegas[a] * dt, -FT.th_lim, FT.th_lim);
              const acc = g * Math.tan(s.th[a]) + (a === 0 ? wx : wy);
              s.p[a] += s.v[a] * dt + 0.5 * acc * dt * dt;
              s.v[a] += acc * dt;
            }
            s.t += dt;
            const errNow = Math.hypot(s.p[0] - s.phat[0], s.p[1] - s.phat[1]);
            s.maxErr = Math.max(s.maxErr, errNow);
            if (Math.round(s.t / dt) % 3 === 0) {
              s.trailQuad.push(s.p.slice());
              s.trailPlan.push(s.phat.slice());
              s.errTrail.push([s.p[0] - s.phat[0], s.p[1] - s.phat[1]]);
            }
            // Check pillar collision
            for (const [cx, cy, r] of A4.pillars) {
              if (Math.hypot(s.p[0] - cx, s.p[1] - cy) < r) s.crashed = true;
            }
            if (s.wpIdx >= wps.length && Math.hypot(s.p[0] - wps[wps.length - 1][0], s.p[1] - wps[wps.length - 1][1]) < 0.5) {
              s.done = true;
            }
          }
          render();
          if (ftSim && !ftSim.crashed && !ftSim.done && ftSim.t < 22) {
            animId = requestAnimationFrame(loop);
          }
        };
        animId = requestAnimationFrame(loop);
      }

      function render() {
        if (tab === "arena") {
          PM.xlim = [0, 24]; PM.ylim = [0, 24];
          PQ.xlim = [-4.3, 4.3]; PQ.ylim = [-4.3, 4.3];
          const nx = A4.xs[2], hx = 24 / (nx - 1);
          const S = new Float32Array(nx * nx);
          let doomCnt = 0;
          for (let i = 0; i < nx; i++) {
            for (let j = 0; j < nx; j++) {
              const val = evalArena(i * hx, j * hx, arenaVel[0], arenaVel[1]);
              S[i * nx + j] = val;
              if (val < 0) doomCnt++;
            }
          }
          PM.clear();
          HJR.drawScalarField(PM, S, nx, nx, 0, 24, 0, 24, HJR.valueColormap(6.0, -2.0));
          PM.clip();
          HJR.strokeSegments(PM, HJR.marchingSquares(S, nx, nx, 0, 24, 0, 24, 0), COL.unsafe, 2.0);
          const c = PM.ctx;
          for (const [cx, cy, r] of A4.pillars) {
            c.fillStyle = COL.ink;
            c.beginPath(); c.arc(PM.X(cx), PM.Y(cy), PM.X(r) - PM.X(0), 0, 2 * Math.PI); c.fill();
          }
          if (showUnionMiss) {
            const hv = (A4.vs[1] - A4.vs[0]) / (A4.vs[2] - 1);
            const iu = Math.round((arenaVel[0] - A4.vs[0]) / hv);
            const iw = Math.round((arenaVel[1] - A4.vs[0]) / hv);
            c.fillStyle = COL["c-gold"];
            for (const [ix, iy, u, w] of A4.missed_idx) {
              if (Math.abs(u - iu) <= 1 && Math.abs(w - iw) <= 1) {
                c.fillRect(PM.X(ix * hx) - 3, PM.Y(iy * hx) - 3, 6, 6);
              }
            }
          }
          if (Math.hypot(...arenaVel) > 0.1) {
            for (const [ax, ay] of [[3, 21], [21, 21], [3, 3]]) {
              HJR.drawArrowPx(c, PM.X(ax), PM.Y(ay), PM.X(ax + arenaVel[0] * 0.65), PM.Y(ay + arenaVel[1] * 0.65), COL.accent, 2.2, 7);
            }
          }
          c.restore();
          PM.axes("x [m]", "y [m]", [0, 6, 12, 18, 24], [0, 6, 12, 18, 24]);

          // Velocity Pad
          PQ.clear();
          const q = PQ.ctx;
          q.strokeStyle = COL.line;
          for (const r of [1, 2, 3, 4]) { q.beginPath(); q.arc(PQ.X(0), PQ.Y(0), PQ.X(r) - PQ.X(0), 0, 2 * Math.PI); q.stroke(); }
          q.beginPath(); q.moveTo(PQ.X(-4.3), PQ.Y(0)); q.lineTo(PQ.X(4.3), PQ.Y(0)); q.moveTo(PQ.X(0), PQ.Y(-4.3)); q.lineTo(PQ.X(0), PQ.Y(4.3)); q.stroke();
          HJR.drawArrowPx(q, PQ.X(0), PQ.Y(0), PQ.X(arenaVel[0]), PQ.Y(arenaVel[1]), COL.accent, 3, 9);
          q.fillStyle = COL.muted; q.font = "11px " + getComputedStyle(document.body).fontFamily;
          q.fillText("Drag velocity vector (m/s)", 14, 22);

          let hovHtml = `<div class="muted">Hover arena to inspect V(x) vs geometric clearance l(x)</div>`;
          if (hoverArena) {
            const [hx_, hy_] = hoverArena;
            const vh = evalArena(hx_, hy_, arenaVel[0], arenaVel[1]);
            let lh = Math.min(hx_, 24 - hx_, hy_, 24 - hy_);
            for (const [cx, cy, r] of A4.pillars) lh = Math.min(lh, Math.hypot(hx_ - cx, hy_ - cy) - r);
            hovHtml = `<div>At (${hx_.toFixed(1)}, ${hy_.toFixed(1)}) m: obstacle dist l = <b>${lh.toFixed(2)} m</b>, HJR value V = <b>${vh.toFixed(2)} m</b> (momentum cost: ${(lh - vh).toFixed(2)} m)</div>`;
          }
          $("#drone-readout").innerHTML =
            `<div class="hud-headline"><span class="pill accent">v = (${arenaVel[0].toFixed(1)}, ${arenaVel[1].toFixed(1)}) m/s</span> <span>${((100 * doomCnt) / S.length).toFixed(1)}% of arena doomed</span></div>` + hovHtml;

        } else if (tab === "dock") {
          PM.xlim = [-6, 6]; PM.ylim = [0, 9.6];
          PQ.xlim = [-3.2, 3.2]; PQ.ylim = [-3.8, 1.8];
          const curVx = dockSim ? dockSim.vx : dockVel[0];
          const curVz = dockSim ? dockSim.vz : dockVel[1];
          const nx = DOCK.xs[2], nz = DOCK.zs[2];
          const S = new Float32Array(nx * nz);
          for (let i = 0; i < nx; i++) {
            const x = DOCK.xs[0] + (i / (nx - 1)) * (DOCK.xs[1] - DOCK.xs[0]);
            for (let j = 0; j < nz; j++) {
              const z = DOCK.zs[0] + (j / (nz - 1)) * (DOCK.zs[1] - DOCK.zs[0]);
              S[i * nz + j] = evalDock(x, z, curVx, curVz);
            }
          }
          PM.clear();
          HJR.drawScalarField(PM, S, nx, nz, -6, 6, 0, 9.6, HJR.valueColormap(1.5, -1.8));
          PM.clip();
          HJR.strokeSegments(PM, HJR.marchingSquares(S, nx, nz, -6, 6, 0, 9.6, 0), COL["c-gold"], 2.4);

          // Draw charging dock pad + coaming walls
          const c = PM.ctx;
          c.fillStyle = COL.safe;
          c.fillRect(PM.X(-0.95), PM.Y(0.28), PM.X(0.95) - PM.X(-0.95), PM.Y(0) - PM.Y(0.28));
          c.fillStyle = COL.ink;
          c.fillRect(PM.X(-1.8), PM.Y(1.25), PM.X(-1.05) - PM.X(-1.8), PM.Y(0) - PM.Y(1.25));
          c.fillRect(PM.X(1.05), PM.Y(1.25), PM.X(1.8) - PM.X(1.05), PM.Y(0) - PM.Y(1.25));

          if (dockSim) {
            HJR.drawPolyline(PM, dockSim.trail, dockSim.mode.includes("go_around") || dockSim.mode === "aborted_safe" ? COL["c-gold"] : COL.safe, 2.8);
            HJR.drawQuadrotorSide(c, PM.X(dockSim.x), PM.Y(dockSim.z), clamp(dockSim.vx * 0.12, -0.35, 0.35), 0.85, COL.ink, dockSim.mode === "go_around" ? COL["c-gold"] : COL.safe, dockSim.t * 25);
          }
          c.restore();
          PM.axes("lateral offset from dock x [m]", "altitude z [m]", [-6, -4, -2, 0, 2, 4, 6], [0, 2, 4, 6, 8]);

          PQ.clear();
          PQ.axes("vx [m/s]", "vz [m/s]", [-3, 0, 3], [-3, -1, 1], true);
          HJR.drawDot(PQ, curVx, curVz, 6, COL.accent, COL["bg-elev"], 2);
          PQ.ctx.fillStyle = COL.muted; PQ.ctx.font = "10.5px " + getComputedStyle(document.body).fontFamily;
          PQ.ctx.fillText("Approach velocity (drag)", 38, 13);

          const stLabel = !dockSim
            ? '<span class="pill accent">READY · CLICK SKY TO DROP</span>'
            : dockSim.mode === "landed"
            ? '<span class="pill safe">SAFE PAD TOUCHDOWN!</span>'
            : dockSim.mode === "go_around" || dockSim.mode === "aborted_safe"
            ? '<span class="pill warn">V_RA &lt; 0 → AUTOMATIC GO-AROUND ABORT!</span>'
            : '<span class="pill safe">DESCENDING IN REACH-AVOID FUNNEL</span>';
          $("#drone-readout").innerHTML =
            `<div class="hud-headline">${stLabel}</div>` +
            `<div>Slice velocity: v<sub>x</sub> = ${curVx.toFixed(2)} m/s, v<sub>z</sub> = ${curVz.toFixed(2)} m/s · Teal funnel guarantees safe touchdown despite ±${DOCK.dx_max} m/s² crosswind gusts!</div>`;

        } else {
          // Tab C: FaSTrack
          PM.xlim = [0, 24]; PM.ylim = [0, 24];
          PQ.xlim = [-2.2, 2.2]; PQ.ylim = [-2.2, 2.2];
          PM.clear();
          PM.clip();
          const c = PM.ctx;
          for (const [cx, cy, r] of A4.pillars) {
            if (ftUseBubble) {
              c.fillStyle = rgba(COL.accent_rgb, 0.14);
              c.strokeStyle = rgba(COL.accent_rgb, 0.65);
              c.lineWidth = 1.4;
              c.setLineDash([5, 4]);
              c.beginPath(); c.arc(PM.X(cx), PM.Y(cy), PM.X(r + FT.teb_m) - PM.X(0), 0, 2 * Math.PI); c.fill(); c.stroke();
              c.setLineDash([]);
            }
            c.fillStyle = COL.ink;
            c.beginPath(); c.arc(PM.X(cx), PM.Y(cy), PM.X(r) - PM.X(0), 0, 2 * Math.PI); c.fill();
          }
          // Planned polyline
          HJR.drawPolyline(PM, ftWaypoints, COL["c-gold"], 1.8, [5, 4]);
          if (ftSim) {
            HJR.drawPolyline(PM, ftSim.trailQuad, ftSim.crashed ? COL.unsafe : COL.safe, 2.8);
            if (ftUseBubble) {
              c.strokeStyle = COL.safe;
              c.fillStyle = rgba(COL.safe_rgb, 0.14);
              c.lineWidth = 2;
              c.beginPath();
              c.arc(PM.X(ftSim.phat[0]), PM.Y(ftSim.phat[1]), PM.X(FT.teb_m) - PM.X(0), 0, 2 * Math.PI);
              c.fill(); c.stroke();
            }
            HJR.drawDot(PM, ftSim.phat[0], ftSim.phat[1], 4.5, COL["c-gold"], COL.ink, 1.2);
            HJR.drawDot(PM, ftSim.p[0], ftSim.p[1], 6, ftSim.crashed ? COL.unsafe : COL.safe, COL["bg-elev"], 2);
          }
          c.restore();
          PM.axes("x [m]", "y [m]", [0, 6, 12, 18, 24], [0, 6, 12, 18, 24]);

          // Relative Tracking Error Phase Pad (ex, ey) vs TEB circle
          PQ.clear();
          PQ.axes("e_x [m]", "e_y [m]", [-2, 0, 2], [-2, 0, 2], true);
          const q = PQ.ctx;
          q.strokeStyle = COL.safe;
          q.lineWidth = 2;
          q.setLineDash([4, 3]);
          q.beginPath();
          q.arc(PQ.X(0), PQ.Y(0), PQ.X(FT.teb_m) - PQ.X(0), 0, 2 * Math.PI);
          q.stroke();
          q.setLineDash([]);
          if (ftSim) {
            HJR.drawPolyline(PQ, ftSim.errTrail, ftSim.crashed ? COL.unsafe : COL.accent, 1.8);
            const lastE = ftSim.errTrail[ftSim.errTrail.length - 1];
            HJR.drawDot(PQ, lastE[0], lastE[1], 4.5, COL.accent, COL["bg-elev"], 1.5);
          }
          q.fillStyle = COL.muted; q.font = "10.5px " + getComputedStyle(document.body).fontFamily;
          q.fillText(`Tracking error ||e|| vs TEB (${FT.teb_m}m)`, 36, 13);

          $("#drone-readout").innerHTML = ftSim
            ? `<div class="hud-headline">${
                ftSim.crashed
                  ? '<span class="pill unsafe">CORNER OVERSHOOT CRASH!</span>'
                  : ftSim.done
                  ? '<span class="pill safe">GOAL REACHED · ZERO COLLISIONS</span>'
                  : '<span class="pill accent">TRACKING VIRTUAL PLANNER</span>'
              } <span>Max tracking error ||e|| = ${ftSim.maxErr.toFixed(2)} m (HJR Bound = ${FT.teb_m} m)</span></div>` +
              `<div>${
                ftUseBubble
                  ? "FaSTrack inflates pillars by TEB (dashed rings) and uses the HJR pitch-rate controller ω*(e, v, θ) to keep the 6D quadrotor inside the green moving bubble!"
                  : "Without FaSTrack's TEB inflation and HJR tracker, pitch lag + 0.5 m/s² wind gusts make the quadrotor swing wide around sharp planner corners into the pillar!"
              }</div>`
            : "";
        }
      }

      // Pointer interactions on cvMain & cvPad
      cvMain.addEventListener("pointermove", (e) => {
        if (tab !== "arena") return;
        PM.xlim = [0, 24]; PM.ylim = [0, 24];
        const [x, y] = PM.inv(...PM.evtPos(e));
        hoverArena = x >= 0 && x <= 24 && y >= 0 && y <= 24 ? [x, y] : null;
        render();
      });
      cvMain.addEventListener("pointerdown", (e) => {
        if (tab !== "dock") return;
        PM.xlim = [-6, 6]; PM.ylim = [0, 9.6];
        const [x, z] = PM.inv(...PM.evtPos(e));
        if (z > 0.6 && z < 9.4 && Math.abs(x) < 5.6) startDockSim(x, z, dockVel[0], dockVel[1]);
      });

      HJR.bindDrag(cvPad, PQ.evtPos, {
        down: (pos) => {
          if (tab === "arena") {
            PQ.xlim = [-4.3, 4.3]; PQ.ylim = [-4.3, 4.3];
            const [vx, vy] = PQ.inv(pos[0], pos[1]);
            arenaVel = [clamp(vx, -4, 4), clamp(vy, -4, 4)];
            render();
          } else if (tab === "dock") {
            cancelAnimationFrame(animId); dockSim = null;
            PQ.xlim = [-3.2, 3.2]; PQ.ylim = [-3.8, 1.8];
            const [vx, vz] = PQ.inv(pos[0], pos[1]);
            dockVel = [clamp(vx, -3.1, 3.1), clamp(vz, -3.7, 1.7)];
            render();
          }
        },
        move: (pos, _e, hover) => {
          if (hover) return;
          if (tab === "arena") {
            PQ.xlim = [-4.3, 4.3]; PQ.ylim = [-4.3, 4.3];
            const [vx, vy] = PQ.inv(pos[0], pos[1]);
            arenaVel = [clamp(vx, -4, 4), clamp(vy, -4, 4)];
            render();
          } else if (tab === "dock") {
            PQ.xlim = [-3.2, 3.2]; PQ.ylim = [-3.8, 1.8];
            const [vx, vz] = PQ.inv(pos[0], pos[1]);
            dockVel = [clamp(vx, -3.1, 3.1), clamp(vz, -3.7, 1.7)];
            render();
          }
        },
      });

      $$("#drone-deck-tabs button").forEach((b) => b.addEventListener("click", () => {
        $$("#drone-deck-tabs button").forEach((x) => x.classList.toggle("active", x === b));
        cancelAnimationFrame(animId);
        tab = b.dataset.droneTab;
        buildControls();
        if (tab === "dock") startDockSim(-1.6, 7.2, 0.5, -1.4);
        else if (tab === "fastrack") startFaSTrackSim();
        else render();
      }));

      HJR.onThemeChange(render);
      buildControls();
      render();
    })();

    /* =====================================================================
       LAB 05: 5-FILTER ARENA RACE & LIVE CONTROL-SPACE (ux, uy) QP MICROSCOPE
       ===================================================================== */
    (function initLab05() {
      const cvArena = $("#filt-arena-cv");
      const cvQP = $("#filt-qp-cv");
      if (!cvArena || !cvQP || !DATA.arena4d) return;

      const A4 = DATA.arena4d;
      const V4 = HJR.decodeQuantU8(A4.V);
      const PA = HJR.createPlot2D(cvArena, { xlim: [0, 24], ylim: [0, 24], margin: { l: 42, r: 12, t: 12, b: 36 } });
      const PQ = HJR.createPlot2D(cvQP, { xlim: [-3.8, 3.8], ylim: [-3.8, 3.8], margin: { l: 44, r: 14, t: 26, b: 38 } });

      const AMAX = 3.0, DMAX = 0.5;
      const CONTROLLERS = [
        { name: "Unfiltered", key: "none", col: "c-none" },
        { name: "APF", key: "apf", col: "c-apf" },
        { name: "HOCBF-QP", key: "hocbf", col: "c-cbf" },
        { name: "HJR Switch", key: "hjsw", col: "c-hjsw" },
        { name: "CBVF (HJR-CBF)", key: "cbvf", col: "c-cbvf" },
      ];

      let scen = "gap";
      let start = [3.0, 4.0], goal = [21.0, 20.0];
      let customPillars = A4.pillars.map((p) => p.slice());
      let simHistory = null;

      function setScenario(s) {
        scen = s;
        if (s === "gap") {
          customPillars = A4.pillars.map((p) => p.slice());
          start = [3.0, 4.0]; goal = [21.0, 20.0];
          $("#f-speed").value = 3.7; $("#f-gust").value = 40; $("#f-cbf-k").value = 2.0;
        } else if (s === "headon") {
          customPillars = A4.pillars.map((p) => p.slice());
          start = [2.5, 9.0]; goal = [20.5, 9.1];
          $("#f-speed").value = 3.9; $("#f-gust").value = 0; $("#f-cbf-k").value = 2.4;
        } else {
          // Concave U-trap of 5 pillars!
          customPillars = [
            [12.5, 10.0, 1.35], [14.2, 11.4, 1.35], [14.6, 13.6, 1.35],
            [13.2, 15.2, 1.35], [11.2, 14.8, 1.35]
          ];
          start = [4.0, 5.0]; goal = [20.5, 20.5];
          $("#f-speed").value = 3.2; $("#f-gust").value = 45; $("#f-cbf-k").value = 1.6;
        }
        syncLabels();
        runAllSimulations();
      }

      function evalV4(s) {
        if (scen === "utrap") {
          // Analytic per-pillar + wall exact relative 2D braking value composition for custom U-trap layout
          const [px, py, vx, vy] = s;
          const ae = AMAX - DMAX;
          let minV = Math.min(
            px - Math.max(-vx, 0) ** 2 / (2 * ae),
            (24 - px) - Math.max(vx, 0) ** 2 / (2 * ae),
            py - Math.max(-vy, 0) ** 2 / (2 * ae),
            (24 - py) - Math.max(vy, 0) ** 2 / (2 * ae)
          );
          for (const [cx, cy, r] of customPillars) {
            const dx = px - cx, dy = py - cy, dist = Math.hypot(dx, dy);
            const h = dist - r;
            const vClose = Math.max(-(dx * vx + dy * vy) / (dist + 1e-6), 0);
            minV = Math.min(minV, h - (vClose * vClose) / (2 * ae));
          }
          return minV;
        }
        return HJR.interp4D(
          V4,
          [A4.xs[2], A4.xs[2], A4.vs[2], A4.vs[2]],
          [[0, 24], [0, 24], [-4, 4], [-4, 4]],
          s
        );
      }

      function gradV4(s) {
        const eps = [0.2, 0.2, 0.25, 0.25];
        const g = [0, 0, 0, 0];
        for (let k = 0; k < 4; k++) {
          const sp = s.slice(), sm = s.slice();
          sp[k] += eps[k]; sm[k] -= eps[k];
          g[k] = (evalV4(sp) - evalV4(sm)) / (2 * eps[k]);
        }
        return g;
      }

      function obsGeom(s) {
        const [x, y] = s;
        const out = [
          [x, [1, 0], Infinity], [24 - x, [-1, 0], Infinity],
          [y, [0, 1], Infinity], [24 - y, [0, -1], Infinity],
        ];
        for (const [cx, cy, r] of customPillars) {
          const rx = x - cx, ry = y - cy, rho = Math.hypot(rx, ry) + 1e-9;
          out.push([rho - r, [rx / rho, ry / rho], rho]);
        }
        return out;
      }

      const sat2 = (u) => {
        const n = Math.hypot(u[0], u[1]);
        return n > AMAX ? [(u[0] * AMAX) / n, (u[1] * AMAX) / n] : u.slice();
      };

      function evalController(key, s, uNom, kGain) {
        if (key === "none") return { u: sat2(uNom), infeas: false, halfspaces: [] };
        const [px, py, vx, vy] = s;
        if (key === "apf") {
          let ux = uNom[0], uy = uNom[1];
          for (const [h, n] of obsGeom(s)) {
            if (h < 3.2) {
              const hc = Math.max(h, 0.02);
              const mag = 12.5 * (1 / hc - 1 / 3.2) / (hc * hc);
              ux += mag * n[0]; uy += mag * n[1];
            }
          }
          return { u: sat2([ux, uy]), infeas: false, halfspaces: [] };
        }

        const A = HJR.DISC16.map((p) => p.slice());
        const b = HJR.DISC16.map(() => AMAX * Math.cos(Math.PI / 16));
        const halfspaces = [];

        if (key === "hocbf") {
          for (const [h, n, rho] of obsGeom(s)) {
            const hd = n[0] * vx + n[1] * vy;
            const curv = isFinite(rho) ? (vx * vx + vy * vy - hd * hd) / rho : 0;
            // Need n . u >= rhs  <=>  -n . u <= -rhs
            const rhs = DMAX - curv - 2 * kGain * hd - kGain * kGain * h;
            A.push([-n[0], -n[1]]);
            b.push(-rhs);
            if (h < 6.5) halfspaces.push({ n: n.slice(), rhs });
          }
          const sol = HJR.solveQP2D(uNom, A, b);
          return { u: sat2(sol.u), infeas: sol.infeasible, halfspaces };
        }

        const V = evalV4(s);
        const g = gradV4(s);
        const gn = Math.hypot(g[2], g[3]);
        if (key === "hjsw") {
          if (V >= 0.65) return { u: sat2(uNom), infeas: false, halfspaces: [] };
          const uStar = gn > 1e-5 ? [(AMAX * g[2]) / gn, (AMAX * g[3]) / gn] : sat2([-vx, -vy]);
          return { u: uStar, infeas: false, halfspaces: [] };
        }

        // CBVF (Smooth HJR-CBF): gv . u >= -gamma*(V - margin) - gp . v + DMAX * ||gv||
        const gamma = 1.6, margin = 0.18;
        const rhs = -gamma * (V - margin) - (g[0] * vx + g[1] * vy) + DMAX * gn;
        if (gn > 1e-5) {
          A.push([-g[2], -g[3]]);
          b.push(-rhs);
          halfspaces.push({ n: [g[2] / gn, g[3] / gn], rhs: rhs / gn });
        }
        const sol = HJR.solveQP2D(uNom, A, b);
        return { u: sat2(sol.u), infeas: sol.infeasible, halfspaces };
      }

      function runAllSimulations() {
        const cruise = +$("#f-speed").value;
        const gustRad = (+$("#f-gust").value * Math.PI) / 180;
        const kGain = +$("#f-cbf-k").value;
        const gust = [DMAX * Math.cos(gustRad), DMAX * Math.sin(gustRad)];
        const dt = 0.025, maxSteps = 720;

        const runs = CONTROLLERS.map((ctrl) => {
          let s = [start[0], start[1], 0, 0];
          const pts = [[s[0], s[1]]];
          const snaps = [];
          let status = "flying", tEnd = maxSteps * dt, minClear = Infinity, infeasSteps = 0, chatter = 0;
          let prevU = [0, 0];

          for (let step = 0; step < maxSteps; step++) {
            const dx = goal[0] - s[0], dy = goal[1] - s[1], dist = Math.hypot(dx, dy) + 1e-9;
            const sp = Math.min(cruise, 0.85 * dist);
            const uNom = sat2([2.2 * ((dx / dist) * sp - s[2]), 2.2 * ((dy / dist) * sp - s[3])]);

            let res;
            if (status === "flying") {
              res = evalController(ctrl.key, s, uNom, kGain);
              if (res.infeas) infeasSteps++;
              chatter += Math.hypot(res.u[0] - prevU[0], res.u[1] - prevU[1]);
              prevU = res.u.slice();

              const ax = res.u[0] + gust[0], ay = res.u[1] + gust[1];
              s = [
                s[0] + s[2] * dt + 0.5 * ax * dt * dt,
                s[1] + s[3] * dt + 0.5 * ay * dt * dt,
                clamp(s[2] + ax * dt, -4, 4),
                clamp(s[3] + ay * dt, -4, 4),
              ];
              let lNow = Math.min(s[0], 24 - s[0], s[1], 24 - s[1]);
              for (const [cx, cy, r] of customPillars) lNow = Math.min(lNow, Math.hypot(s[0] - cx, s[1] - cy) - r);
              minClear = Math.min(minClear, lNow);
              pts.push([s[0], s[1]]);
              if (lNow < 0) { status = "crashed"; tEnd = (step + 1) * dt; }
              else if (Math.hypot(goal[0] - s[0], goal[1] - s[1]) < 0.65) { status = "reached"; tEnd = (step + 1) * dt; }
            } else {
              res = { u: [0, 0], infeas: false, halfspaces: [] };
            }
            if (step % 6 === 0) {
              snaps.push({ t: step * dt, s: s.slice(), uNom: uNom.slice(), u: res.u.slice(), infeas: res.infeas, halfspaces: res.halfspaces });
            }
          }
          if (status === "flying") status = "stuck";
          return { ctrl, pts, snaps, status, tEnd, minClear, infeasSteps, chatter: chatter / pts.length };
        });

        simHistory = { runs, gust, kGain };
        const maxSnap = runs[0].snaps.length - 1;
        $("#f-scrub").max = maxSnap;
        if (+$("#f-scrub").value > maxSnap) $("#f-scrub").value = Math.min(25, maxSnap);
        draw();
      }

      function draw() {
        if (!simHistory) return;
        const scrubIdx = clamp(+$("#f-scrub").value, 0, simHistory.runs[0].snaps.length - 1);
        const tCur = simHistory.runs[0].snaps[scrubIdx].t;
        $("#f-scrub-v").textContent = `t = ${tCur.toFixed(2)} s`;

        /* --- Left Canvas: 2D Arena & 5 Trajectories --- */
        PA.clear();
        PA.clip();
        const ca = PA.ctx;
        for (const [cx, cy, r] of customPillars) {
          ca.fillStyle = COL.ink;
          ca.beginPath(); ca.arc(PA.X(cx), PA.Y(cy), PA.X(r) - PA.X(0), 0, 2 * Math.PI); ca.fill();
        }
        for (const r of simHistory.runs) {
          HJR.drawPolyline(PA, r.pts, COL[r.ctrl.col], r.ctrl.key === "none" ? 1.5 : 2.5, r.ctrl.key === "none" ? [4, 4] : []);
          const endPt = r.pts[r.pts.length - 1];
          if (r.status === "crashed") {
            const sx = PA.X(endPt[0]), sy = PA.Y(endPt[1]);
            ca.strokeStyle = COL.unsafe; ca.lineWidth = 2.6;
            ca.beginPath(); ca.moveTo(sx - 6, sy - 6); ca.lineTo(sx + 6, sy + 6); ca.moveTo(sx + 6, sy - 6); ca.lineTo(sx - 6, sy + 6); ca.stroke();
          }
          // Scrubbed marker on HOCBF & CBVF
          const sn = r.snaps[scrubIdx];
          if (sn && (r.ctrl.key === "hocbf" || r.ctrl.key === "cbvf")) {
            HJR.drawDot(PA, sn.s[0], sn.s[1], 5.5, COL[r.ctrl.col], COL["bg-elev"], 2);
          }
        }
        HJR.drawDot(PA, start[0], start[1], 6, COL["bg-elev"], COL.ink, 2);
        ca.fillStyle = COL["c-gold"]; ca.font = "bold 17px " + getComputedStyle(document.body).fontFamily;
        ca.textAlign = "center"; ca.textBaseline = "middle";
        ca.fillText("★", PA.X(goal[0]), PA.Y(goal[1]));

        // Gust indicator
        HJR.drawArrowPx(ca, PA.X(20.5), PA.Y(2.2), PA.X(20.5 + simHistory.gust[0] * 4.2), PA.Y(2.2 + simHistory.gust[1] * 4.2), COL.warn, 2.2, 7);
        ca.fillStyle = COL.warn; ca.font = "11px " + getComputedStyle(document.body).fontFamily;
        ca.fillText("wind", PA.X(20.5), PA.Y(1.1));
        ca.restore();
        PA.axes("x [m]", "y [m]", [0, 6, 12, 18, 24], [0, 6, 12, 18, 24]);

        /* --- Right Canvas: Control-Space (ux, uy) QP Microscope at tCur --- */
        PQ.clear();
        const cq = PQ.ctx;
        PQ.clip();

        const hocbfRun = simHistory.runs.find((r) => r.ctrl.key === "hocbf");
        const cbvfRun = simHistory.runs.find((r) => r.ctrl.key === "cbvf");
        const snCBF = hocbfRun.snaps[scrubIdx];
        const snCBVF = cbvfRun.snaps[scrubIdx];

        // Shade feasible control pixels inside ||u|| <= AMAX for HOCBF vs CBVF!
        const Rpx = PQ.X(AMAX) - PQ.X(0);
        // Actuator circle background
        cq.fillStyle = rgba(COL.ink_rgb, 0.05);
        cq.strokeStyle = COL.ink;
        cq.lineWidth = 2;
        cq.beginPath();
        cq.arc(PQ.X(0), PQ.Y(0), Rpx, 0, 2 * Math.PI);
        cq.fill(); cq.stroke();

        // Draw HOCBF forbidden half-spaces (violet) & CBVF half-space (sky)
        const drawHalfspace = (hs, rgb, label, dashed) => {
          const [nx_, ny_] = hs.n; // feasible: n . u >= rhs; boundary line: n . u = rhs
          const p0 = [nx_ * hs.rhs, ny_ * hs.rhs];
          const tx = -ny_, ty = nx_;
          // Forbidden side is -n direction
          const poly = [
            [p0[0] - 10 * tx, p0[1] - 10 * ty],
            [p0[0] + 10 * tx, p0[1] + 10 * ty],
            [p0[0] + 10 * tx - 10 * nx_, p0[1] + 10 * ty - 10 * ny_],
            [p0[0] - 10 * tx - 10 * nx_, p0[1] - 10 * ty - 10 * ny_],
          ];
          cq.save();
          // Clip to actuator disc so the user sees how much of the motor disc gets eaten!
          cq.beginPath();
          cq.arc(PQ.X(0), PQ.Y(0), Rpx, 0, 2 * Math.PI);
          cq.clip();
          cq.fillStyle = rgba(rgb, 0.22);
          cq.beginPath();
          poly.forEach((p, idx) => (idx ? cq.lineTo(PQ.X(p[0]), PQ.Y(p[1])) : cq.moveTo(PQ.X(p[0]), PQ.Y(p[1]))));
          cq.closePath();
          cq.fill();
          cq.restore();

          HJR.drawPolyline(PQ, [[p0[0] - 5 * tx, p0[1] - 5 * ty], [p0[0] + 5 * tx, p0[1] + 5 * ty]], rgba(rgb, 0.95), 2.2, dashed ? [6, 4] : []);
          if (label && Math.abs(hs.rhs) < 4.5) {
            cq.fillStyle = rgba(rgb, 1);
            cq.font = "600 11px " + getComputedStyle(document.body).fontFamily;
            cq.fillText(label, PQ.X(clamp(p0[0], -3.2, 2.2)), PQ.Y(clamp(p0[1], -3.2, 3.2)) - 5);
          }
        };

        snCBF.halfspaces.forEach((hs, idx) => drawHalfspace(hs, COL["c-cbf_rgb"], idx === 0 ? "HOCBF constraint" : "", true));
        snCBVF.halfspaces.forEach((hs) => drawHalfspace(hs, COL["c-cbvf_rgb"], "CBVF (∇V constraint)", false));

        // Vectors from (0,0): nominal u_nom, u_HOCBF, u_CBVF
        HJR.drawArrowPx(cq, PQ.X(0), PQ.Y(0), PQ.X(snCBVF.uNom[0]), PQ.Y(snCBVF.uNom[1]), COL.muted, 2.0, 7);
        HJR.drawArrowPx(cq, PQ.X(0), PQ.Y(0), PQ.X(snCBF.u[0]), PQ.Y(snCBF.u[1]), COL["c-cbf"], 2.8, 8);
        HJR.drawArrowPx(cq, PQ.X(0), PQ.Y(0), PQ.X(snCBVF.u[0]), PQ.Y(snCBVF.u[1]), COL["c-cbvf"], 2.8, 8);

        if (snCBF.infeas) {
          cq.fillStyle = COL.unsafe;
          cq.font = "700 12.5px " + getComputedStyle(document.body).fontFamily;
          cq.fillText("⚠ HOCBF QP INFEASIBLE AT THIS STEP! (Constraint outside ||u|| ≤ 3 m/s²)", PQ.px + 8, PQ.py + 18);
        } else {
          cq.fillStyle = COL["ink-2"];
          cq.font = "600 11.5px " + getComputedStyle(document.body).fontFamily;
          cq.fillText(`Actuator Disc ||u|| ≤ 3 m/s² at t = ${tCur.toFixed(2)}s (shaded = forbidden by barrier)`, PQ.px + 8, PQ.py + 16);
        }
        cq.restore();
        PQ.axes("commanded accel u_x [m/s²]", "commanded accel u_y [m/s²]", [-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], true);

        /* --- Telemetry Table Readout --- */
        $("#f-readout").innerHTML = simHistory.runs.map((r) => {
          const badge = r.status === "reached"
            ? `<span class="pill safe">Goal ${r.tEnd.toFixed(1)}s</span>`
            : r.status === "crashed"
            ? `<span class="pill unsafe">CRASH ${r.tEnd.toFixed(1)}s</span>`
            : `<span class="pill warn">STUCK (Local Min)</span>`;
          const infTxt = r.infeasSteps > 0 ? ` · <b style="color:var(--unsafe)">QP Infeasible ${r.infeasSteps} steps!</b>` : "";
          return `<div><span style="color:var(--${r.ctrl.col})">■</span> <b>${r.ctrl.name.padEnd(15, " ")}</b> ${badge} · min clearance <b>${r.minClear.toFixed(2)} m</b> · Δu chatter ${r.chatter.toFixed(2)} m/s²${infTxt}</div>`;
        }).join("");
      }

      const syncLabels = () => {
        $("#f-speed-v").textContent = (+$("#f-speed").value).toFixed(1) + " m/s";
        $("#f-gust-v").textContent = $("#f-gust").value + "°";
        $("#f-cbf-k-v").textContent = (+$("#f-cbf-k").value).toFixed(1) + " /s";
      };

      $$("#filt-scen-tabs button").forEach((b) => b.addEventListener("click", () => {
        $$("#filt-scen-tabs button").forEach((x) => x.classList.toggle("active", x === b));
        setScenario(b.dataset.fscen);
      }));
      ["f-speed", "f-gust", "f-cbf-k"].forEach((id) => $("#" + id).addEventListener("input", () => { syncLabels(); runAllSimulations(); }));
      $("#f-scrub").addEventListener("input", draw);
      $("#f-fly-btn").addEventListener("click", runAllSimulations);

      cvArena.addEventListener("pointerdown", (e) => {
        const [x, y] = PA.inv(...PA.evtPos(e));
        const pt = [clamp(x, 1, 23), clamp(y, 1, 23)];
        for (const [cx, cy, r] of customPillars) if (Math.hypot(pt[0] - cx, pt[1] - cy) < r + 0.4) return;
        if (e.shiftKey) start = pt; else goal = pt;
        runAllSimulations();
      });

      HJR.onThemeChange(draw);
      setScenario("gap");
    })();
  });
})();
