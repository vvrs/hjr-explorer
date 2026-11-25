# Hamilton-Jacobi Reachability & PINNs — Interactive Monograph

An interactive first-principles textbook chapter and simulation laboratory building **Hamilton-Jacobi Reachability (HJR)**, **Control Barrier Functions (CBFs / CBVFs)**, **Artificial Potential Fields (APFs)**, **Physics-Informed Neural Networks (DeepReach)**, **cluttered-environment neural HJB formulations**, and **industrial aerial autonomy patterns** from simple approaches up to modern certified methods.

## Zero-Server Local Usage (`file://` Ready)

Unlike pages that rely on `fetch("data.json")` or cross-origin image pixel reads (which fail under `file://` browser CORS restrictions), this lab works **both** by double-clicking `index.html` directly from your file manager **and** via any local HTTP server:

```bash
# Option 1: Open directly
xdg-open index.html

# Option 2: Local HTTP server
python3 -m http.server 8080
```

## Reproducing All Numerical Solvers & PINN Experiments

All precomputed 2D/3D/4D HJI-VI PDE tensors, Dubins Air3D reachable tubes, 4D Reach-Avoid precision dock landing funnels, FaSTrack tracking error bounds, SIREN PINNs, conformal verification runs, and cluttered-map navigation benchmarks are generated from scratch by:

```bash
python3 generate_data.py
```

This regenerates both `data.json` (inspectable raw metrics and quantized tensors) and `data.js` (`file://`-safe script bundle).

## Interactive Labs Included

1. **Synchronized Quadrotor Flight + Phase-Space Point-of-No-Return Lab** (with pitch tilt, wind gusts, and sensing/actuation latency).
2. **Interactive 3D Level-Set Surface Explorer** ($z = V(x, \tau)$ slicing the $z=0$ plane, **BRS vs. BRT** fly-through slab counterexample, and viscosity kinks/shocks).
3. **3D Fixed-Wing / Dubins Air3D Twisted Reachable Tube Explorer** (rotatable 3D $(x, y, \psi)$ helical tube + heading slice + live evasion rollout).
4. **Drone Planning & Control Lab**:
   - **4D Cluttered Arena BRT** (joint vs. per-obstacle union under-approximation),
   - **4D Reach-Avoid Precision Dock Landing Funnel** (touchdown target vs. go-around abort envelope in gusts),
   - **FaSTrack Nonlinear Lagged Quadrotor Sandbox** (tracking error bound bubble around a fast geometric planner).
5. **Safety Filter & Live Control-Space QP Inspector**: APF vs. HOCBF-QP vs. HJR-Switch vs. Smooth CBVF (HJR-CBF), visualizing live actuator disc $\|u\| \le a_{\max}$ half-space constraints and QP infeasibility.
6. **Live 60fps In-Browser PINN Studio**: Train a real neural network with exact 2nd-order PDE residual backpropagation right in your browser (`SIREN sin` vs. `Tanh` vs. `ReLU`, temporal curriculum, hard boundary anchor vs. steady-state collapse).
7. **Cluttered Navigation Formulation Lab**: Step-by-step progression on a $45\,\text{m} \times 45\,\text{m}$ obstacle field comparing uncoupled collocation losses, $\rho$-triggered 1-step lookahead policies, classical APF, and exact Eikonal HJB wavefronts.
8. **Conformal / Scenario Verification Explorer**: Interactive $\delta$-level calibration and Clopper-Pearson safety certificate calculator.
9. **Industry Feasibility & Drone Architecture Calculator** + **5-Scenario Autonomy Engineering Diagnostic**.
10. **In-Page Session BYOK Study Tutor**: Connect your own ephemeral session API key (Google Gemini, Anthropic Claude, OpenAI, or OpenRouter) right inside the page.
