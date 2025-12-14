# GRMCF: Generalizable-Rapid-Multi-Target-Capture-Framework-Based-on-a-Single-Conventional-PTZ-Camera
A generalizable rapid multi-target capture framework based on a single conventional PTZ camera.

**Workflow figure (PDF):** [`./workflow_figure.pdf`](./workflow_figure.pdf)

> 

---

## Key Components

### 1) Feasible Viewing Volume Back-Projection Modeling
Given wide-angle target bounding boxes, GRMCF back-projects them into the PTZ parameter space and constructs feasible “viewing volumes” (geometric neighborhoods) for each target, enabling constraint-aware scheduling.

### 2) Fast Heuristic Optimization Algorithm (OGSOA-GRO)
GRMCF uses **OGSOA-GRO** to efficiently optimize:
- the visiting order (sequence),
- and the concrete PTZ states (pan, tilt, zoom) under 3D geometric constraints,
to produce a rapid capture schedule suitable for real-time settings.

---

## Repository Layout

- **`grmcf_core/`**  
  Includes:
  - GRMCF core **Linux executables** (modeling + OGSOA-GRO solver)
  - **All ablation experiment code**
  - **Synthetic data generation code**

- **`ptz_time_cost_calibration/`**  
  Includes:
  - PTZ time-cost calibration programs (pan/tilt/zoom/autofocus stabilization fitting)
  - Real measured calibration data from **Hikvision DS-2DC4A423IW-DE(S6)**
  - Scripts/functions to reproduce fitted time-cost curves

- **`workflow_figure.pdf`**  
  A workflow PDF that presents the GRMCF pipeline (from detections to an executable PTZ schedule).

---

## Workflow Overview

A typical capture workflow is:

1. **Input**: target detections as 2D bounding boxes in the wide-angle image  
2. **Back-projection modeling**: map each target box to feasible PTZ neighborhoods (viewing volumes)  
3. **Heuristic scheduling (OGSOA-GRO)**: jointly optimize sequence + feasible PTZ states  
4. **Output**: a PTZ control sequence `[(P1, T1, Z1), (P2, T2, Z2), ...]` for rapid capture

---

## Getting Started

### Clone
```bash
git clone https://github.com/jiaofangxu/GRMCF.git
cd GRMCF
```

