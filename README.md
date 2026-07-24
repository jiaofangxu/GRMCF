# GRMCF: Generalizable-Rapid-Multi-Target-Capture-Framework-Based-on-a-Single-Conventional-PTZ-Camera
A generalizable rapid multi-target capture framework based on a single conventional PTZ camera.

This repository provides the official open-source code for the paper with DOI: [10.1016/j.eswa.2026.131189](https://doi.org/10.1016/j.eswa.2026.131189).

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

- **`core/`**
  Includes the complete source code and a runnable demo for the two core GRMCF components:

  * **Modeling program**: constructs feasible viewing volumes from target detections and PTZ camera parameters.
  * **OGSOA-GRO solver**: optimizes the PTZ capture sequence and executable PTZ states.

  The modeling program and the OGSOA-GRO solver are implemented as two independent programs and are both invoked through **HTTP APIs**.

- **`ablation_studies/`**  
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

> For the full PTZ simulation source code, including the UE4 project files (>10 GB), please contact the paper author at xueshengjiaofangxu@163.com.
