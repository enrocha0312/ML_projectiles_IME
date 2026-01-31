# RAMP-RT Dataset

**Rocket, Artillery, and Mortar Projectile Radar Targets**

Overview

The RAMP-RT dataset (Rocket, Artillery, Mortar Projectiles – Radar Targets) provides a comprehensive collection of synthetic and real radar measurements of ballistic projectiles, along with state estimates obtained through the Square-Root Cubature Kalman Filter (SR-CKF).
It contains a total of 5,048 feature vectors, each associated with one of the projectile classes:

- LM/ML – Mortar Light (≈ 60 mm)

- MM – Mortar Medium (≈ 81–82 mm)

- HM/MH – Mortar Heavy (120 mm)

- LG/GL – Gun/Howitzer Light (105–122 mm)

- HG/GH – Gun/Howitzer Heavy (130–155 mm)

- LR/RL – Rocket Light (< 127 mm)

- MR/RM – Rocket Medium (128–300 mm)

- HR/RH – Rocket Heavy (> 300 mm)

- UN – Unknown / ill-conditioned estimation cases

These classes cover a broad range of calibers and dynamic regimes typically observed in battlefield environments.

A total of 1,650 ballistic trajectories were generated using a point-mass model incorporating gravity and aerodynamic drag.
For each trajectory, synthetic radar detections—range, azimuth, and elevation—were produced to emulate a modern Weapon Locating Radar (WLR) (e.g., Firefinder/Counter-Battery Radar).
Realistic sensor errors were applied, including: range noise (10 m), angular noise (≈ 2 mrad), SNR-consistent scattering fluctuations

SR-CKF Estimation

For every trajectory, the SR-CKF estimated:

projectile position and velocity
unconstrained ballistic coefficient 
	​
No prior knowledge of c_B or drag model was assumed, ensuring realism under uncertain aerodynamic regimes.

After filtering and SR-CKF-based extrapolation, the dataset expanded to 5,048 kinematic feature vectors, each containing multiple physical parameters derived purely from radar observables.

**Dataset Contents**

Each record in RAMP-RT contains:

Class label (9 categories: ML, MM, MH, GL, GH, RL, RM, RH, UN)

Estimated kinematic parameters extracted from SR-CKF, including: firing angle, initial velocity , horizontal range, maximum altitude, mean velocity during the detection interval and estimated ballistic coefficient 	​,RCS statistical descriptors, described below
<img width="1920" height="967" alt="git-dataset" src="https://github.com/user-attachments/assets/56ba0bab-f793-4708-b22b-41f1804613a5" />

**RCS Features**

A The RCS measurements were incorporated into the dataset, comprising:

- **mean RCS** over the observation window
  
- **maximum scatter difference (MSD)**
  

These features were derived from:

- Numerical scattering models in the literature  
  ~\cite{kenyon2015numerical,kenyon2015numericalRock,kenyon2016numericInfl}
- Experimental S-band radar measurements performed at the **Brazilian Army Technological Center (CTEx)**


---

## **Real Radar Measurements**

Beyond the synthetic dataset, **105 real projectile trajectories** were included for the **MM (medium mortar)** and **MH (heavy mortar)** classes.  
These measurements were obtained during live-fire campaigns conducted at:

- **CAEx** – Army Evaluation Center (CAEx)
  
- **CTEx** – (CTEx) and the Brazilian Army Technological Center
  

These real samples serve as an important validation subset for ML classifiers and filter benchmarking.

## **Intended Use**


RAMP-RT is designed for advanced research in: Radar signal processing and nonlinear filtering, machine learning classification of ballistic projectiles, estimation and tracking of launch and impact points, benchmarking of counter-battery radar algorithms and fusion of kinematic and RCS-based features

Example Applications

- Training and evaluating classifiers capable of distinguishing rockets, artillery shells, and mortar rounds from radar-derived features

- Benchmarking Kalman filter variants under uncertain aerodynamic conditions

- Developing integrated pipelines for simultaneous classification and ballistic estimation

- Supporting data-driven studies in WLR performance modeling and sensor fusion


