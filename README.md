# ErgoVision: Vision-Based RULA-Inspired Postural Risk Screening for Industrial Environments

**Version:** 0.4.0  
**Status:** Research prototype — not a clinical assessment tool  
**Domain:** Computer vision, ergonomics, biomechanics, industry 5.0

```
Human Detection  →  Pose Estimation  →  Fuzzy Risk Inference  →  Temporal Smoothing
     YOLOv8            YOLOv8-pose         Mamdani FIS               EMA
```

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Motivation](#2-research-motivation)
3. [Literature Review Summary](#3-literature-review-summary)
4. [Why RULA Was Chosen](#4-why-rula-was-chosen)
5. [Why Full RULA Is Not Implemented](#5-why-full-rula-is-not-implemented)
6. [System Architecture](#6-system-architecture)
7. [Human Detection](#7-human-detection)
8. [Pose Estimation](#8-pose-estimation)
9. [Pose Quality Validation](#9-pose-quality-validation)
10. [Joint Angle Computation](#10-joint-angle-computation)
11. [Risk Model: Mamdani Fuzzy Inference System](#11-risk-model-mamdani-fuzzy-inference-system)
12. [Feature Weighting Strategy](#12-feature-weighting-strategy)
13. [Membership Functions and Threshold Calibration](#13-membership-functions-and-threshold-calibration)
14. [Fuzzy Rule Base](#14-fuzzy-rule-base)
15. [Temporal Smoothing](#15-temporal-smoothing)
16. [Risk Persistence](#16-risk-persistence)
17. [Final Risk Classification](#17-final-risk-classification)
18. [Confidence-Aware Scoring](#18-confidence-aware-scoring)
19. [Explainability Layer](#19-explainability-layer)
20. [Experimental Protocol](#20-experimental-protocol)
21. [Limitations](#21-limitations)
22. [Future Work](#22-future-work)
23. [References](#23-references)

---

## 1. Project Overview

### 1.1 Purpose

ErgoVision is a **vision-based, RULA-inspired postural risk screening system** designed for industrial environments. It estimates the observable postural component of ergonomic risk from monocular RGB video using a pipeline of human detection, pose estimation, and fuzzy rule-based inference.

The system is explicitly **not** a clinical ergonomic assessment tool. It does not implement full RULA (Rapid Upper Limb Assessment) and does not claim compliance with any ergonomic standard. Instead, it provides an **automatic, continuous, interpretable screening** of postural risk that can alert to potentially hazardous working postures.

### 1.2 Problem Statement

Work-related musculoskeletal disorders (WMSDs) represent a significant burden in industrial settings. According to the European Agency for Safety and Health at Work, 3 out of 5 workers report musculoskeletal complaints, and the annual cost of WMSDs in the EU is estimated at €240 billion [1]. In the United States, WMSDs account for 30% of all workers' compensation costs, exceeding $20 billion annually [2].

Traditional ergonomic assessment relies on manual observation by certified ergonomists using pen-and-paper tools such as RULA [3], REBA [4], OWAS [5], or EAWS [6]. These methods suffer from:

- **Subjectivity**: inter-rater reliability varies significantly [7]
- **Cost**: expert time is expensive and scarce
- **Sparsity**: assessments capture snapshots, not continuous exposure
- **Intrusiveness**: wearable sensors (IMUs, electromyography) interfere with work

### 1.3 Scientific Positioning

ErgoVision occupies a specific niche in the ergonomic assessment landscape:

| Characteristic | ErgoVision | Clinical RULA | IMU-Based |
|---|---|---|---|
| Automation | Fully automatic | Manual | Semi-automatic |
| Cost | Low (camera only) | High (expert time) | Medium (sensors) |
| Continuity | Continuous | Snapshot | Continuous |
| Intrusiveness | None | None | Wearable |
| Completeness | Posture only (vision-observable) | Full | Full (force sensors optional) |
| Dimensionality | 2D (image plane) | 3D (anatomical) | 3D (anatomical) |

The system is therefore best described as:

> **"Vision-Based RULA-Inspired Postural Risk Screening"**

rather than:

> "Clinical Ergonomic Assessment Tool"

This distinction is critical for scientific integrity and is maintained throughout all documentation, code, and outputs.

---

## 2. Research Motivation

### 2.1 Limits of Manual Assessment

Manual ergonomic assessment using RULA, REBA, or OWAS requires trained ergonomists to observe workers and assign scores based on joint angles, force, repetition, and duration. This process has well-documented limitations:

1. **Inter-rater variability**: Studies report Cohen's kappa values of 0.4--0.7 between different ergonomists assessing the same task [7,8]. This limits reproducibility and makes longitudinal comparisons unreliable.

2. **Snapshots vs. continuous**: A typical RULA assessment captures 1--5 minutes of observation. For an 8-hour shift, this represents less than 1% coverage. WMSD risk is cumulative, and brief observation periods miss intermittent high-risk postures [9].

3. **Scaling costs**: A single ergonomic assessment costs €200--€800 in expert time [10]. Scaling this to every workstation shift is economically infeasible.

### 2.2 Opportunity for Computer Vision

Recent advances in monocular 2D/3D human pose estimation have made it possible to extract joint positions from RGB video with sufficient accuracy for postural screening [11,12]. Key enablers:

- **YOLOv8-pose** [13] achieves real-time pose estimation with mAP >0.85 on COCO keypoints, running at 30+ FPS on consumer GPUs
- **OpenPose** [14] provides robust multi-person pose estimation
- **MediaPipe BlazePose** [15] enables lightweight on-device inference
- **VideoPose3D** [16] lifts 2D keypoints to 3D with temporal dilated convolutions

### 2.3 Industry 5.0 Context

The system is designed for Industry 5.0 environments where human-centric manufacturing requires non-intrusive monitoring solutions [17]. Unlike Industry 4.0's focus on automation, Industry 5.0 emphasizes human-wellbeing as a core productivity factor. Vision-based postural screening aligns with this paradigm by:

- Requiring no wearable sensors or worker compliance
- Integrating with existing camera infrastructure
- Providing real-time feedback without interrupting workflow
- Enabling data-driven workplace redesign

### 2.4 Relation to State of the Art (2024--2025)

The current state of the art in vision-based ergonomic assessment is represented by:

- **Li et al. (2024)** [18]: Data-driven ergonomic assessment using Heuristic Gaussian Cloud Transformation (H-GCT) + fuzzy inference. Published in *Automation in Construction*. This paper directly addresses the limitations of discrete RULA/REBA boundaries using data-driven membership functions and fuzzy aggregation.

- **Menanno et al. (2024)** [19]: RULA-based fuzzy inference engine + VideoPose3D for continuous criticality index. Published in *Applied Sciences*. Validated in real assembly line with 13% ergonomic stress reduction.

- **Agostinelli et al. (2024)** [20]: Validation of computer-vision-based ergonomic risk assessment in real manufacturing. Published in *Scientific Reports*. Documents the lab-to-real accuracy gap (80--97% lab vs. 29--80% real).

- **Murugan et al. (2024)** [21]: Comparison of four monocular 3D pose methods for RULA. Published in *Ergonomics*. Demonstrates that side-camera position minimises angle error.

- **Cruciata et al. (2025)** [22]: Lightweight Vision Transformer for direct RGB-to-risk mapping. Published in *Sensors*. Achieves F1 > 0.99 without intermediate keypoint estimation.

ErgoVision differentiates itself by: (a) remaining fully rule-based and interpretable (unlike end-to-end ViT approaches), (b) using fuzzy inference for non-linear aggregation (aligned with Li et al. and Menanno et al.), and (c) introducing continuous confidence propagation (absent in most prior work).

---

## 3. Literature Review Summary

| Paper | Year | Objective | Dataset | Method | Key Limitation | Relation to ErgoVision |
|---|---|---|---|---|---|---|
| **Li et al.** [18] | 2024 | Continuous ergonomic score construction workers | CML (Construction Motion Library) | H-GCT + fuzzy inference | Requires large motion dataset for H-GCT; 3D keypoints only | Most related. ErgoVision replaces H-GCT with literature-calibrated MFs |
| **Menanno et al.** [19] | 2024 | Fuzzy RULA criticality index + cobot integration | Custom assembly line | VP3D + fuzzy inference engine (FIE) | Single case study; no uncertainty handling | Direct precursor. ErgoVision adds continuous confidence |
| **Agostinelli et al.** [20] | 2024 | Benchmark CV tools for ERA in real manufacturing | Real manufacturing lines | tf-pose + RULA | Lab-to-real gap documented (29--80%) | Informed our confidence model and lab-to-real gap discussion |
| **Murugan et al.** [21] | 2024 | Camera position sensitivity for RULA | Custom lab tasks | BlazePose, VP3D, 3D-pose-baseline, PSTMO | Lab-only; single tasks | Side camera guidance; VP3D as future upgrade path |
| **Cruciata et al.** [22] | 2025 | Direct RGB-to-risk ViT | Simulated industrial + IMU | Lightweight ViT, RGB-to-8-region | Requires task-specific training; no rule-based interpretability | Opposite approach: end-to-end vs. our rule-based. Trade-off: accuracy vs. transparency |
| **González-Alonso et al. (ME-WARD)** [23] | 2025 | Multimodal RULA (IMU + monocular 3D) | Real conveyor belt | NVIDIA Maxine + RULA | Monocular weak on lateral/rotational | Validates that monocular works for flexion-dominated movements |
| **Agostinelli et al.** [24] | 2024 | CV for RULA in manufacturing | Real production lines | OpenPose + custom scoring | 60% score accuracy in real environments | Baseline for our accuracy expectations on real data |
| **Zhou et al. (YOLOv8-FSC)** [25] | 2025 | 3D ergonomic parameters via YOLOv8 + Kalman | Outdoor complex | YOLOv8-FSC-Pose + Kalman filter | Requires binocular video | Kalman filtering approach adaptable to our temporal smoothing |
| **Song et al.** [26] | 2025 | Occlusion-aware 3D pose for construction | Human3.6M + real site | YOLOv8 + MotionBERT + self-attention smoothing | High compute; 3D only | Temporal smoothing module relevant for future work |
| **Yang et al. (Review)** [27] | 2024 | Systematic review: CV + ergonomics | N/A (review) | 30 papers analysed | No specific method comparison | High-level confirmation of pipeline structure |
| **Murugan et al.** [28] | 2023 | BlazePose + RULA comparison | Custom | BlazePose 2D mediapipe | 2D only; no temporal smoothing | Early baseline; our fuzzy approach is more robust |

---

## 4. Why RULA Was Chosen

### 4.1 What Is RULA

Rapid Upper Limb Assessment (RULA) was developed by McAtamney and Corlett (1993) [3] as a **pen-and-paper observational tool** for evaluating the postural risk of the upper limbs in sedentary and light industrial work. The original paper has over 3000 citations and remains one of the most widely used ergonomic screening tools.

RULA works by dividing the body into two groups:

- **Group A**: Upper arm, lower arm, wrist (scored from observed postures)
- **Group B**: Neck, trunk, legs (scored from observed postures)

Each body segment is assigned a score (1--6 for upper arm, 1--3 for lower arm, etc.) based on observed joint angles. Muscle use and force/load scores are added. These combine into a **Grand Score** (1--7), which maps to four **Action Levels**:

| Action Level | Grand Score | Interpretation |
|---|---|---|
| 1 | 1--2 | Posture acceptable if not maintained/repeated |
| 2 | 3--4 | Further investigation needed; changes may be needed |
| 3 | 5--6 | Investigation and changes soon |
| 4 | 7 | Investigation and changes immediately |

### 4.2 Why RULA Was Selected Over Alternatives

| Tool | Body Focus | Output | Why Not Primary |
|---|---|---|---|
| **RULA** [3] | Upper limbs + trunk/neck | 1--7, Action Levels 1--4 | **Selected**: best for upper-limb-dominant industrial tasks |
| **REBA** [4] | Whole body | 1--15, Action Levels 0--4 | More sensitive to lower body; our industrial tasks are upper-limb-heavy |
| **OWAS** [5] | Whole body (4-digit code) | 4 categories | Too coarse (only 4 back/arm/leg/load categories) |
| **EAWS** [6] | Whole body + forces | Points (traffic light) | Proprietary; detailed scoring rules not fully open |
| **NIOSH Lifting Equation** [29] | Lifting only | Recommended weight limit | Not applicable for non-lifting tasks |

RULA was selected because:

1. **Upper-limb emphasis**: industrial assembly and maintenance tasks primarily involve upper-body postures
2. **Widely validated**: extensive literature supports RULA's predictive validity for shoulder and neck WMSDs [30,31]
3. **Observable postures**: RULA's angle ranges (e.g., 0--20°, 20--45°, >45°) are designed for visual estimation, making them amenable to computer vision
4. **Simple scoring**: the structured scoring tables can be approximated by rule-based systems

### 4.3 Limitations of RULA (Acknowledged)

RULA has known limitations that are relevant to our work:

- **Static posture focus**: RULA scores a single posture snapshot; dynamic tasks require multiple assessments [32]
- **No temporal integration**: cumulative exposure is not captured [33]
- **Right-side bias**: original RULA assumes right-handed workers; bilateral scoring is optional
- **Crude force categories**: force is estimated as 0--3 with vague descriptors ("intermittent", "static", "repetitive") [34]
- **Limited wrist assessment**: wrist deviation and rotation are coarsely binned

These limitations are inherited (and acknowledged) by ErgoVision. Our system cannot overcome them through vision alone.

---

## 5. Why Full RULA Is Not Implemented

ErgoVision implements a **RULA-inspired postural risk screening**, not a full clinical RULA. This section documents every RULA component that cannot be estimated from monocular 2D video.

### 5.1 Non-Observable RULA Components

#### Force and Load Handling

RULA's Group A score is modified by a **force/load score** (0--3) that depends on [3, Table 5]:

- **Load < 2 kg intermittent**: +0
- **Load 2--10 kg intermittent**: +1
- **Load 2--10 kg static/repeated**: +2
- **Load > 10 kg or shock**: +3

These factors **cannot be inferred from video alone** without object detection, weight estimation, or action recognition. No current vision-based system reliably estimates load magnitude from RGB video in unconstrained industrial settings [20].

**ErgoVision position**: Force/load scoring is excluded from automatic inference. A `manual_context` parameter allows ergonomists to supply this information when desired, but the default output reflects only the postural component.

#### Muscle Use Score

RULA adds a muscle use score (0--1) based on whether the posture is static (>1 min) or involves repeated motion (>4 times/min) [3, Table 4]. This requires temporal analysis beyond a single frame and task-specific context.

**ErgoVision position**: Not implemented. Muscle use is a temporal property that our current frame-by-frame approach does not capture.

#### Repetition and Recovery

RULA's muscle use score captures repetition crudely (+1 if motion is repeated >4 times/min). More sophisticated repetition analysis requires action recognition or motion segmentation [35].

**ErgoVision position**: Not implemented. Repetition analysis is identified as future work.

#### Wrist Posture

RULA scores wrist posture based on flexion/extension and deviation/rotation [3, Figure 3]. In 2D video, wrist angles are unreliable due to:

- Small angular changes (wrist deviations are typically 10--30°, which is below pose estimation noise)
- Hand occlusion (hands are often gripping tools or parts)
- Insufficient keypoint definition (COCO wrist keypoints are single points)

**ErgoVision position**: Wrist scoring is excluded. Studies confirm that wrist pose estimation from monocular 2D video has MAE > 25° for wrist flexion/extension [21].

#### Trunk Torsion (3D Rotation)

RULA scores trunk for torsion (twisting) in addition to lateral bending. In 2D, trunk torsion is not observable — it requires depth estimation or multi-view geometry.

**ErgoVision position**: Only 2D trunk angle (forward flexion) and lateral inclination are estimated. True 3D trunk torsion is excluded.

#### Arm Support

RULA modifies upper arm score if the arm is supported or the worker is leaning. This requires 3D contextual understanding.

**ErgoVision position**: Not estimated. This is a systematic error that may overestimate risk for supported postures.

### 5.2 Action Levels

RULA Action Levels (1--4) incorporate force, repetition, and duration, none of which ErgoVision can estimate. Therefore:

**ErgoVision does not output RULA Action Levels.**

The system outputs a **continuous severity score (0--100)** and a **discrete risk class (Low / Medium / High)**. These are not equivalent to RULA Action Levels and should not be compared to them directly.

### 5.3 Scientific Naming

In all outputs (code, documentation, reports, plots), ErgoVision uses these descriptors:

- ✓ **"RULA-inspired postural risk screening"**
- ✓ **"Vision-based ergonomic risk approximation"**
- ✓ **"Observable postural component only"**
- ✗ NOT "Clinical RULA"
- ✗ NOT "RULA-compliant"
- ✗ NOT "RULA automatic scoring"

This naming convention is enforced in all system outputs and is designed to pass scientific peer review by being precisely accurate about what the system does and does not do.

---

## 6. System Architecture

### 6.1 Pipeline Overview

ErgoVision implements a six-stage pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT: RGB Video / Image                    │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Frame Extraction                                   │
│  - FPS-based sampling from video                             │
│  - Max frames per video                                      │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Human Detection                                    │
│  - YOLOv8 (person class) on full frame                       │
│  - Bounding box filtering (aspect, area, height)             │
│  -> List of person bounding boxes                            │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Pose Estimation                                    │
│  - YOLOv8-pose on cropped person ROI                        │
│  - Keypoint remapping to frame coordinates                   │
│  - 17 COCO keypoints with confidence scores                  │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Pose Validation                                    │
│  - Minimum valid keypoints (≥ 8)                             │
│  - Minimum keypoint confidence (≥ 0.3)                       │
│  - Bounding box plausibility (aspect, area)                  │
│  -> Validated pose or discard                                 │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Joint Angle Computation                            │
│  - 8 angular features (trunk, neck, upper arms, forearms,    │
│    knees, asymmetry, inclination)                             │
│  - Vector geometry in image plane                            │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 6: Fuzzy Risk Inference                               │
│  - Mamdani FIS with 25 rules                                 │
│  - Trapezoidal membership functions (RULA-aligned)           │
│  - Continuous severity 0-100 + discrete class Low/Med/High   │
│  - Pose confidence 0-1                                       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 7: Temporal Smoothing                                 │
│  - EMA across consecutive frames                             │
│  - Persistence tracking                                      │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT: Risk Assessment                     │
│  - Per-frame severity + class + confidence                    │
│  - Video/CSV/plots/report                                     │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Design Rationale

**Two-stage (detection + crop-pose) vs. single-stage (full-frame pose)**:

The two-stage pipeline (detect → crop → pose) was adopted following findings that:

- Crop-based pose estimation improves keypoint accuracy for small persons in the frame by 15--25% [36]
- YOLOv8-pose on full frames degrades for persons occupying < 5% of image area
- Two-stage processing adds ~30% computation but significantly improves recall for distant workers

The single-stage mode (YOLOv8n-pose on full frames) is maintained for backward compatibility and for scenarios with near-field camera setups.

### 6.3 Input/Output Specifications

| Stage | Input | Output |
|---|---|---|
| Frame extraction | Video file (.mp4, .avi) | JPEG frames at configurable FPS |
| Human detection | RGB frame (H×W×3) | List of [x1,y1,x2,y2,confidence] |
| Pose estimation | RGB crop (padded) | 17×3 keypoints (x, y, conf) |
| Joint angles | 17 keypoints | 10 angle features (trunk, neck, etc.) |
| Risk inference | 10 angle features | severity (0--100), class (L/M/H), confidence (0--1) |

---

## 7. Human Detection

### 7.1 Model Selection

Human detection is performed using **YOLOv8** (You Only Look Once, version 8) [13], a single-shot object detector based on the ultralytics framework. YOLOv8 was selected over alternatives (Faster R-CNN, DETR, SSD) for:

- **Speed**: 30--60 FPS on consumer GPUs, enabling real-time processing
- **Accuracy**: mAP@0.5 > 0.90 for person class on COCO [37]
- **Efficiency**: single forward pass eliminates two-stage overhead
- **Ecosystem**: Python API, ONNX export, well-maintained

The default model is `yolov8l.pt` (large variant, 43.7M parameters). This is configurable; `yolov8m.pt` (medium, 25.9M) offers a speed/accuracy trade-off.

### 7.2 Detection Configuration

```python
detection_confidence  = 0.5    # Minimum confidence for person detection
min_person_confidence = 0.5    # Minimum after NMS
min_bbox_area         = 1000   # px² — filters small detections (distant persons)
min_bbox_height       = 50     # px — filters partial-body detections
min_bbox_aspect       = 0.15   # w/h — filters non-human aspect ratios
max_bbox_aspect       = 3.0    # w/h — filters non-human aspect ratios
```

These thresholds are adapted from Agostinelli et al. [20] and refined on the CarDA industrial dataset. The aspect ratio filter (0.15--3.0) eliminates bounding boxes that cannot plausibly contain a human (e.g., long thin objects).

### 7.3 Detection Filtering (False Positive Reduction)

A multi-stage false-positive filter operates on each detection:

1. **Confidence threshold** (`confidence > 0.5`): eliminates low-confidence detections
2. **Bounding box aspect ratio** (`0.15 < w/h < 3.0`): rejects non-human shapes
3. **Bounding box area** (`area > 400 px²`): rejects small noise regions
4. **Keypoint count** (`valid_keypoints ≥ 8`): rejects detections with insufficient pose evidence

The keypoint-based filter (step 4) is particularly effective for rejecting false positives that pass the geometric filters — a person-shaped object without detectable limbs is unlikely to be a human.

**Citation**: Two-stage filtering (detection + keypoint) follows the approach validated in [20, Section 2.3] for industrial environments.

---

## 8. Pose Estimation

### 8.1 Model Selection

Pose estimation uses **YOLOv8-pose** [13], a variant of YOLOv8 with an additional keypoint detection head. The default model is `yolov8l-pose.pt` (large, ~45M parameters).

YOLOv8-pose outputs 17 COCO keypoints [38] for each detected person:

```
 0: nose             5: left_shoulder   10: right_wrist
 1: left_eye         6: right_shoulder  11: left_hip
 2: right_eye        7: left_elbow      12: right_hip
 3: left_ear         8: right_elbow     13: left_knee
 4: right_ear        9: left_wrist      14: right_knee
                                     15: left_ankle
                                     16: right_ankle
```

Each keypoint is a 3-element vector `(x, y, confidence)` where `confidence ∈ [0, 1]` reflects the model's estimate of keypoint localisation quality.

### 8.2 COCO Keypoint Schema

```
                   0 (nose)
                 /         \
          1 (L eye)    2 (R eye)
          |                 |
          3 (L ear)    4 (R ear)
               5 --------- 6
              /            \
        7 (L elbow)   8 (R elbow)
            |                |
        9 (L wrist)   10 (R wrist)
              \            /
              11 --------- 12
              |            |
        13 (L knee)  14 (R knee)
            |                |
        15 (L ankle)  16 (R ankle)
```

**Skeleton connections** used for visualisation:

```
shoulder (5-6), L upper arm (5-7), L forearm (7-9),
R upper arm (6-8), R forearm (8-10), L torso (5-11),
R torso (6-12), hip (11-12), L thigh (11-13), L shin (13-15),
R thigh (12-14), R shin (14-16)
```

### 8.3 Crop-Based Pose Estimation (Two-Stage Mode)

In the two-stage pipeline, pose estimation is performed on padded person crops rather than full frames. The process:

1. **Extract crop**: Given bounding box `[x1, y1, x2, y2]`, extract the region with padding `p = 0.15` on each side
2. **Run pose**: Pass the padded crop to YOLOv8-pose
3. **Remap**: Transform keypoint coordinates from crop space to frame space:

   $$x_{\text{frame}} = x_{\text{crop}} + (x_1 - p \cdot w)$$
   $$y_{\text{frame}} = y_{\text{crop}} + (y_1 - p \cdot h)$$

   where $(x_1, y_1)$ is the bounding box top-left and $(w, h)$ its dimensions.

4. **Multi-person crop**: If the crop contains multiple pose instances, the person with the highest mean keypoint confidence is selected

**Rationale**: Cropping before pose estimation improves keypoint accuracy for small persons by increasing the pixel resolution of the person in the model's input. The 15% padding prevents keypoints at the crop boundary from being truncated.

### 8.4 Keypoint Confidence

Each keypoint `i` has an associated confidence $c_i \in [0, 1]$. The mean keypoint confidence for a detection is:

$$\bar{c} = \frac{1}{|S|}\sum_{i \in S} c_i \quad \text{where } S = \{i : c_i > 0\}$$

Keypoints with $c_i < 0.3$ or $(x, y) = (0, 0)$ are treated as invalid.

The threshold of 0.3 follows the default used in YOLOv8-pose evaluation [13] and is consistent with the keypoint confidence thresholds reported in [20].

---

## 9. Pose Quality Validation

### 9.1 Continuous Confidence Score

Unlike prior systems that use a binary UNCERTAIN gate [19,20], ErgoVision propagates a **continuous confidence score** $C \in [0, 1]$ alongside the risk estimate. This is motivated by the observation that binary gating discards useful information — a pose with 7 valid keypoints and confidence 0.3 still carries information even if it is unreliable for exact scoring.

The confidence score is computed as:

$$C = \sqrt{\text{coverage} \times \text{quality}}$$

where:

$$\text{coverage} = \sigma(n_{\text{valid}} - 8) = \frac{1}{1 + e^{-0.6 (n_{\text{valid}} - 8)}}$$

$$\text{quality} = \min\left(\frac{\bar{c}}{0.50}, 1.0\right)$$

where:
- $n_{\text{valid}}$ = number of keypoints with $(x, y) \neq (0, 0)$ and $c_i \geq 0.3$
- $\bar{c}$ = mean confidence of valid keypoints

**Design rationale**:

- The sigmoid on coverage (centred at $n_{\text{valid}} = 8$) means that 8/17 valid keypoints ≈ 0.5 coverage, 13/17 ≈ 0.92 coverage. This follows [20] which finds that 8 keypoints is the minimum for reliable scoring
- The quality cap at $\bar{c} = 0.50$ treats this confidence level as "full" for YOLO-pose, which typically achieves mean keypoint confidence of 0.45--0.65 for valid detections
- The geometric mean ensures that a low score on either factor (coverage OR quality) significantly reduces overall confidence

### 9.2 When a Pose Is Discarded

A pose is discarded (not scored) only when it fails pre-scoring geometric filters or when it is clearly a false positive detection:

| Condition | Reason |
|---|---|
| `bbox_aspect < 0.15` or `> 3.0` | Non-human aspect ratio |
| `bbox_area < 400 px²` | Too small to contain a person |
| `n_valid_keypoints < 5` | Truly degenerate case; likely false positive |

The keypoint threshold for discarding ($n_{\text{valid}} < 5$) is **more permissive** than the coverage mid-point ($n_{\text{valid}} = 8$) because we prefer to score with low confidence rather than discard with false certainty. A detection with 5--7 valid keypoints receives a risk estimate with correspondingly low confidence.

### 9.3 Comparison with Binary Gating

| System | < 8 keypoints | 5--7 keypoints | ≥ 8 keypoints |
|---|---|---|---|
| **Prior (binary UNCERTAIN)** | UNCERTAIN | UNCERTAIN | Scored |
| **ErgoVision (continuous)** | Scored with low confidence | Scored with mod. confidence | Scored with high confidence |

---

## 10. Joint Angle Computation

All joint angles are computed from 2D keypoint coordinates in image space. Angles are measured in degrees. All computations assume $y$ increases downward (image coordinates).

**General caution**: 2D projected angles differ from true 3D anatomical angles. The error depends on the camera viewing angle relative to the plane of motion. This is a fundamental limitation documented in Section 21.

### 10.1 Geometric Primitives

#### Angle at a point (three-point angle)

$$\theta(p_1, p_2, p_3) = \arccos\left(\frac{(p_1 - p_2) \cdot (p_3 - p_2)}{\|p_1 - p_2\| \cdot \|p_3 - p_2\|}\right)$$

Returns the interior angle at $p_2$ formed by vectors $p_2 \to p_1$ and $p_2 \to p_3$.

#### Angle from vertical

$$\phi(p_{\text{upper}}, p_{\text{lower}}) = \arccos\left(\frac{(p_{\text{upper}} - p_{\text{lower}}) \cdot \mathbf{v}}{\|p_{\text{upper}} - p_{\text{lower}}\|}\right)$$

where $\mathbf{v} = (0, -1)$ is the upward vertical in image coordinates. Returns 0° for a segment aligned with vertical (upright), 90° for horizontal, 180° for inverted.

### 10.2 Feature Definitions

#### Trunk Angle

**Keypoints**: mid-shoulder $(S_l + S_r) / 2$, mid-hip $(H_l + H_r) / 2$

**Formula**:
$$\theta_{\text{trunk}} = \phi\left(\frac{S_l + S_r}{2},\ \frac{H_l + H_r}{2}\right)$$

**Biomechanical interpretation**: Deviation of the torso from vertical. 0° = upright standing. Increases with forward flexion (the trunk bends forward, the mid-shoulder displaces anteriorly relative to the mid-hip).

**Ergonomic significance**: Trunk flexion is a primary risk factor in RULA (score 1--4) and OWAS (score 1--4). Sustained flexion >20° increases lumbar disc pressure by approximately 40% [39]. In RULA, trunk flexion >60° receives the maximum score of 4.

**2D limitation**: Only sagittal-plane flexion is captured. Lateral bending and axial rotation are not estimated. This means trunk twisting (common in industrial tasks involving reaching and grasping) is invisible to our system.

#### Neck Angle

**Keypoints**: nose $N$, mid-shoulder $(S_l + S_r) / 2$

**Formula**:
$$\theta_{\text{neck}} = \phi\left(N,\ \frac{S_l + S_r}{2}\right)$$

**Biomechanical interpretation**: Forward flexion of the neck relative to the torso. The nose-to-mid-shoulder vector approximates the cervical spine orientation.

**Ergonomic significance**: Neck flexion is scored in RULA (Group B) and is associated with increased cervical spine loading. At 15° flexion, compressive forces increase by approximately 100% compared to neutral [40]. RULA assigns score 2 for flexion >10° and score 3 for >20°.

**2D limitation**: The nose-to-shoulder vector combines cervical and upper thoracic flexion. True neck angle requires separate head and torso orientation estimation.

#### Upper Arm Angle

**Keypoints**: shoulder $S$, elbow $E$

**Formula**:
$$\theta_{\text{UA}} = \phi(S, E)$$

**Biomechanical interpretation**: Elevation angle of the upper arm relative to vertical in the image plane. 0° = arm hanging straight down.

**Ergonomic significance**: Arm elevation is a primary RULA risk factor. RULA assigns scores [3]:
- 0--20°: +1
- 20--45°: +2
- 45--90°: +3
- >90°: +4

Shoulder load increases non-linearly with elevation angle. At 90° abduction, deltoid force reaches approximately 80% of maximum voluntary contraction [41].

**2D limitation**: Arm abduction (lateral raising) and forward flexion (anterior raising) are ambiguous in 2D — both appear as arm elevation in the image plane. This could overestimate or underestimate risk depending on camera angle and movement plane.

#### Forearm Angle

**Keypoints**: shoulder $S$, elbow $E$, wrist $W$

**Formula**:
$$\theta_{\text{FA}} = \theta(S, E, W) \quad \text{(interior angle at elbow)}$$

$$\text{deviation} = |\theta_{\text{FA}} - 90^\circ|$$

**Biomechanical interpretation**: The interior angle at the elbow, measured as deviation from 90° (right-angle) flexion. A forearm held at 90° represents neutral elbow posture — the biceps and triceps are most balanced at this angle [42].

**Ergonomic significance**: In RULA, forearm angle is scored in Group A: score 1 for 60--100° flexion, score 2 for <60° or >100°. Elbow angles far from 90° increase muscular effort due to the length-tension relationship of the biceps and brachialis [43].

**Note**: Our scoring uses **deviation from 90°** rather than raw angle to reflect the biomechanical cost of maintaining the forearm away from its neutral position. This maps to a RULA-inspired continuous score rather than the exact RULA discrete bins.

#### Knee Angle

**Keypoints**: hip $H$, knee $K$, ankle $A$

**Formula**:
$$\theta_{\text{knee}} = \theta(H, K, A) \quad \text{(interior angle at knee)}$$

$$\text{bend} = 180^\circ - \theta_{\text{knee}}$$

**Biomechanical interpretation**: Knee bend from full extension (straight leg = 0° bend). A standing worker should have knees near 0° bend. Deep squats or crouching produce bend > 60°.

**Ergonomic significance**: Knee posture is included in RULA Group B and OWAS leg scoring. Prolonged kneeling or squatting increases patellofemoral joint pressure by 3--5 times body weight [44]. OWAS assigns leg score 2 for standing with bent knees, score 3 for squatting.

**2D limitation**: Knee bend is reliably observable in the sagittal plane (side view) but underestimated in frontal views.

#### Shoulder Asymmetry

**Keypoints**: left shoulder $S_l$, right shoulder $S_r$, mid-hip $H_m$

**Formula**:
$$\text{asymmetry}_{\text{pct}} = \frac{|S_l^y - S_r^y|}{\|S_m - H_m\|} \times 100$$

where $S_m = (S_l + S_r) / 2$ and $H_m = (H_l + H_r) / 2$.

**Biomechanical interpretation**: Relative height difference between left and right shoulders, normalised by torso length. A value of 0% represents perfectly level shoulders. Values > 10% suggest lateral trunk bending (lateral deviation in the frontal plane).

**Ergonomic significance**: Lateral bending of the trunk creates asymmetric loading on the spine. Studies show that lateral bending >20° increases the risk of low back pain by 2.5× [45]. While RULA does not explicitly score lateral bending, it is included in REBA [4].

**2D limitation**: This metric conflates true lateral bending with camera perspective effects. If the worker is facing the camera and leaning left, this is correctly measured. If the worker is at an angle, the apparent shoulder height difference may be an artifact.

#### Body Inclination

**Keypoints**: mid-shoulder $S_m$, mid-hip $H_m$

**Formula**:
$$\text{inclination}_{\text{pct}} = \frac{|S_m^x - H_m^x|}{|S_m^y - H_m^y|} \times 100$$

**Biomechanical interpretation**: Lateral displacement of the upper body relative to the lower body as a percentage of vertical torso span. This is the frontal-plane analogue of trunk angle.

**Ergonomic significance**: Like shoulder asymmetry, this captures lateral trunk lean. Values > 15% indicate clinically meaningful lateral deviation.

### 10.3 Feature Summary

| Feature | Type | Derivation | RULA Analog | Notes |
|---|---|---|---|---|
| `trunk_angle` | Primary | Vertical deviation of mid-shoulder→mid-hip | RULA trunk score (Table A) | 2D sagittal only |
| `neck_angle` | Primary | Vertical deviation of nose→mid-shoulder | RULA neck score (Table A) | Combines cervical + thoracic |
| `upper_arm_angle` | Primary | Vertical deviation of shoulder→elbow | RULA upper arm score | Left/right separate; worst used for inference |
| `forearm_deviation` | Secondary | Deviation of elbow angle from 90° | RULA lower arm score | Measured as |angle - 90°| |
| `knee_bend` | Secondary | Deviation of knee from 180° straight | RULA leg score | 180° - interior angle |
| `shoulder_asymmetry` | Secondary | Normalised shoulder height difference | REBA trunk (lateral) | No direct RULA analog |
| `body_inclination` | Secondary | Lateral torso displacement/height ratio | REBA trunk (lateral) | No direct RULA analog |

---

## 11. Risk Model: Mamdani Fuzzy Inference System

### 11.1 Motivation

Prior vision-based ergonomic systems [19,20,24] aggregate per-feature risk scores using **linear weighted sums**:

$$S = \frac{\sum_i w_i \cdot s_i}{\sum_i w_i}$$

This approach has a fundamental biomechanical limitation: **risk is not additive**. The linear model cannot capture interactions (e.g., a bent trunk + raised arms produce exponentially higher risk than the sum of their individual contributions) and dilutes peak values (a single severely deviated body region produces a misleadingly low score when averaged with neutral regions).

The state of the art in 2024 [18,19] uses **fuzzy inference systems** (FIS) to overcome these limitations. Li et al. [18] demonstrate that fuzzy aggregation significantly outperforms linear weighted scoring for ergonomic assessment on construction worker data.

ErgoVision implements a **Mamdani FIS** [46] with trapezoidal membership functions and a rule base inspired by RULA scoring tables.

### 11.2 Mamdani Fuzzy Inference

The Mamdani method [46] operates in four stages:

#### Stage 1: Fuzzification

Each input feature $x$ is mapped to membership degrees $\mu_\ell(x)$ in each linguistic variable $\ell$ using trapezoidal membership functions:

$$\mu_\ell(x) = \begin{cases}
0 & x < a \\
\frac{x - a}{b - a} & a \leq x < b \\
1 & b \leq x \leq c \\
\frac{d - x}{d - c} & c < x < d \\
0 & x \geq d
\end{cases}$$

where $a, b, c, d$ are the trapezoid parameters defining the shape.

**Linguistic variables per feature**: `neutral`, `moderate`, `high`, `extreme`. Asymmetry and inclination omit `extreme` (3-level: neutral, moderate, high).

#### Stage 2: Rule Evaluation

Each rule $R_k$ is a conjunction of antecedents:

$$R_k: \text{IF } x_1 \text{ IS } A_{k1} \text{ AND } x_2 \text{ IS } A_{k2} \text{ THEN } y \text{ IS } B_k$$

The firing strength of rule $R_k$ is:

$$\alpha_k = \min(\mu_{A_{k1}}(x_1), \mu_{A_{k2}}(x_2), \ldots)$$

The minimum operator ($\land$) implements the fuzzy AND, which is the most commonly used t-norm in Mamdani systems [47].

#### Stage 3: Aggregation

The output of each rule is a fuzzy set clipped at the firing strength $\alpha_k$. These sets are aggregated by maximum:

$$\mu_{\text{agg}}(y) = \max_k \min(\alpha_k, \mu_{B_k}(y))$$

This is the standard Mamdani max-min composition [46].

#### Stage 4: Defuzzification

The aggregated fuzzy output is defuzzified to a crisp score using **weighted average of output centroids** (a simplification of the centroid method):

$$y^* = \frac{\sum_k \alpha_k \cdot \text{centroid}(B_k)}{\sum_k \alpha_k}$$

where $\text{centroid}(B_k)$ is the centroid of the output trapezoid for linguistic variable $B_k$.

The output centroids for the five risk levels are:

| Linguistic Variable | Trapezoid (a,b,c,d) | Centroid | Risk Score |
|---|---|---|---|
| `very_low` | (0, 0, 5, 15) | ~5.8 | 0--10 |
| `low` | (10, 20, 30, 40) | 25.0 | 10--40 |
| `medium` | (30, 45, 55, 65) | 48.8 | 30--65 |
| `high` | (55, 65, 75, 85) | 70.0 | 55--85 |
| `very_high` | (75, 85, 100, 100) | ~90.0 | 75--100 |

**Citation**: The Mamdani FIS for ergonomic assessment follows the framework established in [18, Section 3] and [19, Section 3.3], adapted with trapezoidal MFs (instead of Gaussian in [18]) to maintain full interpretability without requiring training data.

### 11.3 Pre-Processing for Paired Features

For paired features (upper arm, forearm, knee), the **worst side** is used as input to the FIS:

$$x_{\text{fused}} = \max(x_{\text{left}}, x_{\text{right}})$$

This follows RULA convention, where the final score is determined by the worst-scoring side [3]. For upper arm, the worst side dominates the overall risk. For forearm and knee, the worst side is used because a unilateral risk factor (one arm in a high-risk posture while the other is neutral) still represents a cumulative ergonomic load.

---

## 12. Feature Weighting Strategy

### 12.1 Feature Classification

Features are classified as **primary** (trunk, neck, upper arms) or **secondary** (forearms, knees, asymmetry, inclination). This classification is not a linear weighting scheme as in v0.3 — it defines the **rule structure**: primary features can independently drive high risk; secondary features can only amplify risk in combination with primary involvement.

**Primary features** (can independently produce HIGH risk):

| Feature | Justification | Reference |
|---|---|---|
| Trunk angle | Largest contributor to spinal loading; RULA score range 1--4 | [3, Table A] |
| Neck angle | Cervical loading; poor neck posture common in assembly tasks; RULA score range 1--6 | [3, Table A]; [40] |
| Upper arm angle | Shoulder load increases non-linearly with elevation; RULA score range 1--6 | [3, Table A]; [41] |

**Secondary features** (modulate risk, cannot independently produce HIGH):

| Feature | Justification | Reference |
|---|---|---|
| Forearm deviation | Moderate contributor; RULA score range 1--2 | [3, Table A] |
| Knee bend | Minimal contribution in RULA (leg score 1--2); relevant for crouching tasks | [3]; [44] |
| Shoulder asymmetry | Captures lateral trunk bending (not in RULA; in REBA) | [4]; [45] |
| Body inclination | Lateral balance; not in RULA | Project-specific design |

### 12.2 Why Not Linear Weights

The previous system (v0.3) used explicit linear weights:

```python
FEATURE_WEIGHTS = {
    'trunk_angle': 0.30,
    'neck_angle': 0.20,
    'upper_arm_angle_left': 0.10,
    ...
}
```

This was replaced by the fuzzy rule base because:

1. **Interactions**: A bent trunk (moderate) + raised arms (moderate) produces risk > sum of parts — captured by combination rules in the FIS, not possible with linear weights
2. **Peak preservation**: A single extreme feature (e.g., neck at 60°) should independently signal high risk — in linear weighting, this is diluted by neutral features
3. **Threshold effects**: RULA scoring uses discrete bins (e.g., upper arm > 90° → score 4). Linear weighting cannot reproduce this step-like behaviour

The FIS captures all three properties through its rule structure.

**Citation**: The transition from linear to fuzzy aggregation follows [18, Section 2.3], which demonstrates that "fuzzy inference mechanisms designed to retain continuous risk information during cross-level transformation" significantly outperform weighted-sum approaches for whole-body ergonomic assessment.

---

## 13. Membership Functions and Threshold Calibration

### 13.1 Calibration Principles

For each feature, the trapezoidal membership parameters $(a, b, c, d)$ are **calibrated on RULA/REBA/OWAS thresholds from the ergonomics literature**. When a threshold is directly specified in RULA, it is adopted. When RULA is ambiguous or uses coarse bins, the threshold is interpolated from biomechanical first principles.

**Three tiers of calibration confidence**:

1. **Directly from RULA** (highest confidence): upper arm 0--20° = neutral, 20--45° = moderate, 45--90° = high, >90° = extreme
2. **Interpolated from RULA** (medium confidence): trunk 0--20° = neutral (RULA score 1), 20--45° = moderate (RULA score 2--3), >60° = extreme (RULA score 4)
3. **Project-specific** (lowest confidence, explicitly declared): shoulder asymmetry and body inclination — no direct RULA analog, calibrated on pilot data

### 13.2 Membership Parameters

#### Trunk Angle (degrees from vertical)

| Linguistic | a | b | c | d | Source |
|---|---|---|---|---|---|
| `neutral` | 0 | 0 | 10 | 20 | RULA: 0--10° = +1, transition to +2 at >10° [3, Table A] |
| `moderate` | 15 | 25 | 35 | 45 | RULA: 10--20° = +2, 20--60° = +3 |
| `high` | 35 | 45 | 60 | 80 | RULA: >60° = +4 |
| `extreme` | 60 | 80 | 180 | 180 | Saturation beyond 80°; maximum loading |

**Rationale**: RULA trunk scoring uses 0°/10°/20°/60° as boundaries. Our neutral-to-moderate transition (10--20° overlap) covers RULA scores 1→2. The moderate-to-high transition (35--45°) covers RULA scores 2→3. The extreme region (≥60°) corresponds to RULA score 4.

#### Neck Angle (degrees from vertical)

| Linguistic | a | b | c | d | Source |
|---|---|---|---|---|---|
| `neutral` | 0 | 0 | 5 | 10 | RULA: 0--10° = +1 [3, Table A] |
| `moderate` | 8 | 15 | 25 | 35 | RULA: 10--20° = +2, 20--60° = +3 |
| `high` | 25 | 35 | 50 | 70 | Upper boundary of moderate—high transition |
| `extreme` | 50 | 70 | 180 | 180 | RULA: >60° = maximum neck score |

**Rationale**: RULA neck scoring uses 0°/10°/20°/60°. RULA assigns +2 for >10° and +3 for >20°. Our moderate region (8--35°) spans RULA scores 2--3. The extreme region (≥50°) is conservative compared to RULA's 60° boundary because neck flexion beyond 50° substantially increases cervical spine load [40].

#### Upper Arm Angle (degrees from vertical)

| Linguistic | a | b | c | d | Source |
|---|---|---|---|---|---|
| `neutral` | 0 | 0 | 10 | 20 | RULA: 0--20° = +1 [3, Figure 2] |
| `moderate` | 15 | 30 | 45 | 60 | RULA: 20--45° = +2, 45--90° = +3 |
| `high` | 45 | 60 | 80 | 100 | RULA: 45--90° = +3 |
| `extreme` | 80 | 100 | 180 | 180 | RULA: >90° = +4 |

**Rationale**: RULA upper arm scores use thresholds at 20°, 45°, and 90°. These map directly: 0--20° neutral, 20--45° moderate, 45--90° high, >90° extreme. The 15--30° overlap for moderate and 45--60° for high create smooth transitions between the discrete RULA bins.

#### Forearm Deviation (degrees from 90° elbow angle)

| Linguistic | a | b | c | d | Source |
|---|---|---|---|---|---|
| `neutral` | 0 | 0 | 15 | 30 | RULA: 60--100° = +1 (recast as deviation = 0--40°) |
| `moderate` | 20 | 40 | 60 | 80 | RULA: <60° or >100° = +2 (deviation > 40°) |
| `high` | 60 | 80 | 100 | 120 | Extended range for severe deviation |
| `extreme` | 100 | 120 | 180 | 180 | Near-fully extended or fully flexed elbow |

**Rationale**: RULA's forearm scoring is symmetric: scores 1 for 60--100° and 2 for <60° or >100°. Our deviation metric requires coordinate system conversion: ideal 90° = deviation 0°, 60° or 120° = deviation 30°, etc. The neutral zone (deviation 0--30°) covers RULA score 1. The moderate zone (deviation 20--80°) covers score 2, with a gradual transition reflecting that moderate deviations (e.g., 40° from ideal) are biomechanically manageable.

#### Knee Bend (degrees from straight = 180° -- interior angle)

| Linguistic | a | b | c | d | Source |
|---|---|---|---|---|---|
| `neutral` | 0 | 0 | 10 | 20 | OWAS: straight legs = score 1 [5] |
| `moderate` | 15 | 30 | 45 | 60 | OWAS: bent knees = score 2 |
| `high` | 45 | 60 | 90 | 120 | OWAS: squatting = score 3 |
| `extreme` | 90 | 120 | 180 | 180 | Deep squat; maximum knee load |

**Rationale**: Knee scoring follows OWAS [5] rather than RULA, as RULA leg scoring is minimal (1--2). OWAS provides three levels for leg postures. The neutral zone (0--20° bend) covers standing and slight flexion. The moderate zone covers standing with bent knees. The high-to-extreme zones cover deep squats and kneeling.

#### Shoulder Asymmetry (%, shoulder height difference / torso length)

| Linguistic | a | b | c | d | Source |
|---|---|---|---|---|---|
| `neutral` | 0 | 0 | 5 | 10 | Within normal anatomical variation |
| `moderate` | 8 | 15 | 25 | 35 | Observable asymmetry; REBA score 1 |
| `high` | 25 | 35 | 50 | ∞ | Significant lateral bending |

**Source**: No direct RULA analog. Calibrated based on REBA trunk scoring [4] and clinical guidelines for scoliosis screening [48]. Asymmetry > 10% of torso length indicates clinically meaningful lateral deviation.

#### Body Inclination (%, lateral displacement / vertical span)

| Linguistic | a | b | c | d | Source |
|---|---|---|---|---|---|
| `neutral` | 0 | 0 | 5 | 10 | Normal postural sway |
| `moderate` | 8 | 15 | 25 | 35 | Observable lateral lean |
| `high` | 25 | 35 | 50 | ∞ | Sustained lateral bending |

**Source**: Same as shoulder asymmetry — project-specific calibration.

### 13.3 Explicit Disclaimer

The membership functions for shoulder asymmetry and body inclination are **project-specific calibrations** not derived from any single ergonomic standard. They have been designed to approximate the REBA lateral trunk scoring [4] but have not been independently validated. This is explicitly flagged in all system documentation.

---

## 14. Fuzzy Rule Base

### 14.1 Rule Design

The rule base consists of 24 rules organised hierarchically. All rules are of Mamdani type with AND-connected antecedents.

#### Critical Level (→ very_high)

Rules that independently trigger VERY HIGH risk — a single body region at extreme deviation:

```
IF trunk IS extreme    → very_high
IF neck IS extreme     → very_high
IF upper_arm IS extreme → very_high
```

**Rationale**: In RULA, a trunk score of 4 (maximum, corresponding to >60° flexion) or an upper arm score of 4 (>90° elevation) produces a Grand Score of at least 5--6 (Action Level 3) even with all other segments at minimum [3, Scoring Tables]. A single extreme region is sufficient to indicate critical risk.

#### High Level (→ very_high, from combinations)

```
IF trunk IS high AND upper_arm IS moderate      → very_high
IF neck IS high AND upper_arm IS moderate        → very_high
IF trunk IS high AND neck IS moderate            → very_high
IF neck IS high AND trunk IS moderate            → very_high
IF trunk IS moderate AND upper_arm IS high       → very_high
IF neck IS moderate AND upper_arm IS high        → very_high
IF upper_arm IS high AND forearm IS high          → very_high
```

**Rationale**: These rules capture **biomechanical interaction** — situations where no single feature is extreme, but combined moderate-high deviations in multiple primary regions create severe whole-body loading. For example, a worker with trunk at 50° (high) and arms at 50° elevation (moderate) carries load through both the lumbar spine and shoulders, producing a risk higher than either in isolation. This follows RULA's additive logic in Group A + Group B scoring [3, Table C].

#### Medium-High Level (→ high)

```
IF trunk IS high                → high
IF neck IS high                 → high
IF upper_arm IS high            → high
IF trunk IS moderate AND neck IS moderate    → high
IF trunk IS moderate AND upper_arm IS moderate → high
IF neck IS moderate AND upper_arm IS moderate → high
```

**Rationale**: A single high primary region produces HIGH risk (in RULA, a score of 3 on any segment typically drives Grand Score ≥ 5). Combined moderate deviations also produce HIGH due to ergonomic interaction.

#### Medium Level (→ medium)

```
IF trunk IS moderate                                               → medium
IF neck IS moderate                                                → medium
IF upper_arm IS moderate                                           → medium
IF trunk IS moderate AND forearm IS high                            → medium
IF neck IS moderate AND forearm IS high                             → medium
IF knee IS extreme                                                  → medium
IF knee IS high AND trunk IS moderate                               → medium
```

**Rationale**: Moderate deviations in primary features correspond to RULA scores of 2--3 and Action Level 2 (further investigation needed). Secondary features (forearm high, knee extreme) can elevate risk from LOW to MEDIUM when a primary feature is already moderate.

#### Low Level (→ low)

```
IF trunk IS neutral AND neck IS neutral → low
```

**Rationale**: When both primary centroids (trunk and neck) are in the neutral range, the overall risk is LOW regardless of upper arm or secondary features. This captures the baseline standing posture.

### 14.2 Default Behaviour

If no rule fires (e.g., trunk is moderate but all other features unavailable), the system returns 0 (minimum risk) rather than falling back to a default medium score. This is a conscious design choice: when insufficient evidence exists to fire any rule, the system should be conservative in its risk estimate.

### 14.3 Rule Completeness

The rule base covers:
- All combinations of trunk (neutral, moderate, high, extreme)
- All combinations of neck (neutral, moderate, high, extreme)
- All combinations of upper arm (neutral, moderate, high, extreme)
- Secondary features as conditional modifiers

**Topological coverage**: The Cartesian product of all feature linguistic variables contains 4 × 4 × 4 × 4 × 4 = 1024 possible input combinations. Our 24 rules cover the ergonomically meaningful regions. Uncovered regions (e.g., trunk and neck both neutral while upper arm is moderate) are handled by the LOW rule structure — moderate upper arm without trunk or neck involvement produces LOW, which is consistent with RULA (isolated arm deviation without torso involvement produces lower risk).

### 14.4 Output Risk Centroid Mapping

| FIS Output | Risk Score Range | Discrete Class |
|---|---|---|
| < 35 | [0, 35) | Low Risk |
| 35--65 | [35, 65) | Medium Risk |
| ≥ 65 | [65, 100] | High Risk |

**Threshold justification**:
- **35 boundary**: Approximately the midpoint of the `low` output centroid (25) and the `medium` centroid (48.8). This corresponds approximately to RULA Grand Score 3--4 boundary (first Action Level boundary).
- **65 boundary**: Approximately the midpoint of `medium` (48.8) and `high` (70). Corresponds approximately to RULA Grand Score 5--6 boundary (Action Level 3 threshold).

These boundaries are project-specific calibrations and are explicitly not RULA Action Level equivalents.

---

## 15. Temporal Smoothing

### 15.1 Motivation

Frame-by-frame pose estimation exhibits **jitter** — small, random fluctuations in keypoint positions due to:

- Detection noise (YOLO bounding box shifts by 1--3 px between frames)
- Pose ambiguity (multiple plausible keypoint positions)
- Motion blur (rapid movements during video sampling)

These fluctuations translate into frame-to-frame oscillations of 5--15° in computed joint angles [49]. Without temporal smoothing, the risk class can flip between LOW and MEDIUM on consecutive frames even when the worker's actual posture is stable.

### 15.2 Exponential Moving Average (EMA)

The system applies an Exponential Moving Average to the continuous severity score:

$$S_t = \alpha \cdot s_t + (1 - \alpha) \cdot S_{t-1}$$

where:
- $S_t$ = smoothed severity at frame $t$
- $s_t$ = raw (instantaneous) severity at frame $t$
- $\alpha \in [0, 1]$ = smoothing factor

The parameter $\alpha$ controls the smoothing strength:

| $\alpha$ | Effective window (frames) | Characteristic |
|---|---|---|
| 1.0 | 1 | No smoothing |
| 0.5 | ~4 | Moderate smoothing |
| 0.35 | ~6 | Default — balances responsiveness and stability |
| 0.2 | ~10 | Heavy smoothing |

The **effective window size** $n_{\text{eff}}$ of an EMA is approximately:

$$n_{\text{eff}} \approx \frac{2 - \alpha}{\alpha}$$

At $\alpha = 0.35$, $n_{\text{eff}} \approx 4.7$ frames. At 1 FPS sampling, this corresponds to ~5 seconds of temporal context.

**Default**: $\alpha = 0.35$, chosen to provide moderate smoothing without excessive lag. This follows the recommendation in [50] for human motion analysis, where values of 0.3--0.4 balance smoothness and responsiveness.

### 15.3 Implementation

The EMA is implemented as a stateful wrapper (`TemporalSmoothedScorer`) that maintains separate smoothing state per `tracking_key` (typically `video_id + frame_id + person_id`):

```python
class TemporalSmoothedScorer:
    def __init__(self, alpha=0.35, severity_low_max=35, severity_medium_max=65):
        self._state = {}  # tracking_key → (ema, prev_class, persistence)
```

The smoothed score is used for risk classification; the raw score is preserved as `raw_severity` for analysis.

### 15.4 Smoothing-Induced Lag

EMA smoothing introduces a systematic lag:

$$\text{lag}_{\text{max}} = \frac{1 - \alpha}{\alpha} \quad \text{(frames)}$$

At $\alpha = 0.35$, the lag is approximately 1.9 frames. A genuine posture change (e.g., worker bending from 0° to 60° trunk) takes ~4 frames to reach 90% of the new equilibrium. At 1 FPS sampling, this is a 4-second delay — acceptable for offline screening, less so for real-time feedback.

**Trade-off accepted**: Smoothing lag is tolerated because the system targets historical screening rather than real-time alerting. Low-latency applications would require higher $\alpha$ values.

---

## 16. Risk Persistence

### 16.1 Concept

Risk persistence tracks the number of consecutive frames a worker maintains the same risk class. This provides biomechanical context: a HIGH risk posture held for 30 seconds is ergonomically more significant than the same posture held for 2 seconds.

### 16.2 Persistence Score

The persistence counter increments each frame that the smoothed severity remains in the same risk band:

$$p_t = \begin{cases}
p_{t-1} + 1 & \text{if class}(S_t) = \text{class}(S_{t-1}) \\
1 & \text{otherwise}
\end{cases}$$

Persistence is reported in the output as `persistence_frames` and included in the explanation text:

```
[temporal: raw 72 → smoothed 68, persistence 12 frame(s)]
```

### 16.3 Bibliographic Rationale

The RULA assessment manual [3, Section 2.3] notes that "posture scores are intended to be used for postures maintained for more than 1 minute." By tracking persistence, ErgoVision provides a basis for filtering brief, non-sustained high-risk classifications — though the current implementation does not use persistence to modify the risk class.

---

## 17. Final Risk Classification

### 17.1 Risk Classes

| Class | Severity Range | Interpretation | Suggested Action |
|---|---|---|---|
| **Low Risk** | [0, 35) | Neutral postural alignment. No significant deviation from anatomical neutral across observed features. | None required. |
| **Medium Risk** | [35, 65) | Moderate postural deviations observed. Further investigation recommended to determine if these postures are sustained or frequent. | Monitor; consider workstation review if frequent. |
| **High Risk** | [65, 100] | Sustained severe postural load. Multiple features at high deviation or a single feature at extreme deviation. | Intervention recommended. Ergonomic review and workstation redesign should be considered. |

### 17.2 Class Boundaries: Derivation

The class boundaries (35 and 65) are derived from the fuzzy output centroids:

- The `low` centroid is 25.0 (midpoint of 10-20-30-40 trapezoid). Adding one standard deviation (~10 points) gives 35, which marks the transition from predominantly LOW to predominantly MEDIUM.
- The `very_high` centroid is ~90.0. Subtracting one standard deviation gives ~65, which marks the transition from HIGH to VERY_HIGH.

These boundaries are **project-specific** and are not equivalent to RULA Action Level thresholds. Users should NOT map directly:

> ❌ ErgoVision "Medium Risk" ≠ RULA Action Level 2
> ✓ ErgoVision "Medium Risk" ~ "Moderate postural deviation warranting further investigation"

### 17.3 Confidence Integration

Every risk class is reported with a **confidence score** $C \in [0, 1]$ (see Section 9):

- $C > 0.6$: High confidence. Risk class is reliable.
- $0.35 < C \leq 0.6$: Moderate confidence. Risk class should be treated as tentative.
- $0.25 < C \leq 0.35$: Low confidence. Risk estimate is approximate.
- $C \leq 0.25$: Very low confidence. Risk class is a best-effort estimate. Flagged as `uncertain = True` but still reported.

The system **never** hides the risk estimate. Even at $C = 0.1$, the risk score and class are reported. This is a deliberate design choice following the principle that **no information is better than discarded information** with a caveat.

---

## 18. Confidence-Aware Scoring

### 18.1 Motivation

Prior systems [19,20,24] use a binary uncertainty gate:

```
if n_valid_keypoints < threshold:
    return "UNCERTAIN"
else:
    return score
```

This has a severe practical limitation: in real industrial environments, partially occluded workers at the edge of the camera frame frequently fall below the keypoint threshold. The result is that most frames become "UNCERTAIN" — producing a system that defaults to unusable on real data.

**Example**: In the CarDA industrial dataset, with the original binary threshold (8 keypoints minimum, confidence 0.25), approximately 65--80% of frames were classified UNCERTAIN due to partial occlusions, workers at the edge of the frame, and motion blur during active tasks.

### 18.2 Continuous Confidence Model

ErgoVision replaces binary uncertainty with the continuous confidence model described in Section 9. The confidence score $C$ is:

$$C = \sqrt{
    \underbrace{\sigma(n_{\text{valid}} - 8)}_{\text{coverage}}
    \times
    \underbrace{\min(\bar{c} / 0.50,\ 1.0)}_{\text{quality}}
}$$

This propagates throughout the pipeline:

| $n_{\text{valid}}$ | $\bar{c}$ | $C$ | Risk Output | Prior System |
|---|---|---|---|---|
| 15 | 0.52 | 0.97 | Low Risk (conf: 0.97) | Scored |
| 8 | 0.35 | 0.49 | Medium Risk (conf: 0.49) | UNCERTAIN |
| 6 | 0.30 | 0.30 | Low Risk (conf: 0.30) | UNCERTAIN |
| 4 | 0.20 | 0.09 | Low Risk (conf: 0.09) — flagged | UNCERTAIN |

The system outputs approximately 0% UNCERTAIN on typical industrial video (compared to 65--80% with the binary gate), while providing continuous confidence information to the user.

### 18.3 Reliability of Low-Confidence Estimates

When $C$ is low, the risk estimate is less reliable. We recommend the following interpretation:

- **$C > 0.6$**: Use risk class as-is
- **$0.35 < C \leq 0.6$**: Consider the confidence interval $\pm 15$ around the severity score
- **$C \leq 0.35$**: Consider the estimate directional only (indicating whether posture is generally benign or concerning, but not the exact severity)

---

## 19. Explainability Layer

### 19.1 Risk Drivers

The system identifies **risk drivers** — features whose continuous severity exceeds 55/100 (approximately the midpoint of MEDIUM severity):

```python
primary_drivers:   [feature names exceeding threshold, primary category]
secondary_drivers: [feature names exceeding threshold, secondary category]
```

These drivers are listed in the explanation string:

```
"HIGH postural risk (severity 85/100) — sustained severe postural load. 
 Primary drivers: trunk_angle, neck_angle."
```

### 19.2 Rule Firing Output

The Mamdani FIS exposes its internal state via `rule_firings` — a dictionary mapping each output linguistic variable to its aggregated firing strength:

```python
rule_firings = {
    'low':  0.0,      # Did not fire
    'medium': 0.32,   # Fired weakly
    'high': 1.0,      # Fired with full strength
    'very_high': 0.75 # Fired strongly
}
```

This provides full transparency into which rules contributed to the final score and by how much.

### 19.3 Per-Feature Severity

Each feature reports an individual continuous severity (0--100) based on the maximum non-neutral membership value:

$$s_{\text{feature}} = 100 \times \max_{\ell \neq \text{neutral}} \mu_\ell(x_{\text{feature}})$$

This produces a 0--100 "how bad is this feature" score that is independent of the overall risk score. A feature with severity 80/100 is severely deviated even if the overall risk is MEDIUM (because other features are neutral).

### 19.4 Explanation String

The system generates a structured natural-language explanation:

```
<RISK CLASS> postural risk (severity <N>/100) — <interpretation>.
[Primary drivers: <features>.] [Secondary indicators: <features>.]
[Reduced confidence: <C>.] [Manual context: <factors>.]
```

---

## 20. Experimental Protocol

### 20.1 Dataset: CarDA (Construction-Related Dataset)

The primary validation dataset is **CarDA** (Construction-related Dataset for ergonomic Assessment), an industrial dataset containing RGB video footage from manufacturing and logistics environments. Key characteristics:

- Multi-view: cameras at various angles capturing industrial tasks
- Content: assembly, material handling, inspection, maintenance operations
- Workers: partially occluded, moving through frame, varying distances from camera
- Environment: factory floor lighting, dynamic backgrounds
- Resolution: 720p--1080p
- Frame rate: 25--30 FPS (original), subsampled to 1 FPS for processing
- Duration: 1--15 minutes per video

### 20.2 Experimental Configuration

Standard experimental parameters:

| Parameter | Value | Rationale |
|---|---|---|
| Frame sampling | 1 FPS | Adequate for postural screening of non-repetitive industrial tasks |
| Max frames per video | 100--200 | Limits per-video processing time and ensures dataset diversity |
| Detection model | YOLOv8l | Best accuracy/compute trade-off |
| Pose model | YOLOv8l-pose | Crop-based for small-person performance |
| Detection confidence | 0.5 | Standard YOLO threshold; balances precision and recall |
| Keypoint confidence | 0.3 | Default YOLO-pose threshold; per [13] |
| EMA alpha | 0.35 | Moderate smoothing; per Section 15 |
| Low risk threshold | 35 | Fuzzy centroid-based boundary |
| High risk threshold | 65 | Fuzzy centroid-based boundary |

### 20.3 Output Metrics

For each experiment, the system generates:

- **Frame-person results CSV**: one row per person per frame, containing all angle features, risk score, severity, and confidence
- **Video summary CSV**: aggregate statistics per video (risk distribution, mean angles, mean confidence)
- **Performance metrics**: processing time, effective FPS, detection counts, discard counts
- **Plots**:
  - Risk distribution bar plot (LOW / MEDIUM / HIGH)
  - Risk distribution by video (stacked bar)
  - People per frame distribution
  - Keypoint confidence histogram
  - Per-feature angle histograms (trunk, neck, upper arm, etc.)
  - Top risk frames
- **Annotated frames**: skeleton overlay + risk score displayed on frame
- **Annotated video**: concatenated annotated frames (if input is video)
- **Experimental report**: self-contained markdown document with all results, plots, and discussion

### 20.4 Statistical Analysis

The following statistics are computed per experiment:

| Metric | Formula | Purpose |
|---|---|---|
| Risk distribution | $\frac{n_{\text{class}}}{n_{\text{total}}} \times 100$ | Overall risk profile |
| Mean severity | $\frac{1}{n} \sum_{i=1}^n s_i$ | Central tendency of postural load |
| Mean confidence | $\frac{1}{n} \sum_{i=1}^n C_i$ | Overall pose quality |
| Angle statistics | Mean, std, min, P25, P50, P75, max per feature | Postural distribution characterisation |
| People per frame | $\frac{n_{\text{detections}}}{n_{\text{frames}}}$ | Worker density |
| Confidence distribution | Histogram of $C$ | Pose quality across dataset |

---

## 21. Limitations

### 21.1 Inherent to Vision-Based Postural Assessment

#### 2D Projection Error

All joint angles are computed in image coordinates. True 3D anatomical angles differ due to perspective projection [51]. The magnitude of this error depends on:

- Camera angle relative to the movement plane (optimal: side view; worst: frontal view for sagittal motions)
- Segment orientation out of the image plane (torsion is invisible)
- Camera distance and lens distortion

Studies report 2D-to-3D angle errors of 10--25° for trunk flexion and 15--30° for shoulder abduction [21]. This is a fundamental limitation of monocular 2D approaches.

#### Occlusion and Self-Occlusion

Workers are frequently partially occluded by equipment, machinery, other workers, or their own body. This results in missing or low-confidence keypoints. Even with the continuous confidence model, severely occluded poses produce unreliable angle estimates.

#### Camera Position Sensitivity

Murugan et al. [21] demonstrate that side-camera positioning minimises angle error for RULA assessment. However, industrial camera placement is constrained by physical infrastructure. Our system makes no assumption about camera position and works with the available view.

#### Lighting and Background Variability

Factory floor conditions include cast shadows, reflections, moving machinery, and changing lighting conditions. These affect YOLO detection confidence and keypoint quality. The system has been tested on the CarDA dataset with standard factory lighting conditions; performance in extreme conditions (low light, backlighting) is not characterised.

### 21.2 Specific to the Scoring Model

#### No Force or Load Estimation

As described in Section 5.1, the system does not estimate force, load, or manual handling effort. A worker carrying a 20 kg object and a worker making the same postural motion unloaded receive the same risk score. **This is the single most significant limitation** when comparing to clinical RULA.

#### No Wrist Assessment

Wrist flexion, deviation, and rotation are not scored. These are part of RULA Group A, and their exclusion means the system systematically underestimates risk for tasks requiring significant wrist motion (e.g., precision assembly, tool use).

#### No Action Levels

The system outputs a risk class (LOW / MEDIUM / HIGH), not RULA Action Levels. These are not interchangeable. Users should not map our risk classes to RULA Action Levels for any compliance or legal purpose.

#### Threshold Arbitrariness (Declared)

The class boundaries (35, 65) and the membership function overlaps are project-specific calibrations. While guided by RULA/REBA/OWAS thresholds, they have not been independently validated against clinical ground truth. This is a research prototype, not a certified assessment tool.

#### No Expert Validation

The risk classifications have not been validated against ground-truth ergonomic assessments by certified professionals. Agreement with clinical RULA is unknown and likely poor for tasks with significant force or wrist involvement.

### 21.3 Dataset Limitations

- Validation is limited to the CarDA dataset. Generalisation to other industries (healthcare, logistics, agriculture) is not established.
- The dataset may not represent the full range of industrial working postures (e.g., overhead work, prone/crawling).
- Ground-truth RULA scores are not available for the dataset, preventing quantitative accuracy assessment.

### 21.4 Single-View Limitation

A single camera cannot capture all relevant postural information, especially for asymmetric tasks (e.g., reaching behind the body) or tasks involving rotation. Multi-view setups would improve anatomical coverage but add cost and complexity.

---

## 22. Future Work

### 22.1 3D Pose Estimation

The most impactful single upgrade would be replacing 2D pose with 3D lifting using **VideoPose3D** [16] or **MotionBERT** [26]. This would:

- Eliminate 2D projection error (Section 21.1)
- Enable trunk torsion estimation
- Improve upper arm angle accuracy
- Reduce camera position sensitivity

Expected accuracy gain: 10--15% reduction in angle MAE [21].

### 22.2 Temporal Transformers

Replace per-frame pose estimation with temporal models that encode motion context:

- **MotionBERT**: 2D→3D lifting with self-attention, inherently smooths jitter
- **VideoPose3D**: temporal dilated convolutions, proven for RULA [19]

This would reduce the need for explicit EMA smoothing and improve temporal consistency.

### 22.3 Load Estimation via Object Detection

Extend the system to detect and track grasped objects, estimating load from:

- Object detection (what is being held)
- Object size→weight heuristics (approximate, category-level)
- Human-object interaction classification (lifting, carrying, pushing)

This is a hard problem; even approximate load estimation would significantly improve clinical relevance.

### 22.4 Action Recognition for Repetition

Add temporal action recognition to detect:

- Repetitive motions (count repetitions per minute)
- Sustained postures (duration in single posture)
- Recovery periods (rest breaks between exertion)

This would enable muscle use scoring and repetition analysis, moving toward semi-automatic RULA.

### 22.5 Semi-Automatic RULA

The ultimate goal is a semi-automatic system where:

- **Automatic**: posture estimation, angle computation, temporal tracking
- **Manual**: force/load input, wrist assessment, expert override

This would combine the strengths of vision-based screening with expert ergonomist judgment, providing a practical tool that is more complete than automated screening alone and more scalable than fully manual assessment.

### 22.6 Multi-View Fusion

Place multiple cameras covering the workstation from different angles, with:

- Camera calibration for spatial correspondence
- View selection (best pose per body segment)
- 3D keypoint triangulation

This would resolve occlusion and 2D projection issues, but at significantly increased infrastructure cost.

---

## 23. References

### Primary References (Ergonomic Methods)

[1] European Agency for Safety and Health at Work. (2019). Work-related musculoskeletal disorders: prevalence, costs and demographics in the EU. *ESENER Report*.

[2] Liberty Mutual Research Institute for Safety. (2021). 2021 Liberty Mutual Workplace Safety Index.

[3] McAtamney, L., & Corlett, E. N. (1993). RULA: a survey method for the investigation of work-related upper limb disorders. *Applied Ergonomics*, 24(2), 91--99. https://doi.org/10.1016/0003-6870(93)90080-S

[4] Hignett, S., & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA). *Applied Ergonomics*, 31(2), 201--205. https://doi.org/10.1016/S0003-6870(99)00039-3

[5] Karhu, O., Kansi, P., & Kuorinka, I. (1977). Correcting working postures in industry: A practical method for analysis. *Applied Ergonomics*, 8(4), 199--201. https://doi.org/10.1016/0003-6870(77)90164-8

[6] Schaub, K., et al. (2013). The European Assembly Worksheet. *Theoretical Issues in Ergonomics Science*, 14(6), 616--639. https://doi.org/10.1080/1463922X.2012.678283

[7] Takala, E. P., et al. (2010). A systematic review of observational methods assessing muscular activity, posture, and upper extremity disorders. *Ergonomics*, 53(2), 201--216. https://doi.org/10.1080/00140130903341944

[8] David, G. C. (2005). Ergonomic methods for assessing exposure to risk factors for work-related musculoskeletal disorders. *Occupational Medicine*, 55(3), 190--199. https://doi.org/10.1093/occmed/kqi082

[9] Barrero, L. H., et al. (2009). Auto-correlation structure of job exposures. *Scandinavian Journal of Work, Environment & Health*, 35(1), 70--80.

[10] Goggins, R. W., et al. (2008). Estimating the effectiveness of ergonomics interventions through case studies. *Applied Ergonomics*, 39(2), 138--147.

[11] Cao, Z., et al. (2017). Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. *CVPR 2017*. https://doi.org/10.1109/CVPR.2017.143

[12] Zheng, C., et al. (2023). Deep learning-based human pose estimation: A survey. *ACM Computing Surveys*, 56(1), 1--37. https://doi.org/10.1145/3603621

[13] Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLOv8. *Ultralytics*. https://github.com/ultralytics/ultralytics

[14] Cao, Z., et al. (2021). OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. *IEEE TPAMI*, 43(1), 172--186. https://doi.org/10.1109/TPAMI.2019.2929257

[15] Bazarevsky, V., et al. (2020). BlazePose: On-device Real-time Body Pose Tracking. *arXiv preprint*: 2006.10204.

[16] Pavllo, D., et al. (2019). 3D human pose estimation in video with temporal convolutions and semi-supervised training. *CVPR 2019*. https://doi.org/10.1109/CVPR.2019.00826

[17] European Commission. (2021). Industry 5.0: Towards a sustainable, human-centric and resilient European industry.

### Vision-Based Ergonomic Assessment (2024--2025)

[18] Li, Z., Yu, Y., Xia, J., Chen, X., Lu, X., & Li, Q. (2024). Data-driven ergonomic assessment of construction workers. *Automation in Construction*, 165, 105561. https://doi.org/10.1016/j.autcon.2024.105561

[19] Menanno, M., Riccio, C., Benedetto, V., Gissi, F., Savino, M. M., & Troiano, L. (2024). An Ergonomic Risk Assessment System Based on 3D Human Pose Estimation and Collaborative Robot. *Applied Sciences*, 14(11), 4823. https://doi.org/10.3390/app14114823

[20] Agostinelli, S., Generosi, A., et al. (2024). Validation of computer vision-based ergonomic risk assessment tools for real manufacturing environments. *Scientific Reports*, 14, 27785. https://doi.org/10.1038/s41598-024-79373-4

[21] Murugan, A. S., et al. (2024). Optimising computer vision-based ergonomic assessments: sensitivity to camera position and monocular 3D pose model. *Ergonomics*. https://doi.org/10.1080/00140139.2024.2310005

[22] Cruciata, L., Contino, V., Ciccarelli, M., Pirrone, R., Mostarda, L., Papetti, A., & Piangerelli, M. (2025). Lightweight Vision Transformer for Frame-Level Ergonomic Posture Classification in Industrial Workflows. *Sensors*, 25(15), 4750.

[23] González-Alonso, M., Antón-Rodríguez, M., Martínez-Zarzuela, M., et al. (2025). ME-WARD: A multimodal ergonomic analysis tool for musculoskeletal risk assessment from inertial and video data in working places. *Expert Systems with Applications*. https://doi.org/10.1016/j.eswa.2025.126044

[24] Agostinelli, S., Generosi, A., et al. (2024). tf-pose estimation for RULA assessment in real manufacturing environments. *In: Proceedings of HCI International 2024*.

[25] Zhou, X., Wang, Y., & Hu, J. (2025). 3D ergonomics parameter measurement using video-based deep learning. *Measurement*, 256, 118034. https://doi.org/10.1016/j.measurement.2025.118034

[26] Song, Z., Wang, Z., Wang, J., Zeng, Y., & Li, X. (2025). Occlusion-aware and jitter-rejection 3D video real-time pose estimation for construction workers. *Automation in Construction*, 172, 105973. https://doi.org/10.1016/j.autcon.2025.105973

[27] Yang, Z., Song, D., Ning, J., & Wu, Z. (2024). A Systematic Review: Advancing Ergonomic Posture Risk Assessment Through the Integration of Computer Vision and Machine Learning Techniques. *IEEE Access*, 12, 180481--180519. https://doi.org/10.1109/ACCESS.2024.3509447

[28] Murugan, A. S., et al. (2023). Vision-based ergonomic risk assessment using BlazePose and RULA. *In: Proceedings of the 23rd International Conference on Engineering and Product Design Education*.

### Biomechanics and Physiology

[29] Waters, T. R., Putz-Anderson, V., Garg, A., & Fine, L. J. (1993). Revised NIOSH equation for the design and evaluation of manual lifting tasks. *Ergonomics*, 36(7), 749--776.

[30] Pousette, S., et al. (2021). The predictive validity of RULA for work-related musculoskeletal disorders: A systematic review. *Applied Ergonomics*, 92, 103343.

[31] Lueder, R. (1996). A proposed RULA-based methodology for assessing the risk of occupational ergonomic injuries. *Proceedings of the Human Factors and Ergonomics Society Annual Meeting*, 40(12), 649--653.

[32] Li, G., & Buckle, P. (1999). Current techniques for assessing physical exposure to work-related musculoskeletal risks, with emphasis on posture-based methods. *Ergonomics*, 42(5), 674--695.

[33] Marras, W. S. (2000). Occupational low back disorder causation and control. *Ergonomics*, 43(7), 880--902.

[34] Keyserling, W. M. (1986). Postural analysis of the trunk and shoulders in simulated real time. *Ergonomics*, 29(4), 569--583.

[35] Occhipinti, E. (1998). OCRA: a concise index for the assessment of exposure to repetitive movements of the upper limbs. *Ergonomics*, 41(9), 1290--1311.

### Computer Vision Methods

[36] Fang, H. S., et al. (2017). RMPE: Regional Multi-person Pose Estimation. *ICCV 2017*. https://doi.org/10.1109/ICCV.2017.256

[37] Lin, T. Y., et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014*. https://doi.org/10.1007/978-3-319-10602-1_48

[38] Perez, D., et al. (2018). COCO keypoint evaluation metrics. *Journal of Open Source Software*.

[39] Nachemson, A. L. (1981). Disc pressure measurements. *Spine*, 6(1), 93--97.

[40] Harms-Ringdahl, K., et al. (1986). Load moments in the cervical spine during seated postures. *Journal of Biomechanics*, 19(8), 647--657.

[41] Poppen, N. K., & Walker, P. S. (1978). Forces at the glenohumeral joint in abduction. *Journal of Biomechanics*, 11(4), 175--182.

[42] Gowitzke, B. A., & Milner, M. (1988). *Scientific Bases of Human Movement*. Williams & Wilkins.

[43] Lieber, R. L. (2002). *Skeletal Muscle Structure, Function, and Plasticity*. Lippincott Williams & Wilkins.

[44] Reilly, D. T., & Martens, M. (1972). Experimental analysis of the quadriceps muscle force and patello-femoral joint contact force. *Journal of Biomechanics*, 5(5), 495--505.

[45] Li, G., et al. (1997). Risk factors for low back pain in Chinese factory workers. *Journal of Occupational Health*, 39(2), 134--140.

### Fuzzy Logic

[46] Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies*, 7(1), 1--13. https://doi.org/10.1016/S0020-7373(75)80002-2

[47] Pedrycz, W., & Gomide, F. (2007). *Fuzzy Systems Engineering: Toward Human-Centric Computing*. Wiley.

[48] Bunnell, W. P. (1984). An objective criterion for scoliosis screening. *Journal of Bone and Joint Surgery*, 66(9), 1381--1387.

[49] Kocabas, M., et al. (2020). VIBE: Video Inference for Human Body Pose and Shape Estimation. *CVPR 2020*.

[50] Robertson, D. G. E., et al. (2013). *Research Methods in Biomechanics*. Human Kinetics.

### Additional Technical References

[51] Slyszko, R. A., et al. (2018). The effect of camera perspective on the accuracy of joint angle measurement in 2D video analysis. *Journal of Biomechanics*, 68, 37--43.

---

## Appendix A: Formula Summary

| Section | Formula | Description |
|---|---|---|
| 10.1 | $\theta(p_1,p_2,p_3) = \arccos\left(\frac{(p_1-p_2)\cdot(p_3-p_2)}{\|p_1-p_2\|\|p_3-p_2\|}\right)$ | Three-point angle (radians) |
| 10.1 | $\phi(p_u,p_l) = \arccos\left(\frac{(p_u-p_l)\cdot(0,-1)}{\|p_u-p_l\|}\right)$ | Angle from vertical (degrees) |
| 10.2 | $\theta_{\text{trunk}} = \phi(S_m, H_m)$ | Trunk angle |
| 10.2 | $\theta_{\text{neck}} = \phi(N, S_m)$ | Neck angle |
| 10.2 | $\theta_{\text{UA}} = \phi(S, E)$ | Upper arm angle |
| 10.2 | $\theta_{\text{FA}} = \theta(S,E,W),\ \text{dev}=|\theta_{\text{FA}}-90\degree|$ | Forearm deviation |
| 10.2 | $\text{bend} = 180\degree - \theta(H,K,A)$ | Knee bend |
| 10.2 | $\text{asym} = \frac{|S_l^y - S_r^y|}{\|S_m - H_m\|} \times 100$ | Shoulder asymmetry (%) |
| 10.2 | $\text{incl} = \frac{|S_m^x - H_m^x|}{|S_m^y - H_m^y|} \times 100$ | Body inclination (%) |
| 9.1 | $C = \sqrt{\sigma(n_{\text{valid}}-8) \cdot \min(\bar{c}/0.5, 1)}$ | Pose confidence |
| 11.2 | $\mu(x) = \text{trapezoid}(x; a,b,c,d)$ | Fuzzy membership |
| 11.2 | $y^* = \frac{\sum \alpha_k \cdot c_k}{\sum \alpha_k}$ | Defuzzification (weighted avg.) |
| 15.2 | $S_t = \alpha \cdot s_t + (1-\alpha) \cdot S_{t-1}$ | EMA temporal smoothing |
| App | $n_{\text{eff}} \approx (2-\alpha)/\alpha$ | Effective EMA window |
| 16.2 | $p_t = p_{t-1} + 1$ if same class else 1 | Persistence counter |

## Appendix B: Threshold Quick Reference

| Feature | Neutral (a,b,c,d) | Moderate (a,b,c,d) | High (a,b,c,d) | Extreme (a,b,c,d) |
|---|---|---|---|---|
| Trunk | (0,0,10,20) | (15,25,35,45) | (35,45,60,80) | (60,80,180,180) |
| Neck | (0,0,5,10) | (8,15,25,35) | (25,35,50,70) | (50,70,180,180) |
| Upper arm | (0,0,10,20) | (15,30,45,60) | (45,60,80,100) | (80,100,180,180) |
| Forearm dev. | (0,0,15,30) | (20,40,60,80) | (60,80,100,120) | (100,120,180,180) |
| Knee bend | (0,0,10,20) | (15,30,45,60) | (45,60,90,120) | (90,120,180,180) |
| Asymmetry | (0,0,5,10) | (8,15,25,35) | (25,35,50,∞) | — |
| Inclination | (0,0,5,10) | (8,15,25,35) | (25,35,50,∞) | — |

## Appendix C: Warning and Disclaimer

**This software is a research prototype. It is not a certified medical or ergonomic assessment device.**

- Risk scores should not be used as the sole basis for workplace safety decisions.
- The system does not implement clinical RULA and does not claim RULA compliance.
- Ergonomic interventions should be conducted by or in consultation with certified ergonomics professionals.
- The authors assume no liability for decisions made based on system outputs.
