# ErgoVision

**Vision-based ergonomic risk assessment for Human-Centric Industry 5.0 environments.**

ErgoVision implements a RULA-inspired lightweight ergonomic risk estimation pipeline using pretrained human pose estimation (YOLOv8-pose). It does **not** train or fine-tune any model, and it does **not** claim full clinical RULA compliance or medical/occupational certification.

## Pipeline

```
Image or video frame
  → YOLOv8-pose (pretrained, no training)
  → COCO keypoints (17 keypoints per person)
  → RULA-inspired rule-based scoring
    → torso angle, neck angle, knee angle,
      shoulder asymmetry, body inclination
  → partial scores (1 / 2 / 3)
  → final score = max of available partial scores
  → risk class: Low / Medium / High
  → textual explanation
```

## Project Structure

```
ErgoVision/
├── data/                              # <-- Put your datasets here
│   ├── posture-keypoints-detection/   #     Kaggle images dataset
│   │   ├── images/
│   │   ├── annotations/
│   │   └── ...
│   └── assembly101/                   #     Assembly101 video files
│       ├── P01_mit_12_view_1.mp4
│       └── ...
├── ergovision/
│   ├── __init__.py                  # Package exports
│   ├── config.py                    # Thresholds, paths, constants
│   ├── dataset.py                   # Image discovery (os.walk-based)
│   ├── pose_estimator.py            # YOLOv8-pose inference wrapper
│   ├── ergonomic_scoring.py         # RULA-inspired rule-based scorer
│   ├── visualization.py             # Keypoint skeleton + risk overlay
│   ├── pipeline.py                  # End-to-end pipeline orchestrator
│   ├── data/
│   │   ├── assembly101_loader.py    # Assembly101 video discovery + metadata
│   │   └── video_frame_extractor.py # Frame extraction + quality filtering
│   ├── experiments/
│   │   ├── experiment_01_posture_dataset.py        # Baseline posture images
│   │   ├── experiment_02_assembly101_robustness.py # Assembly101 robustness
│   │   └── experiment_03_qualitative_explainability.py  # Qualitative samples
│   └── evaluation/
│       ├── robustness_metrics.py    # Detection rates, feature availability, etc.
│       └── experimental_reports.py  # Report files, figures, paper summary
│
├── ergovision_kaggle.ipynb          # Kaggle notebook (8 sections)
└── outputs/                         # Results directory
```

## Setup

### Requirements

- Python 3.9+
- `pip install -r requirements.txt`

---

## Dataset 1: Posture Keypoints Detection (Kaggle)

Usato per: **Experiment 1** (baseline posture dataset).

### Download

1. Vai su https://www.kaggle.com/datasets/melsmm/posture-keypoints-detection
2. Clicca **Download** (serve login Kaggle)
3. Estrai la cartella scaricata

### Dove metterlo

```
C:\Users\MarioBove\Desktop\ErgoVision\data\posture-keypoints-detection\
  ├── images\
  ├── annotations\
  └── ...
```

Il codice usa `os.walk` — non importa la struttura esatta delle sottocartelle, purché ci siano file `.jpg/.png`.

### Verifica

```python
from pathlib import Path
assert Path('data/posture-keypoints-detection').exists()
print("OK — dataset trovato")
```

---

## Dataset 2: Assembly101

Usato per: **Experiment 2** (robustness) ed **Experiment 3** (qualitative examples).

Assembly101 è un dataset multi-vista di attività procedurali (assemblaggio e disassemblaggio di oggetti).

- **Sito ufficiale**: https://assembly-101.github.io/
- **Script di download**: https://github.com/Assembly101-2022/assembly101-download
- **Citazione**: Sener et al., "Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities", CVPR 2022

### Opzione A — Download via script ufficiale (consigliata)

```bash
git clone https://github.com/Assembly101-2022/assembly101-download
cd assembly101-download
# Segui le istruzioni nel repository per scaricare i video
```

Poi copia i video scaricati in:

```
C:\Users\MarioBove\Desktop\ErgoVision\data\assembly101\
  ├── P01_mit_12_view_1.mp4
  ├── P01_mit_12_view_2.mp4
  ├── P02_...
  └── ...
```

### Opzione B — File già scaricati

Se hai già i file `.mp4` / `.avi`, copiali direttamente in `data/assembly101/`. Vanno bene anche sottocartelle — il loader usa `os.walk`.

### Opzione C — Kaggle

Carica il dataset Assembly101 su Kaggle e imposta `ASSEMBLY101_VIDEO_PATH` nel notebook.

### Verifica

```python
from pathlib import Path
videos = list(Path('data/assembly101').rglob('*.mp4'))
print(f"Trovati {len(videos)} video Assembly101")
```

---

## Running Experiments

Tutti gli esperimenti accettano il percorso del dataset come parametro. Nessuna modifica a `config.py` necessaria.

### Experiment 1: Baseline Posture Dataset

```python
from ergovision.experiments.experiment_01_posture_dataset import run_experiment
results = run_experiment(
    dataset_path='data/posture-keypoints-detection',
    subset_size=50,
)
```

### Experiment 2: Assembly101 Robustness

```python
from ergovision.experiments.experiment_02_assembly101_robustness import run_experiment
metrics = run_experiment(
    video_folder='data/assembly101',
    max_videos=5,
    sampling_rate=1.0,       # 1 frame al secondo
    max_frames_per_video=100,
)
```

### Experiment 3: Qualitative Explainability

```python
from ergovision.experiments.experiment_03_qualitative_explainability import run_experiment
run_experiment()
```

### All experiments via notebook

Open `ergovision_kaggle.ipynb` and run cells sequentially (8 sections).

## Outputs

```
outputs/
├── ergonomic_assessment.csv          # Kaggle dataset results
├── ergonomic_assessment.json         # Full scoring results
├── visualizations/                   # Annotated images
└── assembly101/
    ├── extracted_frames/             # Per-video extracted frames
    │   └── <video_name>/
    │       ├── _metadata.csv
    │       ├── frame_000000.jpg
    │       └── ...
    ├── predictions/
    │   └── assembly101_predictions.csv  # Per-frame scoring results
    ├── figures/
    │   ├── assembly101_risk_distribution.png
    │   ├── assembly101_feature_availability.png
    │   ├── sample_low_risk.png
    │   ├── sample_medium_risk.png
    │   └── sample_high_risk.png
    ├── metrics/
    │   ├── assembly101_runtime_report.txt
    │   ├── assembly101_robustness_report.csv
    │   ├── assembly101_risk_distribution.csv
    │   ├── assembly101_feature_availability.csv
    │   └── paper_experimental_summary.txt
    └── failure_cases/
        ├── occlusion_failure_case.png
        └── missing_keypoints_failure_case.png
```

## Ergonomic Scoring

The scoring is a **RULA-inspired lightweight ergonomic risk estimation** — it approximates RULA principles using a simplified rule-based approach. It is **not** a full clinical RULA implementation.

### Features and thresholds

| Feature | Score 1 (Low) | Score 2 (Medium) | Score 3 (High) |
|---|---|---|---|
| Torso angle | < 20° | 20–60° | > 60° |
| Neck angle | < 10° | 10–30° | > 30° |
| Knee bend | < 30° (straight) | 30–60° (moderate) | > 60° (deep bend) |
| Shoulder asymmetry | < 10% | 10–30% | > 30% |
| Body inclination | < 10% | 10–30% | > 30% |

**Final score**: maximum of all available partial scores (a single critical posture is not hidden by averaging).

## Limitations

- **No training**: All models are used as-is with pretrained weights. No fine-tuning.
- **No clinical validation**: This is a research prototype. Results are feasibility indicators, not clinical or certified ergonomic assessments.
- **No ground-truth labels**: Accuracy, precision, recall, and F1 are not reported because no ground-truth ergonomic labels are available for these datasets.
- **No temporal modeling**: Each frame is scored independently.
- **Simplified features**: The scoring uses only 5 postural features. A full RULA assessment requires additional measurements (upper arm, lower arm, wrist, etc.).

## Citation

If you use Assembly101 in your research:

```
@inproceedings{sener2022assembly101,
  title={Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities},
  author={Sener, F. and Chatterjee, D. and Shelepov, E. and He, K. and Singhania, D. and Wang, R. and Yao, A. and Gall, J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## License

Research prototype. For academic and research use only.
