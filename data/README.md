# Data

Raw data is **not committed** to this repository.

## HC18 Challenge Dataset

**Source:** Department of Obstetrics, Radboud University Medical Center, Nijmegen, Netherlands

**Contents:**
- 1,334 standard-plane fetal head ultrasound images (800×540 px)
- Acquired from 551 pregnancies, May 2014–May 2015
- Pixel sizes: 0.052–0.326 mm
- Training set (999 images) includes HC annotations by expert sonographers
- Test set (335 images) — no annotations

**Download:**
- Kaggle: https://www.kaggle.com/datasets/sahliz/hc18
- Official challenge: https://hc18.grand-challenge.org/

## Expected Directory Layout

After downloading, organize as follows:

```
data/
└── hc18/
    ├── training_set/
    │   ├── training_set/
    │   │   ├── 000_HC.png
    │   │   ├── 000_HC_Annotation.png
    │   │   └── ...
    │   └── training_set_pixel_size_and_HC.csv
    └── test_set/
        └── ...
```

Update the `INPUT_DIR` path in `src/train.py` and the notebooks accordingly.
