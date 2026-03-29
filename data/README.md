# Data

Raw datasets are **not committed** to this repository. Download from the sources below.

## ECG — MIT-BIH Arrhythmia Database

- **Source:** [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- **Format:** WFDB format (`.dat`, `.atr`, `.hea`)
- **Scale:** 48 recordings, 360 Hz, ~10,000 annotated heartbeat segments
- **Access:** Free, anonymized, ethically cleared

```bash
pip install wfdb
python -c "import wfdb; wfdb.dl_database('mitdb', './data/mitbih')"
```

Place downloaded files at: `data/mitbih/`

## IMU — MobiFall_processed

- **Source:** [Kaggle — MobiFall_processed](https://www.kaggle.com/datasets/)
- **Format:** CSV files per trial, 9 columns (acc_x/y/z · gyro_x/y/z · ori_azimuth/pitch/roll)
- **Scale:** ~40,000 windows from 24 participants, 87–100 Hz
- **Fall types:** FOL · FKL · BSC · SDL
- **ADL types:** Walking, standing, sitting, stairs, etc.

Place downloaded CSVs at: `data/mobifall/`

## Expected Layout

```
data/
├── mitbih/
│   ├── 100.dat
│   ├── 100.atr
│   └── ...
└── mobifall/
    ├── SDL_9_1.csv
    ├── CSI_7_1.csv
    └── ...
```
