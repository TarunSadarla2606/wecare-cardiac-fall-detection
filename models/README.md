# Models

Trained model weights are not committed due to file size. Run the notebooks to generate them.

## Saved Files (generated after training)

| File | Description |
|------|-------------|
| `imu_model.pth` | IMU fall detection CNN weights (PyTorch state_dict) |
| `feature_scaler.gz` | StandardScaler fitted on IMU training data |
| `ecg_model.pth` | ECG arrhythmia CNN weights (PyTorch state_dict) |

## Loading a Model

```python
import torch
from src.imu.model_imu import IMU_CNN

model = IMU_CNN(in_channels=9)
model.load_state_dict(torch.load('models/imu_model.pth', map_location='cpu'))
model.eval()
```

## Edge Export (TorchScript)

```python
scripted = torch.jit.script(model)
scripted.save('models/imu_model_scripted.pt')
```

> TFLite deployment was evaluated but abandoned due to critical dependency conflicts.
> TorchScript is the stable, recommended format for nRF52/ESP32 edge deployment.
