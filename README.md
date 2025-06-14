<h1 align="center">Towards End-to-End Underwater Multi-View Stereo for Real-World Dense Scene Reconstruction</h1>

<br />

<div align="center">

<a href="https://www.youtube.com/watch?v=1N-jPW4yNS8&t=1s" target='_blank'><img src="Video_Page.png" width="70%" /></a>

</div>

## 📦 Underwater Multi-View Stereo Dataset

### ✔  Training and Validation Sets

The training and validation sets are available [here](https://drive.google.com/drive/folders/1WeVMWnPXDBpB948fbhG6xX7O24Mt-gXf?usp=sharing).
```
UwMVS_Training_Validation
 ├── Cameras
 ├── Depths_raw
 ├── Rectified_AIR
 └── Rectified_UW
       ├── Bluish
       ├── Greenish
       ├── Hazy
       └── Lowlight
```

### ✔  Test Set

The test set is available [here](https://drive.google.com/file/d/1qvYtvIKiqpSzLhZGO5mzyJFHsYjWIplC/view?usp=sharing).
```
UwMVS_Test
 ├── Bluish
 ├── Greenish
 ├── Hazy
 └── Lowlight
```

### ✔  Real-World Test Set

The real-world test set is available [here](https://drive.google.com/drive/folders/1TeaEg3Uzq6eEj0VZJSfBkJYNYygpFBA0?usp=sharing).
```
UwMVS_Real_World
 ├── Images
 └── Videos
```

## 🚂 Underwater Multi-View Images Synthesis

### ✔ Underwater Degradation Parameters Estimator
```
python estimator.py
```

### ✔ Underwater Multi-View Images Synthesizer

For synthesizing the training set, run 
```
python synthesizer_train.py
```

For synthesizing the validation set, run 

```
python synthesizer_val.py
```

For synthesizing the test set, run 
```
python synthesize_test.py
```

