<h1 align="center">Towards End-to-End Underwater Multi-View Stereo for Real-World Dense Scene Reconstruction</h1>

<br />

<div align="center">

<a href="https://www.youtube.com/watch?v=1N-jPW4yNS8&t=1s" target='_blank'><img src="assets/Video_Page.png" width="70%" /></a>

</div>

## 📦 Underwater Multi-View Stereo Dataset

The underwater multi-view stereo (UwMVS) dataset represents the **first large-scale synthetic dataset** that preserves real-world underwater degradation characteristics, specifically designed for end-to-end training and evaluation of learning-based UwMVS methods.

### ✔ Comparison with Existing Underwater Stereo Datasets

| Datasets             | Year | Training Set | Validation Set | Test Set | Multi-View* | Depth Map† | Point Cloud‡ |
|----------------------|------|-------------|---------------|----------|------------|-----------|-------------|
| UWbundle [1]         | 2017 | -           | -             | 36       | ✓          | ✗         | ✓           |
| UwStereoNet [2]      | 2019 | 4,047       | -             | 15       | ✓          | ✗         | ✗           |
| DRUVA [3]            | 2023 | 6,000       | -             | 110      | ✓          | ✗         | ✗           |
| FLSea-Stereo [4]     | 2023 | 7,337       | -             | -        | ✓          | ✗         | ✗           |
| SeaThru-NeRF [5]     | 2023 | 74          | 14            | -        | ✓          | ✗         | ✗           |
| UwStereo [6]         | 2024 | 26,611      | -             | 2,957    | ✗          | ✓         | ✗           |
| **UwMVS (Ours)**     | -    | **108,388** | **24,696**    | **4,312**<br>**28,440** | ✓ | ✓ | ✓ |

#### Notes:
- *: Indicates whether the dataset includes multi-view images.
- †: Specifies if per-view ground-truth metric depth maps are available.
- ‡: Denotes the presence of ground-truth point clouds for each reconstruction scenario.

### ✔ Statistics of the Proposed UwMVS Dataset

| Attributes                          | Training Set | Validation Set | Test Set |
|-------------------------------------|--------------|----------------|----------|
| # of Reconstruction Scenarios       | 79           | 18             | 22       |
| # of Underwater Scene Types         | 4            | 4              | 4        |
| # of Underwater Degradation Levels  | 7            | 7              | 1        |
| # of Images in Total                | 108,388      | 24,696         | 4,312    |
| # of Images for Each Scenario       | 1,372        | 1,372          | 196      |
| Ground-Truth Depth Map              | ✓            | ✓              | -        |
| Ground-Truth Point Cloud            | ✓            | ✓              | ✓        |

#### Notes:
Our UwMVS dataset includes 79, 18, and 22 underwater reconstruction scenarios for training, validation, and testing, respectively. Each scenario consists of 1,372 images for both training and validation, and 196 images for testing. The dataset covers 4 underwater scene types per scenario and incorporates 7 levels of underwater degradation for training and validation to enhance model robustness. The most severe degradation level is reserved for the test set to evaluate the generalization capability of the learning-based UwMVS. In total, the training, validation, and test sets comprise 108,388, 24,696, and 4,312 underwater multi-view images, respectively.

In addition, we provide 21 real-world underwater video sequences with a total duration of 948 seconds, recorded at 30 frames per second, from which up to 28,440 multi-view images can be extracted for evaluating reconstruction performance in real-world underwater scenes.

Our synthesis method generates underwater multi-view images with ground-truth depth maps and point clouds directly obtained from the in-air MVS dataset [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36), thereby eliminating the need for labor-intensive depth annotation and costly underwater laser scanning.

*Acknowledgement*: We extend our sincere gratitude to DTU for their foundational dataset and contributions to the research community.

### ✔ UwMVS Training and Validation Sets

The UwMVS training and validation sets are available [here](https://drive.google.com/drive/folders/1WeVMWnPXDBpB948fbhG6xX7O24Mt-gXf?usp=sharing).
```
UwMVS_Training_Validation
 ├── Cameras
 ├── Depths_raw
 ├── MVS_Data
 ├── Rectified_AIR
 └── Rectified_UW
       ├── Bluish
       ├── Greenish
       ├── Hazy
       └── Lowlight
```
#### Folder Structure

- **`Cameras`**: Contains camera intrinsics, extrinsics, and a view selection pair file.
- **`Depth_raw`**: Stores ground-truth depth maps obtained through depth map rendering.
- **`MVS_Data`**: Includes ground-truth point clouds scanned by a structured-light scanner.
- **`Rectified_AIR`**: Holds terrestrial in-air multi-view images.
- **`Rectified_UW`**: Contains underwater multi-view images across four types of underwater scenes.

### ✔ UwMVS Test Set

The UwMVS test set is available [here](https://drive.google.com/file/d/1qvYtvIKiqpSzLhZGO5mzyJFHsYjWIplC/view?usp=sharing).
```
UwMVS_Test
 ├── Bluish
 ├── Greenish
 ├── Hazy
 └── Lowlight
```
#### Each directory contains:
- **Camera parameters** (intrinsics, extrinsics, and view selection pair file)
- **Underwater multi-view images** organized by reconstruction scenario

### ✔ UwMVS Real-World Test Set

The UwMVS real-world test set is available [here](https://drive.google.com/drive/folders/1TeaEg3Uzq6eEj0VZJSfBkJYNYygpFBA0?usp=sharing).
```
UwMVS_Real_World
 ├── Images
 └── Videos
```
#### Folder Structure

- **`Images`**: Contains real-world underwater multi-view images extracted from videos.
- **`Videos`**: Stores real-world underwater video sequences.
- **Note**: Scenarios 1–18 were recorded in the Red Sea, and Scenarios 19–21 near Puerto Galera Island.

## :ocean: Underwater Multi-View Images Synthesis

<div align="center">

<img src="assets/PUIS.png" width="50%" />

</div>

Our two-stage physically-guided underwater multi-view images synthesis approach consists of: a) Underwater Degradation Parameters Estimator: Takes a real-world underwater image as input and estimates backscatter coefficients $\mathbf{S}$, attenuation coefficients $\mathbf{A}$, and global white point $W$. b) Underwater Multi-View Images Synthesizer: Processes real-world in-air multi-view images as input and generates underwater multi-view images by performing the white balance, direct attenuation, and backscatter addition with the estimated $W$, $\mathbf{A}$, $\mathbf{S}$ sequentially. Note that the camera-object depth map are estimated via the monocular depth estimation network [ZoeDepth](https://github.com/isl-org/ZoeDepth).

### ✔ Underwater Degradation Parameters Estimator
```
python ./Physical_Synthesis/estimator.py
```

### ✔ Underwater Multi-View Images Synthesizer

For synthesizing the training set, first run 

```
python ./Physical_Synthesis/synthesize_train.py
```

For synthesizing the validation set, run 

```
python ./Physical_Synthesis/synthesize_val.py
```

For synthesizing the test set, run 
```
python ./Physical_Synthesis/synthesize_test.py
```
## :diving_mask: Underwater Multi-View Stereo

### 📈 Training on UwMVS

#### Modify ``scripts/train.sh``:

* Set ``MVS_TRAINING`` as the path of the UwMVS training set.
* Set ``LOG_DIR`` to save the checkpoints.
* Change ``NGPUS`` to suit your device. By default, we employ the *DistributedDataParallel* mode to train the model. You can also train the model using a single GPU.
  
```bash
# Path to UwMVS training dataset
MVS_TRAINING="/your/path/to/UwMVS_training_set"

# Checkpoint save directory
LOG_DIR="/your/path/to/checkpoints"

# Number of GPUs (adjust to your hardware)
NGPUS=2
```

#### Configure ``datasets/dtu_yao.py``:
Specify the underwater scene type (e.g., Greenish) by updating the image path:

```python
# Example for Greenish water scene:
img_filename = os.path.join(
    self.datapath,
    'Rectified_UW/Greenish/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx)
)

# Alternative scene types available:
# - Bluish: 'Rectified_UW/Bluish/{}/rect...'
# - Hazy: 'Rectified_UW/Hazy/{}/rect...'
# - Lowlight: 'Rectified_UW/Lowlight/{}/rect...'
```
#### Run ``./scripts/train.sh``:
Execute the following scripts to train the model from scratch:
```
bash ./scripts/train.sh
```

## <span style="color:red">❤️</span> Acknowledgements

We gratefully acknowledge the foundational advances that made this research possible: [Sea-Thru](https://openaccess.thecvf.com/content_CVPR_2019/html/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.html) and the [Revised Underwater Image Formation Model](https://openaccess.thecvf.com/content_cvpr_2018/html/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.html) revolutionized underwater imaging; the pioneering [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) dataset established essential MVS benchmarks; the MVSNet series ([MVSNet](https://arxiv.org/pdf/1804.02505), [CasMVSNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Gu_Cascade_Cost_Volume_for_High-Resolution_Multi-View_Stereo_and_Stereo_Matching_CVPR_2020_paper.html), [GC-MVSNet](https://openaccess.thecvf.com/content/WACV2024/html/Vats_GC-MVSNet_Multi-View_Multi-Scale_Geometrically-Consistent_Multi-View_Stereo_WACV_2024_paper.html), [GoMVS](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_GoMVS_Geometrically_Consistent_Cost_Aggregation_for_Multi-View_Stereo_CVPR_2024_paper.html), [RC-MVSNet](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Multi-View_Stereo_Representation_Revist_Region-Aware_MVSNet_CVPR_2023_paper.html), [UniMVSNet](https://arxiv.org/abs/2201.01501), [TransMVSNet](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_TransMVSNet_Global_Context-Aware_Multi-View_Stereo_Network_With_Transformers_CVPR_2022_paper.html), [GeoMVSNet](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_GeoMVSNet_Learning_Multi-View_Stereo_With_Geometry_Perception_CVPR_2023_paper.html),) progressively advanced learning-based terrestrial depth estimation; while [ZoeDepth](https://openaccess.thecvf.com/content/ICCV2023/html/Guizilini_Towards_Zero-Shot_Scale-Aware_Monocular_Depth_Estimation_ICCV_2023_paper.html), [OmniData](https://openaccess.thecvf.com/content/ICCV2021/html/Eftekhar_Omnidata_A_Scalable_Pipeline_for_Making_Multi-Task_Mid-Level_Vision_Datasets_ICCV_2021_paper.html), and [Metric3D](https://openaccess.thecvf.com/content/ICCV2023/html/Yin_Metric3D_Towards_Zero-shot_Metric_3D_Prediction_from_A_Single_Image_ICCV_2023_paper.html) advanced depth and surface normal estimation. These collective breakthroughs created the technical bedrock for our underwater multi-view stereo framework.

## References

[1] K. A. Skinner, E. Iscar, and M. Johnson-Roberson, "Automatic color correction for 3D reconstruction of underwater scenes," in *2017 IEEE International Conference on Robotics and Automation (ICRA)*, 2017, pp. 5140-5147.

[2] K. A. Skinner, J. Zhang, E. A. Olson, and M. Johnson-Roberson, "Uwstereonet: Unsupervised learning for depth estimation and color correction of underwater stereo imagery," in *2019 International Conference on Robotics and Automation (ICRA)*, 2019, pp. 7947-7954.

[3] N. Varghese, A. Kumar, and A. Rajagopalan, "Self-supervised monocular underwater depth recovery, image restoration, and a real-sea video dataset," in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, pp. 12248-12258.

[4] A. Randall and T. Treibitz, "FLSea: Underwater visual-inertial and stereo-vision forward-looking datasets," 2023. [Online]. Available: https://arxiv.org/abs/2302.12772

[5] D. Levy et al., "Seathru-nerf: Neural radiance fields in scattering media," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 56-65.

[6] Q. Lv et al., "Uwstereo: A large synthetic dataset for underwater stereo matching," 2024. [Online]. Available: https://arxiv.org/abs/2409.01782




