# Stereo-Depth-Estimation

## 引用和参考

本项目参考了以下资源：

- StereoNet:[Link](https://gitea.thehfut.top/liangliang/StereoNet)
- CREStereo:[Link](https://github.com/megvii-research/CREStereo) 



## Datasets

### Download dataset:

For simplicity, you can  Download the datasets in https://pan.baidu.com/s/1-qyS6BXRfyF5B_4HA_c7gQ?pwd=A501 and put the folder in ./data.

### Data format:

The RGB dataset is in `./data/flythings3d/frames_cleanpass. `

```
 flythings3d
  └── frames_cleanpass
       └── TRAIN
         └── left
              ├── 0.png
              ├── 1.png
              ├── ...
         └── right
              ├── 0.png
              ├── 1.png
              ├── ...
       └── TEST
         └── left
              ├── 0.png
              ├── 1.png
              ├── ...
         └── right
              ├── 0.png
              ├── 1.png	
              ├── ...
```

The Depth dataset is in ./data/flythings3d/frames_disparity. `

```
 flythings3d
  └── frames_disparity
       └── TRAIN
         └── left
              ├── 0.png
              ├── 1.png
              ├── ...
         └── right
              ├── 0.png
              ├── 1.png
              ├── ...
       └── TEST
         └── left
              ├── 0.png
              ├── 1.png
              ├── ...
         └── right
              ├── 0.png
              ├── 1.png	
              ├── ...
```



## Dependencies

Python Version: 3.8.18 CUDA:11.8

- Pytorch 2.0.1
- tensorboardX v2.1

```
 python -m pip install -r requirements.txt
```

## Training

The network of the model is located in  `./models/model.py. `, the dataset reading is located in `./dataset.py. `, each validation is saved in `./pic `, and the final generated model file is saved in the root directory.

run the following command:

```
python train.py
```

You can launch a TensorBoard to monitor the training process:

```
tensorboard --logdir ./lightning_logs
```

and navigate to the page at `http://localhost:6006` in your browser.

## Result

The pink-colored training is for the  StereoNet+SEAttention model, the blue-colored training is for the StereoNet model, and the yellow-red colored training is for the  StereoNet+SKAttention model.
<div align=center><img src="https://github.com/fyzhang2024/Stereo-Depth-Estimation/blob/main/img/val_loss_epoch.png" width="480" height="360" /></div>


The validation visualization is shown in the figure below：

**StereoNet**：

<div align=center><img src="https://github.com/fyzhang2024/Stereo-Depth-Estimation/blob/main/img/CRE.png" width="480" height="360" /></div>

**StereoNet+SEAttention**：

<div align=center><img src="https://github.com/fyzhang2024/Stereo-Depth-Estimation/blob/main/img/CRE%2BSEK.png" width="480" height="360" /></div>

**StereoNet+SKAttention**：

<div align=center><img src="https://github.com/fyzhang2024/Stereo-Depth-Estimation/blob/main/img/CRE%2BSKA.png" width="480" height="360" /></div>

**Table 1 Comparison of Experimental Data**

| 网络名称 | train_loss(m/pix) | val_loss(m/pix) | Test_loss(m/pix) | FPS |
|:------:|:-------:|:------:|:-------:|:------:|
| StereoNet | 0.01307 | 0.09586 | 0.09467 | 45.55 |
| StereoNet+SKAttention | 0.01635 | 0.08952 | 0.07306 | 31.68 |
| StereoNet+SEAttention | 0.01318 | 0.08618 | 0.06497 | 26.88 |

