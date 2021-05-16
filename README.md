# KDD 2019 Paper
## DeepUrbanEvent: <br> A System for Predicting Citywide Crowd Dynamics at Big Events

## Outline
Under the situations of big events, citywide crowd dynamics would manifest some different patterns from the normal routine. In this study, we propose a highly deployable system to extract the trend from momentary changes, and leverage multitask learning to predict crowd density and crowd flows simultaneously.

<p align="center"><img src="figures/earthquake.png" width="50%" height="50%"></p>
<h4 align="center">Fig 1. Sudden change of human mobility in Tokyo before and after the Great East Japan Earthquake</h4>


<p align="center"><img src="figures/problem.png" width="50%" height="50%"></p>
<h4 align="center">Fig 2. Crowd dynamics prediction</h4>


<p align="center"><img src="figures/multitask_convlstm.png"></p>
<h4 align="center">Fig 3. Multitask ConvLSTM encoder-decoder for simultaneous multi-step prediction</h4>

## Directories
```
├── common		# Parameters, data_utils
├── crowdFlowEncoder		# CNN autoencoder for crowd flow
├── meshdynamic		# meshgrid use
├── predCrowdDensity		# baselines for predicting crowd density
│	├── Historical Average
│	├── Copy Last Frame
│	├── ARIMA
│	├── VAR
│	├── ST-ResNet
│	├── CNN / CNN enc-dec.
│	└── ConvLSTM / ConvLSTM enc-dec.
├── predCrowdEncodedFlow		# baselines for predicting encoded crowd flows
│	├── Historical Average
│	├── Copy Last Frame
│	├── ST-ResNet
│	├── CNN / CNN enc-dec.
│	└── ConvLSTM / ConvLSTM enc-dec.
└── predMultitask(Ours)
	├── Multitask CNN enc-dec.
	└── Multitask ConvLSTM enc-dec.		# our proposed model
```

## Dependencies
Here are the working environment and major dependencies we run the code:
* Ubuntu 16.04.6 LTS
* Python 3 (>= 3.5; Anaconda Distribution)
* SciPy (>= 0.17.0)
* statsmodels (>= 0.8.0)
* NumPy (>= 1.11.0)
* pandas (>= 0.18.0)
* TensorFlow/tensorflow-gpu (>= 1.3.0; <= 1.14.0)
* Keras (>= 2.0.8)

## Citation
If you find anything in this repository useful to your research, please cite our paper :) We sincerely appreciate it. [DeepUrbanEvent: A System for Predicting Citywide Crowd Dynamics at Big Events](https://www.researchgate.net/profile/Renhe_Jiang/publication/334714928_DeepUrbanEvent_A_System_for_Predicting_Citywide_Crowd_Dynamics_at_Big_Events/links/5d417167299bf1995b597f28/DeepUrbanEvent-A-System-for-Predicting-Citywide-Crowd-Dynamics-at-Big-Events.pdf)
```
@inproceedings{jiang2019deepurbanevent,
  title={DeepUrbanEvent: A System for Predicting Citywide Crowd Dynamics at Big Events},
  author={Jiang, Renhe and Song, Xuan and Huang, Dou and Song, Xiaoya and Xia, Tianqi and Cai, Zekun and Wang, Zhaonan and Kim, Kyoung-Sook and Shibasaki, Ryosuke},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2114--2122},
  year={2019},
  organization={ACM}
}
```
