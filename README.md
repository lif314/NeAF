# Coordinate-KANs

## Tasks
- [x] `train_rgb.py`: 2D Image Regression 
- [x] `train_gray.py`: 1D Image Regression
- [x] `train_audio.py`: 1D Audio Signal Regression
- [ ] `train_possion.py`: 1D Image Gradient and Laplace Regression
- [ ] `train_sdf.py`: Sign Distance Functions Regression
- [ ] `train_radiance_field.py`: Inverse Rendering (NeRF)
- More and More!!! Please refer to [Siren](https://github.com/vsitzmann/siren)



## 2D Image Regression

### RGB
| PE | Model | Activation/Basis |Arch | Params| PSNR |
|:-----:|:---:|:----:|:----:|:----:|:-----:|
| - | MLP | ReLU | [2, 256, 256, 256, 3] | 133K |21.10|
| - | MLP | Gaussian | [2, 256, 256, 256, 3] | 133K | 22.30|
| - | MLP | quadratic |  [2, 256, 256, 256, 3] | 133K | 22.30|
| - | MLP | multi-quadratic | [2, 256, 256, 256, 3] | 133K | 21.20 |
| - | MLP | laplacian |  [2, 256, 256, 256, 3] |133K | 21.80 |
| - | MLP | super-gaussian | [2, 256, 256, 256, 3] |133K | 22.20 |
| - | MLP | expsin | [2, 256, 256, 256, 3] |133K | 22.30 |
| NeRF-PE | MLP |  ReLU | [40, 256, 256, 256, 3] | 142K | 11.20 |
| NeRF-PE | MLP |  Gaussian | [40, 256, 256, 256, 3] | 142K | 12.20 |
| NeRF-PE | MLP |  quadratic | [40, 256, 256, 256, 3] | 142K | |
| NeRF-PE | MLP |  multi-quadratic | [40, 256, 256, 256, 3] | 142K | |
| NeRF-PE | MLP |  laplacian | [40, 256, 256, 256, 3] | 142K | |
| NeRF-PE | MLP |  super-gaussian | [40, 256, 256, 256, 3] | 142K | |
| NeRF-PE | MLP |  expsin | [40, 256, 256, 256, 3] | 142K | |
| FFN | MLP | - | [512,  256, 256, 256, 3] | 263K | 22.40 |
| - | Siren(MLP) | - | [2,  256, 256, 256, 256, 3] | 264K | 25.31/21.90 |
| - | Gabor(MLP) | - | [2, 256]*4 + [256, 256] * 4 + [256, 3] | 271K | 31.50/31.50 |
| - | Bacon(MLP) | - | [2, 256]*4 + [256, 256] * 4 + [256, 3] | 269K | 29.20/29.20 |
| - | KAN | B-Spline |  [2, 64, 64, 64, 3]| 85.1K | 22.70 |
| - | KAN | GRBF |  [2, 64, 64, 64, 3]| 77.2K | 22.80 |
| - | KAN | RBF |  [2, 64, 64, 64, 3] | 76.8K | 23.30 |
| - | KAN | RBF |  [2, 128, 128, 3] | 153K | 22.50 |
| - | KAN | RBF |  [2, 128, 128, 128, 3]| 301K | 24.00 |
| - | KAN | Fourier, G=8 |  [2, 32, 32, 32, 3]| 35.4K | 26.02/26.00 |
| - | KAN | Fourier, G=8 |  [2, 64, 64, 64, 3]| 136K | 26.70/25.40 |
| - | KAN | Fourier, G=8 |  [2, 64, 64, 3]| 70.8K | 29.30/29.30 |
| - | KAN | Fourier, G=8 |  [2, 128, 128, 128, 3]| 534K | 32.33/31.80 |
| - | KAN | Fourier, G=8 |  [2, 128, 128, 3]| 272K |  32.80/32.80|
| - | KAN | DoG |  [2, 64, 64, 64, 3]| 34.4K | 22.10 |
| - | KAN | Shannon |  [2, 64, 64, 64, 3]| 34.4K | 21.40 |
| - | KAN | FCN_Interpo |  [2, 64, 64, 64, 3]| 544K |  |
| - | KAN | Mexican Hat |  [2, 64, 64, 64, 3]| 17.9K | 22.09 |
| - | KAN | Morlet |  [2, 64, 64, 64, 3]| 17.9K | 22.60 |
| - | KAN | Morlet |  [2, 128, 128, 128, 3]| 68.6K | 24.30 |
| - | KAN | Morlet |  [2, 256, 256, 256, 3]| 268K | 25.50 |
| KAN-PE | KAN | RBF |  [2, 64, 64, 64, 3]| 137K |  |

### Gray

## Audio Signal
| PE | Model | Activation/Basis |Arch | Params| PSNR $\uparrow$|
|:-----:|:---:|:----:|:----:|:----:|:-----:|
| - | Siren(MLP) | - | [1,  256, 256, 256, 256, 256, 1] | 263K | 16.20 |
| - | KAN | Fourier | [1,  64, 64, 64, 1] | 133K | 23.40 |
| - | KAN | Fourier | [1,  128, 128, 128, 1] | 528K | 28.90 |
| - | KAN | Fourier | [1,  128, 128, 1] | 266K | 16.40 |
| - | KAN | Fourier | [1,  64, 64, 64, 64, 1] | 198K | 34.30  |
| - | KAN | Fourier | [1,  64, 64, 64, 64, 64, 1] | 264K | 43.44  |
| - | KAN | RBF | [1, 64, 64, 64, 1] | 75.1K |  16.20 |
| - | KAN | mexican_hat | [1, 128, 128, 128, 1] | 132K | 16.20 |

## SDF (TODO)
| PE | Model | Activation/Basis |Arch | Params| SDF MSE $\downarrow$|
|:-----:|:---:|:----:|:----:|:----:|:-----:|
| - | Siren(MLP) | - | [3,  256, 256, 256, 256, 256, 1] | 263K | 0.634 |
| - | KAN | RBF | [3,  64, 64, 64, 1] | 76.2K | 0.391 |

## Related papers

*  [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/pdf/2006.09661.pdf)
*  [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739.pdf)
*  [Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs](https://arxiv.org/pdf/2111.15135.pdf)
*  [Gaussian Activated Neural Radiance Fields for High Fidelity Reconstruction & Pose Estimation](https://arxiv.org/pdf/2204.05735.pdf)
*  [Spline Positional Encoding for Learning 3D Implicit Signed Distance Fields](https://arxiv.org/pdf/2106.01553.pdf)
*  [Multiplicative Filter Networks](https://openreview.net/pdf?id=OmtmcPkkhT)
*  [BACON: Band-limited Coordinate Networks for Multiscale Scene Representation](https://arxiv.org/pdf/2112.04645.pdf)
