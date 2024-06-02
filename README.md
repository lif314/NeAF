# ASR-Solver: Unified Continuous Audio Signal Representions via Fourier Kolmogorov-Arnold Networks

## Results

### `Counting`
- `SNR` & `LSD`

|Model| hidden_layers | hidden_features | params | SNR $\uparrow$| LSD $\downarrow$|
|:--:|:--:|:--:|:--:|:--:|:---:|
| Siren | 7 | 256 | 198K | | |
| Fourier | 4 | 64 | 264K | |  |


- `PSNR`

|Model| hidden_layers | hidden_features | params |batch size| lr |PSNR $\uparrow$|
|:--:|:--:|:--:|:--:|:--:|:---:|:---:|
| Siren | 7 | 256 | 198K | 8192 | 1e-4 | 34.10 |
| Fourier | 3 | 64 | 198K | 8192| 1e-4 | 25.40 |
| Fourier | 3 | 64 | 198K | 1024 | 1e-4| 21.30 |
| Fourier(LayerNorm) | 3 | 64 | 199K | 16384 | 1e-4| 23.10 |


### `Bach`

- `SNR` & `LSD`

|Model| hidden_layers | hidden_features | params | SNR $\uparrow$| LSD $\downarrow$|
|:--:|:--:|:--:|:--:|:--:|:---:|
| Siren | 7 | 256 | 198K | 36.22 | 1.16|
| Siren(3K) | 7 | 256 | 198K | **37.30** | 1.020 |
| MLP+ReLU | 4 | 256 | 130K | 0 | 2.86 | 
| Siren | 7 | 256 | 198K | 36.22 | 1.16|
| Fourier(Linear) | 4 | 64 | 198K | 12.20 | 2.40 |
| Fourier,G=8 | 3 | 64 | 198K | 12.00 | 2.40 |
| Fourier,G=8 | 4 | 64 | 264K | 23.00 | 1.65 |
| Fourier,G=8 | 5 | 64 | 330K | 31.40 | 1.23 |
| Fourier,G=8(3K) | 5 | 64 | 330K | 36.30 | **0.963** |
| Fourier,G=8 | 2 | 128 | 528K | 4.13 | 2.68 |
| Fourier,G=5 | 5 | 64 | 206K | 12.60 | 2.33 |

- `PSNR`

|Model| hidden_layers | hidden_features | params |batch size| lr |PSNR $\uparrow$|
|:--:|:--:|:--:|:--:|:--:|:---:|:---:|
| Siren | 7 | 256 | 198K | 8192 | 1e-4 | 53.26 |
| Fourier | 1 | 128 | 266K | 8192| 1e-4 | 17.68 |
| Fourier | 3 | 128 | 790K | 8192| 1e-4 | 29.51 |
| Fourier | 2 | 64 | 133K | 8192| 1e-4 | 17.68 |
| Fourier | 3 | 64 | 198K | 8192| 1e-4 | 28.04 |
| Fourier | 4 | 64 | 264K | 8192 | 1e-4| 39.27 |

### GTZAN (blues/00028)
|Model| hidden_layers | hidden_features | params |batch size| lr |PSNR $\uparrow$|
|:--:|:--:|:--:|:--:|:--:|:---:|:---:|
| Siren | 7 | 256 | 198K | 8192 | 1e-4 |  |


## Related papers

*  [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/pdf/2006.09661.pdf)