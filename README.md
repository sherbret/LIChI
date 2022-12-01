# Unsupervised Linear and Iterative Combinations of Patches for Image Denoising
Sébastien Herbreteau and Charles Kervrann

## Requirements

Here is the list of libraries you need to install to execute the code:
* Python 3.8
* Pytorch 1.12.0
* Torchvision 0.13

## Install

To install in an environment using pip:

```
python -m venv .lichi_env
source .lichi_env/bin/activate
pip install /path/to/LIChI
```

## Demo

To denoise an image with LIChI (remove ``--add_noise`` if it is already noisy):
```
python ./demo.py --sigma 15 --add_noise --in ./test_images/cameraman.png --out ./denoised.png
```

## Results

### Gray denoising
The PSNR (dB) results of different methods on three datasets corrupted with synthetic white Gaussian noise
and sigma = 5, 15, 25, 35 and 50. Best among each category (unsupervised or supervised) is in bold. Best among each
subcategory is underlined

<img width="1066" alt="results_psnr" src="https://user-images.githubusercontent.com/88136310/205091125-6dbbf47c-d639-4485-8a95-f649ccc44efa.png">


### Complexity
We want to emphasize that  LIChI is relatively fast. We report here the execution times of different algorithms. It is
provided for information purposes only, as the implementation, the language used and the machine on which the code is run, highly influence the  results. The CPU used is a 2,3 GHz Intel Core i7 and the GPU is a GeForce RTX 2080 Ti. LIChI has been entirely written in Python with Pytorch so it can run on GPU unlike its traditional counterparts. 


Running time (in seconds) of different methods on images of size 256x256. Run times are given on CPU and GPU if available.

<img width="285" alt="results_running_time" src="https://user-images.githubusercontent.com/88136310/205092027-11aa0770-17fd-40c1-b9d7-1973c56732b3.png">


## Acknowledgements

This work was supported by Bpifrance agency (funding) through the LiChIE contract. Computations  were performed on the Inria Rennes computing grid facilities partly funded by France-BioImaging infrastructure (French National Research Agency - ANR-10-INBS-04-07, “Investments for the future”).
