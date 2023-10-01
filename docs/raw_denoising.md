# RAW Denoising
## Overview
### Directory structure
The following scripts assume or create the following directory structure

```
|-- denoising_expts
|   |-- data
|   |   |-- clean_raw_dngs
|   |   |-- graphics_dngs_graphics2raw
|   |   |-- graphics_dngs_upi
|   `-- expts
|       |-- denoise_graphics2raw_iso1600
|       |   |-- models
|       |   |-- tensorboard
|       |   `-- results
```
### Code
Due to copyright issues, we cannot re-distribute third-party code. 
To run our RAW denoising experiments, please
copy over the [Restormer](https://arxiv.org/abs/2111.09881) architecture from the author's official repository: [here](https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py), 
and place the code in [model_archs/restormer.py](../model_archs/restormer.py)

## Prepare real data
For all methods, we use the [Day-to-Night]((https://openaccess.thecvf.com/content/CVPR2022/papers/Punnappurath_Day-to-Night_Image_Synthesis_for_Training_Nighttime_Neural_ISPs_CVPR_2022_paper.pdf)) Nighttime dataset for testing.
#### Download Day-to-Night dataset
- Download the day-to-night dataset from [here](https://github.com/SamsungLabs/day-to-night#get-started).
- We only need contents from the `night_real` folder, copy subdirectories in `night_real` into `./real_dataset`
#### Package clean RAW images into DNGs
Package each clean RAW image (averaged from 30 ISO 50 frames) into a DNG, so that it can be processed by Photoshop to create the sRGB target.
```
python3 -m data_generation.package_clean_nighttime_raw_to_dng \
--dng_folder_path /path/to/real_dataset/dng/iso_50 \
--raw_folder_path /path/to/real_dataset/clean_raw \
--save_path /path/to/neural_isp_expts/data/clean_raw_dngs
```
> Same as [Neural ISP -> Prepare real data](neural_isp.md#prepare-real-data)

## Graphics2RAW (Our Method)
### Data generation 
#### Invert graphics data to RAW space
Use Graphics2RAW to invert graphics data to RAW space:
```
python3 -m jobs.generate_dataset_isp_denoise_graphics2raw
```
This will generate `graphics_dngs_graphics2raw`.

> Same as [Neural ISP -> Graphics2RAW -> Invert graphics data to RAW space](neural_isp.md#invert-graphics-data-to-raw-space)

### Training & Testing
ISO3200
```
python3 -m jobs.denoise_graphics2raw_iso3200
```
ISO1600
```
python3 -m jobs.denoise_graphics2raw_iso1600
```

## UPI
### Data generation 
Due to copyright issues, we cannot re-distribute third-party code. Please refer to [upi.md](upi.md) before proceeding to the following steps.

#### Invert graphics data to RAW space
Use UPI to invert graphics data to RAW space:
```
python3 -m jobs.generate_dataset_isp_denoise_upi
```
This will generate `graphics_dngs_upi`.

> Same as [Neural ISP -> UPI -> Invert graphics data to RAW space](neural_isp.md#invert-graphics-data-to-raw-space-1)
### Training & Testing
ISO3200
```
python3 -m jobs.denoise_upi_iso3200
```
ISO1600
```
python3 -m jobs.denoise_upi_iso1600
```

## Real
### Data generation 
Already done in [Prepare real data](#prepare-real-data).

### Training & Testing
ISO3200
```
python3 -m jobs.denoise_real_iso3200 --fold 0
python3 -m jobs.denoise_real_iso3200 --fold 1
python3 -m jobs.denoise_real_iso3200 --fold 2
```
ISO1600
```
python3 -m jobs.denoise_real_iso1600 --fold 0
python3 -m jobs.denoise_real_iso1600 --fold 1
python3 -m jobs.denoise_real_iso1600 --fold 2
```
Results reported in paper are averaged over 3 folds.