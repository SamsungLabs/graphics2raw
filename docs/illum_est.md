# Illumination Estimation
## Overview
The following scripts assume or create the following directory structure

```
|-- illum_est_expts
|   |-- data
|   |   |-- SamsungNX2000
|   |   |   |-- ours
|   |   |   |-- real
|   |   |   `-- upi
|   |   |-- Canon1DsMkIII
|   |   |-- Canon600D
|   |   |-- FujifilmXM1
|   |   |-- NikonD40
|   |   |-- NikonD5200
|   |   |-- OlympusEPL6
|   |   |-- PanasonicGX1
|   |   `-- SonyA57
|   `-- expts
|       |-- Canon1DsMkIII_illum_est_ours
|       |   |-- models
|       |   |-- tensorboard
|       |   `-- results
|   |-- nus_metadata
|   |   `-- nus_outdoor_gt_illum_mats
|   `-- synthia
|       `-- SYNTHIA_RAND_CVPR16
            `-- RGB
```
## Prepare real data
For all methods, we use a subset of the SYNTHIA dataset for training and the NUS dataset for testing.
- Prepare the [NUS dataset](https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html)
    - Follow the instructions in [illum_est_nus.pptx](illum_est_nus.pptx) for each camera and put the images under `data/<camera>/real`
    - From the [NUS dataset webpage](https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html), download the groundtruth illuminant (`MAT`) files for each camera and put them under both `data/nus_metadata/nus_outdoor_gt_illum_mats` and `data/<camera>/real` for each camera
- Download the SYNTHIA-RAND (CVPR16) dataset from [link](http://synthia-dataset.net/downloads/)
    - We used [200 images](assets/split_files/illum_est/synthia_train_val_list.p) from `SYNTHIA_RAND_CVPR16/RGB` for training and validation

## Our method 
### Data generation 
```
python3 -m jobs.generate_dataset_illum_est_graphics2raw
```
### Training & Testing
```
python3 -m jobs.illum_est -c <camera1,camera2,...> -m ours
```

## UPI
### Data generation 
Due to copyright issues, we cannot re-distribute third-party code. Please refer to [upi.md](upi.md) before proceeding to the following steps.
```
python3 -m jobs.generate_dataset_illum_est_upi
```
### Training & Testing
```
python3 -m jobs.illum_est_upi
```
## Real
### Data generation 
Already completed in [Prepare real data](#prepare-real-data).
### Training & Testing
```
python3 -m jobs.illum_est -c <camera1,camera2,...> -m real
```



