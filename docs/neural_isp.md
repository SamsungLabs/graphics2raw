# Neural ISP
## Overview
The following scripts assume or create the following directory structure

```
|-- neural_isp_expts
|   |-- data
|   |   |-- clean_raw_dngs
|   |   |-- clean_srgb_tiffs
|   |   |-- graphics_dngs_graphics2raw
|   |   |-- graphics_dngs_upi
|   |   |-- graphics_srgb_graphics2raw
|   |   `-- graphics_srgb_upi
|   `-- expts
|       |-- neural_isp_graphics2raw_clean_raw
|       |   |-- models
|       |   |-- tensorboard
|       |   `-- results
|       |-- neural_isp_upi_clean_raw
|       `-- neural_isp_real_clean_raw
```

## Prepare real data
For all methods, we use the [Day-to-Night](https://openaccess.thecvf.com/content/CVPR2022/papers/Punnappurath_Day-to-Night_Image_Synthesis_for_Training_Nighttime_Neural_ISPs_CVPR_2022_paper.pdf) Nighttime dataset for testing.
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
#### Use Photoshop to process the RAW DNGs into sRGB space.
> Photoshop version: 24.5\
> Camera RAW version: 15.3.1

1. Go to: File -> Scripts -> Image Processor
2. Input folder: `clean_raw_dngs`
    - Check "Open first image to apply settings"
3. Target folder: `clean_srgb_tiffs`
4. File type: "Save as TIFF", uncheck "LZW Compression"
5. Preferences: uncheck everything
6. Click "Run"
7. Under "detail", set sharpening = 40; under "manual noise reduction" set luminance = 0, color = 25
8. Camera RAW preferences: Color Space: sRGB IEC61996-2.1 - 8 bit (12.1MP) - 300 ppi
9. Click "Open"
- Results will be saved into the target folder

## Graphics2RAW (Our Method)
### Data generation
#### Invert graphics data to RAW space
Use Graphics2RAW to invert graphics data to RAW space:
```
python3 -m jobs.generate_dataset_isp_denoise_graphics2raw
```
This will generate `graphics_dngs_graphics2raw`.

#### Use Photoshop to process the RAW DNGs into sRGB space.
> Photoshop version: 24.5\
> Camera RAW version: 15.3.1

1. Go to: File -> Scripts -> Image Processor
2. Input folder: `graphics_dngs_graphics2raw`
    - Check "Open first image to apply settings"
3. Target folder: `graphics_srgb_graphics2raw`
4. File type: "Save as TIFF", uncheck "LZW Compression"
5. Preferences: uncheck everything
6. Click "Run"
7. Under "detail", set sharpening = 40; under "manual noise reduction" set luminance = 0, color = 25
8. Camera RAW preferences: Color Space: sRGB IEC61996-2.1 - 8 bit (12.1MP) - 300 ppi
9. Click "Open"
- Results will be saved into the target folder

### Training & Testing
```
python3 -m jobs.neural_isp_graphics2raw
```

#### Dataset structure 
- real_dataset
    - clean_raw/*.png (30-frame-averaged clean RAW image)
    - clean/*.png     (originally simple ISP processed pngs, now replaced with Photoshop processed pngs)
    - dng/            (not used by Neural ISP)
      - iso_50/*.dng
      - iso_1600/*.dng
      - iso_3200/*.dng
- graphics_dataset
    - train (60 images)
        - clean/*.png (converted from PS-rendered .tifs)
        - clean_raw/*.png (converted from graphics .dngs)
        - metadata_raw/*.p
    - val (10 images)
        - clean/*.png
        - clean_raw/*.png
        - metadata_raw/*.p

## UPI
### Data generation
Due to copyright issues, we cannot re-distribute third-party code. Please refer to [upi.md](upi.md) before proceeding to the following steps.

#### Invert graphics data to RAW space
Use UPI to invert graphics data to RAW space:
```
python3 -m jobs.generate_dataset_isp_denoise_upi
```
This will generate `graphics_dngs_upi`.

#### Use Photoshop to process the RAW DNGs into sRGB space.
Same process as Graphics2RAW
> Photoshop version: 24.5\
> Camera RAW version: 15.3.1

1. Go to: File -> Scripts -> Image Processor
2. Input folder: `graphics_dngs_upi`
    - Check "Open first image to apply settings"
3. Target folder: `graphics_srgb_upi`
4. File type: "Save as TIFF", uncheck "LZW Compression"
5. Preferences: uncheck everything
6. Click "Run"
7. Under "detail", set sharpening = 40; under "manual noise reduction" set luminance = 0, color = 25
8. Camera RAW preferences: Color Space: sRGB IEC61996-2.1 - 8 bit (12.1MP) - 300 ppi
9. Click "Open"
- Results will be saved into the target folder

### Training & Testing
```
python3 -m jobs.neural_isp_upi
```

## Real
### Data generation
Already done in [Prepare real data](#prepare-real-data).

### Training & Testing

```
python3 -m jobs.neural_isp_real --fold 0
python3 -m jobs.neural_isp_real --fold 1
python3 -m jobs.neural_isp_real --fold 2
```
Results reported in paper are averaged over 3 folds.

#### Dataset structure 
- real_dataset
    - clean_raw/*.png (30-frame-averaged clean RAW image)
    - clean/*.png     (originally simple ISP processed pngs, now replaced with Photoshop processed pngs)
    - dng/            (not used by Neural ISP)
      - iso_50/*.dng
      - iso_1600/*.dng
      - iso_3200/*.dng
- real_dataset_k_fold (converted from real_dataset)
    - test (35 images)
        - clean/*.png 
        - clean_raw/*.png 
        - dng/iso_50
    - train (60 images)
        - clean/*.png 
        - clean_raw/*.png 
        - metadata_raw/*.p
        - noisy_raw/*.png
    - val (10 images)
        - clean/*.png
        - clean_raw/*.png
        - metadata_raw/*.p
        - noisy_raw/*.png