## UPI Modifications
Since we used different cameras from those used in the 
[Unprocessing](https://www.timothybrooks.com/tech/unprocessing/) paper,
we followed the authors' suggestion to modify `random_ccm()` and `random_gains()` in unprocess.py 
to best match the distribution of image metadata from the cameras we used. 

Due to copyright issues, we cannot re-distribute third-party code. To reproduce our procedure for UPI, 
please copy over the [official code](https://raw.githubusercontent.com/timothybrooks/unprocessing/master/unprocess.py) 
to [data_generation/unprocess.py](../data_generation/unprocess.py)
and refer to the following modifications. 

### CCM and Gains
We used camera sensors from the NUS dataset and the nighttime dataset of [Day-to-Night](https://openaccess.thecvf.com/content/CVPR2022/papers/Punnappurath_Day-to-Night_Image_Synthesis_for_Training_Nighttime_Neural_ISPs_CVPR_2022_paper.pdf) as our target sensors.
Therefore, the code was modified to use those sensors' CST matrices and gain ranges.

1. Replace `xyz2cams` in `random_ccm()` ([line 32](https://github.com/timothybrooks/unprocessing/blob/master/unprocess.py#L32)) with
```
xyz2cams = pickle.load(open('assets/container_dngs/NUS_S20FE_CST_mats.p', 'rb'))
```
2. Return the sampled `xyz2cam`; this matrix is needed in [package_exr_to_dng_upi.py](../data_generation/package_exr_to_dng_upi.py) for building the DNG.
- Return both `rgb2cam, xyz2cam` in `random_ccm()` ([line 58](https://github.com/timothybrooks/unprocessing/blob/master/unprocess.py#L58), [line 124](https://github.com/timothybrooks/unprocessing/blob/master/unprocess.py#L124))
- [line 141-146](https://github.com/timothybrooks/unprocessing/blob/master/unprocess.py#L141-L146),
change `metadata` to
    ```
    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
        'xyz2cam': xyz2cam,
    }
    ```
3. Replace `red_gain` and `blue_gain` in `random_gains()` ([line 67-68](https://github.com/timothybrooks/unprocessing/blob/master/unprocess.py#L67-L68)) with
```
red_gain = tf.random.uniform((), 1.0, 3.3)
blue_gain = tf.random.uniform((), 1.3, 4.4)
```
4. Comment out `image = mosaic(image)` ([line 139](https://github.com/timothybrooks/unprocessing/blob/master/unprocess.py#L139)) because the illumination estimation experiments need demosaiced images instead of bayer images.

### Syntax modifications
Some small syntax modifications were needed to adjust the code to work with TensorFlow 2.0. Other fixes may work, too.
1. Define `tf.to_float = lambda x: tf.cast(x, tf.float32)` (`tf.to_float` is deprecated in tf v2.0)
2. Change `tf.random_uniform` to `tf.random.uniform`
3. Change `tf.random_normal` to `tf.random.normal`
4. Remove the `None` in `tf.name_scope(None, 'unprocess')`
5. Change `tf.matrix_inverse` to `tf.linalg.inv`