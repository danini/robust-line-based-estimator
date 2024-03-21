# Code cleaning will come soon. Stay tuned!

# robust-line-based-estimator

## TODOs

* [ ] Local Optimization 
* [ ] Advanced sampling

## New Installation needed for the solver part

```
git submodule update --init --recursive
mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=${PATH_TO_YOUR_PYTHON_EXECUTABLE} ..
make -j4
cd ..
```

## Installation

Clone the repository and its submodules:
```
git clone --recurse-submodules git@github.com:danini/robust-line-based-estimator.git
```

Make sure that you have the necessary OpenCV libraries installed:
```
sudo apt-get install libopencv-dev libopencv-contrib-dev libarpack++2-dev libarpack2-dev libsuperlu-dev
```

Install the necessary requirements, third party libraries, and pre-trained models:
```
bash install.sh
```

## Evaluation - Python
Data preparation is kept the same. And then run:
```
python runners/run_scannet.py
```

## Evaluation on the PhotoTourism dataset
Download the data from the CVPR tutorial "RANSAC in 2020":
```
wget http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar
tar -xf  RANSAC-Tutorial-Data-EF.tar
```

Then run the notebook `examples/relative_pose_evaluation_phototourism.ipynb`.

## Evaluation on the ScanNet dataset
Download the data from the test set for relative pose estimation used in SuperGlue (~250Mb for 1500 image pairs only):
```
wget https://www.polybox.ethz.ch/index.php/s/lAZyxm62WUh27Zl/download
unzip ScanNet_test.zip -d <path to extract the ScanNet test set>
```

Then run the notebook `examples/relative_pose_evaluation_scannet.ipynb`.


## Evaluation on the 7Scenes dataset
Download the [7Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and put it where it suits you. You can also only download one scene and later specify this scene in the dataloader constructor.

Then run the script `runners/run_7scenes.py`.


## Evaluation on the ETH3D dataset
Download the [ETH3D dataset](https://www.eth3d.net/datasets) (training split of the high-res multi-view, undistorted images + GT extrinsics & intrinsics should be enough) and put it where it suits you. The input argument 'downsize_factor' can be used to downscale the images, because they can be quite large otherwise.

Then run the script `runners/run_eth3d.py`.


## Evaluation on the LaMAR dataset
Download the [CAB scene of the LaMAR dataset](https://cvg-data.inf.ethz.ch/lamar/CAB.zip), and unzip it to your favourite location. Note that we only use the images in `CAB/sessions/query_val_hololens`.


## Evaluation on the KITTI dataset
Download the [KITTI odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (grayscale images and poses), and unzip them to your favourite location.
