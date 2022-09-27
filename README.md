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

Install the necessary requirements and third party libraries:
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
wget https://www.polybox.ethz.ch/index.php/s/YhVDuO4fDCNM79A/download
unzip ScanNet_test.zip -d <path to extract the ScanNet test set>
```

Then run the notebook `examples/relative_pose_evaluation_scannet.ipynb`.
