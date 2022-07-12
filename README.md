# robust-line-based-estimator

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

## Evaluation on the PhotoTourism dataset
Download the data from the CVPR tutorial "RANSAC in 2020":
```
wget http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar
tar -xf  RANSAC-Tutorial-Data-EF.tar
```

Then run the notebook `examples/relative_pose_evaluation_phototourism.ipynb`.