# Install required Python packages
pip install -r requirements.txt

# Install SOLD2 dependency
pip install -e third_party/SOLD2
mkdir -p third_party/SOLD2/pretrained_models/
wget https://www.polybox.ethz.ch/index.php/s/blOrW89gqSLoHOk/download -P third_party/SOLD2/pretrained_models/
mv third_party/SOLD2/pretrained_models/download third_party/SOLD2/pretrained_models/sold2_wireframe.tar

# Install PytLSD
pip install -e third_party/pytlsd

# Install PytLBD
pip install -e third_party/pytlbd

# Install Progressive-X
pip install -e third_party/progressive-x

# Install the robust line-based estimator
pip install -e .

# Download the pre-trained model for SuperPoint
echo "Downloading SuperPoint model..."
mkdir -p robust_line_based_estimator/line_matching/weights
wget -O superpoint_v1.pth https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth?raw=true
mv superpoint_v1.pth robust_line_based_estimator/line_matching/weights/

# Download the pre-trained model for SuperGlue
echo "Downloading SuperGlue models..."
wget -O superglue_indoor.pth https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_indoor.pth?raw=true
wget -O superglue_outdoor.pth https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_outdoor.pth?raw=true
mv superglue_indoor.pth robust_line_based_estimator/line_matching/weights/
mv superglue_outdoor.pth robust_line_based_estimator/line_matching/weights/

# Install the pre-trained model for GlueStick
echo "Downloading GlueStick model..."
mkdir -p robust_line_based_estimator/line_matching/weights
wget -O gluestick.tar https://www.polybox.ethz.ch/index.php/s/SVE9A3rl2wewuV5/download
mv gluestick.tar robust_line_based_estimator/line_matching/weights/
