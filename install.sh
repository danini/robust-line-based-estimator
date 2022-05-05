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

# Install the robust line-based estimator
pip install -e .