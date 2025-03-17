conda create -n manus python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

git submodule update --init --recursive

cd submodules/diff-gaussian-rasterization
pip install -e .
cd ../..

cd submodules/simple-knn
pip install -e .
cd ../..
pip install pytorch-lightning==2.0.1

## Other pip packages
pip install easydict
pip install joblib
pip install opencv-python
pip install natsort
pip install omegaconf
pip install h5py
pip install scikit-learn
pip install termcolor
pip install imageio
pip install trimesh
pip install setuptools==59.5.0
pip install plotly
pip install hydra-core --upgrade
pip install matplotlib
pip install pymeshlab
pip install scikit-image
pip install taichi
pip install lpips
pip install colorama