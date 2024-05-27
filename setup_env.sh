conda create -n manus python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

mkdir submodules 
cd submodules
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git --recursive

cd diff-gaussian-rasterization
python -m setup.py install

cd simple-knn
python -m setup.py install

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