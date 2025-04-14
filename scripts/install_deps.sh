pip install --force-reinstall ninja
mkdir build
cd build
git clone https://github.com/kipgparker/flash-attention.git kip-flash
cd kip-flash
MAX_JOBS=4 python setup.py install
cd ..
git clone git@github.com:refinement-labs/normalization.git normalization
cd normalization
pip install -e .
cd ..
git clone git@github.com:NVIDIA/apex apex
cd apex
python setup.py install --cuda_ext --cpp_ext --distributed_adam --deprecated_fused_adam
cd ../..