conda create -n python3.12-jax-gpu python=3.12 -y
conda activate python3.12-jax-gpu

conda install -c nvidia cuda-toolkit -y

pip install matplotlib pandas arch tsfeatures utilsforecast numba

# Install GPU supported jax
pip install --upgrade "jax[cuda13]"   

## If no Nvidia GPU, use CPU instead
# pip install --upgrade jax   
