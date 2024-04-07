# DiffAirfoil:  Editable and Controllable Airfoil Generation with Diffusion

## Installation

```bash
conda create --name airfoil python=3.8
conda activate airfoil
pip install -r requirements.txt
```

## Dataset

请将数据集存放在 `data` 文件夹下, 默认数据集为 `data/airfoil/supercritical_airfoil/*.dat`

在项目的跟文件夹下:

```bash
# split train/val/test
python dataload/datasplit.py 

# generate parsec feature
python dataload/parsec_direct.py 
```

## Usage

在项目的根文件夹下:

```bash
# train vae, map airfoil to latent space
python train_vae.py

# train dit condition on keypoint & parsec
python train_dit.py
```



## Resources

I have collected some papers on airfoil design. The [resources are available here](https://github.com/hitcslj/awesome-airfoil-design).

I have created an airfoil editing demo using Gradio. The [demo can be found here](https://github.com/hitcslj/airfoil-demo).

I have also built an airfoil editing baseline using VAE. The [code is available here](https://github.com/hitcslj/Airfoil)

## Acknowledgements

I have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.

- [DDPM](https://github.com/abarankab/DDPM)
- [DIT](https://github.com/facebookresearch/DiT)


## Citations