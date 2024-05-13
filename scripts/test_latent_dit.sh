export CUDA_VISIBLE_DEVICES=1

# python latent.py --model_type dit_latent --data_type interpolated_uiuc --checkpoint_path /home/bingxing2/ailab/scxlab0058/airfoil/DiffAirfoil-dev/new_weights/dit_latent-interpolated_uiuc-/ckpt_epoch_5000.pth

python latent.py --model_type dit_latent --data_type supercritical_airfoil --checkpoint_path /home/bingxing2/ailab/scxlab0058/airfoil/DiffAirfoil-dev/new_weights/dit_latent-supercritical_airfoil-/ckpt_epoch_5000.pth
