# python train.py --sigma 8 --magnitude 0 --n_upscales 2 --log_dir logs/1212-5/base_sig8_mag0_up2 --n_epochs 5 --device cuda:0

# python train.py --sigma 4 --magnitude 0 --n_upscales 2 --log_dir logs/1212-5/base_sig4_mag0_up2 --n_epochs 5 --device cuda:0
# python train.py --sigma 16 --magnitude 0 --n_upscales 2 --log_dir logs/1212-5/base_sig16_mag0_up2 --n_epochs 5 --device cuda:0

# python train.py --sigma 8 --magnitude 2 --n_upscales 2 --log_dir logs/1212-5/base_sig8_mag2_up2 --n_epochs 5 --device cuda:0
# python train.py --sigma 8 --magnitude 4 --n_upscales 2 --log_dir logs/1212-5/base_sig8_mag4_up2 --n_epochs 5 --device cuda:0

# python train.py --sigma 8 --magnitude 0 --n_upscales 3 --log_dir logs/1212-5/base_sig8_mag0_up3 --n_epochs 5 --device cuda:0
# python train.py --sigma 8 --magnitude 0 --n_upscales 4 --log_dir logs/1212-5/base_sig8_mag0_up4 --n_epochs 5 --device cuda:0

# python train.py --sigma 8 --magnitude 0 --n_upscales 2 --log_dir logs/1212-10/base_sig8_mag0_up2 --n_epochs 10 --device cuda:0
# python train.py --sigma 8 --magnitude 2 --n_upscales 2 --log_dir logs/1212-10/base_sig8_mag2_up2 --n_epochs 10 --device cuda:0
# python train_context.py --sigma 8 --magnitude 4 --n_upscales 2 --log_dir logs/1212-10/base_sig8_mag4_up2 --n_epochs 10 --device cuda:0 --context False

# python train_context.py --sigma 8 --magnitude 2 --n_upscales 3 --log_dir logs/1212-20/context_sig8_mag2_up3 --n_epochs 80 --device cuda:0 --context True --f_save=10

python train.py --sigma 8 --magnitude 2 --n_upscales 3 --log_dir logs/1212-20/base_sig8_mag2_up3 --n_epochs 20 --device cuda:1 --f_save=10