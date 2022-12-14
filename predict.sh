# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig8_mag0_up2/best.pt --output_path preds/1212-5/base_sig8_mag0_up2.json

# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig4_mag0_up2/best.pt --output_path preds/1212-5/base_sig4_mag0_up2.json
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig16_mag0_up2/best.pt --output_path preds/1212-5/base_sig16_mag0_up2.json

# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig8_mag2_up2/best.pt --output_path preds/1212-5/base_sig8_mag2_up2.json
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig8_mag4_up2/best.pt --output_path preds/1212-5/base_sig8_mag4_up2.json

# python predict.py --model_name vit_base_patch16_224 --n_upscales 3 --resume_path logs/1212-5/base_sig8_mag0_up3/best.pt --output_path preds/1212-5/base_sig8_mag0_up3.json
# python predict.py --model_name vit_base_patch16_224 --n_upscales 4 --resume_path logs/1212-5/base_sig8_mag0_up4/best.pt --output_path preds/1212-5/base_sig8_mag0_up4.json --device=cuda:1

# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-10/base_sig8_mag0_up2/best.pt --output_path preds/1212-10/base_sig8_mag0_up2.json
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-10/base_sig8_mag2_up2/best.pt --output_path preds/1212-10/base_sig8_mag2_up2.json
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-10/base_sig8_mag4_up2/best.pt --output_path preds/1212-10/base_sig8_mag4_up2.json
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --context --resume_path logs/1212-10/context_sig8_mag0_up2/best.pt --output_path preds/1212-10/context_sig8_mag0_up2.json --device=cuda:1
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --context --resume_path logs/1212-10/context_sig8_mag2_up2/best.pt --output_path preds/1212-10/context_sig8_mag2_up2.json --device=cuda:1
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --context --resume_path logs/1212-10/context_sig8_mag4_up2/best.pt --output_path preds/1212-10/context_sig8_mag4_up2.json --device=cuda:1

# python predict.py --model_name vit_base_patch16_224 --n_upscales 3 --resume_path logs/1212-20/base_sig8_mag2_up3/20.pt --output_path preds/1212-20/base_sig8_mag2_up3.json --device=cuda:1
# python predict.py --model_name vit_base_patch16_224 --n_upscales 3 --context --resume_path logs/1212-20/context_sig8_mag2_up3/20.pt --output_path preds/1212-20/context_sig8_mag2_up3.json --device=cuda:1
# python predict.py --model_name vit_base_patch16_224 --n_upscales 4 --context --resume_path logs/1212-20/context_sig8_mag2_up4/20.pt --output_path preds/1212-20/context_sig8_mag2_up4.json --device=cuda:1
python predict.py --model_name vit_base_patch16_224 --n_upscales 3 --context --resume_path logs/1212-20/context_sig8_mag4_up3/20.pt --output_path preds/1212-20/context_sig8_mag4_up3.json --device=cuda:1
