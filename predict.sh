# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig8_mag0_up2/best.pt --output_path preds/1212-5/base_sig8_mag0_up2.json

# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig4_mag0_up2/best.pt --output_path preds/1212-5/base_sig4_mag0_up2.json
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig16_mag0_up2/best.pt --output_path preds/1212-5/base_sig16_mag0_up2.json

# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig8_mag2_up2/best.pt --output_path preds/1212-5/base_sig8_mag2_up2.json
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-5/base_sig8_mag4_up2/best.pt --output_path preds/1212-5/base_sig8_mag4_up2.json

# python predict.py --model_name vit_base_patch16_224 --n_upscales 3 --resume_path logs/1212-5/base_sig8_mag0_up3/best.pt --output_path preds/1212-5/base_sig8_mag0_up3.json
# python predict.py --model_name vit_base_patch16_224 --n_upscales 4 --resume_path logs/1212-5/base_sig8_mag0_up4/best.pt --output_path preds/1212-5/base_sig8_mag0_up4.json --device=cuda:1

# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-context/base_sig8_mag0_up2/best.pt --output_path preds/1212-context/base_sig8_mag0_up2.json --device=cuda:1 --context True
# python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-context/base_sig8_mag2_up2/best.pt --output_path preds/1212-context/base_sig8_mag2_up2.json --device=cuda:1 --context True
python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-context/base_sig8_mag4_up2/best.pt --output_path preds/1212-context/base_sig8_mag4_up2.json --device=cuda:1 --context True

python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-10/base_sig8_mag0_up2/best.pt --output_path preds/1212-10/base_sig8_mag0_up2.json
python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-10/base_sig8_mag2_up2/best.pt --output_path preds/1212-10/base_sig8_mag2_up2.json
python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1212-10/base_sig8_mag4_up2/best.pt --output_path preds/1212-10/base_sig8_mag4_up2.json