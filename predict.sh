# python predict.py --model_name vit_base_patch16_224 --resume_path logs/base/vit_base_patch16_224_4/best.pt --output_path preds/base/vit_base_patch16_224_4.json
# python predict.py --model_name vit_base_patch16_224 --resume_path logs/base/vit_base_patch16_224_8/best.pt --output_path preds/base/vit_base_patch16_224_8.json
# python predict.py --model_name vit_base_patch16_224 --resume_path logs/base/vit_base_patch16_224_16/best.pt --output_path preds/base/vit_base_patch16_224_16.json
# python predict.py --model_name vit_base_patch16_224 --resume_path logs/randaugment/vit_base_patch16_224_8_2_up2/best.pt --output_path preds/randaugment/vit_base_patch16_224_8_2_up2.json
# python predict.py --model_name vit_base_patch16_224 --resume_path logs/randaugment/vit_base_patch16_224_16_2_up2/best.pt --output_path preds/randaugment/vit_base_patch16_224_16_2_up2.json
# python predict.py --model_name vit_base_patch16_224 --resume_path logs/randaugment/vit_base_patch16_224_32_2_up3/best.pt --output_path preds/randaugment/vit_base_patch16_224_32_2_up3.json
# python predict.py --model_name vit_base_patch16_224 --resume_path logs/randaugment/vit_base_patch16_224_32_2_up4/best.pt --output_path preds/randaugment/vit_base_patch16_224_32_2_up4.json

python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1209/base_sig4_mag0_up2/best.pt --output_path preds/1209/base_sig4_mag0_up2.json
python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1209/base_sig8_mag0_up2/best.pt --output_path preds/1209/base_sig8_mag0_up2.json
python predict.py --model_name vit_base_patch16_224 --n_upscales 2 --resume_path logs/1209/base_sig16_mag0_up2/best.pt --output_path preds/1209/base_sig16_mag0_up2.json

