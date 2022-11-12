@REM python predict.py --model_name hrnet_w18 --resume_path logs/20/hrnet_w18/8.pt --output_path preds/20/val/hrnet_w18.json
@REM python predict.py --model_name hrnet_w32 --resume_path logs/20/hrnet_w32/8.pt --output_path preds/20/val/hrnet_w32.json
@REM python predict.py --model_name hrnet_w48 --resume_path logs/20/hrnet_w48/8.pt --output_path preds/20/val/hrnet_w48.json
@REM python predict.py --model_name hrnet_w64 --resume_path logs/20/hrnet_w64/8.pt --output_path preds/20/val/hrnet_w64.json
@REM python predict.py --model_name vit_base_patch16_224_dino --resume_path logs/20/vit_b16/16.pt --output_path preds/20/val/vit_b16.json
python predict.py --model_name vit_small_patch16_224_dino --resume_path logs/20/vit_s16/16.pt --output_path preds/20/val/vit_s16.json
