python jpgs_to_h5.py --jpgs_dir data/val --json_path data/val_annotation.json --h5_path data/val_384.h5 --size 384
python jpgs_to_h5.py --jpgs_dir data/train --json_path data/train_annotation.json --h5_path data/train_384.h5 --size 384
python jpgs_to_h5.py --jpgs_dir data/test --json_path data/test_prediction.json --h5_path data/test_384.h5 --size 384