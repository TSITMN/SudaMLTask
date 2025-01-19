i=0
GPU=1
SEED=42
type="periodcpos"

python train_other_transformer.py \
    --model_dim=512 \
    --num_heads=8 \
    --num_encoder_layers=1 \
    --num_decoder_layers=1 \
    --output_window=96 \
    --batch_size=32 \
    --num_epochs=100 \
    --learning_rate=1e-4 \
    --train_data_path="data/train_data.csv" \
    --val_data_path="data/val_data.csv" \
    --save_path="save/transformer/" \
    --model_suffix="${type}_96_${i}" \
    --gpu=$GPU \
    --seed=$SEED \
    --model_type="${type}_transformer"

python test_other_transformer.py \
    --model_dim=512 \
    --num_heads=8 \
    --num_encoder_layers=1 \
    --num_decoder_layers=1 \
    --output_window=96 \
    --batch_size=64 \
    --test_data_path="data/original/test_data.csv" \
    --train_data_path="data/train_data.csv" \
    --model_suffix="${type}_96_${i}" \
    --gpu=$GPU \
    --seed=$SEED \
    --model_type="${type}_transformer"


python train_other_transformer.py \
    --model_dim=512 \
    --num_heads=8 \
    --num_encoder_layers=1 \
    --num_decoder_layers=1 \
    --output_window=240 \
    --batch_size=24 \
    --num_epochs=100 \
    --learning_rate=1e-4 \
    --train_data_path="data/train_data.csv" \
    --val_data_path="data/val_data.csv" \
    --save_path="save/transformer/" \
    --model_suffix="${type}_240_${i}" \
    --gpu=$GPU \
    --seed=$SEED \
    --model_type="${type}_transformer"

python test_other_transformer.py \
    --model_dim=512 \
    --num_heads=8 \
    --num_encoder_layers=1 \
    --num_decoder_layers=1 \
    --output_window=240 \
    --batch_size=64 \
    --test_data_path="data/original/test_data.csv" \
    --train_data_path="data/train_data.csv" \
    --model_suffix="${type}_240_${i}" \
    --gpu=$GPU \
    --seed=$SEED \
    --model_type="${type}_transformer"