# 42 510 5424 6657 74751
#96 240

for seed in 42 510 5424 6657 74751 
do
    # for l in 2 3 4
    # do
    #     python train_lstm.py    --seed $seed \
    #                             --num_layers=$l \
    #                             --input_seq_len=96 \
    #                             --output_seq_len=240 \
    #                             --input_feature_len=12 \
    #                             --hidden_size=512 \
    #                             --batch_size=128 \
    #                             --num_epochs=100 \
    #                             --learning_rate=0.0005 \
    #                             --model_suffix=o240_s${seed}_l${l}_lr0.0005 \
    #                             --gpu=0
        
    #     python evaluate_lstm.py --seed $seed \
    #                             --num_layers=$l \
    #                             --input_seq_len=96 \
    #                             --output_seq_len=240 \
    #                             --input_feature_len=12 \
    #                             --hidden_size=512 \
    #                             --batch_size=128 \
    #                             --model_suffix=o240_s${seed}_l${l}_lr0.0005 \
    #                             --gpu=0
    # done
    python train_lstm.py    --seed $seed \
                            --num_layers=1 \
                            --input_seq_len=96 \
                            --output_seq_len=240 \
                            --input_feature_len=12 \
                            --hidden_size=512 \
                            --batch_size=128 \
                            --num_epochs=100 \
                            --learning_rate=0.0005 \
                            --model_suffix=o240_s${seed}_lr0.0005 \
                            --gpu=3
    
    python evaluate_lstm.py --seed $seed \
                            --num_layers=1 \
                            --input_seq_len=96 \
                            --output_seq_len=240 \
                            --input_feature_len=12 \
                            --hidden_size=512 \
                            --batch_size=128 \
                            --model_suffix=o240_s${seed}_lr0.0005 \
                            --gpu=3
done

# python train_lstm.py    --seed 5424 \
#                         --num_layers=1 \
#                         --input_seq_len=96 \
#                         --output_seq_len=96 \
#                         --input_feature_len=12 \
#                         --hidden_size=512 \
#                         --batch_size=128 \
#                         --num_epochs=100 \
#                         --learning_rate=0.0005 \
#                         --model_suffix=o96_s5424_lr0.0005 \
#                         --gpu=0

# python evaluate_lstm.py --seed 5424 \
#                         --num_layers=1 \
#                         --input_seq_len=96 \
#                         --output_seq_len=96 \
#                         --input_feature_len=12 \
#                         --hidden_size=512 \
#                         --batch_size=128 \
#                         --model_suffix=o96_s5424_lr0.0005 \
#                         --gpu=0