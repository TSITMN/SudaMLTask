# 42 510 5424 6657 74751
# "lstm_attention" "lstm_cnn" "lstm_attention_cnn"

for model_type in "lstm_cnn"
do
    for seed in 42 510 5424 6657 74751
    do
    # o96
        # python train_lstm.py    --seed $seed \
        #                         --model_type=$model_type \
        #                         --num_layers=1 \
        #                         --input_seq_len=96 \
        #                         --output_seq_len=96 \
        #                         --input_feature_len=12 \
        #                         --hidden_size=512 \
        #                         --batch_size=128 \
        #                         --num_epochs=100 \
        #                         --learning_rate=0.0005 \
        #                         --model_suffix=${model_type}_o96_s${seed}_lr0.0005 \
        #                         --gpu=2
        # python evaluate_lstm.py --seed $seed \
        #                         --model_type=$model_type \
        #                         --num_layers=1 \
        #                         --input_seq_len=96 \
        #                         --output_seq_len=96 \
        #                         --input_feature_len=12 \
        #                         --hidden_size=512 \
        #                         --batch_size=128 \
        #                         --model_suffix=${model_type}_o96_s${seed}_lr0.0005 \
        #                         --gpu=2
    # o240
        python train_lstm.py    --seed $seed \
                                --model_type=$model_type \
                                --num_layers=2 \
                                --input_seq_len=96 \
                                --output_seq_len=240 \
                                --input_feature_len=12 \
                                --hidden_size=512 \
                                --batch_size=128 \
                                --num_epochs=100 \
                                --learning_rate=0.0005 \
                                --model_suffix=${model_type}_o240_s${seed}_lr0.0005 \
                                --gpu=3
        python evaluate_lstm.py --seed $seed \
                                --model_type=$model_type \
                                --num_layers=2 \
                                --input_seq_len=96 \
                                --output_seq_len=240 \
                                --input_feature_len=12 \
                                --hidden_size=512 \
                                --batch_size=128 \
                                --model_suffix=${model_type}_o240_s${seed}_lr0.0005 \
                                --gpu=3
    done
done