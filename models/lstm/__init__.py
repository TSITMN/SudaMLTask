from models.lstm.lstm import Seq2seqLSTM
from models.lstm.lstm_improve import Seq2seqLSTMWithCNN, Seq2seqLSTMWithAttention, Seq2seqLSTMWithAttentionCNN

def get_model(model_type, **kwargs):
    """
    According to the model type, return the corresponding model instance.
    
    Args:
    - model_type: model type, optional values are "lstm", "lstm_attention", "lstm_cnn", "lstm_attention_cnn".
    - kwargs: initialization parameters of the model.
    """
    # Model type and class mapping
    model_classes = {
        "lstm": Seq2seqLSTM,
        "lstm_attention": Seq2seqLSTMWithAttention,
        "lstm_cnn": Seq2seqLSTMWithCNN,
        "lstm_attention_cnn": Seq2seqLSTMWithAttentionCNN,
    }

    # Check if the model type is supported
    if model_type not in model_classes:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types are: {list(model_classes.keys())}"
        )

    # Return the corresponding model instance
    return model_classes[model_type](**kwargs)