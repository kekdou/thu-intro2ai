from .cnn import TextCNN
from .rnn import TextRNN
from .mlp import MLP
from .attention import TextAttnRNN
from .transformer import Transformer

def get_model(model_name, embedding_matrix, **kwargs):
    if model_name == 'CNN':
        return TextCNN(embedding_matrix, **kwargs)
    elif model_name == 'RNN':
        return TextRNN(embedding_matrix, **kwargs)
    elif model_name == 'MLP':
        return MLP(embedding_matrix, **kwargs)
    elif model_name == "ATTENTION":
        return TextAttnRNN(embedding_matrix, **kwargs)
    elif model_name == 'TRANSFORMER':
        return Transformer(embedding_matrix, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")