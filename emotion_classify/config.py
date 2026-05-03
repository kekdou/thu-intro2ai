MODEL_CONFIG = {
    'CNN': {
        'batch_size': 64, 
        'learning_rate': 1e-3, 
        'num_classes': 2, 
        'filter_sizes': [2, 3, 4],
        'num_filters': 100,
        'dropout': 0.5
    },
    'RNN': {
        'batch_size': 64, 
        'learning_rate': 5e-3, 
        'num_classes': 2, 
        'hidden_size': 64,
        'num_layers': 2,
        'bidirectional': True,
        'dropout': 0.3
    },
    'MLP': {
        'batch_size': 32, 
        'learning_rate': 1e-3, 
        'num_classes': 2, 
        'hidden_dims': [32, 16],
        'dropout': 0.2
    },
    'ATTENTION': {
        'batch_size': 64, 
        'learning_rate': 5e-3, 
        'num_classes': 2, 
        'hidden_size': 64, 
        'num_layers': 2, 
        'dropout': 0.3
    },
    'TRANSFORMER': {
        'batch_size': 64, 
        'learning_rate': 1e-3, 
        'num_classes': 2, 
        'n_heads': 2,
        'n_layers': 3,
        'dropout': 0.4
    }
}