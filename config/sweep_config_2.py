sweep_config = {
        'method': 'random',  
        'metric': {
            'name': 'Iter',  
            'goal': 'minimize' 
        },
        'parameters': {
            'threshold': {              
                'min': 1e-6,
                'max': 1e-1
            },
            'ema_decay': {
                'min': 0.01,
                'max': 0.99
            },
        }
    }