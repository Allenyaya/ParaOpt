sweep_config = {
        'method': 'random',  
        'metric': {
            'name': 'Iter',  
            'goal': 'minimize' 
        },
        'parameters': {
            'P': {  
                'distribution':'q_uniform',
                'q': 1,
                'min': 1,
                'max': 20,                
            },
        }
    }