import os
from itertools import product

os.system('pwd')

base_params = {
    'model': [
        # 'ts2vec',
        'ts2vec_byol',
    ],
    'dataset': [
        'ETTh1',
        # 'ETTh2',
        # 'ETTm1',
        # 'ETTm2'
    ]
}

params = {
    '--seed': [
        1, 42,
        # 2, 3,
        # 6, 7,
    ],
    '--max-train-length': [
        3000,
        # 2000,
        # 1000,
        # 500
    ],
    '--pred-hidden-dims': [
        # 64,
        128,
        # 256,
        # 512
    ],
    '--lr': [
        # 0.001,
        3e-4
    ],
    '--proj-output-dims': [
        32,
        # 64
    ],
    '--arch': [
        # None,
        # 'lstm',
        # 'proj'
        'proj2'
    ],
    '--use-momentum': [
        '',
        None
    ],
    '--augs': [
        '',
        # None
    ],
    '--repr-dims': [
        # 128,
        None
    ]
}

params = {**params, **base_params}

param_names = list(params.keys())
param_values = list(params.values())
configurations = list(product(*param_values))
param_configs = [{param_names[i]: config[i] for i in range(len(param_names))} for config in configurations]
print(len(param_configs))

for i, cfg in enumerate(param_configs):
    print(i, cfg)
    cmd = (
        'python3 main.py ' +
        ' '.join([cfg[k] for k in ['dataset', 'model']]) + ' forecast_csv ' +
        ' '.join([f'{k} {v}' for k, v in cfg.items() if k.startswith('--') and v is not None]) +
        ' --train --eval'
        ' --hier-loss'
    )
    print(cmd)
    os.system(cmd)
