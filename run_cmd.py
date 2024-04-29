import json
import os
from itertools import product

os.system('pwd')

with open('configs.json', 'r') as f:
    id_configs = json.load(f)
for cfg in id_configs.values():
    for k in list(cfg.keys()):
        # if cfg[k] == 0:
        if cfg[k] is None:
            del cfg[k]
        if k == 'hier_loss':
            del cfg[k]

configs = set()
# configs.add(14)
# configs.add(22)

base_params = {
    'model': [
        # 'ts2vec',
        'ts2vec_byol',
    ],
    'dataset': [
        # 'ETTh1',
        'ETTh2',
        # 'ETTm1',
        # 'ETTm2'
    ]
}

if len(configs) > 0:
    keys = set()
    for c in configs:
        keys.update(id_configs[str(c)].keys())
    params = {f'--{k.replace("_", "-")}': list(set([id_configs[str(c)][k] for c in configs])) for k in keys}
    print(params)
else:
    params = {
        # '--epochs': [
        #     400,
        #     # 600,
        #     # 1000
        # ],
        '--seed': [
            1,
            42,
            # 2, 3,
            # 6, 7,
        ],
        '--max-train-length': [
            3000,
            # 2000,
            # 1000, 500
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
            'lstm',
            # 'proj'
        ],
        '--use-momentum': [
            '',
            None
        ],
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
        ' --augs'
    )
    print(cmd)
    os.system(cmd)
