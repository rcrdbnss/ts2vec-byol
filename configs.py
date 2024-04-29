import json
import os.path
from collections import namedtuple


def find_smallest_missing_number(numbers):
    numbers_set = set(numbers)
    for i in range(1, len(numbers) + 2):
        if i not in numbers_set:
            return i


class Configurations:
    _instance = None
    _fpath = './configs.json'
    Config = namedtuple('Config', [
        'max_train_length', 'epochs', 'hier_loss', 'lr', 'repr_dims', 'proj_hidden_dims', 'proj_output_dims',
        'iters', 'pred_hidden_dims',
        # 'augs'
    ])
    defaults = {
        'max_train_length': 3000,
        'epochs': None,
        'iters': None,
        'pred_hidden_dims': 128,
        'hier_loss': False,
        'lr': 1e-3,
        'repr_dims': 320,
        # 'augs': False,
        'proj_hidden_dims': 128,
        'proj_output_dims': 64,
    }

    def __new__(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._configs = dict()
        self.load()

    def load(self):
        if not os.path.exists(self._fpath):
            self._configs = dict()
        else:
            print(f'Loading configurations from {os.path.abspath(self._fpath)}')
            with open(self._fpath, 'r') as f:
                self._configs = {self.Config(**v, **{
                    dk: dv for dk, dv in self.defaults.items() if dk not in v
                }): int(k) for k, v in json.load(f).items()}

    def exists(self, **kwargs):
        kwargs = {**kwargs, **{
            dk: dv for dk, dv in self.defaults.items() if dk not in kwargs
        }}
        key = self.Config(**{f: kwargs[f] for f in self.Config._fields})
        return self._configs.get(key, None) is not None

    def get(self, **kwargs):
        key = self.Config(**{f: kwargs[f] for f in self.Config._fields})
        new = not self.exists(**kwargs)
        if new:
            self._configs[key] = find_smallest_missing_number(list(self._configs.values())) # len(self._configs) + 1
        return self._configs[key], new

    def save(self):
        with open(self._fpath, 'w') as f:
            json.dump({v: k._asdict() for k, v in self._configs.items()}, f, indent=2)

    def clear(self):
        self._configs = dict()
