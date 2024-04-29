import os

from configs import Configurations


def uniquify(path, dir=False):
    counter = 1
    base_path = path

    while True:
        if dir:
            path = f'{base_path}/{counter}'
        else:
            basename, ext = os.path.splitext(base_path)
            path = f'{basename}_{counter}{ext}'
        if not os.path.exists(path):
            break
        counter += 1

    return path, counter


class Paths:

    def __init__(self, args):
        CONFIGS = Configurations()
        cfg_id, new = CONFIGS.get(**args.__dict__)
        if new:
            CONFIGS.save()
        print(f'Config ID: {cfg_id} ({"new" if new else "existing"})')

        dir = f'{args.model}/{args.dataset}/cfg{cfg_id:02d}/seed{args.seed}'
        model_dir = f'./models/{dir}'
        res_dir_path, _ = uniquify(f'./results/{dir}', dir=True)

        self._model_dir = model_dir
        self._res_dir_path = res_dir_path

        self._model_path = f'{model_dir}/{args.model_name}'
        if args.train:
            # self._model_path = f'{model_dir}/{args.model_name}'
            # if os.path.exists(self._model_path):
            #     self._model_path = uniquify(self._model_path)
            self._model_path, c = uniquify(self._model_path)

    def model_path(self):
        # model_dir = os.path.dirname(self._model_path)
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        return self._model_path

    def res_dir_path(self):
        if not os.path.exists(self._res_dir_path):
            os.makedirs(self._res_dir_path)
        return self._res_dir_path
