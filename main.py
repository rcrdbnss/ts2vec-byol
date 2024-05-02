import argparse
import datetime
import json
import os
import pickle
import re
import time

import numpy as np
import torch.cuda

from ts2vec import TS2Vec
from ts2vec import datautils
from ts2vec import tasks
from ts2vec.utils import init_dl_program
from ts2vec_byol import TS2VecBYOL, TS2VecBYOLAugs
from utils import Paths

# def save_checkpoint_callback(
#         save_every=1,
#         unit='epoch'
# ):
#     assert unit in ('epoch', 'iter')
#
#     def callback(model, loss):
#         n = model.n_epochs if unit == 'epoch' else model.n_iters
#         if n % save_every == 0:
#             model.save(f'{run_dir}/model_{n}.pkl')
#
#     return callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    # parser.add_argument('run_name',
    #                     help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('model', choices=['ts2vec', 'ts2vec_byol'],
                        help='Which training model to use. It can be either the original one, provided by TS2Vec, or the one based on BYOL.')
    parser.add_argument('loader', type=str,
                        help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (defaults to 0.0003)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    # parser.add_argument('--save-every', type=int, default=None,
    #                     help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--train', action="store_true",
                        help='Whether to train a new model. If not set, the model would be loaded from the saved file')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    # parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--use-momentum', action='store_true',
                        help='If set, use the SimSiam-based training architecture. Otherwise, use BYOL')
    parser.add_argument('--hier-loss', action='store_true',
                        help='If set, compute the hierarchical loss with the architecture-specific loss function. Otherwise, compute the architecture-specific loss only')
    parser.add_argument('--pred-hidden-dims', type=int, default=128,
                        help='The hidden dimension of the predictor (defaults to 128)')
    parser.add_argument('--augs', action='store_true',
                        help='If set, augment time series with scaling, shifting and jittering. Otherwise, compare the overlapping section of two crops.')
    parser.add_argument('--arch', type=str, choices=['lstm', 'proj', 'proj2'], default=None,
                        help='The training architecture. It can be lstm, proj or proj2. If not set, it defaults to the simple architecture with only the encoder and the predictor')
    parser.add_argument('--proj-hidden-dims', type=int, default=128,
                        help='The hidden dimension of the projector (defaults to 128)')
    parser.add_argument('--proj-output-dims', type=int, default=64,
                        help='The output dimension of the projector (defaults to 64)')

    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads,
                             deterministic=True)

    torch.cuda.empty_cache()

    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)

    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)

    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(
            args.dataset)
        train_data = data[:, train_slice]

    elif args.loader == 'forecast_random':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols = datautils.random_data(1, 250, 5)
        pred_lens = [10]
        train_data = data[:, train_slice]
        args.dataset = 'random'
        args.max_train_length = 50

    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(
            args.dataset, univar=True)
        train_data = data[:, train_slice]

    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(
            args.dataset)
        train_data = data[:, train_slice]

    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(
            args.dataset, univar=True)
        train_data = data[:, train_slice]

    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(
            args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)

    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(
            args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')

    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    # if args.irregular > 0:
    #     if task_type == 'classification':
    #         train_data = data_dropout(train_data, args.irregular)
    #         test_data = data_dropout(test_data, args.irregular)
    #     else:
    #         raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    # print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        arch=args.arch,
    )

    if args.model == 'ts2vec':
        model_cls = TS2Vec
    else:
        if args.augs:
            model_cls = TS2VecBYOLAugs
        else:
            model_cls = TS2VecBYOL
        # adjust the model name to differentiate result folders
        if not args.use_momentum:
            args.model = args.model.replace('byol', 'siam')
        if not args.arch is None:
            args.model = f'{args.model}_{args.arch}'
        if args.augs:
            args.model = f'{args.model}_augs'
    print('Model setup:', args.model)

    if args.model != 'ts2vec':
        config['pred_hidden_dims'] = args.pred_hidden_dims
        config['hier_loss'] = args.hier_loss
        config['proj_hidden_dims'] = args.proj_hidden_dims
        config['proj_output_dims'] = args.proj_output_dims
        config['use_momentum'] = args.use_momentum

    model = model_cls(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )

    # if args.save_every is not None:
    #     unit = 'epoch' if args.epochs is not None else 'iter'
    #     config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    args.model_name = 'model.pt'
    _paths = Paths(args)

    t = time.time()

    if args.train:
        loss = model.fit(
            train_data,
            n_epochs=args.epochs,
            n_iters=args.iters,
        )

        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

        model_path = _paths.model_path()
        model.save(model_path)
        print('Model saved in', os.path.abspath(model_path))

        name, ext = os.path.splitext(args.model_name)
        mt = re.match(fr'.*{name}(_\d+){ext}$', model_path)
        suffix = "" if mt is None else mt.group(1)
        model_dir = os.path.dirname(model_path)
        with open(f'{model_dir}/loss{suffix}.pkl', 'wb') as f:
            pickle.dump(loss, f)
    else:
        print('Skip training.')

    if args.eval:
        model.load(_paths.model_path())
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels,
                                                      eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens,
                                                   n_covariate_cols)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps,
                                                         all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels,
                                                                   all_train_timestamps, all_test_data, all_test_labels,
                                                                   all_test_timestamps, delay)
        else:
            assert False

        with open(f'{_paths.res_dir_path()}/args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        with open(f'{_paths.res_dir_path()}/eval_res.json', 'w') as f:
            json.dump(eval_res, f, indent=2)
        print('Result saved in', os.path.abspath(_paths.res_dir_path()))
        print('A peek:', np.mean([x['norm']['MSE'] for x in eval_res['ours'].values()]))

    print("Finished.")
