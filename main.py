import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default='')
parser.add_argument('--netname', type=str, default='')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--db_path', type=str, default='')
parser.add_argument('--reg', type=str, default='kl')
parser.add_argument('--denoise_train', dest='denoise_train', action='store_true',
                    help='Use denoise training by adding Gaussian/salt and pepper noise')
parser.add_argument('--plot_reconstruction', dest='plot_reconstruction', action='store_true',
                    help='Plot reconstruction')
parser.add_argument('--use_gui', dest='use_gui', action='store_true',
                    help='Display the results with a GUI window')
parser.add_argument('--vis_frequency', type=int, default=1000,
                    help='How many train batches before we perform visualization')
parser.add_argument('--iterations', type=int, default=5000)
parser.add_argument('--restart', type=bool, default=False)
args = parser.parse_args()

import matplotlib
if not args.use_gui:
    matplotlib.use('Agg')
else:
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show()

import os
from dataset import *
from vlae import VLadder
from trainer import NoisyTrainer

if args.gpus is not '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

dataset = MnistDataset()

model = VLadder(dataset, name=args.netname, reg=args.reg, batch_size=args.batch_size, restart=args.restart)
trainer = NoisyTrainer(model, dataset, args)
trainer.train()
# TODO remove
#if args.no_train:
#    trainer.visualize()
#else:
#    trainer.train()