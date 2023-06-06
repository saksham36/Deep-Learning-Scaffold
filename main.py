
#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 04/28/2023
Cleaned up main.py
'''
import os
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.general_tools import get_args, backup_files
from utils.plotting import simple_plot, plot_labels
from utils.train_tools import get_train_valid_test_data
from utils.dataset import sEMGDataset

from model import GRU_MLP_Softmax
from trainer import Trainer
import pdb


def main(args):
    time_string = args.experiment_name + '-' + \
        time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    # Load data
    dataset = sEMGDataset(args)
    train_dataset, valid_dataset, test_dataset, class_weights = get_train_valid_test_data(dataset,
                                                                                          args)
    # Plotting
    if args.plot:
        simple_plot(dataset.data, "60bpm", "Time", "Voltage",
                    "data/plots/60bpm.png", args.sampling_rate)

        plot_labels(dataset.labels, dataset.data, "60bpm", "Time", "Voltage",
                    "data/plots/60bpm_labeled.png", args.sampling_rate)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = GRU_MLP_Softmax(input_dim=args.input_dim,
                            hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers,
                            output_dim=args.output_dim,
                            dropout_prob=args.dropout)

    # Load the trained model, if needed
    if args.load_model or args.eval_only:
        print("Loading model from", args.model_path)
        try:
            model.load_state_dict(torch.load(args.model_path))
        except FileNotFoundError:
            print("Model not found, exiting")
            exit(1)
    # Create create tensorboard logs
    if args.eval_only:
        log_writer_path = './logs/eval/'.format('IParm-' + time_string)
    else:
        log_writer_path = './logs/runs/{}'.format('IParm-' + time_string)
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)
    writer = SummaryWriter(logdir=log_writer_path)

    # Trainer
    trainer = Trainer(model, class_weights, writer, time_string, args, )

    if args.eval_only is False:
        # Backup all py files[great SWE practice]
        backup_files(time_string, args, None)
        # Train
        trainer.train(train_loader, valid_loader, args)

    # Test
    print("Test accuracy", trainer.accuracy(test_loader))

    # Visualize outputs
    trainer.eval_output(train_loader, 'train')
    trainer.eval_output(valid_loader, 'val')
    trainer.eval_output(test_loader, 'test')


if __name__ == "__main__":
    args = get_args()
    main(args)
