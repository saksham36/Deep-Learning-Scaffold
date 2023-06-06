import os
import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, class_weights, writer, timeStr, args):
        self.args = args
        self.model = model
        self.writer = writer
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        self.model.to(device)
        self.class_weights = class_weights.to(device)
        if device == torch.device('cuda'):
            assert next(self.model.parameters()
                        ).is_cuda, 'Model is not on GPU!'
        elif device == torch.device('mps'):
            assert next(self.model.parameters()).is_mps, 'Model is not on MPS!'

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=args.learning_rate,
                                          weight_decay=args.weight_decay)
        self.timeStr = timeStr
        seed = args.seed

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Add Graph to Tensorboard
        # self.writer.add_graph(self.model, torch.rand(1, 10, 3, 25).to(device))

    def train(self, train_loader, val_loader, args):
        batch_size = args.batch_size
        n_epochs = args.n_epochs
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        self.model.to(self.device)
        # Start training
        training_loss = []
        for epoch in tqdm(range(1, n_epochs + 1)):
            running_loss = 0.0
            batch_losses = []
            c = 0
            for x_batch, y_batch in train_loader:
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(x_batch.to(self.device))
                batch_size, sequence_len, n_classes = y_batch.shape
                y_batch_view = y_batch.view(
                    batch_size * sequence_len, n_classes)
                loss = self.criterion(outputs, y_batch_view.to(self.device))
                if loss.isnan().any():
                    print("Loss is NaN!")
                    import pdb
                    pdb.set_trace()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                batch_losses.append(loss.item())
                c += 1
                if False and c % 20 == 19:    # print every 20 mini-batches
                    print(
                        f'[{epoch}, {c + 1:5d}] loss: {running_loss / 20:.3f}')
                    running_loss = 0.0
            training_loss.append(np.mean(batch_losses))

            if epoch % args.val_log_interval == 0:
                # Validation loss
                validation_loss = 0.0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        outputs = self.model(x_val.to(self.device))
                        batch_size, sequence_len, n_classes = y_val.shape
                        y_val = y_val.view(
                            batch_size * sequence_len, n_classes)
                        loss = self.criterion(outputs, y_val.to(self.device))
                        validation_loss += loss.item()

                self.writer.add_scalar('validation_loss',
                                       validation_loss / len(val_loader),
                                       epoch)
                self.writer.add_scalar('validation_accuracy',
                                       self.accuracy(val_loader),
                                       epoch)

                print('[%d] val_loss: %.3f' %
                      (epoch, validation_loss / len(val_loader)))

            # Print statistics & write tensorboard logs.
            if epoch % args.print_log_interval == 0:
                print('[%d] loss: %.3f' %
                      (epoch, np.mean(training_loss) / args.print_log_interval))
                self.writer.add_scalar('training_loss',
                                       np.mean(training_loss),
                                       epoch)
                self.writer.add_scalar('learning_rate',
                                       self.optimizer.param_groups[0]['lr'],
                                       epoch)
                self.writer.add_scalar('training_accuracy',
                                       self.accuracy(train_loader),
                                       epoch)
                training_loss = []

            # Save the trained  model
            if (epoch % args.model_save_interval == 0) and args.model_save_path != "":
                if epoch % args.model_update_interval == 0:
                    sub_time_str = time.strftime(
                        '%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                sub_time_str = time.strftime(
                    '%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                torch.save(self.model.state_dict(), os.path.join(
                    model_save_path, 'IParm-' + self.timeStr + '_' + sub_time_str + ".pt"))

        print('Finished Training')

    def accuracy(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in loader:
                outputs = self.model(x_val.to(self.device))
                predicted = torch.argmax(outputs.data, 1).to(
                    'cpu')  # (batch_size*sequence_len, one_hot)
                batch_size, sequence_len, n_classes = y_val.shape
                y_val = y_val.view(batch_size * sequence_len, n_classes)
                total += y_val.size(0)
                correct += (predicted == torch.argmax(y_val, 1)).sum().item()
        return correct / total

    def eval_output(self, loader, name):
        df = pd.DataFrame(columns=[' ', 'actual'])
        with torch.no_grad():
            for x_val, y_val in tqdm(loader, desc='Processing '+name+' dataset'):
                outputs = self.model(x_val.to(self.device))
                predicted = torch.argmax(outputs.data, 1).to(
                    'cpu')  # (batch_size, one_hot)
                batch_size, sequence_len, n_classes = y_val.shape
                actual = torch.argmax(y_val, 2)
                predicted = predicted.view(batch_size, sequence_len)
                for i in range(batch_size):
                    df.loc[len(df)] = [predicted[i], actual[i]]
        # Save as csv
        df.to_csv(self.args.model_path[:-3] + '-'+name+'-output.csv')
        print('Finished Evaluation')
