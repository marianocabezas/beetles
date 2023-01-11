import time
import itertools
from functools import partial
from copy import deepcopy
from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def time_to_string(time_val):
    """
    Function to convert from a time number to a printable string that
     represents time in hours minutes and seconds.
    :param time_val: Time value in seconds (using functions from the time
     package)
    :return: String with a human format for time
    """

    if time_val < 60:
        time_s = '%ds' % time_val
    elif time_val < 3600:
        time_s = '%dm %ds' % (time_val // 60, time_val % 60)
    else:
        time_s = '%dh %dm %ds' % (
            time_val // 3600,
            (time_val % 3600) // 60,
            time_val % 60
        )
    return time_s


class BaseModel(nn.Module):
    """"
    This is the baseline model to be used for any of my networks. The idea
    of this model is to create a basic framework that works similarly to
    keras, but flexible enough.
    For that reason, I have "embedded" the typical pytorch main loop into a
    fit function and I have defined some intermediate functions and callbacks
    to alter the main loop. By itself, this model can train any "normal"
    network with different losses and scores for training and validation.
    It can be easily extended to create adversarial networks (which I have done
    in other repositories) and probably to other more complex problems.
    The network also includes some print functions to check the current status.
    """
    def __init__(self):
        """
        Main init. By default some parameters are set, but they should be
        redefined on networks inheriting that model.
        """
        super().__init__()
        # Init values
        self.device = None
        self.init = True
        self.optimizer_alg = None
        self.current_task = -1
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.best_loss_tr = np.inf
        self.best_loss_val = np.inf
        self.last_state = None
        self.best_state = None
        self.best_opt = None
        self.train_functions = [
            {'name': 'train', 'weight': 1, 'f': None},
        ]
        self.train_log = {}
        self.val_functions = [
            {'name': 'val', 'weight': 1, 'f': None},
        ]
        self.val_log = {}
        self.acc_functions = {}
        self.acc = None

    def update_logs(self):
        self.train_log = {
            f['name']: []
            for f in self.train_functions
        }
        self.val_log = {
            f['name']: []
            for f in self.val_functions
        }

    def forward(self, *inputs):
        """

        :param inputs: Inputs to the forward function. We are passing the
         contents by reference, so if there are more than one input, they
         will be separated.
        :return: Nothing. This has to be reimplemented for any class.
        """
        return None

    def observe(self, x, y):
        # First, we do a forward pass through the network.
        if isinstance(x, list) or isinstance(x, tuple):
            x_cuda = tuple(x_i.to(self.device) for x_i in x)
            pred_labels = self(*x_cuda)
        else:
            x_cuda = x.to(self.device)
            pred_labels = self(x_cuda)
        if isinstance(y, list) or isinstance(y, tuple):
            y_cuda = tuple(y_i.to(self.device) for y_i in y)
        else:
            y_cuda = y.to(self.device)

        return pred_labels, x_cuda, y_cuda

    def mini_batch_loop(
            self, data, train=True
    ):
        """
        This is the main loop. It's "generic" enough to account for multiple
        types of data (target and input) and it differentiates between
        training and testing. While inherently all networks have a training
        state to check, here the difference is applied to the kind of data
        being used (is it the validation data or the training data?). Why am
        I doing this? Because there might be different metrics for each type
        of data. There is also the fact that for training, I really don't care
        about the values of the losses, since I only want to see how the global
        value updates, while I want both (the losses and the global one) for
        validation.
        :param data: Dataloader for the network.
        :param train: Whether to use the training dataloader or the validation
         one.
        :return:
        """
        losses = list()
        mid_losses = list()
        accs = list()
        n_batches = len(data)
        for batch_i, (x, y) in enumerate(data):
            # In case we are training the the gradient to zero.
            if self.training:
                self.optimizer_alg.zero_grad()

            # First, we do a forward pass through the network.
            pred_labels, x_cuda, y_cuda = self.observe(x, y)

            # After that, we can compute the relevant losses.
            if train:
                # Training losses (applied to the training data)
                batch_losses = [
                    l_f['weight'] * l_f['f'](pred_labels, y_cuda)
                    for l_f in self.train_functions
                ]
                batch_loss = sum(batch_losses)
                if self.training:
                    for l_f, v in zip(self.train_functions, batch_losses):
                        if isinstance(v, torch.Tensor):
                            self.train_log[l_f['name']].append(
                                v.detach().cpu().numpy().tolist()
                            )
                        else:
                            self.train_log[l_f['name']].append(v)
                    try:
                        batch_loss.backward()
                        self.prebatch_update(batch_i, len(data), x_cuda, y_cuda)
                        self.optimizer_alg.step()
                        self.batch_update(batch_i, len(data), x_cuda, y_cuda)
                    except RuntimeError:
                        self.prebatch_update(batch_i, len(data), x_cuda, y_cuda)

            else:
                # Validation losses (applied to the validation data)
                batch_losses = [
                    l_f['f'](pred_labels, y_cuda)
                    for l_f in self.val_functions
                ]
                for l_f, v in zip(self.val_functions, batch_losses):
                    if isinstance(v, torch.Tensor):
                        self.val_log[l_f['name']].append(
                            v.detach().cpu().numpy().tolist()
                        )
                    else:
                        self.val_log[l_f['name']].append(v)
                batch_loss = sum([
                    l_f['weight'] * l
                    for l_f, l in zip(self.val_functions, batch_losses)
                ])
                mid_losses.append([l.tolist() for l in batch_losses])
                batch_accs = [
                    l_f['f'](pred_labels, y_cuda)
                    for l_f in self.acc_functions
                ]
                accs.append([a.tolist() for a in batch_accs])

            # It's important to compute the global loss in both cases.
            loss_value = batch_loss.tolist()
            losses.append(loss_value)

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses)
            )
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Mean loss of the global loss (we don't need the loss for each batch).
        mean_loss = np.mean(losses)

        if train:
            return mean_loss
        else:
            # If using the validation data, we actually need to compute the
            # mean of each different loss.
            mean_losses = np.mean(list(zip(*mid_losses)), axis=1)
            np_accs = np.array(list(zip(*accs)))
            mean_accs = np.mean(np_accs, axis=1) if np_accs.size > 0 else []
            return mean_loss, mean_losses, mean_accs

    def fit(
            self,
            train_loader,
            val_loader,
            epochs=50,
            patience=5,
            verbose=True
    ):
        # Init
        best_e = 0
        no_improv_e = 0
        l_names = ['train', ' val '] + [
            '{:^6s}'.format(l_f['name']) for l_f in self.val_functions
        ]
        acc_names = [
            '{:^6s}'.format(a_f['name']) for a_f in self.acc_functions
        ]
        l_bars = '--|--'.join(
            ['-' * 5] * 2 +
            ['-' * 6] * (len(l_names[2:]) + len(acc_names))
        )
        l_hdr = '  |  '.join(l_names + acc_names)
        # Since we haven't trained the network yet, we'll assume that the
        # initial values are the best ones.
        self.best_state = deepcopy(self.state_dict())
        self.best_opt = deepcopy(self.optimizer_alg.state_dict())
        t_start = time.time()

        # Initial losses
        # This might seem like an unnecessary step (and it actually often is)
        # since it wastes some time checking the output with the initial
        # weights. However, it's good to check that the network doesn't get
        # worse than a random one (which can happen sometimes).
        if self.init:
            # We are looking for the output, without training, so no need to
            # use grad.
            with torch.no_grad():
                self.t_val = time.time()
                # We set the network to eval, for the same reason.
                self.eval()
                # Training losses.
                self.best_loss_tr = self.mini_batch_loop(train_loader)
                # Validation losses.
                self.best_loss_val, best_loss, best_acc = self.mini_batch_loop(
                    val_loader, False
                )
                # Doing this also helps setting an initial best loss for all
                # the necessary losses.
                if verbose:
                    # This is just the print for each epoch, but including the
                    # header.
                    # Mid losses check
                    epoch_s = '\033[32mInit     \033[0m'
                    tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(
                        self.best_loss_tr
                    )
                    loss_s = '\033[32m{:7.4f}\033[0m'.format(
                        self.best_loss_val
                    )
                    losses_s = [
                        '\033[36m{:8.4f}\033[0m'.format(l) for l in best_loss
                    ]
                    # Acc check
                    acc_s = [
                        '\033[36m{:8.4f}\033[0m'.format(a) for a in best_acc
                    ]
                    t_out = time.time() - self.t_val
                    t_s = time_to_string(t_out)

                    print('\033[K', end='')
                    whites = ' '.join([''] * 12)
                    print('{:}Epoch num |  {:}  |'.format(whites, l_hdr))
                    print('{:}----------|--{:}--|'.format(whites, l_bars))
                    final_s = whites + ' | '.join(
                        [epoch_s, tr_loss_s, loss_s] +
                        losses_s + acc_s + [t_s]
                    )
                    print(final_s)
        else:
            # If we don't initialise the losses, we'll just take the maximum
            # ones (inf, -inf) and print just the header.
            print('\033[K', end='')
            whites = ' '.join([''] * 12)
            print('{:}Epoch num |  {:}  |'.format(whites, l_hdr))
            print('{:}----------|--{:}--|'.format(whites, l_bars))
            best_loss = [np.inf] * len(self.val_functions)
            best_acc = [-np.inf] * len(self.acc_functions)

        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            self.train()
            # First we train and check if there has been an improvement.
            loss_tr = self.mini_batch_loop(train_loader)
            improvement_tr = self.best_loss_tr > loss_tr
            if improvement_tr:
                self.best_loss_tr = loss_tr
                tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(loss_tr)
            else:
                tr_loss_s = '{:7.4f}'.format(loss_tr)

            # Then we validate and check all the losses
            with torch.no_grad():
                self.t_val = time.time()
                self.eval()
                loss_val, mid_losses, acc = self.mini_batch_loop(
                    val_loader, False
                )

            # Mid losses check
            losses_s = [
                '\033[36m{:8.4f}\033[0m'.format(l) if bl > l
                else '{:8.4f}'.format(l) for bl, l in zip(
                    best_loss, mid_losses
                )
            ]
            best_loss = [
                l if bl > l else bl for bl, l in zip(
                    best_loss, mid_losses
                )
            ]
            # Acc check
            acc_s = [
                '\033[36m{:8.4f}\033[0m'.format(a) if ba < a
                else '{:8.4f}'.format(a) for ba, a in zip(
                    best_acc, acc
                )
            ]
            best_acc = [
                a if ba < a else ba for ba, a in zip(
                    best_acc, acc
                )
            ]

            # Patience check
            # We check the patience to stop early if the network is not
            # improving. Otherwise we are wasting resources and time.
            improvement_val = self.best_loss_val > loss_val
            loss_s = '{:7.4f}'.format(loss_val)
            if improvement_val:
                self.best_loss_val = loss_val
                epoch_s = '\033[32mEpoch {:03d}\033[0m'.format(self.epoch)
                loss_s = '\033[32m{:}\033[0m'.format(loss_s)
                best_e = self.epoch
                self.best_state = deepcopy(self.state_dict())
                self.best_opt = deepcopy(self.optimizer_alg.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch {:03d}'.format(self.epoch)
                no_improv_e += 1

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                final_s = whites + ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] +
                    losses_s + acc_s + [t_s]
                )
                print(final_s)

            self.epoch_update(epochs, train_loader)

            if no_improv_e == patience:
                break

        t_end = time.time() - t_start
        t_end_s = time_to_string(t_end)
        if verbose:
            print(
                    'Training finished in {:} epochs ({:}) '
                    'with minimum loss = {:f} (epoch {:d})'.format(
                        self.epoch + 1, t_end_s, self.best_loss_val, best_e
                    )
            )

        self.last_state = deepcopy(self.state_dict())
        self.epoch = best_e
        self.load_state_dict(self.best_state)

    def inference(self, data, nonbatched=True, task=None):
        temp_task = task
        if temp_task is not None and hasattr(self, 'current_task'):
            temp_task = self.current_task
            self.current_task = task
        with torch.no_grad():
            if isinstance(data, list) or isinstance(data, tuple):
                x_cuda = tuple(
                    torch.from_numpy(x_i).to(self.device)
                    for x_i in data
                )
                if nonbatched:
                    x_cuda = tuple(
                        x_i.unsqueeze(0) for x_i in x_cuda
                    )

                output = self(*x_cuda)
            else:
                x_cuda = torch.from_numpy(data).to(self.device)
                if nonbatched:
                    x_cuda = x_cuda.unsqueeze(0)
                output = self(x_cuda)
            torch.cuda.empty_cache()

            if len(output) > 1:
                np_output = output.cpu().numpy()
            else:
                np_output = output[0, 0].cpu().numpy()
        if temp_task is not None and hasattr(self, 'current_task'):
            self.current_task = temp_task

        return np_output, np.array([task] * len(output))

    def reset_optimiser(self, model_params=None):
        """
        Abstract function to rest the optimizer.
        :return: Nothing.
        """
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.best_loss_tr = np.inf
        self.best_loss_val = np.inf
        self.best_state = None
        self.best_opt = None
        return None

    def epoch_update(self, epochs, loader):
        """
        Callback function to update something on the model after the epoch
        is finished. To be reimplemented if necessary.
        :param epochs: Maximum number of epochs
        :param loader: Dataloader used for training
        :return: Nothing.
        """
        return None

    def prebatch_update(self, batch, batches, x, y):
        """
        Callback function to update something on the model before the batch
        update is applied. To be reimplemented if necessary.
        :param batches: Maximum number of epochs
        :param x: Training data
        :param y: Training target
        :return: Nothing.
        """
        return None

    def batch_update(self, batch, batches, x, y):
        """
        Callback function to update something on the model after the batch
        is finished. To be reimplemented if necessary.
        :param batches: Maximum number of epochs
        :param x: Training data
        :param y: Training target
        :return: Nothing.
        """
        return None

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss):
        """
        Function to print the progress of a batch. It takes into account
        whether we are training or validating and uses different colors to
        show that. It's based on Keras arrow progress bar, but it only shows
        the current (and current mean) training loss, elapsed time and ETA.
        :param batch_i: Current batch number.
        :param n_batches: Total number of batches.
        :param b_loss: Current loss.
        :param mean_loss: Current mean loss.
        :return: None.
        """
        init_c = '\033[0m' if self.training else '\033[38;5;238m'
        percent = 25 * (batch_i + 1) // n_batches
        progress_s = ''.join(['█'] * percent)
        remainder_s = ''.join([' '] * (25 - percent))
        loss_name = 'train_loss' if self.training else 'val_loss'

        if self.training:
            t_out = time.time() - self.t_train
        else:
            t_out = time.time() - self.t_val
        time_s = time_to_string(t_out)

        t_eta = (t_out / (batch_i + 1)) * (n_batches - (batch_i + 1))
        eta_s = time_to_string(t_eta)
        epoch_hdr = '{:}Epoch {:03} ({:03d}/{:03d} - {:05.2f}%) [{:}] '
        loss_s = '{:} {:f} ({:f}) {:} / ETA {:}'
        batch_s = (epoch_hdr + loss_s).format(
            init_c, self.epoch, batch_i + 1, n_batches,
            100 * (batch_i + 1) / n_batches, progress_s + remainder_s,
            loss_name, b_loss, mean_loss, time_s, eta_s + '\033[0m'
        )
        print('\033[K', end='', flush=True)
        print(batch_s, end='\r', flush=True)

    @staticmethod
    def print_batch(pi, n_patches, i, n_cases, t_in, t_case_in):
        init_c = '\033[38;5;238m'
        percent = 25 * (pi + 1) // n_patches
        progress_s = ''.join(['█'] * percent)
        remainder_s = ''.join([' '] * (25 - percent))

        t_out = time.time() - t_in
        t_case_out = time.time() - t_case_in
        time_s = time_to_string(t_out)

        t_eta = (t_case_out / (pi + 1)) * (n_patches - (pi + 1))
        eta_s = time_to_string(t_eta)
        pre_s = '{:}Case {:03d}/{:03d} ({:03d}/{:03d} - {:06.2f}%) [{:}{:}]' \
                ' {:} ETA: {:}'
        batch_s = pre_s.format(
            init_c, i + 1, n_cases, pi + 1, n_patches,
            100 * (pi + 1) / n_patches,
            progress_s, remainder_s, time_s, eta_s + '\033[0m'
        )
        print('\033[K', end='', flush=True)
        print(batch_s, end='\r', flush=True)

    def freeze(self):
        """
        Method to freeze all the network parameters.
        :return: None
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Method to unfreeze all the network parameters.
        :return: None
        """
        for param in self.parameters():
            param.requires_grad = True

    def set_last_state(self):
        if self.last_state is not None:
            self.load_state_dict(self.last_state)

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(
            torch.load(net_name, map_location=self.device)
        )


class TorchVisionWrapper(BaseModel):
    def __init__(
        self, n_classes, network_function, pretrained=False, lr=1e-3,
        train_functions=None, val_functions=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=True
    ):
        super().__init__()
        # Init
        self.n_classes = n_classes
        self.lr = lr
        self.device = device
        if pretrained:
            try:
                self.net = network_function(weights='IMAGENET1K_V1')
            except TypeError:
                self.net = network_function(pretrained)
        else:
            self.net = network_function()
        if isinstance(self.net, models.resnet.ResNet):
            in_features = self.net.fc.in_features
            self.net.fc = nn.Linear(in_features, n_classes)
        elif isinstance(self.net, models.vision_transformer.SwinTransformer):
            in_features = self.net.head.in_features
            self.net.head = nn.Linear(in_features, n_classes)
        elif isinstance(self.net, models.vision_transformer.VisionTransformer):
            in_features = self.net.heads[0].in_features
            self.net.heads[0] = nn.Linear(in_features, n_classes)
        elif isinstance(self.net, models.convnext.ConvNeXt):
            in_features = self.net.classifier[-1].in_features
            self.net.classifier[-1] = nn.Linear(in_features, n_classes)

        # <Loss function setup>
        if train_functions is None:
            self.train_functions = [
                {
                    'name': 'xentropy',
                    'weight': 1,
                    'f': F.cross_entropy
                }
            ]
        else:
            self.train_functions = train_functions

        if val_functions is None:
            self.val_functions = [
                {
                    'name': 'xent',
                    'weight': 1,
                    'f': F.cross_entropy
                },
            ]
        else:
            self.val_functions = val_functions

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)

    def forward(self, data):
        self.net.to(self.device)
        return self.net(data)