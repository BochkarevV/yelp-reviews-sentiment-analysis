import time
import warnings

import numpy as np
import torch
from transformers import BertForSequenceClassification

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class TransformersGeneric:

    def __init__(self, num_classes,
                 transformers_model=BertForSequenceClassification,
                 model_name='bert-base-cased',
                 output_attentions=False,
                 output_hidden_states=False
                 ):
        """
        Define a HugginFace's transformer model.

        :param num_classes: int
            Number of target classes.
        :param transformers_model: PreTrainedModel, optional (default=BertForSequenceClassification)
            Pre-trained model defined in HuggingFace's transformer model.
            Currently will only work with classification models.
        :param model_name: string, optional (default='bert-base-cased')
            Model name as defined here (shortcut name column):
            https://huggingface.co/transformers/pretrained_models.html
        :param output_attentions: bool, optional (default=False)
            Whether to return attention weights.
        :param output_hidden_states: bool, optional (default=False)
            Whether to return all hidden states.
        """

        self.model = transformers_model.from_pretrained(
            model_name, num_labels=num_classes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        # Define the device (GPU or CPU) to train the model on
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.model.cuda()
            print(f'{torch.cuda.get_device_name(0)} is used...')
        else:
            print('CPU is used...')

    def train(self, train_data_loader, val_data_loader, optimizer, scheduler, epochs):
        """
        Train a previously defined transformer model (e.g., BertForSequenceClassification by default).

        :param train_data_loader: DataLoader
            Iterable over the training dataset.
        :param val_data_loader: DataLoader
            Iterable over the validation dataset.
        :param optimizer: torch.optim.Optimizer
            Model parameters optimizer after each epoch.
        :param scheduler: torch.optim.lr_scheduler.LambdaLR
            Scheduler which adjusts learning rate after each epoch.
        :param epochs: int
            Number of epochs to train train the model.

        :return list of dicts
        [
            {
            'train_loss': float,
            'val_loss': float,
            'val_f1': float,
            'val_accuracy': float
        }
        List of dictionaries containing training and validation losses, as well as validation F1 and accuracy scores.
        More suitable format for logging into WandB.
        """

        # Track training metrics
        training_metrics = []

        # Training loop
        for epoch in range(epochs):

            epoch_metrics = {}

            print()
            print('-'*25 + f'\t{epoch+1}\t' + '-'*25)
            print('Training...')

            # Keep track of epoch training time
            t_start = time.time()

            #####################
            #     Training      #
            #####################

            # Track total train loss, reset every epoch
            total_train_loss = 0

            # Switch to training mode
            self.model.train()

            # Load data for training in batches
            for step, batch in enumerate(train_data_loader):

                # Print out the progress every 25 steps
                if (step + 1) % 25 == 0 and step != 0:
                    elapsed_time = TransformersGeneric._format_elapsed_time(time.time() - t_start)
                    print(f'\tBatch {step + 1}/{len(train_data_loader)}, elapsed {elapsed_time}')

                # Batch unpacking. Each batch consists of 3 tensors:
                # 1. Input sequences of token IDs (preprocessed).
                # 2. Corresponding attention masks.
                # 3. Target labels.
                # Additionally, move the tensors to the available device (GPU or CPU)
                seq, masks, labels = map(lambda x: x.to(self.device), batch)

                # Zero out gradients before backpropagation
                self.model.zero_grad()

                # Forward pass
                outputs = self.model(seq, attention_mask=masks, labels=labels)

                # Extract train loss
                train_loss = outputs[0]

                # Accumulate training loss to obtain average loss at the end.
                total_train_loss += train_loss.item() / batch[0].shape[0]

                # Backward pass
                train_loss.backward()

                # Gradients clipping to prevent exploading gradients problem
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                # Optimize parameters
                optimizer.step()

                # Learning rate update
                scheduler.step()

            # Calculate average training loss and store for history
            avg_train_loss = total_train_loss / len(train_data_loader)
            epoch_metrics['train_loss'] = avg_train_loss

            print()
            print(f'\tAverage training loss: {avg_train_loss}')
            print(f'\tTraining epoch time: {TransformersGeneric._format_elapsed_time(time.time() - t_start)}')

            #####################
            #     Validation    #
            #####################

            print()
            print('Validating...')

            # Keep track of validation time
            t_start = time.time()

            # Switch to evaluation mode.
            self.model.eval()

            # Track validation loss and metrics
            total_val_loss = 0
            val_f1, val_acc = 0, 0

            # Validation loop
            for batch in val_data_loader:

                # Unpack the batch and tensors to device
                seq, masks, labels = map(lambda x: x.to(self.device), batch)

                # Do not compute gradients during validation
                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(seq, attention_mask=masks, labels=labels)

                # Unpack loss and logits from the output
                val_loss, logits = outputs

                # Accumulate the validation loss
                total_val_loss += val_loss.item() / batch[0].shape[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu()
                labels = labels.to('cpu')

                # We can skip softmax for now, since we only need the most likely class.
                # Hence, just finding the index with largest value would be enough.
                y_pred = logits.argmax(dim=1)

                # Calculate and accumulate metrics
                val_f1 += f1_score(labels, y_pred, average='micro')
                val_acc += accuracy_score(labels, y_pred)

            # Calculate and store average validation loss
            steps = len(val_data_loader)
            avg_val_loss = total_val_loss / steps
            epoch_metrics['val_loss'] = avg_val_loss

            epoch_val_f1 = val_f1 / steps
            epoch_val_acc = val_acc / steps

            epoch_metrics['val_f1'] = epoch_val_f1
            epoch_metrics['val_accuracy'] = epoch_val_acc

            training_metrics.append(epoch_metrics)

            print(f'\tAverage validation loss: {avg_val_loss}')
            print(f'\tF1: {epoch_val_f1}')
            print(f'\tAccuracy: {epoch_val_acc}')
            print(f'\tValidation time: {TransformersGeneric._format_elapsed_time(time.time() - t_start)}')

        print('\nTraining finished!')

        return training_metrics

    def predict(self, sequences, attention_masks):
        """
        Makes the prediction and returns logits.

        :param sequences: torch.tensor()
            Encoded input sequences from a batch.
        :param attention_masks: torch.tensor()
            Corresponding attention masks.

        :return: list
            Predicted logits.
        """

        # Make sure the model is in the evaluation mode.
        self.model.eval()

        # Get predictions for the sequences
        with torch.no_grad():
            outputs = self.model(sequences, attention_mask=attention_masks)

        # Extract logits
        logits = outputs[0]

        # Move logits to CPU
        logits = logits.detach().cpu()

        return logits

    def predict_proba(self, sequences, attention_masks):
        """
        Predicts probability distribution over the classes.

        :param sequences: torch.tensor()
            Encoded input sequences from a batch.
        :param attention_masks: torch.tensor()
            Corresponding attention masks.

        :return: torch.tensor()
            Probability distributions in the shape of (# instances, # classes).
        """

        logits = self.predict(sequences, attention_masks)
        softmax = torch.nn.Softmax(dim=1)
        return softmax(logits)

    def evaluate(self, data_loader, **metrics):
        """
        Evaluate the model.

        :param data_loader: DataLoader
            Iterator over a dataset.
        :param metrics: dict
            Dictionary of metrics from sklearn in the following format:
            {
                'metric_name': scoring_function,
                ...
            }

        :return: dict
            Dictionary with metric names as keys and lists of corresponding metrics
            scores per batch.
        """

        if not metrics:
            warnings.warn('Metrics are not provided. The result will be an empty dictionary.')
            return {}

        metrics_scores = dict((metric, []) for metric in metrics)

        for batch in data_loader:

            # Unpack the batch and tensors to device
            seq, attention_masks, y_true = map(lambda x: x.to(self.device), batch)

            # Make predictions and return logits
            logits = self.predict(seq, attention_masks)

            # Find argmax to choose the most likely class
            y_pred = logits.argmax(dim=1)

            # Calculate metric scores and store the results in a dictionary
            for name, metric in metrics.items():
                metrics_scores[name].append(metric(y_true.cpu(), y_pred.cpu()))

        for name, scores in metrics_scores.items():
            metrics_scores[name] = np.average(metrics_scores[name])

        return metrics_scores

    def get_parameters(self):
        """
        Return model parameters for optimization.

        :return: module parameters
        """
        return self.model.parameters()

    @staticmethod
    def _format_elapsed_time(elapsed):
        """
        Convert elapsed time from seconds to hh:mm:ss.s format.

        :param elapsed: float
            Elapsed time in seconds (t_current - t_start).
        :return: string
            Time in human-readable hh:mm:ss.s format, e.g. '04:35:41.05'.
        """

        hours, remainder = divmod(elapsed, 60*60)
        minutes, seconds = divmod(remainder, 60)

        return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
