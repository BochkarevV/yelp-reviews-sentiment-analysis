import time

import torch
from transformers import BertForSequenceClassification

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


class TransformersGeneric:

    def __init__(self, num_labels,
                 transformers_model=BertForSequenceClassification,
                 model_name='bert-large-cased',
                 output_attention=False,
                 output_hidden_states=False
                 ):
        """
        Define a HugginFace's transformer model.

        :param num_labels: int
            Number of target classes.
        :param transformers_model: BertPreTrainedModel, optional (default=BertForSequenceClassification)
            Pre-trained model defined in HuggingFace's transformer model.
            WARNING: currently will only work with ...ForSequenceClassification models.
        :param model_name: string, optional (default='bert-large-cased')
            Model name as defined here (shortcut name column):
            https://huggingface.co/transformers/pretrained_models.html
        :param output_attention: bool, optional (default=False)
            Whether to return attention weights.
        :param output_hidden_states: bool, optional (default=False)
            Whether to return all hidden states.
        """

        self._model = transformers_model.from_pretrained(
            model_name, num_labels=num_labels,
            output_attention=output_attention,
            output_hidden_states=output_hidden_states
        )

        # Define the device (GPU or CPU) to train the model on
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Use GPU if available
        if self.device.type == 'cuda':
            self._model.cuda()

    def train(self, train_data_loader, val_data_loader, optimizer, scheduler, epochs):
        """
        Train a previously defined transformer model (e.g., BertForSequenceClassification by default).

        :param train_data_loader: DataLoader
            Provides an iterable over the training dataset.
        :param val_data_loader: DataLoader
            Provides an iterable over the validation dataset.
        :param optimizer: torch.optim.Optimizer
            Model parameters optimizer after each epoch.
        :param scheduler: torch.optim.lr_scheduler.LambdaLR
            Scheduler which adjusts learning rate after each epoch.
        :param epochs: int
            Number of epochs to train train the model.

        :return dict
        {
            'train_losses': list,
            'val_losses': list,
            'val_f1_scores': list,
            'val_recall_scores': list,
            'val_precision_scores': list
        }
        Dictionary containing train and validation losses, as well as the history of
        validation metrics measured over all epochs.
        """

        # Track training and validation losses
        train_loss_vals = []
        val_loss_vals = []

        # Track F1, recall and precision
        val_f1_scores = []
        val_rec_scores = []
        val_prec_scores = []

        total_train_loss = 0
        # Training loop
        for epoch in range(epochs):

            print()
            print('-'*10 + f'{epoch+1}' + '-'*10)
            print('Training...')

            # Keep track of epoch training time
            t_start = time.time()

            #####################
            #     Training      #
            #####################

            # Switch to training mode
            self._model.train()

            # Load data for training in batches
            for step, batch in enumerate(train_data_loader):

                # Print out the progress every 25 steps
                if step % 25 == 0 and step != 0:
                    elapsed_time = TransformersGeneric._format_elapsed_time(time.time() - t_start)
                    print(f'\tBatch {step+1}/{len(train_data_loader)}, elapsed {elapsed_time}')

                # Batch unpacking. Each batch consists of 3 tensors:
                # 1. Input sequences of token IDs (preprocessed).
                # 2. Corresponding attention masks.
                # 3. Target labels.
                # Additionally, move the tensors to the available device (GPU or CPU)
                seq, masks, labels = map(lambda x: x.to(self.device), batch)

                # Zero out gradients before backpropagation
                self._model.zero_grad()

                # Forward pass
                outputs = self._model(seq, attention_mask=masks, labels=labels)

                # Extract train loss
                train_loss = outputs[0]

                # Accumulate training loss to obtain average loss at the end.
                total_train_loss += train_loss.item() / batch[0].shape[0]

                # Backward pass
                train_loss.backward()

                # Gradients clipping to prevent exploading gradients problem
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)

                # Optimize parameters
                optimizer.step()

                # Learning rate update
                scheduler.step()

            # Calculate average training loss and store for history
            avg_train_loss = total_train_loss / len(train_data_loader)
            train_loss_vals.append(avg_train_loss)

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
            self._model.eval()

            total_val_loss = 0
            val_f1, val_rec, val_prec = 0, 0, 0
            for batch in val_data_loader:

                # Unpack the batch and tensors to device
                seq, masks, labels = map(lambda x: x.to(self.device), batch)

                # Do not compute gradients during validation
                with torch.no_grad():
                    outputs = self._model(seq, attention_mask=masks, labels=labels)

                # Unpack loss and logits from the output
                val_loss, logits = outputs

                # Accumulate the validation loss
                total_val_loss += val_loss.item() / batch[0].shape[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu()
                labels = labels.to('cpu')

                # We can skip softmax for now, since we only need the most likely class.
                # Hence, just finding the index with largest value is enough
                y_pred = logits.argmax(dim=1)

                # Calculate and accumulate metrics
                val_f1 += f1_score(labels, y_pred, average='micro')
                val_rec += recall_score(labels, y_pred, average='micro')
                val_prec += precision_score(labels, y_pred, average='micro')

            # Calculate and store average validation loss
            steps = len(val_data_loader)
            avg_val_loss = total_val_loss / steps
            val_loss_vals.append(avg_val_loss)

            epoch_val_f1 = val_f1 / steps
            epoch_val_rec = val_rec / steps
            epoch_val_prec = val_prec / steps

            val_f1_scores.append(epoch_val_f1)
            val_rec_scores.append(epoch_val_rec)
            val_prec_scores.append(epoch_val_prec)

            print(f'\tAverage validation loss: {avg_val_loss}')
            print(f'\tF1: {epoch_val_f1}')
            print(f'\tRecall: {epoch_val_rec}')
            print(f'\tPrecision: {epoch_val_prec}')
            print(f'\tValidation time: {TransformersGeneric._format_elapsed_time(time.time() - t_start)}')

        print('\nTraining finished!')

        return {
            'train_losses': train_loss_vals,
            'val_losses': val_loss_vals,
            'val_f1_scores': val_f1_scores,
            'val_recall_scores': val_rec_scores,
            'val_precision_scores': val_prec_scores
        }

    def test(self):
        pass

    def predict(self):
        pass

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
