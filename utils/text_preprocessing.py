import warnings
import torch
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences


class TextPreprocessor:

    def __init__(self,
                 tokenizer=BertTokenizer,
                 vocab_file='bert-large-cased',
                 do_lower_case=False,
                 add_special_tokens=True,
                 truncating='post',
                 padding='post'):
        """
        Text preprocessor prepares input raw texts to be used by Transformer models.

        :param tokenizer: PreTrainedTokenizer, optional (default=BertTokenizer)
            Tokenizer as defined in transformers library.
        :param vocab_file: string, optional (default='bert-large-cased')
            Pre-defined vocabulary file to be used by the tokenizer.
            Depends on the model used. A list can be obtained here ('Shortcut name' column):
            https://huggingface.co/transformers/pretrained_models.html
        :param do_lower_case: bool, optional (default=False)
            Indicates if tokenizer should lower case the input texts.
        :param truncating: string, 'pre' or 'post', optional (default='post')
            Remove values from sequences larger than max length found in the training set
            either at the beginning or at the end of the sequences.
        :param padding: string, 'pre' or 'post', optional (default='post')
            Pad either before or after each sequence.

        Example
        -------
        >>> train_texts = ['Short example text here.',
        >>>                'Another one sample string.']
        >>> test_texts = ['One string for testing.']
        >>>
        >>> from utils.text_preprocessing import TextPreprocessor
        >>> prep = TextPreprocessor()
        >>> prep.preprocess(train_texts, fit=True)
        >>> prep.preprocess(test_texts)
        """

        self._max_len = 0   # init

        self._add_special_tokens = add_special_tokens
        self._truncating = truncating
        self._padding = padding

        # Download a pretrained tokenizer
        self.tokenizer = tokenizer.from_pretrained(vocab_file,
                                                   do_lower_case=do_lower_case)

    def preprocess(self, texts, fit=False):
        """
        Preprocess input texts to match the data format required by the transformer models.
        Sequentially performs tokenization, padding and truncating, attention masks generation.

        :param texts: list
            List of string to preprocess.
        :param fit: bool, optional (default=False)
            Whether to fit hyperparams needed by the text preprocessor.
            Should normally be True when used to process training data at first.

        :return: tuple(torch.tensor(), torch.tensor())
            Tuple of two PyTorch tensors.
            The former tensor contains input token ids with transformer tokens ([CLS], [SEP])
            and paddings (0).
            The latter tensor stores corresponding attention masks.
        """

        # Tokenization
        sequences = self.tokenize(texts)

        # Padding and truncating
        sequences = self.pad_truncate(sequences, fit=fit)

        # Attention masks generation
        attention_masks = self.generate_attention_mask(sequences)

        # Convert to torch tensors, needed for further model training
        sequences = torch.tensor(sequences)
        attention_masks = torch.tensor(attention_masks)

        return sequences, attention_masks

    def tokenize(self, texts):
        """
        Use Transformers pre-defined tokenizers. The following steps are performed internally
        by the `encode()` method:
        * Tokenize the input string.
        * Prepend [CLS] token.
        * Append [SEP] token.
        * Map tokens to their IDs according to particular tokenizer used.

        :param texts: list
            List of strings to tokenize.

        :return: list
            List of sequences containing token IDs per each observation.
        """

        id_sequences = []

        for txt in texts:
            encoded = self.tokenizer.encode(text=txt, add_special_tokens=self._add_special_tokens)
            id_sequences.append(encoded)

        return id_sequences

    def pad_truncate(self, id_sequences, fit=False):
        """
        Pad or truncate input sequences of tokens until the max length from the training
        set is reached.

        :param id_sequences: list
            List of lists containing token IDs per each observation.
        :param fit: bool, optional (default=False)
            Whether to fit hyperparams needed by the text preprocessor.
            Should normally be True when used to process training data at first.

        :return: list
            List of sequences having the same length.
        """

        # Find the longest sequence of tokens in the training set ONLY.
        # Use this length to pad/truncate sequences from val and test sets.
        if fit:
            self._max_len = max([len(ids) for ids in id_sequences])

        if self._max_len == 0:
            warnings.warn('max_len is 0. Padding and truncating will result in empty sequences. '
                          'Fit the preprocessor with your training data first.')

        # Pad shorter sequences and truncate longer ones
        padded_token_ids = pad_sequences(id_sequences, value=0, maxlen=self._max_len,
                                         dtype='long', truncating=self._truncating, padding=self._padding)

        return padded_token_ids

    def generate_attention_mask(self, id_sequences):
        """
        Generates attention masks. Attention masks show which tokens represent actual words
        and which ones are paddings.

        :param id_sequences: list
            List of padded or truncated sequences to generate attention masks from.
        :return: list
            List of attention masks corresponding to the input sequences.
        """

        attention_masks = []
        for seq in id_sequences:
            # Paddings are represented by 0, hence they are marked with 0 in the attention masks.
            # All other tokens_ids > 0 are marked with 1.

            mask = [int(token_id > 0) for token_id in seq]
            attention_masks.append(mask)

        return attention_masks
