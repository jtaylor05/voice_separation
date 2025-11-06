from datasets import load_dataset, Audio
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

def create_dataset(data_path = "kylelovesllms/timit_asr_ipa", data_split="train[:100]", test_split=0.2, audio_columns="audio", sr=16000, columns=[]):
    data = load_dataset(data_path, split=data_split)
    data = data.train_test_split(test_size=test_split)
    data = data.select_columns(list(columns))
    return data.cast_column(audio_columns, Audio(sampling_rate=sr))

def prepare_dataset(batch, processor):
    audio = batch["audio"]

    # FIX 1: Remove [0] indexing - this was causing nested structure
    processed = processor(audio["array"], sampling_rate=audio["sampling_rate"])
    
    if isinstance(processed.input_values, list) and len(processed.input_values) > 0:
        batch["input_values"] = processed.input_values[0]
    else:
        batch["input_values"] = processed.input_values
    
    
    labels = processor.tokenizer(batch["ipa_transcription"]).input_ids
    if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], list):
        batch["labels"] = [item[0] for item in labels]
    else:
        batch["labels"] = labels
    
    return batch

class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    
    def __init__(self, processor, padding = True, max_len = None, max_len_lab = None, pad = None, pad_lab = None):
        self.processor = processor
        self.padding = padding
        self.max_length = max_len
        self.max_length_labels = max_len_lab
        self.pad_to_multiple_of = pad
        self.pad_to_multiple_of_labels = pad_lab

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
        
    