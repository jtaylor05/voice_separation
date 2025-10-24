from datasets import load_dataset, Audio
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import torch
from torchcodec.decoders import AudioDecoder
from data import create_dataset, DataCollatorCTCWithPadding
from vocab_maker import make_vocab_file
from evaluate import load

data = create_dataset(data_split="train[:1000]", columns=["audio", "phonetic_detail", "ipa_transcription"])

make_vocab_file(data)

tokenizer = Wav2Vec2PhonemeCTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", do_phonemize=False)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# def prepare_dataset(batch):
#     audio = batch["audio"]

#     # batched output is "un-batched" to ensure mapping is correct
#     batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
#     with processor.as_target_processor():
#         batch["labels"] = processor(batch["ipa_transcription"]).input_ids
#     return batch

def prepare_dataset(batch):
    audio = batch["audio"]

    # FIX 1: Remove [0] indexing - this was causing nested structure
    processed = processor(audio["array"], sampling_rate=audio["sampling_rate"])
    
    if isinstance(processed.input_values, list) and len(processed.input_values) > 0:
        batch["input_values"] = processed.input_values[0]
    else:
        batch["input_values"] = processed.input_values
    
    
    labels = tokenizer(batch["ipa_transcription"]).input_ids
    if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], list):
        batch["labels"] = [item[0] for item in labels]
    else:
        batch["labels"] = labels
    # with processor.as_target_processor():
    #     # FIX 2: Get the actual input_ids without extra nesting
    #     labels = processor(batch["ipa_transcription"]).input_ids
    #     # If labels is a list of lists, flatten it
    #     if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], list):
    #         batch["labels"] = labels[0]
    #     else:
    #         batch["labels"] = labels
    
    return batch

prep_data = data.map(prepare_dataset, remove_columns=data.column_names["train"], num_proc=4)

print("\n=== Data Structure Check ===")
sample = prep_data["train"][0]

print(f"\ninput_values:")
print(f"  Type: {type(sample['input_values'])}")
if hasattr(sample['input_values'], 'shape'):
    print(f"  Shape: {sample['input_values'].shape}")
    print(f"  Dtype: {sample['input_values'].dtype}")
elif isinstance(sample['input_values'], list):
    print(f"  Length: {len(sample['input_values'])}")
    print(f"  First element type: {type(sample['input_values'][0])}")
    # Check if it's nested
    if isinstance(sample['input_values'][0], list):
        print(f"  WARNING: Nested list detected!")
        print(f"  First inner list length: {len(sample['input_values'][0])}")
    else:
        print(f"  Sample values: {sample['input_values'][:5]}")

print(f"\nlabels:")
print(f"  Type: {type(sample['labels'])}")
if hasattr(sample['labels'], 'shape'):
    print(f"  Shape: {sample['labels'].shape}")
elif isinstance(sample['labels'], list):
    print(f"  Length: {len(sample['labels'])}")
    print(f"  Sample: {sample['labels'][:10]}")
    # Check if it's nested
    if len(sample['labels']) > 0 and isinstance(sample['labels'][0], list):
        print(f"  WARNING: Nested list detected!")

print("===========================\n")

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

cer = load("cer")

def compute_metrics(pred):
    print(hasattr(pred['labels']))
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

vocab_size = len(processor.tokenizer)
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base", 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=vocab_size,
    ignore_mismatched_sizes=True
).to("cuda")

model.freeze_feature_encoder()

training_args = TrainingArguments(
  #output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=32,
  #evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True, 
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
  label_names=["labels"]
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=prep_data["train"],
    eval_dataset=prep_data["test"],
    tokenizer=processor.feature_extractor,
)

trainer.train()

