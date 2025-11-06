from datasets import load_dataset, Audio
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import torch
from torchcodec.decoders import AudioDecoder
from data import create_dataset, prepare_dataset, DataCollatorCTCWithPadding
from vocab_maker import make_vocab_file
from evaluate import load
from huggingface_hub import login
import numpy as np

login()

data = create_dataset(data_split="train[:1000]", columns=["audio", "phonetic_detail", "ipa_transcription"])

make_vocab_file(data)

tokenizer = Wav2Vec2PhonemeCTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", do_phonemize=False)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

prep_data = data.map(prepare_dataset, remove_columns=data.column_names["train"], num_proc=4, fn_kwargs={"processor":processor})

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

cer = load("cer")

def compute_metrics(pred):
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

repo_name = "jttaylor01/phon-finetuned"

training_args = TrainingArguments(
  output_dir=repo_name,
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
trainer.push_to_hub()
trainer.evaluate()

