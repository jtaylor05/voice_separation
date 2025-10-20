from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForCTC # Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2Model,
import torch
from torchcodec.decoders import AudioDecoder

timit = load_dataset("kylelovesllms/timit_asr_ipa", split="train[:100]")
timit = timit.train_test_split(test_size=0.2)
timit = timit.select_columns(["audio", "phonetic_detail"])

sampling_rate = 16000 # timit["train"]["audio"].sampling_rate
audio_file = timit["train"][0]["audio"]

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

inputs = processor(audio_file, sampling_rate=16000, return_tensors="pt")

model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base")

with torch.no_grad():
    logits = model(**inputs).logits
    
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)

# from datasets import load_dataset, Audio
# from transformers import (
#     AutoProcessor,
#     AutoModelForAudioClassification,
#     Trainer,
#     TrainingArguments,
# )
# import numpy as np
# import torch
# import evaluate

# # 1. Load Dataset
# dataset = load_dataset("kylelovesllms/timit_asr_ipa", split="train[:100]")  # Keyword Spotting task
# print(
# dataset = dataset.train_test_split(test_size=0.2)
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# # 2. Load Pretrained Model & Processor
# model_name = "facebook/wav2vec2-base"
# processor = AutoProcessor.from_pretrained(model_name)
# model = AutoModelForAudioClassification.from_pretrained(
#     model_name,
#     num_labels=len(set(dataset["train"]["label"]))
# )

# # 3. Preprocessing Function
# def preprocess(batch):
#     audio = batch["audio"]
#     inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
#     inputs["label"] = batch["label"]
#     return inputs

# # 4. Tokenize Dataset
# dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# # 5. Define Evaluation Metric
# accuracy = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return accuracy.compute(predictions=predictions, references=labels)

# # 6. Data Collator (pad inputs correctly)
# def data_collator(features):
#     input_values = [f["input_values"][0] for f in features]
#     labels = [f["label"] for f in features]
#     batch = processor.pad(
#         {"input_values": input_values},
#         return_tensors="pt"
#     )
#     batch["labels"] = torch.tensor(labels)
#     return batch

# # 7. Training Arguments
# training_args = TrainingArguments(
#     output_dir="./wav2vec2-audio-class",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=3e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     logging_dir="./logs",
#     load_best_model_at_end=True,
#     fp16=torch.cuda.is_available(),
# )

# # 8. Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["validation"],
#     tokenizer=processor,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

