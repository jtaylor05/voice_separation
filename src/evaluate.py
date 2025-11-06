from datasets import load_dataset, Audio
from transformers import Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import torch
from torchcodec.decoders import AudioDecoder
from data import create_dataset, prepare_dataset, DataCollatorCTCWithPadding
from vocab_maker import make_vocab_file
from evaluate import load
from huggingface_hub import login
import numpy as np
import sys

def prepare_compute_metrics(processor):
    def compute_metrics(pred):
        nonlocal processor
        cer = load("cer")
        
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        cer_rating = cer.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}
    return compute_metrics

def evaluate_cer(model, dataset):
    processor = Wav2Vec2Processor.from_pretrained(model)
    model = Wav2Vec2ForCTC.from_pretrained(model)
    
    trainer = Trainer(
        model=model,
        args = TrainingArguments(label_names=["labels"]),
        train_dataset=dataset["train"],
        eval_dataset= dataset["test"],
        compute_metrics=prepare_compute_metrics(processor),
        data_collator=DataCollatorCTCWithPadding(processor = processor, padding = True),
        tokenizer=processor.feature_extractor
    )
    trainer.evaluate()
    
def show_cer(model, dataset):
    processor = Wav2Vec2Processor.from_pretrained(model)
    model = Wav2Vec2ForCTC.from_pretrained(model)
    
    def map_to_result(batch):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
            logits = model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = processor.batch_decode(pred_ids)[0]
        batch["text"] = processor.decode(batch["labels"], group_tokens=False)
        
        return batch
        
    cer_metric = load("cer")

    results = dataset["test"].map(map_to_result, remove_columns=dataset["test"].column_names)

    print("Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], references=results["text"])))

if __name__ == "__main__":
    login()
    
    args = sys.argv
    if len(args) != 2:
        raise SystemError("No Input Error") 
    
    model_name = args[1]
    
    data = create_dataset(data_split="train[:1000]", columns=["audio", "phonetic_detail", "ipa_transcription"])
    prep_data = data.map(prepare_dataset, remove_columns=data.column_names["train"], num_proc=4, fn_kwargs={"processor":processor})
    
    evaluate_cer(model_name, prep_data)
    show_cer(model_name, prep_data)