from datasets import load_dataset, Audio
from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2Model, AutoProcessor, AutoModelForCTC
import torch
import soundfile as sf

timit = load_dataset("kylelovesllms/timit_asr_ipa", split="train[:100]")
timit = timit.train_test_split(test_size=0.2)
timit = timit.select_columns(["audio", "phonetic_detail"])
timit = timit.cast_column("audio", Audio(sampling_rate=16000))

print(timit)
print(timit["train"][0])
# timit = timit.cast_column("audio", Audio(sampling_rate=16000))
# sampling_rate = timit.features["audio"].sampling_rate
# audio_file = timit[0]["audio"]
# print(audio_file)

# processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

# inputs = processor(audio_file, sampling_rate=16000, return_tensors="pt")

# model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# with torch.no_grad():
#     logits = model(**inputs).logits
    
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)
# print(transcription)