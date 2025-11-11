"""
HuBERT-based Phoneme Classification Pipeline for Real-Time Audio Processing
Sliding window approach with CTC loss for phoneme embeddings
"""

import torch
import torch.nn as nn
from transformers import (
    HubertModel, 
    HubertConfig,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np

# ============================================================================
# 1. PHONEME VOCABULARY SETUP
# ============================================================================

class PhonemeVocabulary:
    """Standard American English IPA phonemes + special tokens"""
    
    # Consonants
    CONSONANTS = [
        'p', 'b', 't', 'd', 'k', 'g',  # Plosives
        'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h',  # Fricatives
        'tʃ', 'dʒ',  # Affricates
        'm', 'n', 'ŋ',  # Nasals
        'l', 'r',  # Liquids
        'w', 'j'  # Glides
    ]
    
    # Vowels
    VOWELS = [
        'i', 'ɪ', 'e', 'ɛ', 'æ',  # Front
        'ɑ', 'ɔ', 'o', 'ʊ', 'u',  # Back
        'ʌ', 'ə', 'ɚ',  # Central
        'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ'  # Diphthongs
    ]
    
    # Special tokens
    SPECIAL_TOKENS = [
        '[PAD]',      # Padding
        '[UNK]',      # Unknown phoneme
        '[SIL]',      # Silence
        '[SPN]',      # Spoken noise/pause
        '[CTC]',      # CTC blank token
    ]
    
    def __init__(self):
        self.phonemes = self.SPECIAL_TOKENS + self.CONSONANTS + self.VOWELS
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phonemes)}
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}
        self.vocab_size = len(self.phonemes)
        self.pad_token_id = self.phoneme_to_id['[PAD]']
        self.unk_token_id = self.phoneme_to_id['[UNK]']
        self.ctc_token_id = self.phoneme_to_id['[CTC]']
    
    def encode(self, phoneme: str) -> int:
        """Convert phoneme to ID"""
        return self.phoneme_to_id.get(phoneme, self.unk_token_id)
    
    def decode(self, token_id: int) -> str:
        """Convert ID to phoneme"""
        return self.id_to_phoneme.get(token_id, '[UNK]')
    
    def batch_decode(self, token_ids: List[int]) -> List[str]:
        """Convert list of IDs to phonemes"""
        return [self.decode(tid) for tid in token_ids]

class PhonemeVocabularyARPABET:
    """TIMIT ARPABET phoneme set (39 phonemes + special tokens)"""
    
    VOWELS = [
        # Vowels (15)
        'iy', 'ih', 'eh', 'ae', 'ah', 'aw', 'ay', 'ey', 'oy', 
        'ow', 'uh', 'uw', 'er', 'ao', 'aa',
    ]
    
    # Core TIMIT phoneme set (39 phones after reduction)
    CONSONANTS = [
        # Consonants (24)
        'b', 'd', 'g', 'p', 't', 'k',  # Stops
        'jh', 'ch',  # Affricates
        's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'hh',  # Fricatives
        'm', 'n', 'ng',  # Nasals
        'l', 'r', 'w', 'y',  # Semivowels/Liquids
    ]
    
    SPECIAL_TOKENS = [
        '[PAD]', '[UNK]', '[SIL]', '[SPN]', '[CTC]'
    ]
    
    def __init__(self):
        self.phonemes = self.SPECIAL_TOKENS + self.VOWELS + self.CONSONANTS
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phonemes)}
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}
        self.vocab_size = len(self.phonemes)
        self.pad_token_id = self.phoneme_to_id['[PAD]']
        self.unk_token_id = self.phoneme_to_id['[UNK]']
        self.ctc_token_id = self.phoneme_to_id['[CTC]']
    
    def encode(self, phoneme: str) -> int:
        """Convert phoneme to ID"""
        return self.phoneme_to_id.get(phoneme.lower(), self.unk_token_id)
    
    def decode(self, token_id: int) -> str:
        """Convert ID to phoneme"""
        return self.id_to_phoneme.get(token_id, '[UNK]')
    
    def batch_decode(self, token_ids: List[int]) -> List[str]:
        """Convert list of IDs to phonemes"""
        return [self.decode(tid) for tid in token_ids]
    
    def normalize_timit_phone(self, phone: str) -> str:
        """
        Normalize TIMIT phoneme variants to base set
        Maps 61-phone set to 39-phone set
        """
        mapping = {
            'ix': 'ih',    # merge to ih
            'ax': 'ah',    # merge to ah
            'ax-h': 'ah',
            'ux': 'uw',    # merge to uw
            'axr': 'er',   # merge to er
            'el': 'l',     # syllabic l
            'em': 'm',     # syllabic m
            'en': 'n',     # syllabic n
            'eng': 'ng',   # syllabic ng
            'nx': 'n',     # flap
            'dx': 't',     # flap
            'q': 't',      # glottal stop
            'hv': 'hh',    # merge to hh
            'bcl': 'b', 'dcl': 'd', 'gcl': 'g',
            'pcl': 'p', 'tcl': 't', 'kcl': 'k',
            'h#': '[SIL]', 'pau': '[SIL]', 'epi': '[SPN]',
            'sil': '[SIL]', 'spn': '[SPN]'
        }
        phone_lower = phone.lower()
        return mapping.get(phone_lower, phone_lower)


# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

class HuBERTForPhonemeClassification(nn.Module):
    """HuBERT with CTC head for phoneme classification"""
    
    def __init__(
        self,
        vocab_size: int,
        hubert_model_name: str = "facebook/hubert-base-ls960",
        freeze_feature_encoder: bool = True,
        drop_out : float = 0.3,
        freeze_base_model: bool = False
    ):
        super().__init__()
        
        # Load pre-trained HuBERT
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        
        # Freeze feature encoder (CNN) for efficiency if needed
        if freeze_feature_encoder:
            self.hubert.feature_extractor._freeze_parameters()
        
        # Optionally freeze entire base model (only train classification head)
        if freeze_base_model:
            for param in self.hubert.parameters():
                param.requires_grad = False
        
        # CTC Classification head
        self.dropout = nn.Dropout(drop_out)
        self.classifier = nn.Linear(self.hubert.config.hidden_size, vocab_size)
        
        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=vocab_size - 1, zero_infinity=True)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_values: (batch, sequence_length) - raw audio waveform
            attention_mask: (batch, sequence_length)
            labels: (batch, target_length) - phoneme IDs
            label_lengths: (batch,) - actual length of each label sequence
        """
        #print(input_values, attention_mask, sep="\n")
        # Get HuBERT hidden states
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden_size)
        hidden_states = self.dropout(hidden_states)
        
        # Phoneme logits
        logits = self.classifier(hidden_states)  # (batch, time, vocab_size)
        
        loss = None
        if labels is not None:
            # Calculate sequence lengths for CTC
            if attention_mask is not None:
                input_lengths = attention_mask.sum(-1)
            else:
                input_lengths = torch.full(
                    (logits.size(0),), 
                    logits.size(1), 
                    dtype=torch.long,
                    device=logits.device
                )
            
            # CTC expects log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # (time, batch, vocab)
            
            loss = self.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )
        
        return {
            'loss': loss,
            'logits': logits,  # (batch, time, vocab_size)
            'hidden_states': hidden_states  # Can extract for downstream use
        }
    
    def get_phoneme_probabilities(self, input_values: torch.Tensor) -> torch.Tensor:
        """Get phoneme probability distribution (for real-time inference)"""
        with torch.no_grad():
            outputs = self.forward(input_values)
            probs = torch.nn.functional.softmax(outputs['logits'], dim=-1)
        return probs


# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

class DataCollater:
    """
    Processes audio into sliding windows and prepares phoneme labels for CTC
    """
    
    y_ignored = 0
    
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        vocab: PhonemeVocabularyARPABET,
        window_size_ms: int = 100,
        overlap_frames: int = 100,
        sampling_rate: int = 16000,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None
    ):
        self.processor = processor
        self.vocab = vocab
        self.window_size = int(window_size_ms * sampling_rate / 1000)  # samples
        self.overlap_frames = overlap_frames
        self.sampling_rate = sampling_rate
        self.padding = padding
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples
        Each feature contains: audio array, phoneme sequence, timing info
        """
        # Extract audio
        input_features = []
        label_features = []
        label_lengths = []
        
        for feature in features:
            audio = feature['audio']['array']
            #print(audio)
            
            if len(audio) < self.window_size:
                audio = np.pad(audio, (0, self.window_size - len(audio)))

            # Apply sliding windows if audio is longer than window_size
            #if len(audio) > self.window_size:
                #windows = self._create_windows(audio)
                # For now, use first window (can be extended for full processing)
                #audio = windows[0]
            
            input_features.append(audio)
            
            # Process phoneme labels
            phonemes = self._parse_phoneme_sequence(feature)
            label_ids = [self.vocab.encode(p) for p in phonemes]
            label_features.append(label_ids)
            label_lengths.append(len(label_ids))
        
        # Pad and batch audio
        batch = self.processor(
            input_features,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length
        )
        
        # Pad labels
        max_label_len = max(label_lengths)
        labels_padded = []
        for labels in label_features:
            padded = labels + [self.vocab.pad_token_id] * (max_label_len - len(labels))
            labels_padded.append(padded)
        
        batch['labels'] = torch.tensor(labels_padded, dtype=torch.long)
        batch['label_lengths'] = torch.tensor(label_lengths, dtype=torch.long)
        
        return batch
    
    def _create_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        """Create overlapping sliding windows"""
        windows = []
        stride = self.window_size - self.overlap_frames
        
        for start in range(0, len(audio) - self.window_size + 1, stride):
            window = audio[start:start + self.window_size]
            windows.append(window)
        
        return windows
    
    def _parse_phoneme_sequence(self, feature: Dict) -> List[str]:
        if 'phonetic_detail' in feature and feature['phonetic_detail']:
            phonemes = []
            prev_phon = ""
            for phone_info in feature['phonetic_detail']:
                phone = phone_info.get('utterance', '[UNK]')
                normalized = self.vocab.normalize_timit_phone(phone)
                if normalized == 'y' and not (prev_phon in self.vocab.VOWELS and 'y' in prev_phon) :
                    phonemes.append(normalized)
                else: self.y_ignored += 1
                prev_phon = normalized
            return phonemes if phonemes else ['[UNK]']
        return ['[UNK]']
    
    def _map_timit_to_ipa(self, timit_phone: str) -> str:
        """Map TIMIT phoneme notation to IPA (customize as needed)"""
        # Basic mapping - expand based on TIMIT phoneme set
        mapping = {
            'sil': '[SIL]',
            'spn': '[SPN]',
            # Add more mappings as needed
        }
        return mapping.get(timit_phone.lower(), timit_phone)


# ============================================================================
# 4. TRAINING SETUP
# ============================================================================

def setup_training(
    model: HuBERTForPhonemeClassification,
    train_dataset,
    eval_dataset,
    data_collator: DataCollater,
    output_dir: str = "./phon-embedder"
):
    """Configure training arguments and trainer"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=10000,
        fp16=True,  # For faster training
        optim="adamw_torch",
        push_to_hub=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(3)],
    )
    
    return trainer


# ============================================================================
# 5. REAL-TIME INFERENCE
# ============================================================================

class RealtimePhonemeClassifier:
    """Real-time phoneme classification with sliding windows"""
    
    def __init__(
        self,
        model: HuBERTForPhonemeClassification,
        processor: Wav2Vec2Processor,
        vocab: PhonemeVocabularyARPABET,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device).eval()
        self.processor = processor
        self.vocab = vocab
        self.device = device
        self.window_size = 1600  # 100ms at 16kHz
    
    def classify_window(self, audio_chunk: np.ndarray) -> Dict:
        """
        Classify a single audio window
        
        Returns:
            Dict with 'probabilities' and 'predicted_phonemes'
        """
        # Ensure correct window size
        if len(audio_chunk) != self.window_size:
            # Pad or truncate
            if len(audio_chunk) < self.window_size:
                audio_chunk = np.pad(
                    audio_chunk, 
                    (0, self.window_size - len(audio_chunk))
                )
            else:
                audio_chunk = audio_chunk[:self.window_size]
        
        # Process audio
        inputs = self.processor(
            audio_chunk,
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(**inputs)['logits']
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Average over time dimension for window-level prediction
        window_probs = probs.mean(dim=1).squeeze().cpu().numpy()
        
        # Get top predictions
        top_k = 5
        top_indices = np.argsort(window_probs)[-top_k:][::-1]
        top_phonemes = [
            (self.vocab.decode(int(idx)), float(window_probs[idx]))
            for idx in top_indices
        ]
        
        return {
            'probabilities': window_probs,  # Full distribution
            'top_predictions': top_phonemes,
            'predicted_phoneme': top_phonemes[0][0]
        }
    
    def process_stream(
        self, 
        audio_stream: np.ndarray,
        overlap: int = 100
    ) -> List[Dict]:
        """Process audio stream with sliding windows"""
        results = []
        stride = self.window_size - overlap
        
        for start in range(0, len(audio_stream) - self.window_size + 1, stride):
            window = audio_stream[start:start + self.window_size]
            result = self.classify_window(window)
            result['start_sample'] = start
            result['end_sample'] = start + self.window_size
            results.append(result)
        
        return results


# ============================================================================
# 6. USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of the pipeline"""
    
    # Initialize vocabulary
    vocab = PhonemeVocabularyARPABET()
    print(f"Vocabulary size: {vocab.vocab_size}")
    
    # Initialize processor (using Wav2Vec2 processor for compatibility)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Initialize model
    model = HuBERTForPhonemeClassification(
        vocab_size=vocab.vocab_size,
        freeze_feature_encoder=True,
        freeze_base_model=False
    )
    
    # Load dataset
    print("Loading TIMIT dataset...")
    dataset = load_dataset("kylelovesllms/timit_asr_ipa")
    sample = dataset["train"][0]
    print("Audio shape:", sample["audio"]["array"].shape)
    print("Audio length:", len(sample["audio"]["array"]))
    print("Sample rate:", sample["audio"]["sampling_rate"])
    print("Duration (seconds):", len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"])
    
    # Cast audio to correct sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    sample = dataset["train"][0]
    print("Audio shape:", sample["audio"]["array"].shape)
    print("Audio length:", len(sample["audio"]["array"]))
    print("Sample rate:", sample["audio"]["sampling_rate"])
    print("Duration (seconds):", len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"])
    
    # Initialize data collator
    data_collator = DataCollater(
        processor=processor,
        vocab=vocab,
        window_size_ms=100,
        overlap_frames=100
    )
    
    # Setup training
    trainer = setup_training(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model("./final_model")
    
    # Real-time inference example
    # print("\nTesting real-time inference...")
    # classifier = RealtimePhonemeClassifier(model, processor, vocab)
    
    print(f"\nNumber of y phonemes deleted: {data_collator.y_ignored}")


if __name__ == "__main__":
    main()
