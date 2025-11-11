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
from collections import Counter

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
        self.phonemes = self.VOWELS + self.CONSONANTS + self.SPECIAL_TOKENS
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
        self.layer_norm = nn.LayerNorm(self.hubert.config.hidden_size)
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
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Phoneme logits
        logits = self.classifier(hidden_states)  # (batch, time, vocab_size)
        
        loss = None
        if labels is not None:
            # Calculate sequence lengths for CTC
            if attention_mask is not None:
                input_lengths = attention_mask.sum(-1)
                input_lengths = input_lengths // 320
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
        sampling_rate: int = 16000,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None
    ):
        self.processor = processor
        self.vocab = vocab
        self.sampling_rate = sampling_rate
        self.padding = padding
        self.max_length = max_length
        self.phoneme_stats = Counter()
    
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
            
            input_features.append(audio)
            
            # Process phoneme labels
            phonemes = self._parse_phoneme_sequence(feature)
            
            self.phoneme_stats.update(phonemes)
            
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
    
    def _parse_phoneme_sequence(self, feature: Dict) -> List[str]:
        if 'phonetic_detail' not in feature or not feature['phonetic_detail']:
            return ['[UNK]']
        
        phonemes = []
        prev_phone = None
        
        for phone_info in feature['phonetic_detail']:
            phone = phone_info.get('utterance', '[UNK]')
            normalized = self.vocab.normalize_timit_phone(phone)
            
            # Remove consecutive duplicates (common in TIMIT)
            if normalized != prev_phone:
                phonemes.append(normalized)
                prev_phone = normalized
        
        return phonemes if phonemes else ['[UNK]']
    
    def print_statistics(self):
        """Print phoneme distribution"""
        print("\n" + "="*60)
        print("PHONEME STATISTICS")
        print("="*60)
        total = sum(self.phoneme_stats.values())
        print(f"Total phonemes: {total}")
        print(f"Unique phonemes: {len(self.phoneme_stats)}")
        print("\nTop 20 most common:")
        for phone, count in self.phoneme_stats.most_common(20):
            print(f"  {phone:6s}: {count:6d} ({100*count/total:5.2f}%)")


# ============================================================================
# 4. TRAINING SETUP
# ============================================================================

def ctc_greedy_decode(logits: torch.Tensor, blank_id: int) -> List[List[int]]:
    """
    Greedy CTC decoding: collapse repeated tokens and remove blanks
    
    Args:
        logits: (batch, time, vocab_size)
        blank_id: ID of the CTC blank token
    
    Returns:
        List of decoded sequences (list of token IDs)
    """
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)  # (batch, time)
    
    decoded_sequences = []
    for pred_seq in predictions:
        decoded = []
        prev_token = None
        
        for token in pred_seq.cpu().numpy():
            # Skip blanks
            if token == blank_id:
                prev_token = None
                continue
            
            # Skip repeated tokens (CTC collapse)
            if token != prev_token:
                decoded.append(int(token))
                prev_token = token
        
        decoded_sequences.append(decoded)
    
    return decoded_sequences

def compute_metrics(pred_ids: List[List[int]], 
                    label_ids: List[List[int]], 
                    vocab: PhonemeVocabularyARPABET,
                    blank_id: int) -> Dict:
    """
    Compute PER (Phoneme Error Rate) using edit distance
    """
    from Levenshtein import distance as levenshtein_distance
    
    total_distance = 0
    total_length = 0
    exact_matches = 0
    
    for pred, label in zip(pred_ids, label_ids):
        # Remove padding from labels
        label_clean = [l for l in label if l != vocab.pad_token_id]
        
        # Decode predictions (already CTC-decoded)
        pred_phones = vocab.batch_decode(pred)
        label_phones = vocab.batch_decode(label_clean)
        
        # Convert to strings for edit distance
        pred_str = ' '.join(pred_phones)
        label_str = ' '.join(label_phones)
        
        dist = levenshtein_distance(pred_str, label_str)
        total_distance += dist
        total_length += len(label_str)
        
        if pred_str == label_str:
            exact_matches += 1
    
    per = total_distance / max(total_length, 1)
    accuracy = exact_matches / len(pred_ids)
    
    return {
        'phoneme_error_rate': per,
        'accuracy': accuracy,
        'total_samples': len(pred_ids)
    }

def setup_training(
    model: HuBERTForPhonemeClassification,
    train_dataset,
    eval_dataset,
    data_collator: DataCollater,
    vocab : PhonemeVocabularyARPABET,
    output_dir: str = "./phon-embedder"
):
    """Configure training arguments and trainer"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # Increased from 4
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
        learning_rate=5e-5,  # Reduced from 1e-4
        warmup_steps=1000,
        max_steps=15000,
        fp16=True,
        optim="adamw_torch",
        weight_decay=0.01,
        push_to_hub=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Preprocess logits to extract only what we need for metrics
    def preprocess_logits_for_metrics(logits, labels):
        """
        The logits come as (batch, time, vocab_size) but time dimension varies.
        We need to pad to max length in the batch for stacking.
        """
        if isinstance(logits, torch.Tensor):
            # Already a tensor, return as is
            return logits.detach().cpu()
        
        # If it's already processed, return
        return logits
    
    # Custom compute metrics function
    def compute_metrics_wrapper(eval_pred):
        predictions, labels = eval_pred
        
        # Debug: print what we received
        # print(f"Predictions type: {type(predictions)}")
        # if isinstance(predictions, np.ndarray):
        #     print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
        
        decoded_preds = []
        
        # The Trainer concatenates predictions across batches into a numpy array
        # but with variable time dimensions, it creates an object array
        if isinstance(predictions, np.ndarray):
            # Check if it's an object array (ragged)
            if predictions.dtype == object:
                # Each element is a separate array with potentially different time dim
                for pred in predictions:
                    pred_tensor = torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred
                    if pred_tensor.dim() == 2:  # (time, vocab)
                        pred_tensor = pred_tensor.unsqueeze(0)  # (1, time, vocab)
                    sample_decoded = ctc_greedy_decode(pred_tensor, blank_id=vocab.ctc_token_id)
                    decoded_preds.extend(sample_decoded)
            else:
                # Regular array - can process normally
                predictions_tensor = torch.from_numpy(predictions)
                decoded_preds = ctc_greedy_decode(predictions_tensor, blank_id=vocab.ctc_token_id)
        elif isinstance(predictions, torch.Tensor):
            decoded_preds = ctc_greedy_decode(predictions, blank_id=vocab.ctc_token_id)
        else:
            # Last resort: iterate
            for pred in predictions:
                pred_tensor = torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred
                if pred_tensor.dim() == 2:
                    pred_tensor = pred_tensor.unsqueeze(0)
                sample_decoded = ctc_greedy_decode(pred_tensor, blank_id=vocab.ctc_token_id)
                decoded_preds.extend(sample_decoded)
        
        # Convert labels to list if needed
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        
        # Compute metrics
        metrics = compute_metrics(decoded_preds, labels, vocab, blank_id=vocab.ctc_token_id)
        return metrics
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        #compute_metrics=compute_metrics_wrapper,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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
        vocab=vocab
    )
    
    # Setup training
    trainer = setup_training(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        vocab=vocab,
        data_collator=data_collator
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model("./final_model")
    
    data_collator.print_statistics()


if __name__ == "__main__":
    main()
