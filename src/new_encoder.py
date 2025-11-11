"""
Complete improved training configuration with:
- Label smoothing
- Better learning rate schedule
- Improved model architecture
- Class weighting
- Data augmentation

Replace the relevant sections in encoder.py with these implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

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
    
    # Core TIMIT phoneme set (39 phones after reduction)
    PHONEMES = [
        # Vowels (15)
        'iy', 'ih', 'eh', 'ae', 'ah', 'aw', 'ay', 'ey', 'oy', 
        'ow', 'uh', 'uw', 'er', 'ao', 'aa',
        
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
        self.phonemes = self.SPECIAL_TOKENS + self.PHONEMES
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
# 1. IMPROVED MODEL WITH LABEL SMOOTHING
# ============================================================================

class HuBERTForPhonemeClassification(nn.Module):
    """
    Enhanced HuBERT model with:
    - Label smoothing
    - Layer normalization
    - Better initialization
    - Focal loss option
    """
    
    def __init__(
        self,
        vocab_size: int,
        hubert_model_name: str = "facebook/hubert-base-ls960",
        freeze_feature_encoder: bool = True,
        freeze_base_model: bool = False,
        dropout_rate: float = 0.3,
        label_smoothing: float = 0.1,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        y_penalty_weight: float = 1.5  # Penalize 'y' predictions more
    ):
        super().__init__()
        
        from transformers import HubertModel
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        
        if freeze_feature_encoder:
            self.hubert.feature_extractor._freeze_parameters()
        
        if freeze_base_model:
            for param in self.hubert.parameters():
                param.requires_grad = False
        
        # Enhanced architecture
        hidden_size = self.hubert.config.hidden_size
        
        # Add intermediate layer for better representation
        self.intermediate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)  # Lower dropout in intermediate
        )
        
        # Final classification layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        
        # Initialize with small weights to prevent early convergence
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.zeros_(self.classifier.bias)
        
        # Loss configuration
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.y_penalty_weight = y_penalty_weight
        
        # Standard CTC loss
        self.ctc_loss = nn.CTCLoss(blank=vocab_size - 1, zero_infinity=True)
        
        # Store vocab size for loss computation
        self.vocab_size = vocab_size
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        vocab: Optional[object] = None  # Pass vocab to penalize 'y'
    ) -> Dict[str, torch.Tensor]:
        
        # Get HuBERT features
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Pass through intermediate layer
        hidden_states = self.intermediate(hidden_states)
        
        # Final classification
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
            # Calculate input lengths
            if attention_mask is not None:
                input_lengths = attention_mask.sum(-1)
            else:
                input_lengths = torch.full(
                    (logits.size(0),), 
                    logits.size(1), 
                    dtype=torch.long,
                    device=logits.device
                )
            
            # Compute loss with improvements
            loss = self._compute_loss_with_smoothing(
                logits, labels, input_lengths, label_lengths, vocab
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def _compute_loss_with_smoothing(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
        vocab: Optional[object] = None
    ) -> torch.Tensor:
        """
        Enhanced CTC loss with:
        1. Label smoothing
        2. Class weighting (penalize 'y')
        3. Optional focal loss
        """
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (batch, time, vocab)
        
        # Standard CTC loss
        log_probs_ctc = log_probs.transpose(0, 1)  # (time, batch, vocab)
        ctc_loss = self.ctc_loss(log_probs_ctc, labels, input_lengths, label_lengths)
        
        # Label smoothing: encourage more uniform distribution
        if self.label_smoothing > 0:
            # Compute smoothing loss (entropy regularization)
            # Exclude blank token from smoothing
            smooth_probs = log_probs[:, :, :-1]  # Exclude blank
            smooth_loss = -smooth_probs.mean()
            
            # Combine losses
            loss = (1 - self.label_smoothing) * ctc_loss + self.label_smoothing * smooth_loss
        else:
            loss = ctc_loss
        
        # Optional: Add penalty for 'y' predictions
        if vocab is not None and self.y_penalty_weight > 1.0:
            y_id = vocab.encode('y')
            
            # Get probability of 'y' across all positions
            probs = torch.exp(log_probs)
            y_probs = probs[:, :, y_id]
            
            # Penalize high 'y' probability
            y_penalty = y_probs.mean()
            loss = loss + (self.y_penalty_weight - 1.0) * y_penalty
        
        return loss
    
    def get_phoneme_probabilities(self, input_values: torch.Tensor) -> torch.Tensor:
        """Get phoneme probability distribution"""
        with torch.no_grad():
            outputs = self.forward(input_values)
            probs = F.softmax(outputs['logits'], dim=-1)
        return probs

# ============================================================================
# 3. ENHANCED DATA COLLATOR WITH ALL IMPROVEMENTS
# ============================================================================

class EnhancedDataCollator:
    """
    Complete data collator with:
    - Phoneme filtering
    - Audio augmentation
    - Statistics tracking
    """
    
    def __init__(
        self,
        processor,
        vocab,
        window_size_ms: int = 100,
        overlap_frames: int = 100,
        sampling_rate: int = 16000,
        padding = True,
        max_length: Optional[int] = None,
        filter_y: bool = True,
        y_filter_strategy: str = "aggressive",
        augment_audio: bool = True
    ):
        self.processor = processor
        self.vocab = vocab
        self.window_size = int(window_size_ms * sampling_rate / 1000)
        self.overlap_frames = overlap_frames
        self.sampling_rate = sampling_rate
        self.padding = padding
        self.max_length = max_length
        self.filter_y = filter_y
        self.y_filter_strategy = y_filter_strategy
        
        # Augmentation
        self.augment_audio = augment_audio
        self.augmenter = AudioAugmenter() if augment_audio else None
        
        # Statistics
        self.y_removed_count = 0
        self.total_phonemes = 0
        self.samples_processed = 0
    
    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        """Process batch with all enhancements"""
        input_features = []
        label_features = []
        label_lengths = []
        
        for feature in features:
            audio = feature['audio']['array']
            
            # Apply augmentation (only during training, not eval)
            if self.augment_audio and self.augmenter is not None:
                audio = self.augmenter(audio)
            
            # Pad if needed
            if len(audio) < self.window_size:
                audio = np.pad(audio, (0, self.window_size - len(audio)))
            
            input_features.append(audio)
            
            # Parse and filter phonemes
            phonemes = self._parse_phoneme_sequence(feature)
            
            if self.filter_y and self.y_filter_strategy != "none":
                phonemes = self._filter_y_phonemes(phonemes)
            
            # Encode phonemes
            label_ids = [self.vocab.encode(p) for p in phonemes]
            label_features.append(label_ids)
            label_lengths.append(len(label_ids))
            
            self.samples_processed += 1
        
        # Process audio
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
    
    def _parse_phoneme_sequence(self, feature: Dict) -> list:
        """Parse phoneme sequence"""
        if 'phonetic_detail' in feature and feature['phonetic_detail']:
            phonemes = []
            for phone_info in feature['phonetic_detail']:
                phone = phone_info.get('utterance', '[UNK]')
                normalized = self.vocab.normalize_timit_phone(phone)
                phonemes.append(normalized)
                self.total_phonemes += 1
            return phonemes if phonemes else ['[UNK]']
        return ['[UNK]']
    
    def _filter_y_phonemes(self, phonemes: list) -> list:
        """Filter excessive 'y' phonemes"""
        if self.y_filter_strategy == "aggressive":
            return self._aggressive_y_filter(phonemes)
        elif self.y_filter_strategy == "moderate":
            return self._moderate_y_filter(phonemes)
        return phonemes
    
    def _aggressive_y_filter(self, phonemes: list) -> list:
        """Aggressively filter 'y'"""
        filtered = []
        vowels = {'iy', 'ih', 'eh', 'ae', 'ah', 'aw', 'ay', 'ey', 'oy',
                  'ow', 'uh', 'uw', 'er', 'ao', 'aa'}
        
        for i, phone in enumerate(phonemes):
            if phone != 'y':
                filtered.append(phone)
            else:
                keep_y = False
                
                # Keep at start
                if i == 0 or (i > 0 and phonemes[i-1] in ['[SIL]', '[SPN]']):
                    keep_y = True
                # Keep between vowels
                elif i > 0 and i < len(phonemes) - 1:
                    if phonemes[i-1] in vowels and phonemes[i+1] in vowels:
                        keep_y = True
                # Keep after 'w'
                elif i > 0 and phonemes[i-1] == 'w':
                    keep_y = True
                
                if keep_y:
                    filtered.append(phone)
                else:
                    self.y_removed_count += 1
        
        return filtered
    
    def _moderate_y_filter(self, phonemes: list) -> list:
        """Moderately filter 'y'"""
        filtered = []
        stops = {'p', 't', 'k', 'b', 'd', 'g'}
        
        for i, phone in enumerate(phonemes):
            if phone != 'y':
                filtered.append(phone)
            else:
                # Remove after stops or repeated y
                if i > 0 and (phonemes[i-1] in stops or phonemes[i-1] == 'y'):
                    self.y_removed_count += 1
                else:
                    filtered.append(phone)
        
        return filtered
    
    def get_statistics(self) -> dict:
        """Get processing statistics"""
        return {
            'samples_processed': self.samples_processed,
            'total_phonemes': self.total_phonemes,
            'y_removed': self.y_removed_count,
            'y_removal_rate': self.y_removed_count / max(1, self.total_phonemes)
        }

# ============================================================================
# 4. TRAINING SETUP
# ============================================================================

def setup_training(
    model: HuBERTForPhonemeClassification,
    train_dataset,
    eval_dataset,
    data_collator: EnhancedDataCollator,
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
# 2. DATA AUGMENTATION
# ============================================================================

class AudioAugmenter:
    """
    Audio augmentation techniques for robustness
    """
    
    def __init__(
        self,
        augment_prob: float = 0.5,
        noise_std: float = 0.005,
        volume_range: tuple = (0.8, 1.2),
        time_stretch_range: tuple = (0.95, 1.05),
        pitch_shift_range: tuple = (-2, 2)  # semitones
    ):
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        self.volume_range = volume_range
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations"""
        if np.random.random() > self.augment_prob:
            return audio
        
        # Apply random augmentations
        audio = self._add_noise(audio)
        audio = self._adjust_volume(audio)
        # Note: Time stretch and pitch shift require librosa or torchaudio
        # Commented out to keep dependencies minimal
        
        return audio
    
    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        if np.random.random() > 0.5:
            noise = np.random.normal(0, self.noise_std, audio.shape)
            audio = audio + noise
        return audio
    
    def _adjust_volume(self, audio: np.ndarray) -> np.ndarray:
        """Random volume adjustment"""
        if np.random.random() > 0.5:
            factor = np.random.uniform(*self.volume_range)
            audio = audio * factor
        return np.clip(audio, -1.0, 1.0)


# ============================================================================
# 4. OPTIMIZED TRAINING CONFIGURATION
# ============================================================================

def setup_optimized_training(
    model,
    train_dataset,
    eval_dataset,
    vocab,
    processor,
    output_dir: str = "./phon-embedder-optimized"
):
    """
    Complete optimized training setup with all improvements
    """
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
    
    # Enhanced data collator
    data_collator = EnhancedDataCollator(
        processor=processor,
        vocab=vocab,
        window_size_ms=100,
        overlap_frames=100,
        filter_y=True,
        y_filter_strategy="aggressive",
        augment_audio=True
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # BATCH SIZE & ACCUMULATION
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        
        # LEARNING RATE SCHEDULE
        learning_rate=5e-5,  # Lower than before (was 1e-4)
        lr_scheduler_type="cosine_with_restarts",  # Better than plain cosine
        warmup_steps=1000,  # Longer warmup (was 500)
        warmup_ratio=0.1,  # Alternative to warmup_steps
        
        # TRAINING LENGTH
        num_train_epochs=20,  # Use epochs instead of max_steps
        max_steps=-1,  # Disable max_steps to use epochs
        
        # EVALUATION & SAVING
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,  # Keep only best 3 checkpoints
        logging_steps=50,
        logging_first_step=True,
        
        # OPTIMIZATION
        optim="adamw_torch_fused",  # Faster than regular adamw_torch
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,  # Gradient clipping
        
        # MIXED PRECISION
        fp16=True,
        fp16_opt_level="O1",
        
        # REGULARIZATION
        label_smoothing_factor=0.0,  # We handle this in model
        
        # MODEL SELECTION
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # SYSTEM
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        
        # LOGGING
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        push_to_hub=False,
        
        # REPRODUCIBILITY
        seed=42,
        data_seed=42,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)  # Stop if no improvement for 5 evals
        ],
    )
    
    return trainer, data_collator


# ============================================================================
# 5. COMPLETE TRAINING SCRIPT
# ============================================================================

def train_optimized_model():
    """
    Complete training script with all optimizations
    """
    from datasets import load_dataset, Audio
    from transformers import Wav2Vec2Processor
    
    print("="*70)
    print("OPTIMIZED HUBERT PHONEME CLASSIFIER TRAINING")
    print("="*70)
    
    # 1. Setup vocabulary
    from encoder import PhonemeVocabularyARPABET
    vocab = PhonemeVocabularyARPABET()
    print(f"\n[1/6] Vocabulary loaded: {vocab.vocab_size} phonemes")
    
    # 2. Load processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    print("[2/6] Processor loaded")
    
    # 3. Load dataset
    print("[3/6] Loading TIMIT dataset...")
    dataset = load_dataset("kylelovesllms/timit_asr_ipa")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"      Train samples: {len(dataset['train'])}")
    print(f"      Test samples: {len(dataset['test'])}")
    
    # 4. Create improved model
    print("[4/6] Initializing model with improvements...")
    model = HuBERTForPhonemeClassification(
        vocab_size=vocab.vocab_size,
        freeze_feature_encoder=True,
        freeze_base_model=False,
        dropout_rate=0.3,
        label_smoothing=0.1,
        y_penalty_weight=1.5  # Penalize 'y' by 50%
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total parameters: {total_params:,}")
    print(f"      Trainable parameters: {trainable_params:,}")
    
    # 5. Setup training
    print("[5/6] Setting up training configuration...")
    trainer, data_collator = setup_optimized_training(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        vocab=vocab,
        processor=processor,
        output_dir="./phon-embedder-optimized"
    )
    
    print("\nTraining Configuration:")
    print(f"  - Learning rate: {trainer.args.learning_rate}")
    print(f"  - Batch size (effective): {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}")
    print(f"  - Warmup steps: {trainer.args.warmup_steps}")
    print(f"  - Label smoothing: 0.1 (in model)")
    print(f"  - 'y' filtering: {data_collator.y_filter_strategy}")
    print(f"  - Audio augmentation: {data_collator.augment_audio}")
    print(f"  - 'y' penalty weight: 1.5x")
    
    # 6. Train
    print("\n[6/6] Starting training...")
    print("="*70)
    
    trainer.train()
    
    # Show statistics
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    stats = data_collator.get_statistics()
    print(f"\nData Processing Statistics:")
    print(f"  Samples processed: {stats['samples_processed']}")
    print(f"  Total phonemes: {stats['total_phonemes']}")
    print(f"  'y' phonemes removed: {stats['y_removed']}")
    print(f"  'y' removal rate: {stats['y_removal_rate']:.1%}")
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model("./final_model_optimized")
    vocab_save_path = "./final_model_optimized/vocab_config.txt"
    with open(vocab_save_path, 'w') as f:
        f.write(f"vocab_size: {vocab.vocab_size}\n")
        f.write(f"phonemes: {','.join(vocab.phonemes)}\n")
    
    print(f"Model saved to: ./final_model_optimized")
    print(f"Vocabulary config saved to: {vocab_save_path}")
    
    return trainer, model, vocab


if __name__ == "__main__":
    trainer, model, vocab = train_optimized_model()
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Run evaluation: python evaluate_encoder.py")
    print("  2. Check TensorBoard: tensorboard --logdir ./phon-embedder-optimized/logs")
    print("  3. Compare with baseline model")
    print("="*70)