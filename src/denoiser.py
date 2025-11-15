"""
FlowAVSE Speech Enhancement with Phoneme Embedding Conditioning
Replaces text encoder with frozen HuBERT phoneme embeddings
Based on: https://github.com/kaistmm/FlowAVSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Audio, concatenate_datasets
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import torchaudio
import math

# Import the phoneme encoder from previous pipeline
from encoder import (
    HuBERTForPhonemeClassification,
    PhonemeVocabularyARPABET,
    RealtimePhonemeClassifier
)


# ============================================================================
# 1. PHONEME ADAPTER LAYER
# ============================================================================

class PhonemeConditioningAdapter(nn.Module):
    """
    Projects HuBERT phoneme embeddings to FlowAVSE conditioning space
    Maintains temporal resolution while adapting dimensionality
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # HuBERT hidden size
        output_dim: int = 512,  # FlowAVSE conditioning dimension
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Multi-layer projection with residual connections
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else output_dim
            layers.extend([
                nn.Linear(in_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.projection = nn.Sequential(*layers)
        
        # Optional: Add positional encoding for better temporal modeling
        self.pos_encoding = PositionalEncoding(output_dim)
    
    def forward(self, phoneme_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phoneme_embeddings: (batch, time, 768) from HuBERT
        Returns:
            conditioned: (batch, time, output_dim)
        """
        # Project to conditioning space
        conditioned = self.projection(phoneme_embeddings)
        
        # Add positional information
        conditioned = self.pos_encoding(conditioned)
        
        return conditioned


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


# ============================================================================
# 2. FLOW MATCHING COMPONENTS
# ============================================================================

class ConditionalFlowMatching(nn.Module):
    """
    Flow Matching for speech enhancement conditioned on phoneme embeddings
    Uses Optimal Transport path for training
    """
    
    def __init__(self, sigma_max : float = 0.5, sigma_min: float = 1e-4, epsilon : float = 1e-5):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.epsilon = epsilon
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps uniformly"""
        return torch.rand(batch_size, device=device)
    
    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Variance-preserving schedule: σ_t = sqrt(t * (1-t)) * σ_max"""
        return torch.sqrt(t * (1 - t) + self.epsilon) * self.sigma_max
    
    def compute_conditional_flow(
        self,
        x0: torch.Tensor,  # Clean
        x1: torch.Tensor,  # Noisy
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_shape = t.view(-1, 1, 1)
        
        # OT path with variance preservation
        mu_t = t_shape * x0 + (1 - t_shape) * x1
        sigma_t = self.compute_sigma_t(t).view(-1, 1, 1)
        noise = torch.randn_like(x0)
        xt = mu_t + sigma_t * noise
        
        # Velocity with noise correction
        noise_correction = (1 - 2 * t_shape) * (self.sigma_max ** 2) * noise / (2 * sigma_t + self.epsilon)
        ut = (x0 - x1) + noise_correction
        
        return xt, ut
    
    def _compute_conditional_flow(
        self,
        x0: torch.Tensor,  # Clean speech
        x1: torch.Tensor,  # Noisy speech
        t: torch.Tensor,   # Time
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow path using Optimal Transport
        
        Args:
            x0: Clean speech (batch, channels, time)
            x1: Noisy speech (batch, channels, time)
            t: Timesteps (batch,)
        
        Returns:
            xt: Interpolated state
            ut: Target velocity (conditional flow)
        """
        t = t.view(-1, 1, 1)  # (batch, 1, 1)
        
        # OT path: x_t = (1-t)*x_0 + t*x_1 + sigma_t * noise
        mu_t = (1 - t) * x0 + t * x1
        sigma_t = self.sigma_min
        
        # Sample noise
        noise = torch.randn_like(x0)
        xt = mu_t + sigma_t * noise
        
        # Target velocity: dx/dt = x_1 - x_0
        ut = x0 - x1
        
        return xt, ut
    
    @torch.no_grad()
    def sample_ode(
        self,
        x1: torch.Tensor,  # Noisy speech
        model: nn.Module,
        phoneme_condition: torch.Tensor,
        steps: int = 50,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Sample from the learned flow using ODE solver
        
        Args:
            x1: Starting point (noisy speech)
            model: Velocity field estimator
            phoneme_condition: Phoneme embeddings
            steps: Number of integration steps
            method: 'euler' or 'heun'
        
        Returns:
            x0: Denoised speech
        """
        dt = 1.0 / steps
        x = x1.clone()
        
        for i in range(steps):
            t = torch.ones(x.shape[0], device=x.device) * (1 - i * dt)
            
            # Predict velocity
            v = model(x, t, phoneme_condition)
            
            if method == 'euler':
                x = x - dt * v
            elif method == 'heun':
                # Heun's method (2nd order)
                x_temp = x - dt * v
                t_next = t - dt
                v_next = model(x_temp, t_next, phoneme_condition)
                x = x - dt * (v + v_next) / 2
        
        return x


# ============================================================================
# 3. AUDIO ENCODER (for noisy input)
# ============================================================================

class AudioEncoder(nn.Module):
    """
    Encodes noisy audio input into latent representation
    Uses strided convolutions for efficient processing
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 4,
        stride: int = 2
    ):
        super().__init__()
        
        layers = []
        channels = base_channels
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else channels // 2
            out_ch = channels
            
            layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding=kernel_size//2),
                nn.GroupNorm(8, out_ch),
                nn.SiLU()
            ))
            
            channels *= 2
        
        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, time)
        Returns:
            encoded: (batch, channels, reduced_time)
        """
        return self.encoder(x)


class AudioDecoder(nn.Module):
    """
    Decodes latent representation back to audio
    Uses transposed convolutions for upsampling
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 4,
        stride: int = 2,
        out_channels: int = 1
    ):
        super().__init__()
        
        layers = []
        channels = in_channels
        
        for i in range(num_layers):
            out_ch = channels // 2 if i < num_layers - 1 else out_channels
            
            layers.append(nn.Sequential(
                nn.ConvTranspose1d(channels, out_ch, kernel_size, stride, padding=kernel_size//2, output_padding=1),
                nn.GroupNorm(8, out_ch) if i < num_layers - 1 else nn.Identity(),
                nn.SiLU() if i < num_layers - 1 else nn.Tanh()
            ))
            
            channels = out_ch
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, reduced_time)
        Returns:
            decoded: (batch, 1, time)
        """
        return self.decoder(x)


# ============================================================================
# 4. CROSS-ATTENTION FOR PHONEME CONDITIONING
# ============================================================================

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between audio features and phoneme embeddings
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,  # Audio features (query)
        condition: torch.Tensor,  # Phoneme embeddings (key, value)
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, time, dim)
            condition: (batch, time_cond, dim)
        Returns:
            out: (batch, time, dim)
        """
        # Cross-attention
        attn_out, _ = self.attention(
            query=self.norm1(x),
            key=condition,
            value=condition,
            key_padding_mask=mask
        )
        x = x + attn_out
        
        # Feedforward
        x = x + self.ffn(self.norm2(x))
        
        return x


# ============================================================================
# 5. MAIN FLOWAVSE MODEL
# ============================================================================

class FlowAVSEPhonemeConditioned(nn.Module):
    """
    Complete FlowAVSE model with phoneme conditioning
    Replaces text encoder with frozen HuBERT phoneme embeddings
    """
    
    def __init__(
        self,
        phoneme_encoder: HuBERTForPhonemeClassification,
        phoneme_adapter: PhonemeConditioningAdapter,
        d_model: int = 512,
        num_cross_attention_layers: int = 4,
        freeze_phoneme_encoder: bool = True,
        audio_encoder_layers: int = 6
    ):
        super().__init__()
        
        # Phoneme encoder (frozen)
        self.phoneme_encoder = phoneme_encoder
        if freeze_phoneme_encoder:
            for param in self.phoneme_encoder.parameters():
                param.requires_grad = False
        
        # Adapter for phoneme embeddings
        self.phoneme_adapter = phoneme_adapter
        
        # Audio encoder/decoder
        self.audio_encoder = AudioEncoder(num_layers=audio_encoder_layers)
        self.audio_decoder = AudioDecoder(
            in_channels=self.audio_encoder.out_channels,
            num_layers=audio_encoder_layers
        )
        
        # Project encoded audio to d_model for attention
        self.audio_projection = nn.Linear(self.audio_encoder.out_channels, d_model)
        
        # Time embedding for flow matching
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(d_model)
            for _ in range(num_cross_attention_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, self.audio_encoder.out_channels)
        
        # Flow matching
        self.flow_matching = ConditionalFlowMatching()
    
    def encode_phonemes(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract phoneme embeddings from audio
        
        Args:
            audio: (batch, time) raw waveform
        Returns:
            phoneme_embeddings: (batch, time_frames, 768)
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.phoneme_encoder(
                input_values=audio,
                attention_mask=attention_mask
            )
            phoneme_embeddings = outputs['hidden_states']  # (batch, time, 768)
        
        return phoneme_embeddings
    
    def forward(
        self,
        noisy_audio: torch.Tensor,
        clean_audio: Optional[torch.Tensor] = None,
        phoneme_embeddings: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference
        
        Args:
            noisy_audio: (batch, time) noisy waveform
            clean_audio: (batch, time) clean waveform (for training)
            phoneme_embeddings: (batch, time, 768) pre-computed embeddings
            return_loss: Whether to compute and return loss
        
        Returns:
            Dictionary with 'loss', 'denoised_audio', etc.
        """
        batch_size = noisy_audio.shape[0]
        device = noisy_audio.device
        
        # Get phoneme embeddings (either pre-computed or on-the-fly)
        if phoneme_embeddings is None:
            phoneme_embeddings = self.encode_phonemes(noisy_audio)
        
        # Project phoneme embeddings to conditioning space
        phoneme_condition = self.phoneme_adapter(phoneme_embeddings)  # (batch, time, d_model)
        
        # Encode noisy audio
        noisy_audio_2d = noisy_audio.unsqueeze(1)  # (batch, 1, time)
        audio_encoded = self.audio_encoder(noisy_audio_2d)  # (batch, channels, reduced_time)
        
        if return_loss and clean_audio is not None:
            # Training mode: Flow matching loss
            clean_audio_2d = clean_audio.unsqueeze(1)
            clean_encoded = self.audio_encoder(clean_audio_2d)
            
            # Sample time and compute flow
            t = self.flow_matching.sample_time(batch_size, device)
            xt, ut = self.flow_matching.compute_conditional_flow(
                clean_encoded,
                audio_encoded,
                t
            )
            
            # Predict velocity
            velocity_pred = self._predict_velocity(xt, t, phoneme_condition)
            
            # Flow matching loss
            loss = F.mse_loss(velocity_pred, ut)
            
            return {
                'loss': loss,
                'velocity_pred': velocity_pred,
                'velocity_target': ut
            }
        else:
            # Inference mode: Sample from flow
            denoised_encoded = self.flow_matching.sample_ode(
                audio_encoded,
                self._predict_velocity_wrapper(phoneme_condition),
                steps=50
            )
            
            # Decode to audio
            denoised_audio = self.audio_decoder(denoised_encoded)
            denoised_audio = denoised_audio.squeeze(1)  # (batch, time)
            
            return {
                'denoised_audio': denoised_audio,
                'phoneme_embeddings': phoneme_embeddings
            }
    
    def _predict_velocity(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        phoneme_condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field for flow matching
        
        Args:
            xt: (batch, channels, time) current state
            t: (batch,) timestep
            phoneme_condition: (batch, time_cond, d_model)
        """
        # Embed time
        t_emb = self.time_embedding(t.unsqueeze(-1))  # (batch, d_model)
        
        # Project audio features to d_model
        # xt: (batch, channels, time) -> (batch, time, channels)
        xt_transposed = xt.transpose(1, 2)
        audio_features = self.audio_projection(xt_transposed)  # (batch, time, d_model)
        
        # Add time embedding
        audio_features = audio_features + t_emb.unsqueeze(1)
        
        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            audio_features = layer(audio_features, phoneme_condition)
        
        # Project back to audio space
        velocity = self.output_projection(audio_features)  # (batch, time, channels)
        velocity = velocity.transpose(1, 2)  # (batch, channels, time)
        
        return velocity
    
    def _predict_velocity_wrapper(self, phoneme_condition: torch.Tensor):
        """Create a wrapper function for ODE sampling"""
        def predict(xt, t, condition):
            return self._predict_velocity(xt, t, condition)
        
        return lambda xt, t: predict(xt, t, phoneme_condition)


# ============================================================================
# 6. DATA PREPROCESSING WITH NOISE AUGMENTATION
# ============================================================================

class NoiseAugmentation:
    """Add noise to clean speech for training"""
    
    def __init__(
        self,
        noise_dataset_paths: Optional[List[str]] = None,
        snr_range: Tuple[float, float] = (-5, 20),  # dB
        sampling_rate: int = 16000
    ):
        self.snr_range = snr_range
        self.sampling_rate = sampling_rate
        self.noise_samples = []
        
        # Load noise dataset if provided
        if noise_dataset_paths:
            for path in noise_dataset_paths:
                # Could load from file, dataset, etc.
                pass
    
    def add_noise(
        self,
        clean_audio: np.ndarray,
        noise_type: str = 'white'
    ) -> np.ndarray:
        """
        Add noise to clean audio
        
        Args:
            clean_audio: Clean speech signal
            noise_type: 'white', 'babble', 'hardware', etc.
        
        Returns:
            noisy_audio: Noisy speech signal
        """
        # Sample SNR
        snr_db = np.random.uniform(*self.snr_range)
        
        # Generate or select noise
        if noise_type == 'white':
            noise = np.random.randn(len(clean_audio))
        else:
            # Use pre-loaded noise samples
            noise = np.random.randn(len(clean_audio))  # Placeholder
        
        # Calculate noise power for desired SNR
        signal_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        snr_linear = 10 ** (snr_db / 10)
        noise_scaling = np.sqrt(signal_power / (snr_linear * noise_power))
        
        noisy_audio = clean_audio + noise_scaling * noise
        
        return noisy_audio


class FlowAVSEDataCollator:
    """
    Data collator for FlowAVSE training
    Handles windowing, noise augmentation, and embedding caching
    """
    
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        phoneme_encoder: HuBERTForPhonemeClassification,
        vocab: PhonemeVocabularyARPABET,
        window_size_ms: int = 1000,  # Window size in milliseconds
        overlap: float = 0.2,
        sampling_rate: int = 16000,
        use_cached_embeddings: bool = False,
        noise_augmentation: Optional[NoiseAugmentation] = None
    ):
        self.processor = processor
        #self.phoneme_encoder = phoneme_encoder
        self.phoneme_encoder = None
        self.vocab = vocab
        self.window_size = int(window_size_ms * sampling_rate / 1000)
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.use_cached_embeddings = use_cached_embeddings
        self.noise_augmentation = noise_augmentation or NoiseAugmentation()
        
        # Cache for phoneme embeddings
        self.embedding_cache = {}
        
        if phoneme_encoder is not None:
            self.phoneme_encoder = phoneme_encoder.cpu()
            self.phoneme_encoder.eval()
        else:
            self.phoneme_encoder = None
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples
        
        Each feature should contain:
        - audio: clean speech
        - (optionally) phoneme_embeddings: pre-computed embeddings
        """
        clean_audios = []
        noisy_audios = []
        phoneme_embeddings_list = []
        
        for feature in features:
            clean_audio = feature['clean']['array']
            noisy_audio = feature['noisy']['array']
            
            # Apply windowing
            clean_windows = self._create_windows(clean_audio)
            noisy_windows = self._create_windows(noisy_audio)
            
            for clean_window, noisy_window in zip(clean_windows, noisy_windows):
                clean_audios.append(clean_window)
                noisy_audios.append(noisy_window)
                
                # Get phoneme embeddings (cached or computed)
                if self.use_cached_embeddings:
                    # In practice, you'd compute and cache these during preprocessing
                    embedding = self._compute_phoneme_embedding(noisy_window)
                else:
                    embedding = None  # Will be computed on-the-fly
                
                phoneme_embeddings_list.append(embedding)
        
        # Convert to tensors
        clean_batch = torch.tensor(np.array(clean_audios), dtype=torch.float32)
        noisy_batch = torch.tensor(np.array(noisy_audios), dtype=torch.float32)
        
        batch = {
            'clean_audio': clean_batch,
            'noisy_audio': noisy_batch,
        }
        
        if self.use_cached_embeddings and phoneme_embeddings_list[0] is not None:
            # Stack embeddings if available
            batch['phoneme_embeddings'] = torch.stack(phoneme_embeddings_list)
        
        return batch
    
    def _create_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        """Create overlapping windows from audio"""
        stride = int(self.window_size * (1 - self.overlap))
        windows = []
        
        for start in range(0, len(audio) - self.window_size + 1, stride):
            window = audio[start:start + self.window_size]
            windows.append(window)
        
        # Pad last window if needed
        if len(audio) >= self.window_size and len(windows) == 0:
            windows.append(audio[:self.window_size])
        
        return windows if windows else [np.pad(audio, (0, self.window_size - len(audio)))]
    
    @torch.no_grad()
    def _compute_phoneme_embedding(self, audio: np.ndarray) -> torch.Tensor:
        """Compute phoneme embedding for caching"""
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        
        # Get embeddings from phoneme encoder
        outputs = self.phoneme_encoder(audio_tensor)
        embeddings = outputs['hidden_states']  # (1, time, 768)
        
        return embeddings.squeeze(0)  # (time, 768)


# ============================================================================
# 7. TRAINING SETUP
# ============================================================================

def setup_flowavse_training(
    phoneme_encoder_path: str,
    output_dir: str = "./flowavse_phoneme",
    window_size_ms: int = 1000,
    use_cached_embeddings: bool = False,
    d_model: int = 512
):
    """Setup FlowAVSE training pipeline"""
    
    # Load vocabulary and processor
    vocab = PhonemeVocabularyARPABET()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Load pre-trained phoneme encoder
    phoneme_encoder = HuBERTForPhonemeClassification(
        vocab_size=vocab.vocab_size
    )
    from safetensors.torch import load_file
    phoneme_encoder.load_state_dict(load_file(f"{phoneme_encoder_path}/model.safetensors"))
    
    # Create phoneme adapter
    phoneme_adapter = PhonemeConditioningAdapter(
        input_dim=768,
        output_dim=d_model
    )
    
    # Create FlowAVSE model
    model = FlowAVSEPhonemeConditioned(
        phoneme_encoder=phoneme_encoder,
        phoneme_adapter=phoneme_adapter,
        d_model=d_model,
        freeze_phoneme_encoder=True
    )
    
    # Load dataset
    print("Loading dataset...")
    #dataset = load_dataset("kylelovesllms/timit_asr_ipa")
    #dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    dataset = dataset.cast_column("noisy", Audio(sampling_rate=16000))
    dataset = dataset.cast_column("clean", Audio(sampling_rate=16000))

    # Create data collator
    data_collator = FlowAVSEDataCollator(
        processor=processor,
        phoneme_encoder=phoneme_encoder,
        vocab=vocab,
        window_size_ms=window_size_ms,
        use_cached_embeddings=use_cached_embeddings
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        logging_steps=50,
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=10000,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Custom trainer for FlowAVSE
    trainer = FlowAVSETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    return trainer, model


class FlowAVSETrainer(Trainer):
    """Custom trainer for FlowAVSE"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        """Compute flow matching loss"""
        outputs = model(
            noisy_audio=inputs['noisy_audio'],
            clean_audio=inputs['clean_audio'],
            return_loss=True
        )
        
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# 8. INFERENCE PIPELINE
# ============================================================================

class RealtimeFlowAVSE:
    """Real-time inference pipeline for FlowAVSE"""
    
    def __init__(
        self,
        model: FlowAVSEPhonemeConditioned,
        processor: Wav2Vec2Processor,
        window_size_ms: int = 1000,
        overlap: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device).eval()
        self.processor = processor
        self.window_size = int(window_size_ms * 16000 / 1000)
        self.overlap = overlap
        self.device = device
    
    @torch.no_grad()
    def denoise(self, noisy_audio: np.ndarray) -> np.ndarray:
        """
        Denoise audio stream
        
        Args:
            noisy_audio: Noisy audio array (16kHz)
        
        Returns:
            denoised_audio: Cleaned audio
        """
        stride = int(self.window_size * (1 - self.overlap))
        denoised_chunks = []
        
        for start in range(0, len(noisy_audio) - self.window_size + 1, stride):
            window = noisy_audio[start:start + self.window_size]
            
            # Convert to tensor
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Denoise
            outputs = self.model(window_tensor, return_loss=False)
            denoised_window = outputs['denoised_audio'].cpu().numpy()[0]
            
            denoised_chunks.append(denoised_window)
        
        # Overlap-add reconstruction
        output = self._overlap_add(denoised_chunks, stride)
        
        return output
    
    def _overlap_add(self, chunks: List[np.ndarray], stride: int) -> np.ndarray:
        """Reconstruct signal from overlapping windows"""
        if not chunks:
            return np.array([])
        
        output_len = (len(chunks) - 1) * stride + self.window_size
        output = np.zeros(output_len)
        window_sum = np.zeros(output_len)
        
        # Hann window for smooth blending
        window = np.hanning(self.window_size)
        
        for i, chunk in enumerate(chunks):
            start = i * stride
            end = start + self.window_size
            output[start:end] += chunk * window
            window_sum[start:end] += window
        
        # Normalize by window overlap
        window_sum = np.maximum(window_sum, 1e-8)
        output = output / window_sum
        
        return output


# ============================================================================
# 9. COMPARATIVE BASELINE (Original FlowAVSE with Text)
# ============================================================================

class TextEncoder(nn.Module):
    """Simple text encoder for baseline comparison"""
    
    def __init__(
        self,
        vocab_size: int = 100,  # Character vocabulary
        d_model: int = 512,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_ids: (batch, seq_len) character IDs
        Returns:
            encoded: (batch, seq_len, d_model)
        """
        x = self.embedding(text_ids)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return x


class FlowAVSEBaseline(nn.Module):
    """Original FlowAVSE with text conditioning (for comparison)"""
    
    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 512,
        num_cross_attention_layers: int = 4,
        audio_encoder_layers: int = 6
    ):
        super().__init__()
        
        # Text encoder instead of phoneme encoder
        self.text_encoder = TextEncoder(vocab_size, d_model)
        
        # Audio encoder/decoder
        self.audio_encoder = AudioEncoder(num_layers=audio_encoder_layers)
        self.audio_decoder = AudioDecoder(
            in_channels=self.audio_encoder.out_channels,
            num_layers=audio_encoder_layers
        )
        
        # Rest is similar to phoneme-conditioned version
        self.audio_projection = nn.Linear(self.audio_encoder.out_channels, d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(d_model)
            for _ in range(num_cross_attention_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, self.audio_encoder.out_channels)
        self.flow_matching = ConditionalFlowMatching()
    
    def forward(
        self,
        noisy_audio: torch.Tensor,
        text_ids: torch.Tensor,
        clean_audio: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with text conditioning"""
        # Encode text
        text_condition = self.text_encoder(text_ids)
        
        # Rest is similar to phoneme version...
        # (Implementation would follow same pattern as FlowAVSEPhonemeConditioned)
        pass


# ============================================================================
# 10. EVALUATION METRICS
# ============================================================================

class SpeechEnhancementMetrics:
    """Compute standard speech enhancement metrics"""
    
    @staticmethod
    def pesq(reference: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
        """
        Perceptual Evaluation of Speech Quality
        Requires: pip install pesq
        """
        try:
            from pesq import pesq
            return pesq(sr, reference, enhanced, 'wb')
        except ImportError:
            print("Warning: pesq not installed. Install with: pip install pesq")
            return 0.0
    
    @staticmethod
    def stoi(reference: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
        """
        Short-Time Objective Intelligibility
        Requires: pip install pystoi
        """
        try:
            from pystoi import stoi
            return stoi(reference, enhanced, sr, extended=False)
        except ImportError:
            print("Warning: pystoi not installed. Install with: pip install pystoi")
            return 0.0
    
    @staticmethod
    def snr(reference: np.ndarray, enhanced: np.ndarray) -> float:
        """Signal-to-Noise Ratio in dB"""
        noise = reference - enhanced
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power < 1e-10:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def si_sdr(reference: np.ndarray, enhanced: np.ndarray) -> float:
        """Scale-Invariant Signal-to-Distortion Ratio"""
        # Remove mean
        reference = reference - np.mean(reference)
        enhanced = enhanced - np.mean(enhanced)
        
        # Scale-invariant projection
        alpha = np.dot(enhanced, reference) / (np.dot(reference, reference) + 1e-8)
        target = alpha * reference
        noise = enhanced - target
        
        signal_power = np.sum(target ** 2)
        noise_power = np.sum(noise ** 2)
        
        if noise_power < 1e-10:
            return float('inf')
        
        return 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    @classmethod
    def compute_all_metrics(
        cls,
        reference: np.ndarray,
        enhanced: np.ndarray,
        sr: int = 16000
    ) -> Dict[str, float]:
        """Compute all available metrics"""
        return {
            'pesq': cls.pesq(reference, enhanced, sr),
            'stoi': cls.stoi(reference, enhanced, sr),
            'snr': cls.snr(reference, enhanced),
            'si_sdr': cls.si_sdr(reference, enhanced)
        }


# ============================================================================
# 11. MAIN USAGE EXAMPLE
# ============================================================================

def main_train():
    """Main training script"""
    
    print("="*70)
    print("FlowAVSE with Phoneme Conditioning - Training Pipeline")
    print("="*70)
    
    # Configuration
    PHONEME_ENCODER_PATH = "./final_model"  # Path to trained phoneme encoder
    OUTPUT_DIR = "./flowavse_phoneme_output"
    WINDOW_SIZE_MS = 1000  # 1 second windows
    USE_CACHED_EMBEDDINGS = False
    D_MODEL = 512
    
    print(f"\nConfiguration:")
    print(f"  - Phoneme encoder: {PHONEME_ENCODER_PATH}")
    print(f"  - Window size: {WINDOW_SIZE_MS}ms")
    print(f"  - Cached embeddings: {USE_CACHED_EMBEDDINGS}")
    print(f"  - Model dimension: {D_MODEL}")
    
    # Setup training
    trainer, model = setup_flowavse_training(
        phoneme_encoder_path=PHONEME_ENCODER_PATH,
        output_dir=OUTPUT_DIR,
        window_size_ms=WINDOW_SIZE_MS,
        use_cached_embeddings=USE_CACHED_EMBEDDINGS,
        d_model=D_MODEL
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}/final_model")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    
    print("\nTraining complete!")


def main_inference():
    """Main inference script"""
    
    print("="*70)
    print("FlowAVSE with Phoneme Conditioning - Inference")
    print("="*70)
    
    # Load model
    PHONEME_ENCODER_PATH = "./final_model"
    FLOWAVSE_MODEL_PATH = "./flowavse_phoneme_output/final_model"
    
    # Initialize components
    vocab = PhonemeVocabularyARPABET()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Load phoneme encoder
    phoneme_encoder = HuBERTForPhonemeClassification(vocab_size=vocab.vocab_size)
    phoneme_encoder.load_state_dict(torch.load(f"{PHONEME_ENCODER_PATH}/model.safetensors", weights_only=False))
    
    # Load FlowAVSE model
    phoneme_adapter = PhonemeConditioningAdapter()
    model = FlowAVSEPhonemeConditioned(
        phoneme_encoder=phoneme_encoder,
        phoneme_adapter=phoneme_adapter,
        freeze_phoneme_encoder=True
    )
    model.load_state_dict(torch.load(f"{FLOWAVSE_MODEL_PATH}/model.safetensors", weights_only=False))
    
    # Create inference pipeline
    inference_pipeline = RealtimeFlowAVSE(
        model=model,
        processor=processor,
        window_size_ms=1000,
        overlap=0.2
    )
    
    # Example: Denoise audio file
    print("\nLoading noisy audio...")
    # noisy_audio, sr = torchaudio.load("noisy_speech.wav")
    # noisy_audio = noisy_audio.numpy()[0]  # Convert to numpy
    
    # For demo, generate random noisy audio
    noisy_audio = np.random.randn(16000 * 3)  # 3 seconds
    
    print("Denoising audio...")
    denoised_audio = inference_pipeline.denoise(noisy_audio)
    
    print(f"Input length: {len(noisy_audio)} samples")
    print(f"Output length: {len(denoised_audio)} samples")
    
    # Save output
    # torchaudio.save("denoised_speech.wav", torch.tensor(denoised_audio).unsqueeze(0), 16000)
    
    print("\nInference complete!")


def main_evaluation():
    """Evaluate model on test set"""
    
    print("="*70)
    print("FlowAVSE Evaluation")
    print("="*70)
    
    # Load test dataset
    dataset = load_dataset("kylelovesllms/timit_asr_ipa", split="test")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Initialize model (same as inference)
    # ... (load model code here)
    
    # Evaluate
    metrics = SpeechEnhancementMetrics()
    all_results = []
    
    print("\nEvaluating on test set...")
    for i, example in enumerate(dataset):
        if i >= 10:  # Evaluate on subset
            break
        
        clean_audio = example['audio']['array']
        
        # Add noise
        noise_aug = NoiseAugmentation()
        noisy_audio = noise_aug.add_noise(clean_audio)
        
        # Denoise
        # denoised_audio = inference_pipeline.denoise(noisy_audio)
        
        # Compute metrics
        # results = metrics.compute_all_metrics(clean_audio, denoised_audio)
        # all_results.append(results)
        
        # print(f"Sample {i+1}: PESQ={results['pesq']:.3f}, STOI={results['stoi']:.3f}")
    
    # Average results
    # avg_metrics = {
    #     key: np.mean([r[key] for r in all_results])
    #     for key in all_results[0].keys()
    # }
    
    # print("\nAverage Metrics:")
    # for key, value in avg_metrics.items():
    #     print(f"  {key.upper()}: {value:.3f}")


# ============================================================================
# 12. EXPERIMENT CONFIGURATIONS
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for different experiments"""
    
    name: str
    window_size_ms: int
    use_cached_embeddings: bool
    d_model: int
    num_cross_attention_layers: int
    audio_encoder_layers: int
    freeze_phoneme_encoder: bool
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'window_size_ms': self.window_size_ms,
            'use_cached_embeddings': self.use_cached_embeddings,
            'd_model': self.d_model,
            'num_cross_attention_layers': self.num_cross_attention_layers,
            'audio_encoder_layers': self.audio_encoder_layers,
            'freeze_phoneme_encoder': self.freeze_phoneme_encoder
        }


# Experiment configurations to test
EXPERIMENTS = [
    ExperimentConfig(
        name="baseline_100ms",
        window_size_ms=100,
        use_cached_embeddings=True,
        d_model=512,
        num_cross_attention_layers=4,
        audio_encoder_layers=6,
        freeze_phoneme_encoder=True
    ),
    ExperimentConfig(
        name="baseline_1000ms",
        window_size_ms=1000,
        use_cached_embeddings=True,
        d_model=512,
        num_cross_attention_layers=4,
        audio_encoder_layers=6,
        freeze_phoneme_encoder=True
    ),
    ExperimentConfig(
        name="large_2000ms",
        window_size_ms=2000,
        use_cached_embeddings=True,
        d_model=768,
        num_cross_attention_layers=6,
        audio_encoder_layers=8,
        freeze_phoneme_encoder=True
    ),
    ExperimentConfig(
        name="realtime_100ms",
        window_size_ms=100,
        use_cached_embeddings=False,  # On-the-fly encoding
        d_model=512,
        num_cross_attention_layers=4,
        audio_encoder_layers=6,
        freeze_phoneme_encoder=True
    ),
]


def run_experiments():
    """Run multiple experiments with different configurations"""
    
    print("="*70)
    print("Running Multiple Experiments")
    print("="*70)
    
    for config in EXPERIMENTS:
        print(f"\n{'='*70}")
        print(f"Experiment: {config.name}")
        print(f"{'='*70}")
        print(f"Configuration: {config.to_dict()}")
        
        # Setup and train
        trainer, model = setup_flowavse_training(
            phoneme_encoder_path="./final_model",
            output_dir=f"./experiments/{config.name}",
            window_size_ms=config.window_size_ms,
            use_cached_embeddings=config.use_cached_embeddings,
            d_model=config.d_model
        )
        
        print(f"\nTraining {config.name}...")
        trainer.train()
        
        print(f"Saving model for {config.name}...")
        trainer.save_model(f"./experiments/{config.name}/final_model")
        
        print(f"\nCompleted: {config.name}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FlowAVSE with Phoneme Conditioning")
    parser.add_argument(
        "mode",
        choices=["train", "inference", "evaluate", "experiments"],
        help="Mode to run"
    )
    parser.add_argument(
        "--phoneme-encoder",
        type=str,
        default="./final_model",
        help="Path to trained phoneme encoder"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./flowavse_phoneme_output/final_model",
        help="Path to trained FlowAVSE model"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Window size in milliseconds"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main_train()
    elif args.mode == "inference":
        main_inference()
    elif args.mode == "evaluate":
        main_evaluation()
    elif args.mode == "experiments":
        run_experiments()
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
