"""
Loading and Evaluating Trained HuBERT Phoneme Classifier
Includes CER, PER, and other phoneme-level metrics
"""

import torch, os
import torch.nn as nn
from transformers import Wav2Vec2Processor
from datasets import load_dataset, Audio
import numpy as np
from typing import List, Dict, Tuple
import evaluate
from jiwer import wer, cer
from collections import Counter


# ============================================================================
# 1. LOAD TRAINED MODEL
# ============================================================================

def load_trained_model(
    model_path: str,
    vocab,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load a trained HuBERTForPhonemeClassification model
    
    Args:
        model_path: Path to saved model directory (from trainer.save_model())
        vocab: PhonemeVocabulary instance
        device: Device to load model on
    
    Returns:
        Loaded model ready for evaluation
    """
    from encoder import HuBERTForPhonemeClassification  # Import your model class
    
    # Initialize model architecture
    model = HuBERTForPhonemeClassification(
        vocab_size=vocab.vocab_size,
        freeze_feature_encoder=False,  # Can keep frozen or not
        freeze_base_model=False
    )
    
    safetensors_path = os.path.join(model_path, "model.safetensors")
    pytorch_path = os.path.join(model_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        # Load from safetensors format (newer default)
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        print(f"Loading from safetensors: {safetensors_path}")
    elif os.path.exists(pytorch_path):
        # Load from pytorch format (older default)
        state_dict = torch.load(pytorch_path, map_location=device)
        print(f"Loading from pytorch: {pytorch_path}")
    else:
        raise FileNotFoundError(
            f"No model file found in {model_path}. "
            f"Expected either 'model.safetensors' or 'pytorch_model.bin'"
        )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


# Alternative: If you saved with trainer.save_model(), you can also use:
def load_model_from_checkpoint(
    checkpoint_path: str,
    vocab,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Load from a specific checkpoint"""
    from encoder import HuBERTForPhonemeClassification
    
    model = HuBERTForPhonemeClassification(vocab_size=vocab.vocab_size)
    
    # Load checkpoint
    checkpoint = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=device)
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================================
# 2. CTC DECODING
# ============================================================================

def ctc_greedy_decode(logits: torch.Tensor, vocab, blank_id: int) -> List[str]:
    """
    Greedy CTC decoding: take argmax and collapse repetitions
    
    Args:
        logits: (time, vocab_size) or (batch, time, vocab_size)
        vocab: PhonemeVocabulary instance
        blank_id: ID of the CTC blank token
    
    Returns:
        List of decoded phoneme sequences
    """
    if len(logits.shape) == 3:
        # Batch decoding
        predictions = torch.argmax(logits, dim=-1)  # (batch, time)
        decoded = []
        for pred_seq in predictions:
            decoded.append(_decode_single_sequence(pred_seq, vocab, blank_id))
        return decoded
    else:
        # Single sequence
        predictions = torch.argmax(logits, dim=-1)
        return [_decode_single_sequence(predictions, vocab, blank_id)]


def _decode_single_sequence(predictions: torch.Tensor, vocab, blank_id: int) -> str:
    """Decode a single sequence with CTC collapse"""
    # Convert to list
    pred_list = predictions.cpu().tolist()
    
    # Remove consecutive duplicates and blanks
    collapsed = []
    prev = None
    for pred in pred_list:
        if pred != blank_id and pred != prev:
            collapsed.append(pred)
        prev = pred
    
    # Convert to phonemes
    phonemes = [vocab.decode(pid) for pid in collapsed]
    
    # Join into string (space-separated for phonemes)
    return " ".join(phonemes)


def ctc_beam_search_decode(
    logits: torch.Tensor,
    vocab,
    blank_id: int,
    beam_width: int = 10
) -> List[str]:
    """
    Beam search decoding for better accuracy
    This is a simplified version - for production use a library like ctcdecode
    """
    # For simplicity, using greedy decode here
    # In practice, install and use: pip install ctcdecode
    # Or use libraries like pyctcdecode
    return ctc_greedy_decode(logits, vocab, blank_id)


# ============================================================================
# 3. EVALUATION METRICS
# ============================================================================

class PhonemeEvaluator:
    """Compute various phoneme recognition metrics"""
    
    def __init__(self, vocab):
        self.vocab = vocab
    
    def compute_per(self, predictions: List[str], references: List[str]) -> float:
        """
        Phoneme Error Rate (PER) - similar to WER but for phonemes
        PER = (Substitutions + Deletions + Insertions) / Total Reference Phonemes
        """
        total_errors = 0
        total_phonemes = 0
        
        for pred, ref in zip(predictions, references):
            pred_phones = pred.split()
            ref_phones = ref.split()
            
            # Use WER function from jiwer (works for any tokens)
            error_rate = wer(ref, pred)
            total_errors += error_rate * len(ref_phones)
            total_phonemes += len(ref_phones)
        
        return total_errors / total_phonemes if total_phonemes > 0 else 0.0
    
    def compute_cer(self, predictions: List[str], references: List[str]) -> float:
        """
        Character Error Rate (CER) - treats each phoneme character as a unit
        Useful when phonemes are multi-character (like 'tʃ')
        """
        # Concatenate all predictions and references
        all_preds = "".join(predictions)
        all_refs = "".join(references)
        
        return cer(all_refs, all_preds)
    
    def compute_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Frame-level accuracy"""
        correct = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_phones = pred.split()
            ref_phones = ref.split()
            
            # Pad to same length for comparison
            max_len = max(len(pred_phones), len(ref_phones))
            pred_phones += ['[PAD]'] * (max_len - len(pred_phones))
            ref_phones += ['[PAD]'] * (max_len - len(ref_phones))
            
            correct += sum(p == r for p, r in zip(pred_phones, ref_phones))
            total += max_len
        
        return correct / total if total > 0 else 0.0
    
    def compute_confusion_matrix(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Counter]:
        """
        Compute phoneme confusion matrix
        Shows which phonemes are commonly confused
        """
        confusion = {}
        
        for pred, ref in zip(predictions, references):
            pred_phones = pred.split()
            ref_phones = ref.split()
            
            # Align sequences (simple alignment - could use edit distance)
            for p, r in zip(pred_phones, ref_phones):
                if r not in confusion:
                    confusion[r] = Counter()
                confusion[r][p] += 1
        
        return confusion
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute all metrics at once"""
        return {
            'per': self.compute_per(predictions, references),
            'cer': self.compute_cer(predictions, references),
            'accuracy': self.compute_accuracy(predictions, references),
        }


# ============================================================================
# 4. EVALUATION PIPELINE
# ============================================================================

def evaluate_model(
    model,
    processor: Wav2Vec2Processor,
    vocab,
    test_dataset,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8
):
    """
    Full evaluation pipeline
    
    Args:
        model: Trained HuBERTForPhonemeClassification
        processor: Wav2Vec2Processor
        vocab: PhonemeVocabulary
        test_dataset: HuggingFace dataset with audio and phoneme labels
        device: Device to run on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    evaluator = PhonemeEvaluator(vocab)
    
    all_predictions = []
    all_references = []
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    # Process in batches
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i + batch_size]
        
        # Prepare audio inputs
        audio_arrays = [sample['audio']['array'] for sample in batch]
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs['logits']
        
        # Decode predictions
        predictions = ctc_greedy_decode(logits, vocab, vocab.ctc_token_id)
        
        # Get references
        references = []
        for sample in batch:
            # Parse reference phonemes from dataset
            ref_phonemes = parse_reference_phonemes(sample, vocab)
            references.append(ref_phonemes)
        
        all_predictions.extend(predictions)
        all_references.extend(references)
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i + len(batch)}/{len(test_dataset)} samples")
    
    # Compute metrics
    metrics = evaluator.compute_all_metrics(all_predictions, all_references)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Phoneme Error Rate (PER): {metrics['per']:.4f}")
    print(f"Character Error Rate (CER): {metrics['cer']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("="*50)
    
    # Show some examples
    print("\nExample predictions (first 5):")
    for i in range(min(5, len(all_predictions))):
        print(f"\nSample {i}:")
        print(f"Reference:  {all_references[i]}")
        print(f"Prediction: {all_predictions[i]}")
    
    # Compute confusion matrix
    confusion = evaluator.compute_confusion_matrix(all_predictions, all_references)
    print("\nMost confused phonemes:")
    for phoneme, confusions in list(confusion.items())[:10]:
        top_confusion = confusions.most_common(2)
        if len(top_confusion) > 1 and top_confusion[0][0] != phoneme:
            print(f"  {phoneme} -> {top_confusion[0][0]} ({top_confusion[0][1]} times)")
    
    return {
        'metrics': metrics,
        'predictions': all_predictions,
        'references': all_references,
        'confusion_matrix': confusion
    }


def parse_reference_phonemes(sample: Dict, vocab) -> str:
    """
    Parse reference phonemes from TIMIT dataset sample
    Adapt this to your actual dataset structure
    """
    if 'phonetic_detail' in sample:
        phonemes = []
        for phone_info in sample['phonetic_detail']:
            phone = phone_info.get('utterance', '[UNK]')
            # Map to IPA if needed
            phonemes.append(phone)
        return " ".join(phonemes)
    elif 'text' in sample:
        # If phonemes are in text field
        return sample['text']
    else:
        return "[UNK]"


# ============================================================================
# 5. USAGE EXAMPLE
# ============================================================================

def main():
    """Complete example of loading and evaluating"""
    
    # 1. Setup vocabulary (same as training)
    from encoder import PhonemeVocabulary
    vocab = PhonemeVocabulary()
    
    # 2. Load processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # 3. Load trained model
    model_path = "./final_model"  # or "./phon-embedder/checkpoint-XXXX"
    model = load_trained_model(model_path, vocab)
    
    # 4. Load test dataset
    print("Loading test dataset...")
    dataset = load_dataset("kylelovesllms/timit_asr_ipa")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = dataset['test']
    
    # 5. Evaluate
    results = evaluate_model(
        model=model,
        processor=processor,
        vocab=vocab,
        test_dataset=test_dataset,
        batch_size=8
    )
    
    # 6. Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'metrics': results['metrics'],
            'num_samples': len(results['predictions'])
        }, f, indent=2)
    
    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()