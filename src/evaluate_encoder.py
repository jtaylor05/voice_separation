"""
Loading and Evaluating Trained HuBERT Phoneme Classifier
Includes CER, PER, and other phoneme-level metrics - FIXED VERSION
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
# 1. LOAD TRAINED MODEL (unchanged)
# ============================================================================

def load_trained_model(
    model_path: str,
    vocab,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Load a trained HuBERTForPhonemeClassification model"""
    from encoder import HuBERTForPhonemeClassification
    
    model = HuBERTForPhonemeClassification(
        vocab_size=vocab.vocab_size,
        freeze_feature_encoder=False,
        freeze_base_model=False
    )
    
    safetensors_path = os.path.join(model_path, "model.safetensors")
    pytorch_path = os.path.join(model_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        print(f"Loading from safetensors: {safetensors_path}")
    elif os.path.exists(pytorch_path):
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


# ============================================================================
# 2. IMPROVED CTC DECODING
# ============================================================================

def ctc_greedy_decode(logits: torch.Tensor, vocab, blank_id: int) -> List[str]:
    """
    Improved greedy CTC decoding with post-processing
    """
    if len(logits.shape) == 3:
        predictions = torch.argmax(logits, dim=-1)
        decoded = []
        for pred_seq in predictions:
            decoded.append(_decode_single_sequence_improved(pred_seq, vocab, blank_id))
        return decoded
    else:
        predictions = torch.argmax(logits, dim=-1)
        return [_decode_single_sequence_improved(predictions, vocab, blank_id)]


def _decode_single_sequence_improved(predictions: torch.Tensor, vocab, blank_id: int) -> str:
    """Decode with CTC collapse and noise filtering"""
    pred_list = predictions.cpu().tolist()
    
    # Step 1: Standard CTC collapse
    collapsed = []
    prev = None
    for pred in pred_list:
        if pred != blank_id and pred != prev:
            collapsed.append(pred)
        prev = pred
    
    if not collapsed:
        return "[UNK]"
    
    # Step 2: Remove excessive repetitions (max 2 in a row)
    filtered = []
    count = 1
    for i, pid in enumerate(collapsed):
        if i > 0 and pid == collapsed[i-1]:
            count += 1
            if count <= 2:
                filtered.append(pid)
        else:
            count = 1
            filtered.append(pid)
    
    # Step 3: Convert to phonemes and clean up 'y' artifacts
    phonemes = [vocab.decode(pid) for pid in filtered]
    
    # Remove 'y' if it appears in alternating pattern (likely spurious)
    cleaned = []
    for i, phone in enumerate(phonemes):
        if phone == 'y' and i > 0 and i < len(phonemes) - 1:
            # Check if surrounded by same non-y phoneme
            if phonemes[i-1] == phonemes[i+1] and phonemes[i-1] != 'y':
                continue  # Skip this spurious 'y'
        cleaned.append(phone)
    
    return " ".join(cleaned) if cleaned else "[UNK]"


# ============================================================================
# 3. EVALUATION METRICS (unchanged)
# ============================================================================

class PhonemeEvaluator:
    """Compute various phoneme recognition metrics"""
    
    def __init__(self, vocab):
        self.vocab = vocab
    
    def compute_per(self, predictions: List[str], references: List[str]) -> float:
        """Phoneme Error Rate"""
        total_errors = 0
        total_phonemes = 0
        
        for pred, ref in zip(predictions, references):
            pred_phones = pred.split()
            ref_phones = ref.split()
            error_rate = wer(ref, pred)
            total_errors += error_rate * len(ref_phones)
            total_phonemes += len(ref_phones)
        
        return total_errors / total_phonemes if total_phonemes > 0 else 0.0
    
    def compute_cer(self, predictions: List[str], references: List[str]) -> float:
        """Character Error Rate"""
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
        """Compute phoneme confusion matrix"""
        confusion = {}
        
        for pred, ref in zip(predictions, references):
            pred_phones = pred.split()
            ref_phones = ref.split()
            
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
# 4. FIXED EVALUATION PIPELINE
# ============================================================================

def evaluate_model(
    model,
    processor: Wav2Vec2Processor,
    vocab,
    test_dataset,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8
):
    """Full evaluation pipeline with FIXED reference parsing"""
    model.eval()
    evaluator = PhonemeEvaluator(vocab)
    
    all_predictions = []
    all_references = []
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i + batch_size]
        
        # Get audio arrays from batch
        audio_arrays = [sample['array'] for sample in batch['audio']]
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
        
        # Decode with improved decoder
        predictions = ctc_greedy_decode(logits, vocab, vocab.ctc_token_id)
        
        # FIXED: Get references correctly
        references = []
        for phonetic_detail_list in batch['phonetic_detail']:
            # phonetic_detail_list is already the list of phone dicts
            ref_phonemes = parse_reference_phonemes(phonetic_detail_list, vocab)
            references.append(ref_phonemes)
        
        all_predictions.extend(predictions)
        all_references.extend(references)
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i + len(audio_arrays)}/{len(test_dataset)} samples")
    
    # Analyze prediction patterns
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    analysis = analyze_predictions(all_predictions, vocab)
    print(f"Total phonemes predicted: {analysis['total_phonemes']}")
    print(f"Unique phonemes used: {analysis['vocab_coverage']}")
    print(f"Average prediction length: {analysis['avg_length']:.1f} phonemes")
    print(f"Repetition rate: {analysis['repetition_rate']:.2%}")
    print(f"\nMost common phonemes:")
    for phone, count in analysis['most_common']:
        pct = 100 * count / analysis['total_phonemes']
        print(f"  {phone}: {count} ({pct:.1f}%)")
    
    # Compute metrics
    metrics = evaluator.compute_all_metrics(all_predictions, all_references)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Phoneme Error Rate (PER): {metrics['per']:.4f}")
    print(f"Character Error Rate (CER): {metrics['cer']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("="*60)
    
    # Show examples
    print("\nExample predictions (first 5):")
    for i in range(min(5, len(all_predictions))):
        print(f"\nSample {i}:")
        print(f"Reference:  {all_references[i]}")
        print(f"Prediction: {all_predictions[i]}")
    
    # Confusion matrix
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
        'confusion_matrix': confusion,
        'analysis': analysis
    }


def parse_reference_phonemes(phonetic_detail_list, vocab) -> str:
    """
    FIXED: Parse reference phonemes from TIMIT phonetic_detail
    
    Args:
        phonetic_detail_list: List of dicts like [{'start': X, 'stop': Y, 'utterance': 'ph'}, ...]
        vocab: PhonemeVocabularyARPABET with normalize_timit_phone method
    """
    if isinstance(phonetic_detail_list, list) and len(phonetic_detail_list) > 0:
        phonemes = []
        for phone_info in phonetic_detail_list:
            if isinstance(phone_info, dict) and 'utterance' in phone_info:
                phone = phone_info['utterance']
                normalized = vocab.normalize_timit_phone(phone)
                phonemes.append(normalized)
        return " ".join(phonemes) if phonemes else "[UNK]"
    return "[UNK]"


def analyze_predictions(predictions: List[str], vocab) -> dict:
    """Analyze prediction patterns"""
    all_phonemes = []
    phoneme_counts = Counter()
    
    for pred in predictions:
        phones = pred.split()
        all_phonemes.extend(phones)
        phoneme_counts.update(phones)
    
    total_phonemes = len(all_phonemes)
    unique_phonemes = len(set(all_phonemes))
    
    most_common = phoneme_counts.most_common(10)
    
    repetition_count = 0
    for pred in predictions:
        phones = pred.split()
        for i in range(len(phones)-1):
            if phones[i] == phones[i+1]:
                repetition_count += 1
    
    return {
        'total_phonemes': total_phonemes,
        'unique_phonemes': unique_phonemes,
        'vocab_coverage': f"{unique_phonemes}/{vocab.vocab_size}",
        'most_common': most_common,
        'repetition_rate': repetition_count / max(1, total_phonemes - len(predictions)),
        'avg_length': total_phonemes / len(predictions)
    }


# ============================================================================
# 5. USAGE EXAMPLE
# ============================================================================

def main():
    """Complete example of loading and evaluating"""
    
    # Use ARPABET vocab (matching training)
    from encoder import PhonemeVocabularyARPABET
    vocab = PhonemeVocabularyARPABET()
    print(f"Vocabulary size: {vocab.vocab_size}")
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Load trained model
    model_path = "./final_model"
    model = load_trained_model(model_path, vocab)
    
    # Load test dataset
    print("Loading test dataset...")
    dataset = load_dataset("kylelovesllms/timit_asr_ipa")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = dataset['test']
    
    # Quick diagnostic
    test_audio = dataset['test'][0]['audio']['array']
    inputs = processor(test_audio, sampling_rate=16000, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
    
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    print(f"Average entropy: {entropy:.3f}")
    print(f"Max entropy possible: {torch.log(torch.tensor(vocab.vocab_size)):.3f}")
    
    predictions = torch.argmax(logits, dim=-1).flatten()
    unique_preds = torch.unique(predictions)
    print(f"Unique phonemes predicted: {len(unique_preds)} out of {vocab.vocab_size}")
    for pred_id in unique_preds[:10]:
        print(f"  - {vocab.decode(pred_id.item())}")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        processor=processor,
        vocab=vocab,
        test_dataset=test_dataset,
        batch_size=8
    )
    
    # Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'metrics': results['metrics'],
            'num_samples': len(results['predictions']),
            'analysis': {k: str(v) for k, v in results['analysis'].items()}
        }, f, indent=2)
    
    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()
