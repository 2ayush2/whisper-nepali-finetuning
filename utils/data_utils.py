import torch
import numpy as np
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Load WER metric
metric = evaluate.load("wer")

def create_formatting_function(tokenizer):
    def formatting_prompts_func(example):
        """Preprocesses audio and text for the model."""
        try:
            audio = example["utterance"]["array"]
            rate = example["utterance"]["sampling_rate"]
            features = tokenizer.feature_extractor(audio, sampling_rate=rate).input_features[0]
            labels = tokenizer.tokenizer(example["transcription"], add_special_tokens=True, truncation=True).input_ids
            return {"input_features": features, "labels": labels}
        except Exception:
            return {"input_features": None, "labels": None}
    return formatting_prompts_func
    
def remove_bad_examples(example):
    """Filter out examples that failed preprocessing."""
    return example["input_features"] is not None and example["labels"] is not None

def create_compute_metrics(tokenizer):
    def compute_metrics(pred):
        """Computes Word Error Rate (WER)."""
        pred_ids = np.argmax(pred.predictions[0], axis=-1)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}
    return compute_metrics

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper that pads input_features and labels.
    """
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        if "input_ids" in batch:
            del batch["input_ids"]
        return batch

