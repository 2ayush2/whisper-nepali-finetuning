import torch
import numpy as np
import unicodedata
import evaluate
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Unicode NFC - Canonical form for Devanagari
    text = unicodedata.normalize('NFC', text)
    # Remove Zero-Width and Hidden characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    # Standardize Danda (punctuation) spacing
    text = text.replace(' ।', '।').replace(' ॥', '॥')
    # Clean redundant whitespace
    text = ' '.join(text.strip().split())
    return text
def create_formatting_function(processor):
    def formatting_prompts_func(example):
        try:
            audio = example["utterance"]["array"]
            rate = example["utterance"]["sampling_rate"]
            features = processor.feature_extractor(
                audio, 
                sampling_rate=rate
            ).input_features[0]
            normalized_text = normalize_text(example["transcription"])
            if not normalized_text or len(normalized_text.strip()) == 0:
                return {"input_features": None, "labels": None}
            labels = processor.tokenizer(
                normalized_text,
                add_special_tokens=True,
                truncation=True,
                max_length=448
            ).input_ids
            return {
                "input_features": features,
                "labels": labels
            }
        except Exception as e:
            return {"input_features": None, "labels": None}
    return formatting_prompts_func

def remove_bad_examples(example):
    if example["input_features"] is None or example["labels"] is None:
        return False
    if len(example["labels"]) < 2:
        return False
    if len(example["labels"]) > 448:
        return False
    return True

def create_compute_metrics(tokenizer):
    def compute_metrics(pred):
        output_ids = pred.predictions
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalize_text(s) for s in pred_str]
        label_str = [normalize_text(s) for s in label_str]
        wer_score = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
        cer_score = 100 * metric_cer.compute(predictions=pred_str, references=label_str)
        return {"wer": wer_score, "cer": cer_score}
    return compute_metrics

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), 
            -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        if "input_ids" in batch:
            del batch["input_ids"]
        return batch