import unsloth
from unsloth import FastModel, is_bf16_supported
import os, gc, argparse, torch, datasets, config
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from utils.logger import ExperimentLogger
from utils.data_utils import create_formatting_function, remove_bad_examples, DataCollatorSpeechSeq2SeqWithPadding, create_compute_metrics
from utils.file_utils import setup_experiment_dir

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

class WhisperFinetuner:
    def __init__(self, args):
        self.args = args
        self.output_dir, self.run_name = setup_experiment_dir(config, self.args)
        self.logger = ExperimentLogger.setup(self.output_dir)
        self.logger.info(f"SOTA Nepali ASR Training Session: {self.run_name}")

    def load_model(self):
        self.logger.info(f"Loading SOTA Base Model: {config.MODEL_ID}")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=config.MODEL_ID,
            load_in_4bit=False,
            auto_model=WhisperForConditionalGeneration,
            whisper_language=config.LANGUAGE,
            whisper_task=config.TASK
        )
        
        self.model = FastModel.get_peft_model(
            self.model,
            r=self.args.lora_rank,
            target_modules=config.LORA_TARGET_MODULES,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            use_gradient_checkpointing="unsloth",
            use_rslora=config.USE_RSLORA
        )

        # Accuracy Configuration from Config
        self.model.config.use_cache = False
        self.model.config.apply_spec_augment = config.APPLY_SPEC_AUGMENT
        
        # Generation Parameters for Accuracy
        self.model.generation_config.language = "<|ne|>"
        self.model.generation_config.task = config.TASK
        self.model.generation_config.num_beams = config.NUM_BEAMS
        self.model.generation_config.max_length = config.MAX_LABEL_LENGTH

    def prepare_data(self):
        self.logger.info(f"Loading Dataset: {config.DATASET_ID}")
        dataset = datasets.load_dataset(config.DATASET_ID, name="cleaned", split="train", trust_remote_code=True)
        dataset = dataset.cast_column("utterance", datasets.Audio(sampling_rate=config.SAMPLING_RATE))
        
        if config.MAX_SAMPLES:
            max_samples = min(config.MAX_SAMPLES, len(dataset))
            dataset = dataset.select(range(max_samples))
        
        split_data = dataset.train_test_split(test_size=config.TEST_SIZE, seed=3407)
        format_fn = create_formatting_function(self.tokenizer, max_length=config.MAX_LABEL_LENGTH)
        
        self.train_dataset = split_data["train"].map(format_fn, remove_columns=split_data["train"].column_names).filter(remove_bad_examples)
        self.test_dataset = split_data["test"].map(format_fn, remove_columns=split_data["test"].column_names).filter(remove_bad_examples)
        self.logger.info(f"Final Data sizes -> Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")

    def train(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=config.GRAD_ACCUMULATION,
            learning_rate=self.args.learning_rate,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=self.args.epochs,
            logging_steps=config.LOGGING_STEPS,
            save_steps=config.SAVE_STEPS,
            eval_steps=config.EVAL_STEPS,
            optim="adamw_torch",
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            weight_decay=config.WEIGHT_DECAY,
            label_smoothing_factor=config.LABEL_SMOOTHING,
            predict_with_generate=config.PREDICT_WITH_GENERATE,
            load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
            metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
            greater_is_better=config.GREATER_IS_BETTER,
            eval_strategy="steps",
            eval_accumulation_steps=1,
            remove_unused_columns=True,
            report_to=config.REPORT_TO
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=self.tokenizer),
            # processing_class is the replacement for tokenizer in newer transformers
            processing_class=self.tokenizer,
            compute_metrics=create_compute_metrics(self.tokenizer),
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)]
        )
        
        last_ckpt = get_last_checkpoint(self.output_dir)
        if last_ckpt:
            self.logger.info(f"Resuming from checkpoint: {last_ckpt}")
            self.trainer.train(resume_from_checkpoint=last_ckpt)
        else:
            self.logger.info("Starting fresh training session...")
            self.trainer.train()

    def export_model(self):
        save_path = os.path.join(self.output_dir, "final_model_merged")
        self.model.save_pretrained_merged(save_path, self.tokenizer, save_method="merged_16bit")
        self.logger.info(f"SOTA Model Saved and Merged: {save_path}")

def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description="Whisper SOTA Fine-Tuning Script")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lora_rank", type=int, default=config.LORA_R)
    args = parser.parse_args()
    
    finetuner = WhisperFinetuner(args)
    finetuner.load_model()
    finetuner.prepare_data()
    finetuner.train()
    finetuner.export_model()

if __name__ == "__main__":
    main()