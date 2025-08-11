from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftConfig,
    PeftModel
)
import numpy as np
import wandb
import os
import torch
from typing import Optional, List, Dict, Any
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "dlp/extended_json_data/validation.jsonl"      # JSONL with {"sequence": "...", "spans": "11-12_34-34"}

def get_the_latest_checkpoint(output_dir="./qlora_checkpoints"):
    """Get the latest checkpoint from the checkpoints directory"""
    if not os.path.exists(output_dir):
        return None

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None

    steps = [int(d.split("-")[1]) for d in checkpoints]
    latest_index = np.argmax(steps)
    latest_checkpoint = checkpoints[latest_index]

    return os.path.join(output_dir, latest_checkpoint)

def setup_qlora_model(model_name: str, checkpoint_path: Optional[str] = None):
    """
    Loads a QLoRA model and tokenizer, and if a checkpoint is provided, loads the adapter weights from it.
    """
    # Always load tokenizer from base model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    peft_model_path = "qlora_checkpoints/checkpoint-1050"
    peft_config = PeftConfig.from_pretrained(peft_model_path)

    # 2. Load base model (same model used before QLoRA fine-tuning)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype="auto",
        device_map="auto"
    )

    # 3. Load QLoRA adapter
    model = PeftModel.from_pretrained(base_model, peft_model_path)

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    # 5. Set to evaluation
    model.eval()

    # # 4-bit quantization config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=False,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float32
    # )

    # # Load base model with quantization
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     model_name,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    # )

    # # Prepare for k-bit training (required for LoRA)
    # model = prepare_model_for_kbit_training(model)

    # # LoRA config (should match training)
    # lora_config = LoraConfig(
    #     r=4,
    #     lora_alpha=8,
    #     target_modules=["q", "v"],
    #     lora_dropout=0.0,
    #     bias="none",
    #     task_type=TaskType.SEQ_2_SEQ_LM,
    #     inference_mode=True,  # Set to True for inference!
    # )

    # # Apply LoRA wrapper
    # model = get_peft_model(model, lora_config)
    
    # # Load checkpoint if provided
    # if checkpoint_path and os.path.exists(checkpoint_path):
    #     print(f"Loading checkpoint from: {checkpoint_path}")
        
    #     # Try loading from safetensors first (newer format)
    #     adapter_path_safetensors = os.path.join(checkpoint_path, "adapter_model.safetensors")
        
    #     if os.path.exists(adapter_path_safetensors):
    #         print(f"Loading from safetensors: {adapter_path_safetensors}")
    #         adapter_weights = load_file(adapter_path_safetensors)
    #         model.load_state_dict(adapter_weights, strict=False)

    #     else:
    #         print(f"Warning: No adapter weights found in {checkpoint_path}")
    
    # # Print model info
    # model.print_trainable_parameters()
    
    return model, tokenizer
class QLoRAInference:
    def __init__(self):
        checkpoint_path = get_the_latest_checkpoint()
        base_model_name = "t5-3b"
        self.model, self.tokenizer = setup_qlora_model(base_model_name, checkpoint_path)
    
    def predict(self, sequence: str, max_length: int = 128) -> str:
        """
        Predict spans for a given sequence
        
        Args:
            sequence: Input protein sequence
            max_length: Maximum length of generated output
            
        Returns:
            Predicted spans as string
        """
        inputs = self.tokenizer(
            sequence,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction
    
    def batch_predict(self, sequences: List[str], max_length: int = 128) -> List[str]:
        """
        Predict spans for multiple sequences
        
        Args:
            sequences: List of input protein sequences
            max_length: Maximum length of generated output
            
        Returns:
            List of predicted spans
        """
        inputs = self.tokenizer(
            sequences,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions
    
    def predict_with_confidence(self, sequence: str, max_length: int = 128, num_samples: int = 5) -> Dict[str, Any]:
        """
        Predict with confidence by sampling multiple times
        
        Args:
            sequence: Input protein sequence
            max_length: Maximum length of generated output
            num_samples: Number of samples for confidence estimation
            
        Returns:
            Dictionary with prediction and confidence score
        """
        predictions = []
        
        for _ in range(num_samples):
            inputs = self.tokenizer(
                sequence,
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred)
        
        # Calculate confidence based on agreement
        most_common = max(set(predictions), key=predictions.count)
        confidence = predictions.count(most_common) / len(predictions)
        
        return {
            "prediction": most_common,
            "confidence": confidence,
            "all_predictions": predictions
        }

    def compute_metrics(self, eval_pred):
        input_ids = eval_pred.input_ids.to(device)
        preds = eval_pred.predictions.to(device)
        labels = eval_pred.label_ids.to(device)
        preds = np.clip(preds, 0, self.tokenizer.vocab_size - 1)

        decoded_inputs  = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_preds  = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Row-wise accuracy
        row_accuracies = []
        for i in range(len(decoded_inputs)):
            corr = 0
            for j in range(len(decoded_labels[i])):
                if j >= len(decoded_preds[i]):
                    continue
                if decoded_preds[i][j] == decoded_labels[i][j]:
                    corr += 1
            if len(decoded_labels[i]) > 0:
                row_accuracies.append(float(corr / len(decoded_labels[i])))
        
        table = wandb.Table(columns=["input", "prediction", "label", "token_accuracy"])
        for inp, pr, lab, acc in zip(decoded_inputs, decoded_preds, decoded_labels, row_accuracies):
            table.add_data(inp, pr, lab, acc)
        wandb.log({"eval_samples": table})
        
        return {"token_accuracy": sum(row_accuracies) / len(row_accuracies)}


def main():
    """Example usage of the QLoRA inference"""
    wandb.init(project="ted-qlora-inference", name="inference")
    inferencer = QLoRAInference()
    
    # Example sequences
    test_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKLLIVLMLGTVFGNSLQADAVGNTQIFFLLNQSFTVGLFLSCKLTMGYFVHGEVFFGLLFLGAGL",
    ]
    
    print("=== Single Prediction ===")
    for i, seq in enumerate(test_sequences):
        prediction = inferencer.predict(seq)
        print(f"Sequence {i+1}: {seq[:50]}...")
        print(f"Prediction: {prediction}")
        print()
    
    print("=== Batch Prediction ===")
    batch_predictions = inferencer.batch_predict(test_sequences)
    for i, (seq, pred) in enumerate(zip(test_sequences, batch_predictions)):
        print(f"Sequence {i+1}: {seq[:50]}...")
        print(f"Prediction: {pred}")
        print()
    
    print("=== Prediction with Confidence ===")
    for i, seq in enumerate(test_sequences):
        result = inferencer.predict_with_confidence(seq)
        print(f"Sequence {i+1}: {seq[:50]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"All predictions: {result['all_predictions']}")
        print()

if __name__ == "__main__":
    main() 