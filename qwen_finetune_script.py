#!/usr/bin/env python3
"""
Fine-tuning script untuk Qwen2.5-0.5B-Instruct
Dataset: Informasi tentang Andey Koiwai
Hardware: Core i5-6300U, RAM 8GB
"""

import json
import torch
import gc
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# Konfigurasi untuk hardware dengan RAM terbatas
CONFIG = {
    "model_name": "C:/Users/MOGI/Documents/PENTING/huggingface/Qwen/Qwen2.5-0.5B-Instruct",      #"Qwen/Qwen2.5-0.5B-Instruct",
    "data_file": "mydata(2).txt",
    "output_dir": "./qwen-andey-assistant",
    "max_length": 512,
    "batch_size": 1,  # Sangat kecil untuk RAM 8GB
    "gradient_accumulation_steps": 8,  # Simulasi batch size lebih besar
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "save_steps": 100,
    "eval_steps": 100,
    "logging_steps": 50,
    "warmup_steps": 100,
}

def load_data(file_path):
    """Load dan parse data dari file JSON lines"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        print(f"Berhasil memuat {len(data)} data dari {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def format_conversation(item):
    """Format data menjadi format percakapan untuk Qwen"""
    question = item['question']
    answer = item['answer']
    
    # Format sesuai Qwen chat template
    formatted = f"<|im_start|>system\nAnda adalah asisten AI yang membantu menjawab pertanyaan tentang Andey Koiwai, seorang insinyur AI dan robotika.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
    
    return formatted

def create_dataset(data, tokenizer):
    """Buat dataset dari data yang sudah diformat"""
    formatted_texts = [format_conversation(item) for item in data]
    
    # Tokenize data
    tokenized = tokenizer(
        formatted_texts,
        truncation=False,
        padding=True,
        max_length=CONFIG["max_length"],
        return_tensors="pt"
    )
    
    # Buat dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    })
    
    return dataset

def setup_model_and_tokenizer():
    """Setup model dan tokenizer dengan optimasi memory"""
    print("Memuat tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    # Set pad token jika belum ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Memuat model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu",  # Force CPU usage
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing untuk menghemat memory
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def main():
    print("=== Fine-tuning Qwen2.5-0.5B untuk Asisten Andey Koiwai ===")
    print(f"Hardware: CPU only, RAM optimization enabled")
    
    # Clear memory
    gc.collect()
    
    # Load data
    print("\n1. Memuat data...")
    data = load_data(CONFIG["data_file"])
    if not data:
        print("Gagal memuat data. Pastikan file mydata(2).txt ada dan formatnya benar.")
        return
    
    # Setup model dan tokenizer
    print("\n2. Setup model dan tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Buat dataset
    print("\n3. Membuat dataset...")
    dataset = create_dataset(data, tokenizer)
    
    # Split dataset (80% train, 20% eval)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Dataset train: {len(train_dataset)} samples")
    print(f"Dataset eval: {len(eval_dataset)} samples")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments - optimized untuk CPU dan RAM terbatas
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=0.01,
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_steps=CONFIG["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,  # Hemat RAM
        dataloader_num_workers=0,     # Hindari multiprocessing
        fp16=False,  # Tidak menggunakan FP16 di CPU
        gradient_checkpointing=True,
        report_to=[],  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )
    
    # Setup trainer
    print("\n4. Setup trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n5. Mulai training...")
    print("Ini akan memakan waktu cukup lama di CPU. Harap bersabar...")
    
    try:
        trainer.train()
        print("\nTraining selesai!")
        
        # Save model
        print("\n6. Menyimpan model...")
        trainer.save_model()
        tokenizer.save_pretrained(CONFIG["output_dir"])
        print(f"Model disimpan di: {CONFIG['output_dir']}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Test model
    print("\n7. Testing model...")
    test_model(CONFIG["output_dir"])

def test_model(model_path):
    """Test model yang sudah di-fine-tune"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Test questions
        test_questions = [
            "Siapakah Andey Koiwai?",
            "Apa itu proyek newmogiserver?",
            "Bagaimana karakteristik AI Mogi?"
        ]
        
        print("\n=== Testing Model ===")
        for question in test_questions:
            prompt = f"<|im_start|>system\nAnda adalah asisten AI yang membantu menjawab pertanyaan tentang Andey Koiwai.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("assistant\n")[-1]
            
            print(f"\nQ: {question}")
            print(f"A: {answer}")
            
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    main()