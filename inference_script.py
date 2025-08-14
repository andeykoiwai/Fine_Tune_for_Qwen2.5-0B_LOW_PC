#!/usr/bin/env python3
"""
Script inference untuk model Qwen2.5-0.5B yang sudah di-fine-tune
Untuk menjadi asisten Q&A tentang Andey Koiwai
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

class AndeyAssistant:
    def __init__(self, model_path="./qwen-andey-assistant"):
        """Inisialisasi asisten dengan model yang sudah di-fine-tune"""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load model dan tokenizer"""
        try:
            print("Memuat model dan tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            print("Model berhasil dimuat!")
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Pastikan model sudah di-fine-tune dan path benar.")
    
    def generate_response(self, question, max_tokens=150, temperature=0.7):
        """Generate response untuk pertanyaan"""
        if self.model is None or self.tokenizer is None:
            return "Model belum dimuat dengan benar."
        
        # Format prompt sesuai template Qwen
        # prompt = f"<|im_start|>system\nAnda adalah asisten AI yang membantu menjawab pertanyaan tentang Andey Koiwai, seorang insinyur AI dan robotika. Berikan jawaban yang akurat dan informatif.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        prompt = f"<|im_start|>system<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract hanya bagian assistant response
            if "assistant\n" in full_response:
                response = full_response.split("assistant\n")[-1].strip()
                return response
            else:
                return full_response.strip()
                
        except Exception as e:
            return f"Error generating response: {e}"
    
    def chat(self):
        """Interactive chat loop"""
        print("=== Asisten Andey Koiwai ===")
        print("Tanyakan apa saja tentang Andey Koiwai dan proyeknya!")
        print("Ketik 'quit' atau 'exit' untuk keluar.\n")
        
        while True:
            try:
                question = input("Anda: ").strip()
                
                if question.lower() in ['quit', 'exit', 'keluar']:
                    print("Terima kasih! Sampai jumpa!")
                    break
                
                if not question:
                    continue
                
                print("Asisten: ", end="", flush=True)
                response = self.generate_response(question)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\nTerima kasih! Sampai jumpa!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function untuk menjalankan asisten"""
    assistant = AndeyAssistant()
    
    # Test beberapa pertanyaan terlebih dahulu
    test_questions = [
        "Siapakah Andey Koiwai?",
        "Apa itu proyek newmogiserver?",
        "Bagaimana karakteristik AI Mogi?",
        "Apa teknologi yang digunakan dalam proyek compres_pdf?",
        "Bagaimana Andey Koiwai mendukung komunitas AI?"
    ]
    
    print("=== Testing Model dengan Pertanyaan Sample ===")
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. {question}")
        response = assistant.generate_response(question)
        print(f"Jawaban: {response}")
    
    print("\n" + "="*50)
    
    # Start interactive chat
    assistant.chat()

if __name__ == "__main__":
    main()