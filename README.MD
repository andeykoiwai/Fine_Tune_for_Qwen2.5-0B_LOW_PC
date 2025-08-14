# Fine-tuning Qwen2.5-0.5B untuk Asisten Andey Koiwai

### python dan VSCODE
1 python -m venv hugface 
2 hugface\Scripts\activate
3 ctrl+shif+p pilih interpreter di venv kamu ya ^^

## Persyaratan Sistem

### Hardware
- **CPU**: Core i5-6300U (sudah memadai)
- **RAM**: 8GB (minimal untuk model 0.5B)
- **Storage**: ~5GB untuk model dan dataset

### Software
- Python 3.8+
- CUDA tidak diperlukan (akan menggunakan CPU)

## Instalasi Dependencies

Buat file `requirements.txt`:

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.20.0
tokenizers>=0.14.0
numpy
tqdm
```

Install dengan:
```bash
pip install -r requirements.txt
```

## Langkah-langkah Eksekusi

### 1. Persiapan Data
Pastikan file `mydata(2).txt` berada di direktori yang sama dengan script.

### 2. Download Model Base
```bash
# Jika belum ada, download model base terlebih dahulu
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"
```

### 3. Jalankan Fine-tuning
```bash
python qwen_finetune_script.py
```

**Catatan**: Proses ini akan memakan waktu 2-4 jam di CPU Core i5-6300U.

### 4. Test Model
Setelah fine-tuning selesai:
```bash
python inference_script.py
```

## Optimasi untuk Hardware Terbatas

### Memory Management
- Batch size: 1 (sangat kecil)
- Gradient accumulation: 8 steps
- Gradient checkpointing: enabled
- Dataloader workers: 0 (hindari multiprocessing)

### CPU Optimizations
- Mixed precision: disabled (FP16 tidak optimal di CPU)
- Pin memory: disabled
- Multiprocessing: disabled

## Struktur Output

Setelah fine-tuning, struktur folder akan seperti ini:
```
./
├── qwen-andey-assistant/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
├── mydata(2).txt
├── qwen_finetune_script.py
└── inference_script.py
```

## Monitoring Progress

Script akan menampilkan:
- Loss training dan evaluation
- Perplexity metrics
- Sample responses selama training
- Memory usage warnings

## Troubleshooting

### Out of Memory
Jika masih OOM, kurangi:
- `max_length` dari 512 ke 256
- `gradient_accumulation_steps` dari 8 ke 4
- Tutup aplikasi lain saat training

### Model Tidak Merespons dengan Baik
- Tingkatkan `num_epochs` ke 5-10
- Adjust `learning_rate` ke 1e-5
- Tambah data training jika memungkinkan

### Training Terlalu Lambat
- Kurangi `max_length` ke 256
- Kurangi `eval_steps` dan `save_steps`
- Disable evaluation selama training

## Performance Expectations

### Training Time
- **Estimasi**: 2-4 jam untuk 3 epochs
- **Dataset**: ~45 samples Q&A
- **Hardware**: Core i5-6300U, RAM 8GB

### setPython dan VS CODE
1 python -m venv hugface 
2 hugface\Scripts\activate
3 ctrl+shif+p pilih interpreter di venv kamu ya ^^

### Model Quality
- Model akan dapat menjawab pertanyaan spesifik tentang Andey Koiwai
- Akurasi tergantung kualitas dan variasi data training
- Untuk hasil terbaik, pertimbangkan tambah data training

## Usage Examples

Setelah training selesai, model dapat:

```python
# Import dan load model
assistant = AndeyAssistant("./qwen-andey-assistant")

# Single question
response = assistant.generate_response("Siapakah Andey Koiwai?")

# Interactive chat
assistant.chat()
```

## Tips Optimasi Lanjutan

1. **Data Augmentation**: Tambah variasi pertanyaan dengan parafrasa
2. **Context Length**: Sesuaikan dengan kebutuhan (256-512 tokens)
3. **Learning Rate Scheduling**: Gunakan cosine annealing untuk hasil lebih stabil
4. **Evaluation Metrics**: Monitor BLEU score untuk kualitas response

## Catatan Penting

- Training di CPU akan lambat tapi tetap feasible untuk model 0.5B
- Monitor usage RAM selama training
- Backup model checkpoints secara berkala
- Test model dengan berbagai jenis pertanyaan sebelum deploy