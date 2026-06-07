# 1. GEREKLİ KÜTÜPHANELERİ KUR
# !pip install -q -U transformers datasets trl peft bitsandbytes accelerate
# from huggingface_hub import notebook_login
# notebook_login() # Hugging Face Token'ını gireceksin

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# NOT: Bu kod Google Colab üzerinde A100 GPU ile çalıştırılmıştır.
# Local'de veya başka bir sunucuda çalıştırmak için Drive yollarını güncelleyiniz.

# 2. GOOGLE DRIVE'A BAĞLAN
# from google.colab import drive
# print("Google Drive'a bağlanılıyor...")
# drive.mount('/content/drive')
KAYIT_YERI = "models/CSE4078_Qwen_Medical_LoRA_PRO"

# 3. VERİ SETİNİ YÜKLE
print("Eğitim verisi yükleniyor...")
VERI_YOLU = 'data/train_dataset.csv'
df = pd.read_csv(VERI_YOLU, lineterminator='\n')
dataset = Dataset.from_pandas(df)
print(f"Veri başarıyla okundu! Toplam {len(df)} satır işleniyor.")

# 4. TOKENİZER
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def format_prompts(ornek):
    messages = [
        {"role": "system", "content": "Sen uzman bir tıbbi asistansın. Hastaların sorularına ilgili tıbbi alana (kategoriye) uygun, profesyonel ve anlaşılır cevaplar ver."},
        {"role": "user", "content": f"Kategori: {ornek['kategori']}\nSoru: {ornek['soru']}"},
        {"role": "assistant", "content": ornek['cevap']}
    ]
    ornek["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return ornek

print("Veriler ChatML formatına dönüştürülüyor...")
train_dataset = dataset.map(format_prompts)

# 5. MODELİ YÜKLE (A100 İÇİN MÜKEMMEL bfloat16 DESTEĞİ AÇIK!)
print("Model A100 GPU gücüyle yükleniyor...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16   # A100'de mükemmel çalışır
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 6. LoRA AYARLARI
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 7. EĞİTİM PARAMETRELERİ (PRO AYARLAR)
training_args = SFTConfig(
    output_dir=KAYIT_YERI,
    dataset_text_field="text",
    per_device_train_batch_size=4,    # A100 Hafızası yüksek olduğu için 2'den 4'e çıkardık, hız katlanacak!
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=50,
    num_train_epochs=1,
    optim="paged_adamw_32bit",
    fp16=False,
    bf16=True,    # A100'de bizi uçuracak donanımsal hızlandırıcı
    save_strategy="epoch",
    report_to="none"
)

# 8. EĞİTİMİ BAŞLAT
print("A100 Motorları ateşlendi! Eğitim başlıyor...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)
trainer.train()

# 9. KAYDET
print(f"Eğitim tamamlandı! Model şu konuma kaydediliyor: {KAYIT_YERI}")
trainer.save_model(KAYIT_YERI)
tokenizer.save_pretrained(KAYIT_YERI)
print("İşlem Başarılı! Projenizin modeli tamamen hazır!")
