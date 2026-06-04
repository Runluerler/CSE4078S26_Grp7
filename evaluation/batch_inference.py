import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Modelimizi yüklüyoruz
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"{model_id} yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

# İsmail veriyi getirene kadar kullanacağımız SAHTE (Mock) veri setimiz
dummy_test_set = [
    {"id": 1, "soru": "Sürekli baş ağrısı ve mide bulantım var, hangi bölüme gitmeliyim?"},
    {"id": 2, "soru": "Dedemin kalbinde ritim bozukluğu var, hangi doktordan randevu alalım?"},
    {"id": 3, "soru": "Çocuğumun kulağı çok ağrıyor ve ateşi var, nereye başvurmalıyım?"}
]

sonuclar = []

print("Toplu test başlıyor...\n")

# Veri setindeki her bir soru için döngü başlatıyoruz
for item in dummy_test_set:
    prompt = f"Soru: {item['soru']}\nCevap:"
    
    messages = [
        {"role": "system", "content": "Sen uzman bir tıbbi asistansın."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Modelden cevabı üretiyoruz
    outputs = model.generate(**inputs, max_new_tokens=100)
    
    # SADECE YENİ ÜRETİLEN CEVABI ALMAK İÇİN GÜNCELLEDİĞİMİZ KISIM:
    # Girdi tokenlerinin uzunluğunu buluyoruz
    input_length = inputs.input_ids.shape[1] 
    
    # Sadece o uzunluktan sonrasını (modelin ürettiği yeni tokenleri) decode ediyoruz
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # Sonucu listeye kaydediyoruz
    sonuclar.append({
        "id": item["id"],
        "soru": item["soru"],
        "model_cevabi": response.strip() # .strip() ile baştaki/sondaki boşlukları temizliyoruz
    })
    print(f"Soru {item['id']} tamamlandı.")

# İşlem bitince sonuçları bir DataFrame'e çevirip ekranda gösteriyoruz
df = pd.DataFrame(sonuclar)
print("\n--- TEST TAMAMLANDI. İLK 3 SONUÇ: ---")
print(df.head())

# İsmail'in verisi geldiğinde bunu CSV olarak kaydedeceğiz:
# df.to_csv("qwen_baseline_sonuclari.csv", index=False)