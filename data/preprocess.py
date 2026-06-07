import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

print("1. 'alibayram/doktorsitesi' veri seti Hugging Face'ten indiriliyor...")
dataset = load_dataset("alibayram/doktorsitesi", split="train")
df = dataset.to_pandas()

print(f"Orijinal veri seti boyutu: {len(df)} satır.")

# Projede istenen tıp alanları
hedef_alanlar = [
    "beyin-ve-sinir-cerrahisi",
    "uroloji",
    "ortopedi-ve-travmatoloji",
    "dahiliye-ve-ic-hastaliklari",
    "genel-cerrahi",
    "kulak-burun-bogaz-hastaliklari",
    "fiziksel-tip-ve-rehabilitasyon",
    "kardiyoloji",
    "kalp-damar-cerrahisi"
]

print("\n2. İstenen tıp bölümlerine göre filtreleniyor...")
# BURASI DÜZELTİLDİ: Artık 'doctor_speciality' sütununa bakıyoruz
df_filtered = df[df['doctor_speciality'].isin(hedef_alanlar)].copy()

print("\n3. Veri temizliği yapılıyor (Boş ve tekrarlı veriler siliniyor)...")
df_filtered = df_filtered.dropna(subset=['question_content', 'question_answer', 'doctor_speciality'])
df_filtered = df_filtered.drop_duplicates(subset=['question_content'])

print(f"Temizlenmiş veri boyutu: {len(df_filtered)} satır.")

print("\n4. Sınıf oranları korunarak Eğitim ve Test setlerine bölünüyor...")
# Proje Kuralı: Test seti en az 1500 olmalı.
train_df, test_df = train_test_split(
    df_filtered,
    test_size=1500,
    random_state=42,
    stratify=df_filtered['doctor_speciality']
)

print("\n--- SONUÇLAR ---")
print(f"Eğitim (Train) Seti Boyutu: {len(train_df)}")
print(f"Test Seti Boyutu: {len(test_df)}")

# Eğitim modelinde (Qwen) rahat kullanmak için sütun isimlerini Türkçeleştirip sadeleştiriyoruz
train_df = train_df[['question_content', 'question_answer', 'doctor_speciality']].rename(columns={'question_content': 'soru', 'question_answer': 'cevap', 'doctor_speciality': 'kategori'})
test_df = test_df[['question_content', 'question_answer', 'doctor_speciality']].rename(columns={'question_content': 'soru', 'question_answer': 'cevap', 'doctor_speciality': 'kategori'})

# CSV olarak kaydetme işlemi
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print("\n'train_dataset.csv' ve 'test_dataset.csv' dosyaları başarıyla oluşturuldu!")
