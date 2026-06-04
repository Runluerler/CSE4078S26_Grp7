import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model(model_id, prompt):
    print(f"\n{'='*50}")
    print(f"Yükleniyor: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    messages = [
        {"role": "system", "content": "Sen uzman bir tıbbi asistansın."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Model cevap üretiyor...\n")
    outputs = model.generate(**inputs, max_new_tokens=150)
    
    # Sadece yeni üretilen cevabı almak için
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    print(f"--- {model_id} ÇIKTISI ---")
    print(response.strip())
    print(f"{'='*50}\n")

    # Modeli hafızadan siliyoruz ki diğer modele yer açılsın
    del model
    del tokenizer
    del inputs
    del outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Test için sorumuz
test_sorusu = "Soru: Sürekli baş ağrısı çekiyorum ve midem bulanıyor. Hangi bölüme gitmeliyim?\nCevap:"

# 1. Qwen Modelini Test Et
test_model("Qwen/Qwen2.5-1.5B-Instruct", test_sorusu)

# 2. Phi-3.5 Modelini Test Et
test_model("microsoft/Phi-3.5-mini-instruct", test_sorusu)

print("Tüm baseline testleri başarıyla tamamlandı!")