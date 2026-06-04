{\rtf1\ansi\ansicpg1254\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red188\green135\blue186;\red30\green31\blue33;}
{\*\expandedcolortbl;;\cssrgb\c78824\c61176\c77647;\cssrgb\c15686\c16471\c17255;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import torch\
import gc\
from transformers import AutoModelForCausalLM, AutoTokenizer\
\
def test_model(model_id, prompt):\
    print(f"\\n\{'='*50\}")\
    print(f"Y\'fckleniyor: \{model_id\}...")\
    \
    tokenizer = AutoTokenizer.from_pretrained(model_id)\
    model = AutoModelForCausalLM.from_pretrained(\
        model_id,\
        torch_dtype="auto",\
        device_map="auto"\
    )\
\
    messages = [\
        \{"role": "system", "content": "Sen uzman bir t\uc0\u305 bbi asistans\u305 n."\},\
        \{"role": "user", "content": prompt\}\
    ]\
\
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\
    inputs = tokenizer([text], return_tensors="pt").to(model.device)\
\
    print("Model cevap \'fcretiyor...\\n")\
    outputs = model.generate(**inputs, max_new_tokens=150)\
    \
    # Sadece yeni \'fcretilen cevab\uc0\u305  almak i\'e7in\
    input_length = inputs.input_ids.shape[1]\
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)\
\
    print(f"--- \{model_id\} \'c7IKTISI ---")\
    print(response.strip())\
    print(f"\{'='*50\}\\n")\
\
    # Modeli haf\uc0\u305 zadan siliyoruz ki di\u287 er modele yer a\'e7\u305 ls\u305 n\
    del model\
    del tokenizer\
    del inputs\
    del outputs\
    gc.collect()\
    if torch.cuda.is_available():\
        torch.cuda.empty_cache()\
\
# Test i\'e7in sorumuz\
test_sorusu = "Soru: S\'fcrekli ba\uc0\u351  a\u287 r\u305 s\u305  \'e7ekiyorum ve midem bulan\u305 yor. Hangi b\'f6l\'fcme gitmeliyim?\\nCevap:"\
\
# 1. Qwen Modelini Test Et\
test_model("Qwen/Qwen2.5-1.5B-Instruct", test_sorusu)\
\
# 2. Phi-3.5 Modelini Test Et\
test_model("microsoft/Phi-3.5-mini-instruct", test_sorusu)\
\
print("T\'fcm baseline testleri ba\uc0\u351 ar\u305 yla tamamland\u305 !")}