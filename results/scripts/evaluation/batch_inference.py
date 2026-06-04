{\rtf1\ansi\ansicpg1254\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red188\green135\blue186;\red30\green31\blue33;\red202\green202\blue202;
\red212\green212\blue212;\red113\green171\blue89;\red194\green126\blue101;\red212\green214\blue154;\red88\green147\blue206;
\red167\green197\blue152;\red113\green184\blue255;\red70\green137\blue204;}
{\*\expandedcolortbl;;\cssrgb\c78824\c61176\c77647;\cssrgb\c15686\c16471\c17255;\cssrgb\c83137\c83137\c83137;
\cssrgb\c86275\c86275\c86275;\cssrgb\c50980\c71765\c42353;\cssrgb\c80784\c56863\c47059;\cssrgb\c86275\c86275\c66667;\cssrgb\c41176\c64706\c84314;
\cssrgb\c70980\c80784\c65882;\cssrgb\c50980\c77647\c100000;\cssrgb\c33725\c61176\c83922;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import\cf4 \strokec4  torch\cb1 \
\cf2 \cb3 \strokec2 import\cf4 \strokec4  pandas \cf2 \strokec2 as\cf4 \strokec4  pd\cb1 \
\cf2 \cb3 \strokec2 from\cf4 \strokec4  transformers \cf2 \strokec2 import\cf4 \strokec4  AutoModelForCausalLM\cf5 \strokec5 ,\cf4 \strokec4  AutoTokenizer\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 # Modelimizi y\'fckl\'fcyoruz\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 model_id = \cf7 \strokec7 "Qwen/Qwen2.5-1.5B-Instruct"\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 \strokec8 print\cf5 \strokec5 (\cf9 \strokec9 f\cf7 \strokec7 "\cf5 \strokec5 \{\cf4 \strokec4 model_id\cf5 \strokec5 \}\cf7 \strokec7  y\'fckleniyor..."\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 tokenizer = AutoTokenizer.from_pretrained\cf5 \strokec5 (\cf4 \strokec4 model_id\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\cb3 model = AutoModelForCausalLM.from_pretrained\cf5 \strokec5 (\cf4 \cb1 \strokec4 \
\cb3     model_id\cf5 \strokec5 ,\cf4 \cb1 \strokec4 \
\cb3     torch_dtype=\cf7 \strokec7 "auto"\cf5 \strokec5 ,\cf4 \cb1 \strokec4 \
\cb3     device_map=\cf7 \strokec7 "auto"\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 # \uc0\u304 smail veriyi getirene kadar kullanaca\u287 \u305 m\u305 z SAHTE (Mock) veri setimiz\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 dummy_test_set = \cf5 \strokec5 [\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 \{\cf7 \strokec7 "id"\cf5 \strokec5 :\cf4 \strokec4  \cf10 \strokec10 1\cf5 \strokec5 ,\cf4 \strokec4  \cf7 \strokec7 "soru"\cf5 \strokec5 :\cf4 \strokec4  \cf7 \strokec7 "S\'fcrekli ba\uc0\u351  a\u287 r\u305 s\u305  ve mide bulant\u305 m var, hangi b\'f6l\'fcme gitmeliyim?"\cf5 \strokec5 \},\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 \{\cf7 \strokec7 "id"\cf5 \strokec5 :\cf4 \strokec4  \cf10 \strokec10 2\cf5 \strokec5 ,\cf4 \strokec4  \cf7 \strokec7 "soru"\cf5 \strokec5 :\cf4 \strokec4  \cf7 \strokec7 "Dedemin kalbinde ritim bozuklu\uc0\u287 u var, hangi doktordan randevu alal\u305 m?"\cf5 \strokec5 \},\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 \{\cf7 \strokec7 "id"\cf5 \strokec5 :\cf4 \strokec4  \cf10 \strokec10 3\cf5 \strokec5 ,\cf4 \strokec4  \cf7 \strokec7 "soru"\cf5 \strokec5 :\cf4 \strokec4  \cf7 \strokec7 "\'c7ocu\uc0\u287 umun kula\u287 \u305  \'e7ok a\u287 r\u305 yor ve ate\u351 i var, nereye ba\u351 vurmal\u305 y\u305 m?"\cf5 \strokec5 \}\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 ]\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 sonuclar = \cf5 \strokec5 []\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 \strokec8 print\cf5 \strokec5 (\cf7 \strokec7 "Toplu test ba\uc0\u351 l\u305 yor...\\n"\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 # Veri setindeki her bir soru i\'e7in d\'f6ng\'fc ba\uc0\u351 lat\u305 yoruz\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 for\cf4 \strokec4  item \cf11 \strokec11 in\cf4 \strokec4  dummy_test_set\cf5 \strokec5 :\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3     prompt = \cf9 \strokec9 f\cf7 \strokec7 "Soru: \cf5 \strokec5 \{\cf4 \strokec4 item\cf5 \strokec5 [\cf7 \strokec7 'soru'\cf5 \strokec5 ]\}\cf7 \strokec7 \\nCevap:"\cf4 \cb1 \strokec4 \
\cb3     \cb1 \
\cb3     messages = \cf5 \strokec5 [\cf4 \cb1 \strokec4 \
\cb3         \cf5 \strokec5 \{\cf7 \strokec7 "role"\cf5 \strokec5 :\cf4 \strokec4  \cf7 \strokec7 "system"\cf5 \strokec5 ,\cf4 \strokec4  \cf7 \strokec7 "content"\cf5 \strokec5 :\cf4 \strokec4  \cf7 \strokec7 "Sen uzman bir t\uc0\u305 bbi asistans\u305 n."\cf5 \strokec5 \},\cf4 \cb1 \strokec4 \
\cb3         \cf5 \strokec5 \{\cf7 \strokec7 "role"\cf5 \strokec5 :\cf4 \strokec4  \cf7 \strokec7 "user"\cf5 \strokec5 ,\cf4 \strokec4  \cf7 \strokec7 "content"\cf5 \strokec5 :\cf4 \strokec4  prompt\cf5 \strokec5 \}\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 ]\cf4 \cb1 \strokec4 \
\cb3     \cb1 \
\cb3     text = tokenizer.apply_chat_template\cf5 \strokec5 (\cf4 \strokec4 messages\cf5 \strokec5 ,\cf4 \strokec4  tokenize=\cf12 \strokec12 False\cf5 \strokec5 ,\cf4 \strokec4  add_generation_prompt=\cf12 \strokec12 True\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\cb3     inputs = tokenizer\cf5 \strokec5 ([\cf4 \strokec4 text\cf5 \strokec5 ],\cf4 \strokec4  return_tensors=\cf7 \strokec7 "pt"\cf5 \strokec5 )\cf4 \strokec4 .to\cf5 \strokec5 (\cf4 \strokec4 model.device\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\cb3     \cb1 \
\cb3     \cf6 \strokec6 # Modelden cevab\uc0\u305  \'fcretiyoruz\cf4 \cb1 \strokec4 \
\cb3     outputs = model.generate\cf5 \strokec5 (\cf4 \strokec4 **inputs\cf5 \strokec5 ,\cf4 \strokec4  max_new_tokens=\cf10 \strokec10 100\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\cb3     \cb1 \
\cb3     \cf6 \strokec6 # SADECE YEN\uc0\u304  \'dcRET\u304 LEN CEVABI ALMAK \u304 \'c7\u304 N G\'dcNCELLED\u304 \u286 \u304 M\u304 Z KISIM:\cf4 \cb1 \strokec4 \
\cb3     \cf6 \strokec6 # Girdi tokenlerinin uzunlu\uc0\u287 unu buluyoruz\cf4 \cb1 \strokec4 \
\cb3     input_length = inputs.input_ids.shape\cf5 \strokec5 [\cf10 \strokec10 1\cf5 \strokec5 ]\cf4 \strokec4  \cb1 \
\cb3     \cb1 \
\cb3     \cf6 \strokec6 # Sadece o uzunluktan sonras\uc0\u305 n\u305  (modelin \'fcretti\u287 i yeni tokenleri) decode ediyoruz\cf4 \cb1 \strokec4 \
\cb3     response = tokenizer.decode\cf5 \strokec5 (\cf4 \strokec4 outputs\cf5 \strokec5 [\cf10 \strokec10 0\cf5 \strokec5 ][\cf4 \strokec4 input_length\cf5 \strokec5 :],\cf4 \strokec4  skip_special_tokens=\cf12 \strokec12 True\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\cb3     \cb1 \
\cb3     \cf6 \strokec6 # Sonucu listeye kaydediyoruz\cf4 \cb1 \strokec4 \
\cb3     sonuclar.append\cf5 \strokec5 (\{\cf4 \cb1 \strokec4 \
\cb3         \cf7 \strokec7 "id"\cf5 \strokec5 :\cf4 \strokec4  item\cf5 \strokec5 [\cf7 \strokec7 "id"\cf5 \strokec5 ],\cf4 \cb1 \strokec4 \
\cb3         \cf7 \strokec7 "soru"\cf5 \strokec5 :\cf4 \strokec4  item\cf5 \strokec5 [\cf7 \strokec7 "soru"\cf5 \strokec5 ],\cf4 \cb1 \strokec4 \
\cb3         \cf7 \strokec7 "model_cevabi"\cf5 \strokec5 :\cf4 \strokec4  response.strip\cf5 \strokec5 ()\cf4 \strokec4  \cf6 \strokec6 # .strip() ile ba\uc0\u351 taki/sondaki bo\u351 luklar\u305  temizliyoruz\cf4 \cb1 \strokec4 \
\cb3     \cf5 \strokec5 \})\cf4 \cb1 \strokec4 \
\cb3     \cf8 \strokec8 print\cf5 \strokec5 (\cf9 \strokec9 f\cf7 \strokec7 "Soru \cf5 \strokec5 \{\cf4 \strokec4 item\cf5 \strokec5 [\cf7 \strokec7 'id'\cf5 \strokec5 ]\}\cf7 \strokec7  tamamland\uc0\u305 ."\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 # \uc0\u304 \u351 lem bitince sonu\'e7lar\u305  bir DataFrame'e \'e7evirip ekranda g\'f6steriyoruz\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf4 \cb3 df = pd.DataFrame\cf5 \strokec5 (\cf4 \strokec4 sonuclar\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf8 \cb3 \strokec8 print\cf5 \strokec5 (\cf7 \strokec7 "\\n--- TEST TAMAMLANDI. \uc0\u304 LK 3 SONU\'c7: ---"\cf5 \strokec5 )\cf4 \cb1 \strokec4 \
\cf8 \cb3 \strokec8 print\cf5 \strokec5 (\cf4 \strokec4 df.head\cf5 \strokec5 ())\cf4 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 # \uc0\u304 smail'in verisi geldi\u287 inde bunu CSV olarak kaydedece\u287 iz:\cf4 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 # df.to_csv("qwen_baseline_sonuclari.csv", index=False)\cf4 \cb1 \strokec4 \
}