import argparse
import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_id, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_id, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question, max_new_tokens=180):
    messages = [
        {"role": "system", "content": "Sen Türkçe konuşan uzman bir tıbbi asistansın. Soruyu kısa, doğru ve güvenli şekilde cevapla."},
        {"role": "user", "content": f"Soru: {question}\nCevap:"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/test_dataset.csv")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter folder, e.g. models/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Use small number for quick test first, e.g. 20")
    parser.add_argument("--max_new_tokens", type=int, default=180)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)

    tokenizer, model = load_model(args.model_id, args.adapter_path)
    rows = []
    start = time.time()

    for i, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row["soru"])
        pred = generate_answer(tokenizer, model, q, args.max_new_tokens)
        rows.append({
            "id": i,
            "kategori": row.get("kategori", ""),
            "soru": q,
            "reference_cevap": row.get("cevap", ""),
            "model_cevabi": pred,
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Examples: {len(rows)} | Time: {(time.time()-start)/60:.2f} minutes")


if __name__ == "__main__":
    main()
