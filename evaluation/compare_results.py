import argparse
import os
import pandas as pd

try:
    import evaluate
except ImportError:
    evaluate = None


def simple_stats(df):
    return {
        "num_examples": len(df),
        "avg_answer_len_chars": df["model_cevabi"].fillna("").str.len().mean(),
        "empty_answers": int((df["model_cevabi"].fillna("").str.strip() == "").sum()),
    }


def compute_rouge(df):
    if evaluate is None:
        return {}
    rouge = evaluate.load("rouge")
    preds = df["model_cevabi"].fillna("").astype(str).tolist()
    refs = df["reference_cevap"].fillna("").astype(str).tolist()
    return rouge.compute(predictions=preds, references=refs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    ft = pd.read_csv(args.finetuned)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for name, df in [("base", base), ("fine_tuned", ft)]:
        stats = simple_stats(df)
        stats.update(compute_rouge(df))
        stats["model"] = name
        rows.append(stats)

    summary = pd.DataFrame(rows)
    metric_cols = [c for c in summary.columns if c != "model"]
    summary = summary[["model"] + metric_cols]
    summary.to_csv(os.path.join(args.output_dir, "comparison_summary.csv"), index=False)

    merged = base[["id", "kategori", "soru", "reference_cevap", "model_cevabi"]].rename(columns={"model_cevabi": "base_answer"}).merge(
        ft[["id", "model_cevabi"]].rename(columns={"model_cevabi": "fine_tuned_answer"}),
        on="id",
        how="inner",
    )
    merged.to_csv(os.path.join(args.output_dir, "before_after_outputs.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"Saved files in: {args.output_dir}")


if __name__ == "__main__":
    main()
