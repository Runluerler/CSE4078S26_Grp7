import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before_after", default="results/before_after_outputs.csv")
    parser.add_argument("--output", default="results/error_analysis_sample.csv")
    parser.add_argument("--n_per_class", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.before_after)
    samples = []
    for cat, g in df.groupby("kategori"):
        samples.append(g.sample(min(args.n_per_class, len(g)), random_state=42))
    out = pd.concat(samples).reset_index(drop=True)
    out["human_label"] = ""  # improved / same_correct / still_wrong / worse
    out["error_type"] = ""   # hallucination / incomplete / wrong_specialty / vague / unsafe / language
    out["comment"] = ""
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved manual review file: {args.output}")


if __name__ == "__main__":
    main()
