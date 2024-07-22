import argparse

def get_args():
    parser = argparse.ArgumentParser("Transformer-based Translator From CN To EN")
    # Log file & data
    parser.add_argument("--save-dir", type=str, default="run", help="")
    parser.add_argument("--save-dir-model", type=str, default="model", help="")
    # DataLoader
    parser.add_argument("--min-cnt", type=int, default=0, help="words whose occurred less than min_cnt are encoded as <UNK>")
    parser.add_argument("--max-len", type=int, default=50, help="maximum number of words in a sentence")
    parser.add_argument("--source-train", type=str, default="train.source.cn.txt", help="")
    parser.add_argument("--target-train", type=str, default="train.target.en.txt", help="")
    parser.add_argument("--source-test", type=str, default="test.source.cn.txt", help="")
    parser.add_argument("--target-test", type=str, default="test.target.en.txt", help="")
    parser.add_argument("--source-vocab", type=str, default="vocab.cn.tsv", help="")
    parser.add_argument("--target-vocab", type=str, default="vocab.en.tsv", help="")
    # Transformer
    parser.add_argument("--n-epoch", type=int, default=50, help="")
    parser.add_argument("--batch-size", type=int, default=64, help="")
    parser.add_argument("--n-layers", type=int, default=6, help="")
    parser.add_argument("--n-head", type=int, default=8, help="")
    parser.add_argument("--d-model", type=int, default=512, help="")
    parser.add_argument("--ffn-hidden", type=int, default=2048, help="")
    parser.add_argument("--lr", type=float, default=1e-4, help="")
    parser.add_argument("--lr-decay", type=float, default=5e-4, help="")
    parser.add_argument("--drop-prob", type=float, default=0.1, help="")
    parser.add_argument("--adam-eps", type=float, default=5e-9, help="")
    parser.add_argument("--clip", type=float, default=1.0, help="")
    parser.add_argument("--save-rate", type=int, default=10, help="")

    args = parser.parse_args()
    return args