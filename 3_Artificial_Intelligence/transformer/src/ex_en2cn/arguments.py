import argparse

def get_args():
    parser = argparse.ArgumentParser("Transformer-based Translator From CN To EN")
    # Program settings
    parser.add_argument("--save-dir", type=str, default="run", help="")
    parser.add_argument("--save-dir-model", type=str, default="model", help="")
    parser.add_argument("--model-path", type=str, default="", help="")
    parser.add_argument("--evaluate", type=bool, default=False, help="")
    parser.add_argument("--inference", type=bool, default=False, help="")
    parser.add_argument("--n-inference", type=int, default=10, help="")
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
    parser.add_argument("--n-epoch", type=int, default=500, help="")
    parser.add_argument("--batch-size", type=int, default=16, help="")
    parser.add_argument("--batch-size-evaluate", type=int, default=64, help="")
    parser.add_argument("--n-layers", type=int, default=3, help="")
    parser.add_argument("--n-head", type=int, default=8, help="")
    parser.add_argument("--d-model", type=int, default=512, help="")
    parser.add_argument("--ffn-hidden", type=int, default=512, help="")
    parser.add_argument("--lr", type=float, default=1e-4, help="")
    parser.add_argument("--lr-decay", type=float, default=1e-8, help="")
    parser.add_argument("--drop-prob", type=float, default=0.4, help="")
    parser.add_argument("--adam-eps", type=float, default=1e-9, help="")
    parser.add_argument("--clip", type=float, default=1.0, help="")
    parser.add_argument("--sinusoid", type=bool, default=True, help="")
    parser.add_argument("--save-rate", type=int, default=5, help="")
    args = parser.parse_args()
    return args