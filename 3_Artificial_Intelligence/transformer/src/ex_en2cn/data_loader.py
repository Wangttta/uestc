import codecs
import regex
import random
import numpy as np
from collections import Counter

from _common.util import *


class DataLoader:

    def __init__(self, args):
        self.args = args
        self.abs_path = get_abs_path(root=None, rel_path="ex_en2cn/data")
        self.cn2idx, self.idx2cn = self.load_cn_vocab()
        self.en2idx, self.idx2en = self.load_en_vocab()
        args.src_pad_idx = self.cn2idx["<PAD>"]
        args.trg_pad_idx = self.en2idx["<PAD>"]
        args.trg_sos_idx = self.en2idx["<S>"]
        args.vocab_size_source = len(self.cn2idx)
        args.vocab_size_target = len(self.en2idx)

    def _load_vocab(self, language):
        assert language in ["cn", "en"]
        vocab = [
            line.split()[0]
            for line in codecs.open(
                get_abs_path(root=self.abs_path, rel_path="vocab.{}.tsv".format(language)), "r", "utf-8"
            )
            .read()
            .splitlines()
            if int(line.split()[1]) >= self.args.min_cnt
        ]
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        return word2idx, idx2word

    def load_cn_vocab(self):
        word2idx, idx2word = self._load_vocab("cn")
        return word2idx, idx2word


    def load_en_vocab(self):
        word2idx, idx2word = self._load_vocab("en")
        return word2idx, idx2word

    def create_data(self, source_sents, target_sents):
        cn2idx, _ = self.load_cn_vocab()
        en2idx, _ = self.load_en_vocab()
        x_list, y_list, sources, targets = [], [], [], []
        for source_sent, target_sent in zip(source_sents, target_sents):
            x = [
                cn2idx.get(word, 1) for word in (source_sent + " </S>").split()
            ]  # 1: OOV, </S>: End of Text
            y = [en2idx.get(word, 1) for word in (target_sent + " </S>").split()]
            if max(len(x), len(y)) <= self.args.max_len:
                x_list.append(np.array(x))
                y_list.append(np.array(y))
                sources.append(source_sent)
                targets.append(target_sent)
        X = np.zeros([len(x_list), self.args.max_len], np.int32)
        Y = np.zeros([len(y_list), self.args.max_len], np.int32)
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            X[i] = np.lib.pad(
                x, [0, self.args.max_len - len(x)], "constant", constant_values=(0, 0)
            )
            Y[i] = np.lib.pad(
                y, [0, self.args.max_len - len(y)], "constant", constant_values=(0, 0)
            )
        return X, Y, sources, targets

    def _load_data(self, data_type):
        assert data_type in ["train", "test"]
        if data_type == "train":
            source = get_abs_path(root=self.abs_path, rel_path=self.args.source_train)
            target = get_abs_path(root=self.abs_path, rel_path=self.args.target_train)
        else:
            source = get_abs_path(root=self.abs_path, rel_path=self.args.source_test)
            target = get_abs_path(root=self.abs_path, rel_path=self.args.target_test)
        cn_sents = [
            regex.sub("[^\s\p{L}']", "", line)
            for line in codecs.open(source, "r", "utf-8").read().split("\n")
            if line and line[0] != "<"
        ]
        en_sents = [
            regex.sub("[^\s\p{L}']", "", line)
            for line in codecs.open(target, "r", "utf-8").read().split("\n")
            if line and line[0] != "<"
        ]
        X, Y, sources, targets = self.create_data(cn_sents, en_sents)
        return X, Y, sources, targets

    def load_train_data(self):
        X, Y, _, _ = self._load_data("train")
        return X, Y


    def load_test_data(self):
        X, _, Sources, Targets = self._load_data("test")
        return X, Sources, Targets

    def get_batch_indices(self, total_length, batch_size):
        assert (batch_size <= total_length), "Batch size is large than total data length. Check your data or change batch size."
        current_index = 0
        indexs = [i for i in range(total_length)]
        random.shuffle(indexs)
        while 1:
            if current_index + batch_size >= total_length:
                break
            current_index += batch_size
            yield indexs[current_index : current_index + batch_size], current_index

    def make_vocabulary(self):
        source = get_abs_path(root=self.abs_path, rel_path=self.args.source_train)
        target = get_abs_path(root=self.abs_path, rel_path=self.args.target_train)
        for from_file, to_file in [(source, self.args.source_vocab), (target, self.args.target_vocab)]:
            text = codecs.open(from_file, "r", "utf-8").read()
            text = regex.sub("[^\s\p{L}']", "", text)
            words = text.split()
            word2cnt = Counter(words)
            with codecs.open(to_file, "w", "utf-8") as fout:
                fout.write(
                    "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format(
                        "<PAD>", "<UNK>", "<S>", "</S>"
                    )
                )
            for word, cnt in word2cnt.most_common(len(word2cnt)):
                fout.write(u"{}\t{}\n".format(word, cnt))

    def text2embedding(self, text):
        input_seq = list(text)
        embedding = np.zeros(self.args.max_len, dtype=np.int64)
        for i, letter in enumerate(input_seq):
            assert letter in self.cn2idx, f"词表中无法检索到该字符：{letter}"
            embedding[i] = self.cn2idx[letter]
        embedding[len(input_seq):] = self.cn2idx["<PAD>"]
        embedding[len(input_seq)] = self.cn2idx["</S>"]
        return embedding

    def embedding2text(self, embedding):
        output_seq = []
        for idx in embedding:
            letter = self.idx2en[idx]
            if letter == "</S>":
                break
            output_seq.append(letter)
        return ' '.join(output_seq)