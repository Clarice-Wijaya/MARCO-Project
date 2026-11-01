import argparse
import glob
import os
from collections import Counter

import tensorflow as tf


def convert(shard_dir, out_dir):
    counts = Counter()
    for path in sorted(glob.glob(os.path.join(shard_dir, "test-*"))):
        for raw in tf.data.TFRecordDataset(path):
            f = tf.train.Example.FromString(raw.numpy()).features.feature
            img = f["image/encoded"].bytes_list.value[0]
            img_id = f["image/id"].int64_list.value[0]
            label = f["image/class/text"].bytes_list.value[0].decode()

            folder = os.path.join(out_dir, label)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, f"{img_id}.jpg"), "wb") as fp:
                fp.write(img)
            counts[label] += 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    counts = convert(args.shard_dir, args.out_dir)
    for label, n in counts.most_common():
        print(label, n)
