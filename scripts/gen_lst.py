"""Generate cross-sentence lst file from LibriSpeech test-clean.json."""
import argparse
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="LibriSpeech data root directory")
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    JSON_PATH = os.path.join(DATA_ROOT, "test-clean.json")

    entries = []
    with open(JSON_PATH) as f:
        for line in f:
            entries.append(json.loads(line))

    by_spk_chap = defaultdict(list)
    for e in entries:
        parts = e["audio_filepath"].split("/")
        spk, chap = parts[1], parts[2]
        utt_id = os.path.splitext(parts[3])[0]
        by_spk_chap[(spk, chap)].append({
            "id": utt_id,
            "dur": e["duration"],
            "text": e["text_raw"],
        })

    pairs = []
    for (spk, chap), utts in sorted(by_spk_chap.items()):
        utts.sort(key=lambda x: x["id"])
        for i in range(len(utts) - 1):
            src = utts[i]
            tgt = utts[i + 1]
            if 3.0 <= src["dur"] <= 10.0 and 3.0 <= tgt["dur"] <= 10.0:
                line = "\t".join([
                    src["id"], f"{src['dur']:.3f}", src["text"],
                    tgt["id"], f"{tgt['dur']:.3f}", tgt["text"],
                ])
                pairs.append(line)

    print(f"Total pairs: {len(pairs)}")
    out_path = os.path.join(DATA_ROOT, "test_clean_cross_sentence.lst")
    with open(out_path, "w") as f:
        f.write("\n".join(pairs) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
