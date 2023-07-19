import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm

def preprocess_one_dir(in_data_dir, out_dir, data_type, spk):
    """Create .json file for one condition."""
    file_infos = []
    in_dir = os.path.abspath(os.path.join(in_data_dir, data_type, spk))
    wav_list = os.listdir(in_dir)
    wav_list.sort()
    for wav_file in tqdm(wav_list):
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples = sf.SoundFile(wav_path)
        if spk == "mix":
            file_infos.append((wav_path, len(samples)))
        else:
            file_infos.append(
                (
                    wav_path,
                    len(samples),
                )
            )
    if not os.path.exists(os.path.join(out_dir, data_type)):
        os.makedirs(os.path.join(out_dir, data_type))
    with open(os.path.join(out_dir, data_type, spk + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)


def preprocess_wham_audio(inp_args):
    """Create .json files for all conditions."""
    speaker_list = ["mix_both", "s1", "s2"]
    for data_type in ["tr", "cv", "tt"]:
        for spk in speaker_list:
            preprocess_one_dir(
                inp_args.in_dir, inp_args.out_dir, data_type, spk,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LRS2 audio data preprocessing")
    parser.add_argument(
        "--in_dir",
        type=str,
        default=None,
        help="Directory path of audio including tr, cv and tt",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Directory path to put output files"
    )
    args = parser.parse_args()
    print(args)
    preprocess_wham_audio(args)