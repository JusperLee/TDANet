# An efficient encoder-decoder architecture with top-down attention for speech separation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-efficient-encoder-decoder-architecture/speech-separation-on-libri2mix)](https://paperswithcode.com/sota/speech-separation-on-libri2mix?p=an-efficient-encoder-decoder-architecture) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-efficient-encoder-decoder-architecture/speech-separation-on-wham)](https://paperswithcode.com/sota/speech-separation-on-wham?p=an-efficient-encoder-decoder-architecture)

This repository is the official implementation of [An efficient encoder-decoder architecture with top-down attention for speech separation](https://cslikai.cn/project/TDANet) [Paper link](https://openreview.net/pdf?id=fzberKYWKsI). 

```
@inproceedings{tdanet2023iclr,
  title={An efficient encoder-decoder architecture with top-down attention for speech separation},
  author={Li, Kai and Yang, Runxuan and Hu, Xiaolin},
  booktitle={ICLR},
  year={2023}
}
```

## Datasets

The [LRS2 dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) contains thousands of video clips acquired through BBC. LRS2 contains a large amount of noise and reverberation interference, which is more challenging and closer to the actual environment than the WSJ0 and LibriSpeech corpora. 

**LRS2-2Mix** is created by using the LRS2 corpus, where the training set, validation set and test set contain 20000, 5000 and 3000 utterances, respectively. The two different speaker audios from different scenes with 16 kHz sample rate were randomly selected from the LRS2 corpus and were mixed with signal-to-noise ratios sampled between -5 dB and 5 dB. The length of mixture audios is 2 seconds.

Dataset Download Link: [Google Driver](https://drive.google.com/file/d/1dCWD5OIGcj43qTidmU18unoaqo_6QetW/view?usp=sharing)

## Training and evaluation

- You can refer to this repository [Conv-TasNet](https://github.com/JusperLee/Conv-TasNet)

## Results

Our model achieves the following performance on :

![](./results.png)

## Demo Page

- [Demo](https://cslikai.cn/project/TDANet/)

## Reference

- [A-FRCNN](https://github.com/JusperLee/AFRCNN-For-Speech-Separation)
- [SudoRM-RF](https://github.com/etzinis/sudo_rm_rf)
