# BLEU + COMET summary — v6b conditions on FLORES-200 devtest (N=50)

Streaming inference, check_argmax policy, Gemma-4-E2B-it backbone. cond-A only on 4 directions.

For each (direction, latency): BLEU / COMET (wmt22-comet-da) / AL.  '—' = not run.

## de-en

| latency | merged3 BLEU/COMET/AL | rebucket BLEU/COMET/AL | cond-A BLEU/COMET/AL |
|---|---|---|---|
| low | 30.60 / 0.8543 / 4.48 | 29.78 / 0.8395 / 2.48 | 29.17 / 0.8339 / 2.50 |
| low_medium | 31.88 / 0.8554 / 4.78 | 30.61 / 0.8427 / 2.79 | 30.90 / 0.8475 / 2.88 |
| medium | 34.08 / 0.8584 / 4.64 | 31.96 / 0.8613 / 4.27 | 32.11 / 0.8571 / 3.53 |
| medium_high | 33.49 / 0.8556 / 4.57 | 32.74 / 0.8649 / 5.36 | 33.46 / 0.8612 / 4.53 |
| high | 34.58 / 0.8652 / 9.16 | 34.85 / 0.8751 / 11.59 | 35.54 / 0.8733 / 6.07 |

## en-de

| latency | merged3 BLEU/COMET/AL | rebucket BLEU/COMET/AL | cond-A BLEU/COMET/AL |
|---|---|---|---|
| low | 26.20 / 0.7883 / 5.43 | 23.99 / 0.7993 / 3.07 | 26.69 / 0.7752 / 2.82 |
| low_medium | 27.52 / 0.8100 / 6.22 | 24.03 / 0.7958 / 3.19 | 27.74 / 0.7962 / 3.93 |
| medium | 25.83 / 0.8090 / 5.02 | 25.97 / 0.8323 / 4.75 | 29.71 / 0.8371 / 6.12 |
| medium_high | 26.81 / 0.8075 / 4.80 | 26.80 / 0.8178 / 5.44 | 31.08 / 0.8476 / 9.61 |
| high | 27.88 / 0.8157 / 8.57 | 27.74 / 0.8141 / 13.11 | 30.32 / 0.8412 / 15.90 |

## ar-en

| latency | merged3 BLEU/COMET/AL | rebucket BLEU/COMET/AL | cond-A BLEU/COMET/AL |
|---|---|---|---|
| low | 27.74 / 0.8444 / 2.87 | 28.14 / 0.8391 / 2.68 | — |
| low_medium | 27.67 / 0.8443 / 2.78 | 30.01 / 0.8403 / 2.75 | — |
| medium | 27.97 / 0.8437 / 2.89 | 26.35 / 0.8449 / 3.39 | — |
| medium_high | 27.93 / 0.8465 / 3.06 | 26.70 / 0.8450 / 3.92 | — |
| high | 27.63 / 0.8335 / 4.26 | 27.30 / 0.8424 / 6.71 | — |

## en-ar

| latency | merged3 BLEU/COMET/AL | rebucket BLEU/COMET/AL | cond-A BLEU/COMET/AL |
|---|---|---|---|
| low | 19.38 / 0.8533 / 3.97 | 21.96 / 0.8512 / 3.33 | — |
| low_medium | 19.47 / 0.8449 / 3.75 | 22.24 / 0.8493 / 3.17 | — |
| medium | 18.75 / 0.8503 / 4.33 | 20.68 / 0.8479 / 4.41 | — |
| medium_high | 20.29 / 0.8586 / 4.38 | 18.61 / 0.8463 / 5.50 | — |
| high | 18.09 / 0.8572 / 8.27 | 18.25 / 0.8416 / 13.54 | — |

## ru-en

| latency | merged3 BLEU/COMET/AL | rebucket BLEU/COMET/AL | cond-A BLEU/COMET/AL |
|---|---|---|---|
| low | 27.66 / 0.8424 / 3.14 | 29.62 / 0.8544 / 2.83 | 30.24 / 0.8565 / 2.92 |
| low_medium | 28.47 / 0.8506 / 3.24 | 28.83 / 0.8543 / 2.92 | 30.87 / 0.8634 / 3.38 |
| medium | 29.12 / 0.8520 / 3.03 | 31.02 / 0.8563 / 3.69 | 30.10 / 0.8609 / 3.89 |
| medium_high | 29.29 / 0.8535 / 2.98 | 30.15 / 0.8575 / 4.06 | 30.40 / 0.8618 / 5.03 |
| high | 31.04 / 0.8584 / 6.17 | 31.90 / 0.8495 / 7.49 | 31.75 / 0.8652 / 7.88 |

## en-ru

| latency | merged3 BLEU/COMET/AL | rebucket BLEU/COMET/AL | cond-A BLEU/COMET/AL |
|---|---|---|---|
| low | 27.34 / 0.8463 / 4.57 | 25.93 / 0.8453 / 3.18 | 25.34 / 0.8394 / 3.06 |
| low_medium | 28.42 / 0.8532 / 4.52 | 25.83 / 0.8419 / 3.17 | 29.09 / 0.8615 / 4.09 |
| medium | 28.03 / 0.8443 / 4.14 | 28.29 / 0.8408 / 4.75 | 31.34 / 0.8692 / 5.59 |
| medium_high | 26.88 / 0.8389 / 4.22 | 28.82 / 0.8563 / 4.80 | 32.27 / 0.8843 / 8.68 |
| high | 27.89 / 0.8503 / 9.48 | 29.32 / 0.8540 / 14.07 | 32.08 / 0.8743 / 11.44 |

## vi-en

| latency | merged3 BLEU/COMET/AL | rebucket BLEU/COMET/AL | cond-A BLEU/COMET/AL |
|---|---|---|---|
| low | 27.17 / 0.8386 / 3.94 | 27.27 / 0.8452 / 3.22 | — |
| low_medium | 29.12 / 0.8498 / 3.48 | 28.46 / 0.8494 / 3.38 | — |
| medium | 30.13 / 0.8548 / 4.85 | 28.69 / 0.8460 / 4.27 | — |
| medium_high | 30.07 / 0.8540 / 4.05 | 29.16 / 0.8510 / 4.92 | — |
| high | 31.86 / 0.8591 / 7.62 | 32.58 / 0.8660 / 9.86 | — |

## en-vi

| latency | merged3 BLEU/COMET/AL | rebucket BLEU/COMET/AL | cond-A BLEU/COMET/AL |
|---|---|---|---|
| low | 42.88 / 0.8605 / 3.59 | 42.03 / 0.8753 / 3.01 | — |
| low_medium | 42.28 / 0.8687 / 3.51 | 41.60 / 0.8738 / 3.03 | — |
| medium | 42.44 / 0.8668 / 4.44 | 42.65 / 0.8703 / 3.90 | — |
| medium_high | 42.55 / 0.8668 / 4.16 | 41.77 / 0.8832 / 5.49 | — |
| high | 41.94 / 0.8765 / 7.66 | 40.29 / 0.8703 / 13.62 | — |

## Overall aggregates

| aggregate | merged3 BLEU/COMET | rebucket BLEU/COMET | cond-A BLEU/COMET |
|---|---|---|---|
| All 8 directions (40 cells) | 29.46 / 0.8470 | 29.32 / 0.8483 | 30.51 / 0.8503 |
| Matched-to-condA (20 cells) | 29.15 / 0.8405 | 28.91 / 0.8427 | 30.51 / 0.8503 |