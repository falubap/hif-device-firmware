# Python baseline (SNR 40 dB, fixed soil-grouped split)

Test set: 1645 files from 7 held-out soils. Random rows: mean over draws; f1_p95 is the 95th percentile of the random draws.

| model | feature set | k | acc | macro-F1 | F1 sd | F1 p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| forest | all | 7056 | 0.771 | 0.754 | | |
| forest | thesis_train | 15 | 0.793 | 0.781 | | |
| forest | thesis_published | 15 | 0.766 | 0.755 | | |
| forest | random | 15 | 0.472 | 0.468 | 0.068 | 0.564 |
| logreg | all | 7056 | 0.674 | 0.653 | | |
| logreg | thesis_train | 15 | 0.684 | 0.680 | | |
| logreg | thesis_published | 15 | 0.702 | 0.694 | | |
| logreg | random | 15 | 0.407 | 0.401 | 0.053 | 0.477 |
| mlp | all | 7056 | 0.657 | 0.632 | | |
| mlp | thesis_train | 15 | 0.718 | 0.689 | | |
| mlp | thesis_published | 15 | 0.762 | 0.737 | | |
| mlp | random | 15 | 0.443 | 0.412 | 0.060 | 0.503 |
| tree | all | 7056 | 0.793 | 0.780 | | |
| tree | thesis_train | 15 | 0.753 | 0.741 | | |
| tree | thesis_published | 15 | 0.711 | 0.706 | | |
| tree | random | 15 | 0.438 | 0.432 | 0.070 | 0.538 |
