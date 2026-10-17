| comparison                            | metric            | level   |   n_pairs |   mean_a |   mean_b |   mean_diff |   wins_a |      p |   p_holm |
|:--------------------------------------|:------------------|:--------|----------:|---------:|---------:|------------:|---------:|-------:|---------:|
| AttentionNet vs 2D-CNN (aug)          | f1_weighted       | fold    |        25 |   0.8818 |   0.8745 |      0.0073 |       11 | 0.8532 |   1      |
| AttentionNet vs ResNet1D (aug)        | f1_weighted       | fold    |        25 |   0.8818 |   0.8779 |      0.004  |       13 | 0.6071 |   1      |
| AttentionNet vs Bi-LSTM (aug)         | f1_weighted       | fold    |        25 |   0.8818 |   0.8921 |     -0.0103 |       13 | 1      |   1      |
| AttentionNet: aug vs no-aug           | f1_weighted       | fold    |        25 |   0.8818 |   0.8908 |     -0.009  |        8 | 0.1209 |   1      |
| 2D-CNN: aug vs no-aug                 | f1_weighted       | fold    |        25 |   0.8745 |   0.8704 |      0.0041 |       12 | 0.7843 |   1      |
| ResNet1D: aug vs no-aug               | f1_weighted       | fold    |        25 |   0.8779 |   0.8752 |      0.0027 |       12 | 0.7578 |   1      |
| Bi-LSTM: aug vs no-aug                | f1_weighted       | fold    |        25 |   0.8921 |   0.9311 |     -0.039  |        5 | 0.013  |   0.1431 |
| AttentionNet (aug) vs Engineered + RF | f1_weighted       | fold    |        25 |   0.8818 |   0.872  |      0.0098 |       14 | 0.5249 |   1      |
| 2D-CNN (aug) vs Engineered + RF       | f1_weighted       | fold    |        25 |   0.8745 |   0.872  |      0.0025 |        9 | 0.8757 |   1      |
| ResNet1D (aug) vs Engineered + RF     | f1_weighted       | fold    |        25 |   0.8779 |   0.872  |      0.0058 |       12 | 0.6143 |   1      |
| Bi-LSTM (aug) vs Engineered + RF      | f1_weighted       | fold    |        25 |   0.8921 |   0.872  |      0.0201 |       12 | 0.4077 |   1      |
| AttentionNet vs 2D-CNN (aug)          | balanced_accuracy | fold    |        25 |   0.6678 |   0.6552 |      0.0126 |       11 | 0.9368 |   1      |
| AttentionNet vs ResNet1D (aug)        | balanced_accuracy | fold    |        25 |   0.6678 |   0.6869 |     -0.019  |       12 | 0.7971 |   1      |
| AttentionNet vs Bi-LSTM (aug)         | balanced_accuracy | fold    |        25 |   0.6678 |   0.7057 |     -0.0379 |        9 | 0.0588 |   0.5372 |
| AttentionNet: aug vs no-aug           | balanced_accuracy | fold    |        25 |   0.6678 |   0.6596 |      0.0082 |       10 | 0.4291 |   1      |
| 2D-CNN: aug vs no-aug                 | balanced_accuracy | fold    |        25 |   0.6552 |   0.6493 |      0.0059 |       11 | 0.9757 |   1      |
| ResNet1D: aug vs no-aug               | balanced_accuracy | fold    |        25 |   0.6869 |   0.6852 |      0.0017 |       12 | 0.7332 |   1      |
| Bi-LSTM: aug vs no-aug                | balanced_accuracy | fold    |        25 |   0.7057 |   0.6983 |      0.0074 |        8 | 0.6849 |   1      |
| AttentionNet (aug) vs Engineered + RF | balanced_accuracy | fold    |        25 |   0.6678 |   0.6365 |      0.0313 |       13 | 0.2872 |   1      |
| 2D-CNN (aug) vs Engineered + RF       | balanced_accuracy | fold    |        25 |   0.6552 |   0.6365 |      0.0187 |       10 | 0.5202 |   1      |
| ResNet1D (aug) vs Engineered + RF     | balanced_accuracy | fold    |        25 |   0.6869 |   0.6365 |      0.0504 |       15 | 0.0537 |   0.5372 |
| Bi-LSTM (aug) vs Engineered + RF      | balanced_accuracy | fold    |        25 |   0.7057 |   0.6365 |      0.0692 |       14 | 0.0041 |   0.0447 |
| AttentionNet vs 2D-CNN (aug)          | f1_macro          | seed    |         5 |   0.6396 |   0.61   |      0.0297 |        4 | 0.3125 |   1      |
| AttentionNet vs ResNet1D (aug)        | f1_macro          | seed    |         5 |   0.6396 |   0.6321 |      0.0075 |        3 | 1      |   1      |
| AttentionNet vs Bi-LSTM (aug)         | f1_macro          | seed    |         5 |   0.6396 |   0.6331 |      0.0065 |        2 | 0.8125 |   1      |
| AttentionNet: aug vs no-aug           | f1_macro          | seed    |         5 |   0.6396 |   0.605  |      0.0347 |        4 | 0.1875 |   1      |
| 2D-CNN: aug vs no-aug                 | f1_macro          | seed    |         5 |   0.61   |   0.6113 |     -0.0013 |        3 | 1      |   1      |
| ResNet1D: aug vs no-aug               | f1_macro          | seed    |         5 |   0.6321 |   0.6172 |      0.0149 |        3 | 0.625  |   1      |
| Bi-LSTM: aug vs no-aug                | f1_macro          | seed    |         5 |   0.6331 |   0.6318 |      0.0013 |        2 | 0.8125 |   1      |
| AttentionNet (aug) vs Engineered + RF | f1_macro          | seed    |         5 |   0.6396 |   0.5822 |      0.0575 |        5 | 0.0625 |   0.6875 |
| 2D-CNN (aug) vs Engineered + RF       | f1_macro          | seed    |         5 |   0.61   |   0.5822 |      0.0278 |        3 | 0.3125 |   1      |
| ResNet1D (aug) vs Engineered + RF     | f1_macro          | seed    |         5 |   0.6321 |   0.5822 |      0.05   |        5 | 0.0625 |   0.6875 |
| Bi-LSTM (aug) vs Engineered + RF      | f1_macro          | seed    |         5 |   0.6331 |   0.5822 |      0.051  |        5 | 0.0625 |   0.6875 |