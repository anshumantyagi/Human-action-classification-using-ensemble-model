Classifying human actions from still images or video sequences is a demanding
task owing to issues, like lighting, backdrop clutter, variations in scale, partial occlusion,
viewpoint, and appearance. A lot of appliances, together with video systems,
human–computer interfaces, and surveillance necessitate a compound action recognition
system. Here, the proposed system develops a novel scheme for HAR. Initially,
filtering as well as background subtraction is done during preprocessing. Then, the features
including local binary pattern (LBP), bag of the virtual word (BOW), and the proposed
local spatio-temporal features are extracted. Then, in the recognition phase, an
ensemble classification model is introduced that includes Recurrent Neural networks
(RNN 1 and RNN 2) and Multi-Layer Perceptron (MLP 1 and MLP 2). The features are
classified using RNN 1 and RNN 2, and the outputs from RNN 1 and RNN 2 are further
classified using MLP 1 and MLP 2, respectively. Finally, the outputs attained from
MLP 1 and MLP 2 are averaged and the final classified output is obtained. At last, the
superiority of the developed approach is proved on varied measures.

Original Paper : Tyagi A, Singh P, Dev H., "Proposed spatio-temporal features for human activity classification using ensemble
classification model." Concurrency Computat Pract Exper. 2023;e7588. doi: 10.1002/cpe.7588
Dataset: UCF101 Human Action Video.
