This contains all the files used to define and train the models
=================================================================

>  This Folder is arranged as follows:

    .
    ├── models                         # Contains all the models
    ├── utils                          # Contains the dataloader, metrics, loss functions, earlystopping procedure.
    └── main_conv3D.py                 # Main script for 3DCNN pipeline
    └── main_convESN.py                # Main script for the convESN pipeline
    └── main_convRNNs.py               # Main script for the convLSTM/GRU/RNN pipeline
    └── get_prediction_images.py       # Script to generated test sample images
    └── main_convESN_hparam.py         # Script for hyperoptization of the convESN
    └── run_convESN.sh		       # Script to run convESN experiments on the cluster
    └── run_convRNNs.sh                # Script to run convLSTM/GRU/RNN experiemnts on the cluster
    └── run_conv3D.sh                  # Script to run 3DCNN experiments on the cluster
    └── run_getpredictions.sh          # 
    └── run_hparam_esn.sh              # Script to run convESN different parameters
    └── run_matlab.sh                  # Script to run matlab to generated the data
    └── run_tensorboard_over_cluster.txt # File with info for running tensorboard over a cluster
    └── multi_tb.sh                      # Script to run multiple tensorboard over a cluster 
    └── README.md



run_conv3D.sh            - runs the 3dCNN  model 
run_convESN.sh         - runs the convolutional ESN model
run_convRNNs.sh       - runs the convolutional RNN, LSTM and GRU
run_getpredictions.sh  - runs the saved models to makes the predictions on the testing dataset
run_hparam_esn.sh    - runs the hyperparameter search of the convolutional ESN
