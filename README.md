# GrabChallenge-Safety

Instructions for inferring test set:
1) Clone this repo
2) copy test_set file (.csv format) into submission_data folder
3) open command prompt at src folder
4) run command: python model_infer2.py test_file.csv

Requirements:
torch==1.0.0
sklearn==0.0
pandas==0.23.4
numpy==1.15.4
matplotlib==2.2.3 

# Training
Perform 5-fold split for validation testing
![alt text](https://github.com/plkmo/GrabChallenge-Safety/blob/master/src/submission_data/test_loss_vs_epoch_1.png) Loss vs Epoch
![alt text](https://github.com/plkmo/GrabChallenge-Safety/blob/master/src/submission_data/train_auc_vs_epoch_1.png) Training AUC vs Epoch
![alt text](https://github.com/plkmo/GrabChallenge-Safety/blob/master/src/submission_data/test_auc_vs_epoch_1.png) Test AUC vs Epoch
![alt text](https://github.com/plkmo/GrabChallenge-Safety/blob/master/src/submission_data/test_Accuracy_vs_epoch_1.png) Test Accuracy vs Epoch

Model starts to overfit after 40 epochs. For final model, all datasets are included for training and early stopping is implemented after 30 epochs.




