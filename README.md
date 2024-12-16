# CS528
Group project for CS528

Collaborators:

Ahmad Shahrour
Alex Goldstone
Sreekanth Cherukuri


All files are in the IMU-main folder. Files and folders are organized as such below:

~~ csv_files/ ~~
Contains folders with csv files for each different motion.

~~ main/ ~~
Contains i2c_simple_main.c, which has the code that organizes data from the IMU sensor and is uploaded into the microcontroller. It contains sleep(0.008) between fetches to set a sampling rate of 125 Hz.

~~ SVM_eval.ipynb ~~
Contains code for creation of SVM model, mainly the evaluation. Uses scikit-learn's test_train_split to create an 80/20 split and evaluate the model.

~~ collect_data.py ~~
Code that, when run, will collect data for each of the motions in an automated fashion. csv files created used for training of SVM model.

~~ streaming_data.py ~~
Code that, when run, will make live predictions using the SVM generated in svm_predict. 

~~ svm_predict.py ~~
Code that uses files in csv_files to train an SVM model, which is stored in 
