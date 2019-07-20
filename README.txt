INTRODUCTION
deal_with_data_tg_invitro.R plays the code for feature selection of open_tggate_liver_single_rat_invitro
deal_with_data_tg_invivo.R plays the code for feature selection of open_tggate_liver_single_rat_invivo
deal_with_data_dm_invivo.R plays the code for feature selection of drug_matrix
model_tg_invitro.py is responsible for running the model on gene selection data of open_tggate_liver_single_rat_invitro
cnn_deal_with_data_tg_invitro .py is used to process images for training in convolutional neural networks
cnn_model_tg_invitro.py is used to train pictures with different kinds of convolutional neural networks

R version 3.4.0
Python version 3.6

USE:
You need to download the gene exprssion files of open_tggate_liver_single_rat_invitro and open_tggate_liver_single_rat_invivo, from the website 'https://toxygates.nibiohn.go.jp/toxygates/'
download the drug_matrix data from 'ftp://anonftp.niehs.nih.gov/drugmatrix/Affymetrix_data/'
You need to download the corresponding drug data according to our drug toxicity list.
Then you need to put them into the corresponding dirs.
First, you can get the genetic data after feature selection by running the R file code in the file. After that, you need to run the python code to get the results of the gene data under different parameters and classifiers.
You need to use the python file in the directory to process the generated image and train it using different convolutional neural networks.
