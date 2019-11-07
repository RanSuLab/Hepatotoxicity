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
First, you can get the genetic data after feature selection by running the R file code in the file. 
Once you have the data you need to experiment, you need to process the data using deal_with_data_tg_rat_invivo.R. 
1.Use the dose_split function to divide the drug data into three batches of high, medium and low concentrations. 
2.The omitInf function is used to remove missing data items.
3.The doseMatch function is used to obtain the test concentration of each drug. 
4.The gene and paramExtr functions are responsible for combining the three concentrations of data, and performing curve fitting to extract the Rmax parameter. 
5.The geneMean2 function is used to average the probes representing the same gene. 
6.The findDiffGene function is used to find differentially expressed genes. 
7.The egoSel function is used to count the number of biological processes in which each gene appears. 
8.The batchFS function is used to obtain genetically selected gene data under different thresholds and parameters.

After that, you need to run the python code model_tg_invivo.py to get the results of the gene data under different parameters and classifiers.
1.batchFS function is used to calculate the predicted probability of genes under various thresholds after loocv verification.
2.The cptRes function is used to calculate the predictions spe, sen, auc, acc corresponding to each probability. 
3. loocvSVC, loocvLR, loocvGBDT, loocvKnn, loocvRF, loocvRF, loocvLGB are used to calculate the probability that each drug is predicted to be toxic by the corresponding algorithm.
You need to use the python file in the directory(cnn_deal_with_data_tg_invivo.py,cnn_model_tg_invivo.py) to process the generated image and train it using different convolutional neural networks.
1.The trainTestVal_split function is used to divide the input data into training sets and test sets. 
2.The batchCsvToPicture function is used to batch convert the input csv file into a picture. 
3.Use the batchCrop function to batch cut the image.
4.Use the batchResize function to modify the image to the desired size. 
5. Use the batchTrain function to train the image with different network structures.

