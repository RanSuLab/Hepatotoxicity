library(org.Rn.eg.db)
library(org.Hs.eg.db)
library(xlsxjars)
library(rJava)
library(xlsx)
library(DOSE)
library(clusterProfiler)
library(psych)
library("readr")  
library(caret)
library(randomForest)
library(e1071)
library(gbm)
library(mlbench)
library(caretEnsemble)
library(topGO)
library(drc)
library(parallel)
source('./function.R')



data1=read.csv("./toxygates-2019-4-21-4285.csv",header = TRUE)
dose<-read.csv("./open_tggates_cel_file_attribute.csv",header = TRUE)


rat_invivo_1<-dose_select(data1)
rat_invivo_Low_1<-doseSplit(rat_invivo_1,'L')

rat_invivo_Middle_1<-doseSplit(rat_invivo_1,'M')
rat_invivo_High_1<-doseSplit(rat_invivo_1,'H')
rat_invivo_Low_2<-omitInf(rat_invivo_Low_1)
rat_invivo_Middle_2<-omitInf(rat_invivo_Middle_1)
rat_invivo_High_2<-omitInf(rat_invivo_High_1)
rat_invivo_Low_3<-tData(rat_invivo_Low_2)
rat_invivo_Middle_3<-tData(rat_invivo_Middle_2)
rat_invivo_High_3<-tData(rat_invivo_High_2)
#rat_invitro_Low_1_3<-doseMatch2(dose,rat_invitro_Low_1_2,'Low')
drugs1=colnames(rat_invivo_Low_2)

dose_low=doseMatch(dose,drugs1,'Low','Rat','in vivo','Single','Liver')
dose_Middle=doseMatch(dose,drugs1,'Middle','Rat','in vivo','Single','Liver')
dose_High=doseMatch(dose,drugs1,'High','Rat','in vivo','Single','Liver')


rat_invivo_Low_4<-cbind(rat_invivo_Low_3,dose_low)
rat_invivo_Low_4<-as.data.frame(rat_invivo_Low_4)
#rat_invitro_Middle_1_3<-doseMatch(dose,rat_invitro_Middle_1_2,'Middle','in vitro','Rat')
rat_invivo_Middle_4<-cbind(rat_invivo_Middle_3,dose_Middle)
rat_invivo_Middle_4<-as.data.frame(rat_invivo_Middle_4)
#rat_invitro_High_1_3<-doseMatch(dose,rat_invitro_High_1_2,'High','in vitro','Rat')
rat_invivo_High_4<-cbind(rat_invivo_High_3,dose_High)
rat_invivo_High_4<-as.data.frame(rat_invivo_High_4)
totalGene<-geneCo(rat_invivo_Low_4,rat_invivo_Middle_4,rat_invivo_High_4)
#totalGene1<-toList(totalGene)
#parComp(3,totalGene1,3)
rat_invivo_rmax<-paramExtr(totalGene,'Rinf')



rat_invivo_rmax_1<-nameOmit(rat_invivo_rmax)
rat_invivo_rmax_2=geneMean2(rat_invivo_rmax_1)
#rat_invivo_rmax_3<-geneMerge(rat_invivo_rmax_2)

rat_invivo_rmax_3=naColDel(rat_invivo_rmax_2)
rat_invivo_rmax_3=rat_invivo_rmax_3[,2:length(rat_invivo_rmax_3[1,])]


difGene_TG_rat_invivo=findDiffGene(rat_invivo_rmax_3)
tgCount_rat_invivo=egoSel(rat_invivo_rmax_3,difGene_TG_rat_invivo,colnames(rat_invivo_rmax_3))
batchFS(rat_invivo_rmax_3,tgCount_rat_invivo,0,20,'./mean/','mean')
batchFS(rat_invivo_rmax_3,tgCount_rat_invivo,0,20,"./sum/",'sum')
batchFS(rat_invivo_rmax_3,tgCount_rat_invivo,0,20,"./sd/",'sd')
batchFS(rat_invivo_rmax_3,tgCount_rat_invivo,0,20,"./max/",'max')


selData=countSel(rat_invivo_rmax_3,tg_count,12)

selData=cbind(selData[,1],selData)
pic_score<-picGener_score(selData,'ego')
pic<-picGener(selData,pic_score)
min_matrix<-getMost(pic,'min')
max_matrix<-getMost(pic,'max')
max_matrix=max_matrix+0.00001
pic_normal<-lapply(pic,MatrixNormal,max_matrix,min_matrix)
listWrite(pic_normal,'./tg_rat_invivo/pic/')
