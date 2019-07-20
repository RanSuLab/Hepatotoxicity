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




data1<-read.csv("./toxygates-2019-4-9-1477.csv",header = TRUE)
dose<-read.csv("./open_tggates_cel_file_attribute.csv",header = TRUE)
rat_invitro_1<-dose_select(data1)
rat_invitro_Low_1<-doseSplit(rat_invitro_1,'L')
rat_invitro_Middle_1<-doseSplit(rat_invitro_1,'M')
rat_invitro_High_1<-doseSplit(rat_invitro_1,'H')
rat_invitro_Low_2<-omitInf(rat_invitro_Low_1)
rat_invitro_Middle_2<-omitInf(rat_invitro_Middle_1)
rat_invitro_High_2<-omitInf(rat_invitro_High_1)
rat_invitro_Low_3<-tData(rat_invitro_Low_2)
rat_invitro_Middle_3<-tData(rat_invitro_Middle_2)
rat_invitro_High_3<-tData(rat_invitro_High_2)
drugs1=colnames(rat_invitro_Low_2)

dose_low=doseMatch(dose,drugs1,'Low','Rat','in vitro','Single','Liver')
dose_Middle=doseMatch(dose,drugs1,'Middle','Rat','in vitro','Single','Liver')
dose_High=doseMatch(dose,drugs1,'High','Rat','in vitro','Single','Liver')


rat_invitro_Low_4<-cbind(rat_invitro_Low_4,dose_low)
rat_invitro_Low_4<-as.data.frame(rat_invitro_Low_4)

rat_invitro_Middle_4<-cbind(rat_invitro_Middle_3,dose_Middle)
rat_invitro_Middle_4<-as.data.frame(rat_invitro_Middle_4)

rat_invitro_High_4<-cbind(rat_invitro_High_3,dose_High)
rat_invitro_High_4<-as.data.frame(rat_invitro_High_4)
totalGene<-geneCo(rat_invitro_Low_4,rat_invitro_Middle_4,rat_invitro_High_4)
totalGene1<-toList(totalGene)
Rmax<-paramExtr(totalGene,'Rinf')


data_rmax_1<-nameOmit(Rmax)
#data_rmax_2<-geneMerge(data_rmax_1)
data_rmax_3<-geneMean2(data_rmax_1)
data_rmax_4=data_rmax_3[,2:length(data_rmax_3[1,])]

difGene_TG=findDiffGene(data_rmax_4)
tgCount=egoSel(data_rmax_4,difGene_TG,colnames(data_rmax_4))
batchFS(data_rmax_4,tgCount,0,20,"./tg_invitro/mean/",'mean')
batchFS(data_rmax_4,tg_count,0,20,"/tg_invitro/sum/",'sum')
batchFS(data_rmax_4,tg_count,0,20,"./tg_invitro/sd/",'sd')
batchFS(data_rmax_4,tg_count,0,20,"./tg_invitro/max/",'max')

selData=countSel(data_rmax_4,tg_count,8)
#selData_TG=countSel(data_rmax_4,tgCount,5)
#selData_DM=mergeDM[,colnames(selData_TG)]
selData=cbind(selData[,1],selData)
pic_score<-picGener_score(selData,'ego')
pic<-picGener(selData,pic_score)
min_matrix<-getMost(pic,'min')
max_matrix<-getMost(pic,'max')
max_matrix=max_matrix+0.00001
pic_normal<-lapply(pic,MatrixNormal,max_matrix,min_matrix)
listWrite(pic_normal,'./tg_rat_inviro/pic/')
