import model_function


random.seed(1)
tg_rat_invivo_mean_prop=batchFS("./mean_TG_","./tg_invivo_label.csv",0,108)
tg_invivo_mean_res=cptRes(tg_rat_invivo_mean_prop,label)
tg_invivo_mean_res=pd.DataFrame(tg_invivo_mean_res)
tg_invivo_mean_res.to_csv("./tg_invivo_mean_res.csv")

data=pd.read_csv(open("./tg_invivo_mean/mean_TG_12.csv"))
data=data.values
data=data[:,1:len(data[1,:])]
label=pd.read_csv(open("./tg_invivo_mean/tg_invivo_label.csv"))
label=label.values
label=label[:,1]


p_svc=loocvSVC(data,label)
p_lr=loocvLR(data,label)
p_gbdt=loocvGBDT(data,label)
p_knn=loocvKnn(data,label)
p_RF=loocvRF(data,label)
p_lgb=loocvLGB(data,label)

lr_fpr,lr_tpr,lr_threshold= roc_curve(label,p_lr)
lr_roc_auc = auc(lr_fpr,lr_tpr)
lr_roc_auc

lgb_fpr,lgb_tpr,lgb_threshold = roc_curve(label,p_lgb)
lgb_roc_auc = auc(lgb_fpr,lgb_tpr)
lgb_roc_auc

gbdt_fpr,gbdt_tpr,gbdt_threshold = roc_curve(label,p_gbdt)
gbdt_roc_auc = auc(gbdt_fpr,gbdt_tpr)
gbdt_roc_auc

knn_fpr,knn_tpr,knn_threshold= roc_curve(label,p_knn)
knn_roc_auc = auc(knn_fpr,knn_tpr)
knn_roc_auc

svc_fpr,svc_tpr,svc_threshold= roc_curve(label,p_svc)
svc_roc_auc = auc(svc_fpr,svc_tpr)
svc_roc_auc

RF_fpr,RF_tpr,RF_threshold= roc_curve(label,p_RF)
RF_roc_auc = auc(RF_fpr,RF_tpr)
RF_roc_auc
