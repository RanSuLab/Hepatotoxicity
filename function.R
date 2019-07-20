#remove drugs whose dose is not complete
dose_select<-function(data){
  name<-colnames(data)
  drug<-name[3:length(name)]
  drug<-strsplit(drug,'\\.')
  mark<-c()
  cat(drug[[1]])
  for(i in 1:(length(drug)-5)){
    if(drug[[i]][1]==drug[[i+5]][1]){
      mark<-c(mark,i+2,i+3,i+4,i+5,i+6,i+7)
    }
  }
  mark<-c(1,2,mark)
  result<-data[,c(mark)]
  return(result)
}
#####split data by dose#######
doseSplit<-function(data,dose){
  name<-colnames(data)
  drug<-name[3:length(name)]
  drug<-strsplit(drug,'\\.')
  mark<-c()
  for(i in 1:length(drug)){
    if(drug[[i]][2]==dose&is.na(drug[[i]][5])){
      mark<-c(mark,i+2)
    }
  }
  mark<-c(1,2,mark)
  result<-data[,c(mark)]
  return(result)
}
#####del the inf data
omitInf<-function(data){
  
  mark1<-c()
  for(i in 1:length(data[,1])){
    if(!data[i,2]==''){
      mark1<-c(mark1,i)
    }
    cat(i,'genes\n')
  }
  result1<-data[c(mark1),]
  mark2<-c()
  for(i in 1:length(result1[,1])){
    mark<-0
    for(j in 3:length(result1[1,])){
      if(is.infinite(result1[i,j])){
        mark<-1
        break()
      }
    }
    if(mark==0){
      mark2<-c(mark2,i)
    }
    cat(i,'genes finished\n')
  }
  result2<-result1[c(mark2),]
  return(result2)
}

tData<-function(data){
  colname<-colnames(data)
  colname<-colname[3:length(colname)]
  data[,2]<-as.character(data[,2])
  rowname<-data[,2]
  rowname=as.character(rowname)
  rowname=strsplit(rowname,split = ' ')
  gene=c()
  for(i in 1:length(rowname)){
    gene=c(gene,rowname[[i]][1])
  }
  result<-t(data)
  rownames(result)<-NULL
  result<-result[3:length(result[,1]),]
  result<-cbind(colname,result)
  colnames(result)<-c('drug',gene)
  return(result)
}

######get the dose of drug
doseMatch<-function(data,drugs,type,spe,invi,re,organ){
  drugs=strsplit(drugs,split = '\\.')
  a=c()
  name=as.character(data[,8])
  for(i in 1:length(drugs)){
    a=c(a,drugs[[i]][1])
  }
  dose=c()
  for(i in 1:length(drugs)){
    for(j in 1:length(data[,8])){
      if(length(str_subset(name[j],a[i]))>0&data[j,21]==type&data[j,11]==spe&data[j,12]==invi&data[j,13]==re&data[j,6]==organ){
        dose=c(dose,data[j,19])
        cat(i,'drug find\n')
        break()
        
      }
    }
  }
  return(dose)
}
##########Get genes that appear in three data simultaneously
geneCo<-function(data1,data2,data3){
  result<-list()  
  name<-colnames(data1)
  name<-name[2:length(name)]
  mark<-c()
  for(i in 1:length(name)){
    if(((!is.null(data1[1,c(name[i])]))&(!is.null(data2[1,c(name[i])])))){
      if(!is.null(data3[1,name[i]])){
        
        
        mark<-c(mark,name[i])
      }
    }
  }
  result[[1]]<-data1[,c(mark)]
  result[[1]]<-cbind(data1[,1],result[[1]],data1[,length(data1[1,])])
  result[[2]]<-data2[,c(mark)]
  result[[2]]<-cbind(data2[,1],result[[2]],data2[,length(data2[1,])])
  result[[3]]<-data3[,c(mark)]
  result[[3]]<-cbind(data3[,1],result[[3]],data3[,length(data3[1,])])
  return(result)
  
}

##########select the liver data
liverData<-function(data){
  mark<-c()
  for(i in 1:length(data[,1])){
    if(data[i,7]=='LIVER'){
      mark=c(mark,i)
    }
  }
  return(data[mark,])
}

##########Divide each drug into a list
drug_spl<-function(data){
  total_drug=as.character(data[,2])
  drugs<-c('a')
  col=length(data[1,])
  rows=length(data[,1])
  for(i in 1:rows){
    if(total_drug[i]==''){
      next()
    }
    mark=1
    for(j in 1:length(drugs)){
      if(total_drug[i]==drugs[j]){
        mark=0
      }
    }
    if(mark==1){
      drugs=c(drugs,total_drug[i])
      cat('drug:',drugs[length(drugs)],'\n')
    }
  }
  res=list()
  for(i in 1:length(drugs)){
    mark=c()
    for(j in 1:rows){
      if(drugs[i]==total_drug[j]){
        mark=c(mark,j)
      }
      cat(i,'drugs',j,'rows finished\n')
    }
    res[[drugs[i]]]=data[mark,]
  }
  return(res)
}


##########Choose a drug that has been applied for a specific length of time
timeFilt<-function(data){
  mark<-c()
  name=names(data)
  res=list()
  for(i in 1:length(data)){
    mark1=c()
    if(is.na(data[[i]][1,1])){
      next()
    }
    for(j in 1:length(data[[i]][,1])){
      if(data[[i]][j,6]=='1 d'){
        mark1=c(mark1,j)
      }
    }
    if(length(unique(data[[i]][mark1,3]))>0){
      res[[name[i]]]=data[[i]][mark1,]
    }
    #if(length(mark1)>1){
    #res[[name[i]]]=data[[i]][mark1,]
    #}
  }
  return(res)
}



##########Obtain data for each drug at a higher concentration
getData<-function(data){
  rows=length(data)
  cols=length(data[[1]][1,])
  res=data.frame(matrix(0,rows,cols))
  for(i in 1:rows){
    dose=as.character(data[[i]][,3])
    b=gsub('\\ .*','',dose)
    b=as.numeric(b)
    max=0
    mark=0
    for(j in 1:cols){
      data[[i]][,j]=as.character(data[[i]][,j])
    }
    for(j in 1:length(b)){
      if(b[j]>max){
        max=b[j]
        mark=j
      }
    }
    #data[[i]][mark,]=as.character(data[[i]][mark,])
    res[i,]=data[[i]][mark,]
  }
  return(res)
}

##########Get genetic data for each drug
drug_data<-function(data,path){
  drug=as.character(data[,3])
  cel=as.character(data[,2])
  res=c()
  name=c()
  dirList=c()
  for(i in 1:length(data[,1])){
    
    drug[i]=gsub(' ','_',drug[i])
    #path1=paste(path,'/',drug[i],'/',cel[i],sep = '')
    path1=str_c(path,'/',drug[i],'/',cel[i],sep = '')
    path1=as.character(path1)
    t=c(path1)
    dirList=c(dirList,path1)
    cat(t)
    #a=read.csv(dirList[i])
    a=read.affybatch(t)
    a=mas5(a)
    b=data.frame(a)
    name=colnames(b)
    res=rbind(res,b)
    cat(i,'drugs finished\n')
  }
  names(res)=name
  return(res)
}


##########Obtain genetic data under normal conditions
control_sel<-function(data,match,control,gene){
  treat=as.character(data[,2])
  name=colnames(gene)
  name[1:31042]=gsub('\\X','',name[1:31042])
  for(i in 1:length(control[1,])){
    control[,i]=as.character(control[,i])
  }
  data[,1]=as.character(data[,1])
  c_col=as.character(control[1,])
  c_col=c_col[2:length(c_col)]
  c_row=as.character(control[,1])
  c_row=c_row[2:length(c_row)]
  c_row=gsub('\\/','.',c_row)
  c_row=gsub('\\-','.',c_row)
  n_c_col=length(control[1,])
  n_c_row=length(control[,1])
  control=control[2:n_c_row,2:n_c_col]
  colnames(control)=c_col
  #rownames(control)=c_row
  cat(c_col)
  for(i in 1:(n_c_col-1)){
    control[,i]=as.character(control[,i])
    control[,i]=as.numeric(control[,i])
  }
  res=c()
  match[,1]=as.character(match[,1])
  match[,2]=as.character(match[,2])
  cat('1\n')
  for(i in 1:length(treat)){
    n_match=c()
    a=control[,1]
    for(j in 1:length(match[,1])){
      if(match[j,1]==treat[i]){
        
        n_match=c(n_match,match[j,2])
        cat(i,'drug',match[j,2],'cel matched\n')
      }
    }
    cat('2\n')
    a=control[,1]*0
    
    cat(length(control[,1]),length(a))
    a=as.matrix(a)
    for(j in 1:length(n_match)){
      a=a+control[,n_match[j]]
    }
    
    cat('2.5\n')
    a=a/length(n_match)
    a=t(a)
    cat(a,'\n')
    res=rbind(res,a)
  }
  cat(length(c_row),length(res[1,]))
  colnames(res)=c_row
  cat('3.1\n')
  #rownames(res)=c_col
  final=data.frame(matrix(0,length(treat),length(name)-1))
  for(i in 1:(length(name)-1)){
    final[,i]=res[,name[i]]
    cat(i,'combed\n')
  }
  cat('2\n')
  #final=final[,2:length(final[1,])]
  colnames(final)=name[1:length(name)-1]
  #rownames(final)=c_col
  return(final)
  
}



##########Convert the gene probe id to entr id
convertID<-function(a,b){
  a<-IDprocess(a)
  cat(a,'\n')
  
  b$Gene<-as.character(b$Gene)
  
  c<-c()
  b$X<-as.character(b$X)
  b$X<-IDprocess(b$X)
  cat('1')
  for(i in 1:length(a)){
    flag=0
    for(j in 1:length(b$X)){
      if(a[i]==b$X[j]){
        flag=1
        c<-rbind(c,b$Gene[j])
        break()
      }
    }
    if(flag==0){
      c<-rbind(c,'a b')
    }
    
    cat(i,"IDs has been converted",c[i],'\n')
    
  }
  
  c<-as.character(c)
  return(c)
}


IDprocess<-function(a){
  c<-c()
  for(i in 1:length(a)){
    c<-rbind(c,gsub('(\\X)','',a[i]))
    c[i]<-chartr("A","a",c[i])
    c[i]<-chartr("T","t",c[i])
  }
  c<-as.character(c)
  return(c)
}

##########gene select
getCol2<-function(data,universe){
  result<-c()
  name=c()
  tmp=strsplit(universe,split = ' ')
  for(i in 1:length(tmp)){
    if(length(tmp[[i]])==1){
      name=c(name,universe[i])
      result<-c(result,i)
    }
  }
  data=data[,result]
  colnames(data)=name
  return(data)
}

##########Average and combine genes of the same id
geneMean2<-function(data){
  name<-colnames(data)
  name2<-gsub("\\..*","",name)
  name3<-c()
  for(j in 1:length(name2)){
    name3<-c(name3,name2[[j]][1])
  }
  name4<-name3
  name4<-as.factor(name4)
  #col=levels(name4)
  #row<-length(data[,1])
  #result<-data.frame(matrix(0,row,col))
  #result[,1]<-data[,1]
  mark1<-c()
  for(i in 1:length(data[1,])){
    mark1<-c(mark1,0)
  }
  mark2<-c()
  for(i in 1:(length(name2))){
    mark<-0
    if(i==length(name2)){
      if(mark1[i]==1){
        break()
      }
      if(mark1[i]==0){
        mark2=c(mark2,i)
        break()
      }
    }
    if(mark1[i]==0){
      mark2<-c(mark2,i)
    }
    cnt=1
    for(j in (i+1):length(name2)){
      if((name2[i]==name2[j])&(mark1[j]==0)){
        data[,i]<-(data[,i]+data[,j])
        mark1[j]<-1
        cnt=cnt+1
      }
    }
    if(cnt>1){
      data[,i]=data[,i]/cnt
    }
    cat(i,'genes finished\n')
  }
  cat(length(mark2),'genes\n')
  result<-data[,c(mark2)]
  
  return(result)
}

##########Find differentially expressed genes
findDiffGene<-function(data){
  result<-list()
  name<-colnames(data)
  #name[1:length(name)]
  for(i in 1:length(data[,1])){
    temp<-c()
    for(j in 1:length(data[1,])){
      if((abs(data[i,j])>2)){
        temp<-c(temp,name[j])
        
      }
      cat(i,'drugs',j,'genes finished\n')
    }
    if(length(temp)>0){
      result[[i]]<-temp
    }
    else{
      result[[i]]<-'000'
    }
    
  }
  return(result)
}

##########Obtain statistics on gene enrichment matrix
egoSel<-function(data,difGene,universe){
  
  rows<-length(data[,1])
  cols<-length(data[1,])
  name<-colnames(data)
  countMatrix<-data.frame(matrix(0,rows,cols))
  names(countMatrix)<-c(name)
  for(i in 1:rows){
    geneCount<-list()
    difGene[[i]]<-as.character(difGene[[i]])
    if(difGene[[i]]=='000'){
      
      next()
    }
    ego2 <- enrichGO(gene          = difGene[[i]],
                     universe      = universe,
                     OrgDb         = org.Rn.eg.db,
                     ont           = "BP",
                     pAdjustMethod = "BH",
                     pvalueCutoff  = 1,
                     qvalueCutoff  = 1,
                     minGSSize=0,
                     readable      = FALSE,
                     pool='ALL'
    )
    j=1
    if(is.null(ego2[1])){
      next()
    }
    while(!is.na(ego2[j]$ID)){
      
      a<-strsplit(ego2[j]$geneID,'/')
      countMatrix[i,a[[1]]]=countMatrix[i,a[[1]]]+1
      j=j+1
      cat(i,'drug',j,'terms finished\n')
    }
    
    
  }
  return(countMatrix)
}

##########Batch feature selection for different thresholds
batchFS<-function(data,count,start,end,dir,param){
  id=c()
  for(i in start:end){
    id<-c(id,i)
  }
  id=as.character(id)
  fileDir=c()
  for(i in 1:length(id)){
    tmp=paste(dir,'mean_TG_',id[i],'.csv',sep = '')
    cat(tmp,'\n')
    fileDir=c(fileDir,tmp)
  }
  cnt=1
  for(i in start:end){
    selData=countSel(data,count,i)
    selData=cbind(selData[,1],selData)
    pic_score<-picGener_score(selData,'ego')
    pic<-picGener(selData,pic_score)
    if(param=='mean'){
      meanData=meanMatrix(pic)
      write.csv(meanData,fileDir[cnt])
    }
    if(param=='max'){
      maxData=maxMatrix(pic)
      write.csv(maxData,fileDir[cnt])
    }
    if(param=='sd'){
      sdData=sdMatrix(pic)
      write.csv(sdData,fileDir[cnt])
    }
    if(param==sum){
      sumData=sumMatrix(pic)
      write.csv(sumData,fileDir[cnt])
    }
    cnt=cnt+1
  }
}


picGener_score<-function(data,type){
  difGene<-findDiffGene(data[,2:length(data[1,])])
  universe<-colnames(data)
  universe<-universe[2:length(universe)]
  score<-list()
  a<-c()
  if(type=='ego'){
    ego1<-enrichGO(gene          = universe,
                   universe      = universe,
                   OrgDb         = org.Rn.eg.db,
                   ont           = "BP",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = 1,
                   qvalueCutoff  = 1,
                   minGSSize=0,
                   readable      = FALSE,
                   pool='ALL'
    )
    a<-ego1
  }
  if(type=='pathway'){
    ego1 <- enrichKEGG(gene = universe,
                       organism = 'rno',pvalueCutoff = 1)
    a<-ego1
  }
  for(i in 1:length(data[,1])){
    b<-c()
    if(type=='ego'){
      ego2<-enrichGO(gene          = difGene[[i]],
                     universe      = universe,
                     OrgDb         = org.Rn.eg.db,
                     ont           = "BP",
                     pAdjustMethod = "BH",
                     pvalueCutoff  = 1,
                     qvalueCutoff  = 1,
                     minGSSize=0,
                     readable      = FALSE,
                     pool='ALL'
      )
      score[[i]]<-egoScore(ego2,ego1,universe,difGene[[i]])
      
    }
    if(type=='pathway'){
      
      ego2=enrichKEGG(gene         = difGene[[i]],organism     = 'rno',pvalueCutoff = 1)
      score[[i]]=kegg_score(ego2,ego1)
      #cat(score[[i]][,1])
    }
    #score[[i]]<-egoScore(b,a,universe,difGene[[i]])
    cat(i,'scores finished\n')
  }
  return (score)
  row<-length(score[[1]][,1])
  col<-length(data[1,])
  mark<-data.frame(matrix(0,row,col))
  colnames(mark)<-c('go',universe)
  mark[,1]<-score[[1]][,1]
  TermGene<-list()
  count<-1
  while(!is.na(ego1[count]$geneID)){
    tmp<-ego1[count]$geneID
    tmp<-strsplit(tmp,split = '/')
    TermGene[[count]]<-tmp
    count<-count+1
  }
  for(i in 1:length(mark[,1])){
    for(j in 1:length(TermGene[[i]])){
      mark[i,c(TermGene[[i]][j])]<-1
    }
  }
  result<-list()
  for(i in 1:length(data[,1])){
    result[[i]]<-data[1,]
    for(j in 1:(length(score[[1]][,1])-1)){
      result[[i]]<-rbind(result[[i]],data[i,])
    }
    a<-as.matrix(mark[,2:length(mark[1,])])
    b<-as.matrix(result[[i]][,2:length(data[1,])])
    result[[i]]<-a*b
    for(k in 1:length(result[[i]][,1])){
      result[[i]][k,]<-score[[i]][k,2]*result[[i]][k,]
    }
    cat(i,'pic finished\n')
  }
  for(i in 1:length(result)){
    result[[i]]<-cbind(score[[1]][,1])
    result[[i]]<-as.data.frame(result[[i]])
    colnames(result[[i]])<-c('go',universe)
  }
  return(result)
}


egoScore<-function(ego1,egoUni,total,dif){
  TotalgeneNum<-length(total)
  difNum<-length(dif)
  termNum<-c()
  selTermNum<-c()
  countUni<-1
  UniId<-c()
  while(!is.na(egoUni[countUni]$geneID)){
    temp<-egoUni[countUni]$geneID
    temp<-strsplit(egoUni[countUni]$geneID,split='/')
    n<-length(temp[[1]])
    termNum<-c(termNum,n)
    UniId<-c(UniId,egoUni[countUni]$ID)
    countUni<-countUni+1
  }
  count1<-1
  val=rep(1,length(UniId))
  res=data.frame(UniId,val)
  rownames(res)=UniId
  selId<-c()
  if(!is.null(ego1[1]$ID)){
    while((!is.na(ego1[count1]$ID))){
      selId<-ego1[count1]$ID
      res[selId,2]=ego1[count1]$pvalue
      count1<-count1+1
    }
  }
  return(res)
  cat('uni:',length(UniId),'\n')
  cat('sel:',length(selId),'\n')
  for(i in 1:length(UniId)){
    mark<-0
    mark1<-0
    if(length(selId)>0){
      for(j in 1:length(selId)){
        
        if(selId[j]==UniId[i]){
          mark<-1
          mark1<-j
          break()
        }
        
      }
    }
    if(mark==1){
      temp1<-ego1[mark1]$geneID
      temp1<-strsplit(temp1,split = '/')
      n<-length(temp1[[1]])
      selTermNum<-c(selTermNum,n)
    }
    if(mark==0){
      selTermNum<-c(selTermNum,0)
    }
    #cat(i,'id finished\n')
  }
  result<-c()
  cat('termNum:',termNum,'\n')
  cat('selTermNum:',selTermNum,'\n')
  for(i in 1:length(selTermNum)){
    a<-selTermNum[i]
    b<-termNum[i]
    c<-TotalgeneNum
    d<-difNum
    #cat('a:',a,'b:',b,"c:",c,'d:',d,'\n')
    #x<-choose(b,a)
    #y<-choose(c-b,d-a)
    #z<-choose(c,d)
    #cat('x:',x,'y:',y,'z:',z,'\n')
    e<-(1-phyper(a-1,b,c-b,d,))
    result<-c(result,e)
  }
  p<-result
  goName<-UniId
  result<-data.frame(goName,p)
  return(result)
}

##########Generate bp-gene matrix
picGener<-function(data,score){
  for(i in 1:length(score)){
    score[[i]][,2]<-(-log(score[[i]][,2],10))+1
    score[[i]][,1]<-as.character(score[[i]][,1])
  }
  difGene<-findDiffGene(data)
  universe<-colnames(data)
  universe<-universe[2:length(universe)]
  
  ego1<-enrichGO(gene          = universe,
                 universe      = universe,
                 OrgDb         = org.Rn.eg.db,
                 ont           = "BP",
                 pAdjustMethod = "BH",
                 pvalueCutoff  = 1,
                 qvalueCutoff  = 1,
                 minGSSize=0,
                 readable      = FALSE,
                 pool='ALL'
  )
  
  
  row<-length(score[[1]][,1])
  col<-length(data[1,])
  mark<-data.frame(matrix(0,row,col))
  colnames(mark)<-c('go',universe)
  mark[,1]<-score[[1]][,1]
  TermGene<-list()
  count<-1
  cat('1\n')
  while(!is.na(ego1[count]$geneID)){
    tmp<-ego1[count]$geneID
    tmp<-strsplit(tmp,split = '/')
    TermGene[[count]]<-tmp[[1]]
    count<-count+1
  }
  cat('2\n')
  for(i in 1:length(mark[,1])){
    for(j in 1:length(TermGene[[i]])){
      mark[i,c(TermGene[[i]][j])]<-1
    }
  }
  cat('3\n')
  result<-list()
  for(i in 1:length(data[,1])){
    result[[i]]<-data.frame(matrix(0,length(score[[1]][,1]),length(data[1,])))
    rows<-length(result[[i]][,1])
    result[[i]][1:length(score[[1]][,1]),]<-data[i,]
    cols<-length(data[1,])
    #cat('4\n')
    a<-as.matrix(mark[,2:length(mark[1,])])
    b<-as.matrix(result[[i]][,2:length(data[1,])])
    result[[i]]<-a*b
    for(k in 1:length(result[[i]][,1])){
      result[[i]][k,]<-score[[i]][k,2]*result[[i]][k,]
    }
    #cat('5\n')
    cat(i,'pic finished\n')
  }
  for(i in 1:length(result)){
    cat(i,'\n')
    #result[[i]]<-cbind(score[[1]][,1],result[[i]])
    #result[[i]]<-as.data.frame(result[[i]])
    colnames(result[[i]])<-c(universe)
  }
  return(result)
}


meanMatrix<-function(data){
  res=data.frame(matrix(0,length(data),length(data[[1]][1,])))
  for(i in 1:length(data)){
    for(j in 1:length(data[[1]][1,])){
      total=sum(data[[i]][,j])
      num=sum(data[[i]][,j]!=0)+0.0001
      res[i,j]=total/num
      cat(i,'drug',j,'genes','finished\n')
    }
  }
  return(res)
}
sumMatrix<-function(data){
  res=data.frame(matrix(0,length(data),length(data[[1]][1,])))
  for(i in 1:length(data)){
    for(j in 1:length(data[[1]][1,])){
      total=sum(data[[i]][,j])
      num=sum(data[[i]][,j]!=0)+0.0001
      res[i,j]=total
      cat(i,'drug',j,'genes','finished\n')
    }
  }
  return(res)
}

varMatrix<-function(data){
  res=data.frame(matrix(0,length(data),length(data[[1]][1,])))
  for(i in 1:length(data)){
    for(j in 1:length(data[[1]][1,])){
      t=c()
      for(k in 1:length(data[[1]][,1])){
        if(data[[i]][k,j]!=0){
          t=c(t,data[[i]][k,j])
        }
      }
      if(length(t)>0){
        res[i,j]=var(t)
      }
      else{
        res[i,j]=0
      }
      cat(i,'drug',j,'genes','finished\n')
      
    }
  }
  return(res)
}


maxMatrix<-function(data){
  res=data.frame(matrix(0,length(data),length(data[[1]][1,])))
  for(i in 1:length(data)){
    for(j in 1:length(data[[1]][1,])){
      max_value=max(data[[i]][,j])
      res[i,j]=max_value
      
      cat(i,'drug',j,'genes','finished\n')
    }
  }
  return(res)
}

##########Get the best value for each location in the list
getMost<-function(data,opt){
  row<-length(data[[1]][,1])
  col<-length(data[[1]][1,])
  result<-matrix(0,row,col)
  if(opt=='min'){
    for(i in 1:row){
      for(j in 1:col){
        a<-c()
        for(k in 1:length(data)){
          a<-c(a,data[[k]][i,j])
        }
        b<-min(a)
        result[i,j]<-b
        cat(i,'row',j,'cols:',b,'\n')
      }
    }
  }
  if(opt=='max'){
    for(i in 1:row){
      for(j in 1:col){
        a<-c()
        for(k in 1:length(data)){
          a<-c(a,data[[k]][i,j])
        }
        b<-max(a)
        result[i,j]<-b
        
        cat(i,'row',j,'cols:',b,'\n')
      }
    }
  }
  return(result)
}

##########Standardize the list
MatrixNormal<-function(data,max_data,min_data){
  
  a<-(max_data-min_data)
  
  result<-((data-min_data)/a)
  result<-result*255
  cat('drugs finished\n')
  
  return(result)
}

##########Batch upload images to folder
listWrite<-function(list,dir){
  for(i in 1:length(list)){
    a<-i
    a<-as.character(a)
    name<-paste(dir,a,'.csv')
    write.csv(list[[i]],file = name,row.names = F,quote = F,col.names = F)
  }
}