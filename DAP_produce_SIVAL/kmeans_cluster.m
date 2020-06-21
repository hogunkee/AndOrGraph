function [kcenter,LabelList,sqrerr,CluMaxDist,DistList,CluSize]=kmeans_cluster(data,kcenter,start,MaxIterNum,Learning)
PtNum=size(data,2);
OldLabelList=zeros(PtNum,1);
for iter=1:MaxIterNum
    [DistList,LabelList,CluSize,CluMaxDist,sqrerr,kcenter]=kmeans_getDistUpdate(data,kcenter,start,Learning);
    fprintf(' %f\n',sqrerr);
    if(sum(abs(OldLabelList-LabelList))==0)
        break;
    end
    OldLabelList=LabelList;
end


function [DistList,LabelList,CluSize,CluMaxDist,sqrerr,kcenter]=kmeans_getDistUpdate(data,kcenter,start,Learning)
fprintf('doing kmeans_getDistUpdate ... ');
[W,PtNum]=size(data);
K=size(kcenter,2);
Learning.ThreadNum=min(Learning.ThreadNum,PtNum);
list=round(linspace(1,PtNum+1,Learning.ThreadNum+1));
list1=list(1:end-1);
list2=list(2:end)-1;
trd(Learning.ThreadNum).data=[];
trd(Learning.ThreadNum).DistList=[];
trd(Learning.ThreadNum).LabelList=[];
for i=1:Learning.ThreadNum
    trd(i).data=data(:,list1(i):list2(i));
    trd(i).DistList=[];
    trd(i).LabelList=[];
    trd(i).CluSize=[];
    trd(i).CluMaxDist=[];
    trd(i).sqrerr=[];
end
parfor par=1:Learning.ThreadNum
    [trd(par).DistList,trd(par).LabelList,trd(par).CluSize,trd(par).CluMaxDist,trd(par).sqrerr]=kmeans_getDistUpdate_mex(start,kcenter,trd(par).data);
end
DistList=zeros(PtNum,1);
LabelList=zeros(PtNum,1);
CluSize=zeros(K,1);
CluMaxDist=zeros(K,1);
sqrerr=0;
c=0;
for i=1:Learning.ThreadNum
    num=size(trd(i).DistList,1);
    DistList(c+1:c+num)=trd(i).DistList;
    LabelList(c+1:c+num)=trd(i).LabelList;
    CluSize=CluSize+trd(i).CluSize;
    CluMaxDist=max(CluMaxDist,trd(i).CluMaxDist);
    sqrerr=sqrerr+trd(i).sqrerr;
    c=c+num;
end
clear trd
list=find(CluSize>0);
K=sum(CluSize>0);
CluSize=CluSize(list);
CluMaxDist=CluMaxDist(list);
for i=1:K
    LabelList(LabelList==list(i))=i;
end
Learning.ThreadNum=min(Learning.ThreadNum,K);
list=round(linspace(1,K+1,Learning.ThreadNum+1));
list1=list(1:end-1);
list2=list(2:end)-1;
TheData(Learning.ThreadNum).data=[];
for i=1:Learning.ThreadNum
    tmplist=find((LabelList>=list1(i)).*(LabelList<=list2(i))>0);
    TheData(i).data=data(:,tmplist);
    TheData(i).LabelList=LabelList(tmplist);
    TheData(i).kcenter=[];
end
clear data
parfor par=1:Learning.ThreadNum
    label_min=min(TheData(par).LabelList);
    label_max=max(TheData(par).LabelList);
    TheData(par).kcenter=zeros(W,label_max-label_min+1);
    c=0;
    for label=label_min:label_max
        c=c+1;
        TheData(par).kcenter(:,c)=mean(TheData(par).data(:,TheData(par).LabelList==label),2);
    end
end
kcenter=zeros(W,K);
c=0;
for i=1:Learning.ThreadNum
    num=size(TheData(i).kcenter,2);
    kcenter(:,c+1:c+num)=TheData(i).kcenter;
    c=c+num;
end
fprintf('done\n');
