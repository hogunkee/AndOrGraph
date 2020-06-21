function theUnaryF=getTerminalDivision(unaryF,N_pos,N_neg,matched_pos,matched_neg,unaryW,Para)
getTheLambda=@(paraLambda,N_pos,N_neg,matched_pos,matched_neg)(paraLambda/(1/N_pos+matched_neg/(N_neg*max(matched_pos,1))));
atriNum=length(unaryF);
if(Para.Lambda<=0)
    theUnaryF(atriNum).f=[];
    for atri=1:atriNum
        theUnaryF(atri).f=mean(unaryF(atri).f,2);
    end
    if(Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri)
        theUnaryF(1).DiscriminativeW=getDiscriminativeW(theUnaryF(1).f);
    end
    if(Para.LearningItems.IsWeightTraining_.IsTrainCoreSVMForFirstLocalAtri)
        theUnaryF(1).CoreSVM(size(theUnaryF(1).f,2)).model=[];
    end
    return;
end
TheLambda=getTheLambda(Para.Lambda,N_pos,N_neg,matched_pos,matched_neg);
num=size(unaryF(1).f,2);
fDim=0;
for atri=1:atriNum
    fDim=fDim+size(unaryF(atri).f,1);
end
f=zeros(fDim,num);
fDim=0;
for atri=1:atriNum
    d=size(unaryF(atri).f,1);
    f(fDim+1:fDim+d,:)=unaryF(atri).f.*sqrt(unaryW(atri));
    fDim=fDim+d;
end
distMap=reshape(sum((repmat(reshape(f,[fDim,num,1]),[1,1,num])-repmat(reshape(f,[fDim,1,num]),[1,num,1])).^2,1),[num,num]);
TheLarge=10000000000000;
clusterMap=distMap./2;
clusterMap(linspace(1,num^2,num))=TheLarge;
cluster(num).list=[];
for i=1:num
    cluster(i).list=i;
    cluster(i).sumSqrDist=0;
end
while(length(cluster)>1)
    [d,rank]=min(clusterMap,[],1);
    [~,r2]=min(d,[],2);
    r1=rank(r2);
    TheCluster.list=[cluster(r1).list,cluster(r2).list];
    NewLen=length(TheCluster.list);
    TheCluster.sumSqrDist=sum(sum(distMap(TheCluster.list,TheCluster.list)))/(2*NewLen);
    if((TheCluster.sumSqrDist>cluster(r1).sumSqrDist+cluster(r2).sumSqrDist+TheLambda)&&(length(cluster)<=Para.MaxTerminalNum))
        break;
    end
    cluster(end+1)=TheCluster;
    list=setdiff(1:length(cluster),[r1,r2]);
    clusterMap(end+1,end+1)=0;
    for i=list(1:end-1)
        tmp=sum(sum(distMap(cluster(i).list,TheCluster.list)))/(2*(NewLen+length(cluster(i).list)))-(cluster(i).sumSqrDist+TheCluster.sumSqrDist);
        clusterMap(i,end)=tmp;
        clusterMap(end,i)=tmp;
    end
    clusterMap(end,end)=TheLarge;    
    cluster=cluster(list);
    clusterMap=clusterMap(list,list);
end
theUnaryF(atriNum).f=[];
for atri=1:atriNum
    TerNum=length(cluster);
    theUnaryF(atri).f=zeros(size(unaryF(atri).f,1),TerNum);
    for i=1:TerNum
        theUnaryF(atri).f(:,i)=mean(unaryF(atri).f(:,cluster(i).list),2);
    end
end
if(Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri)
    theUnaryF(1).DiscriminativeW=getDiscriminativeW(theUnaryF(1).f);
end
if(Para.LearningItems.IsWeightTraining_.IsTrainCoreSVMForFirstLocalAtri)
    theUnaryF(1).CoreSVM(size(theUnaryF(1).f,2)).model=[];
end


function w=getDiscriminativeW(f)
w=f-repmat(mean(f,1),[size(f,1),1]);
w=w./repmat(mean(abs(w),1),[size(w,1),1]);
