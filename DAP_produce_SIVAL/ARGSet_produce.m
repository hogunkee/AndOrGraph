function ARGSet_produce(Name_batch)
ARGSet=producing(Name_batch,'');
save(sprintf('./mat/ARGs_%s.mat',Name_batch),'ARGSet');
ARGSet_neg=producing(Name_batch,'neg_');
save(sprintf('./mat/ARGs_neg_%s.mat',Name_batch),'ARGSet_neg');


function ARGSet=producing(Name_batch,PreName)
Small=0.0000000000001;
load(sprintf('./mat/FinalPatches_%s%s.mat',PreName,Name_batch),'FinalPatches');
TotalImgNum=size(FinalPatches,2);
ARGSet(TotalImgNum).arg=[];
c=0;
for ImgNo=1:TotalImgNum
    arg=[];
    arg.ARGInfo.name=sprintf('%s%03d',Name_batch,ImgNo);
    num=size(FinalPatches(ImgNo).HOGFeature,2);
    arg.nodeNum=num;
    arg.NodeType.type='MidLevel';
    arg.NodeType.nodeList=1:num;
    arg.NodeTypeInfo.unaryF(1).f=FinalPatches(ImgNo).HOGFeature;
    if(num>1)
        HW=FinalPatches(ImgNo).HWScaleVal(1:2,:);
        Scale=FinalPatches(ImgNo).HWScaleVal(3,:);
        diff=repmat(reshape(HW,[2,num,1]),[1,1,num])-repmat(reshape(HW,[2,1,num]),[1,num,1]);
        dist=max(sqrt(reshape(sum(diff.^2,1),[num,num])),sqrt(Scale'*Scale).*0.1);
        arg.pairwiseF(1).f=zeros(2,num,num);
        arg.pairwiseF(1).f(1,:,:)=reshape(repmat(Scale,[num,1])./dist,[1,num,num]);
        arg.pairwiseF(1).f(2,:,:)=reshape(repmat(Scale',[1,num])./dist,[1,num,num]);
        arg.pairwiseF(1).f=log(max(arg.pairwiseF(1).f,Small));
        arg.pairwiseF(2).f=reshape(repmat(Scale,[num,1])./repmat(max(Scale,Small)',[1,num]),[1,num,num]);
        arg.pairwiseF(2).f=log(max(arg.pairwiseF(2).f,Small));
        arg.pairwiseF(3).f=diff./max(repmat(sqrt(sum(diff.^2,1)),[2,1,1]),Small);
        c=c+1;
        ARGSet(c).arg=arg;
        ARGSet(c).flip=false;
    end
end
ARGSet=ARGSet(1:c);
