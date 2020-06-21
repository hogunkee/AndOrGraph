function LearnDiscriminativePatches(Name_batch)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Parameters
[HOG,Kmeans,Learning,ImgRoot]=ParaSetting_ImgSet_SIVAL();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ProducePatchSet(ImgRoot,Name_batch,HOG,'');
ProduceBackgroundPatchSet(ImgRoot,Name_batch,HOG);
KmeansPatches(Name_batch,Kmeans,Learning);
% ViewPatches(ImgRoot,Name_batch,HOG,Learning);
TrainClusters(Name_batch,Learning);
SelectBestClusterAndSamples(Name_batch,Learning);
getFinalPatches('',Name_batch,Learning,1);
getFinalPatches('neg_',Name_batch,Learning,-1);
% viewFinalPatches(ImgRoot,Name_batch);


function getFinalPatches(PreName,Name_batch,Learning,PosNeg)
load(sprintf('./mat/Final_PatchSet_%s.mat',Name_batch),'TopSamples','TheModel','posValPerImg','posDetectPerImg','negValPerImg','negDetectPerImg');
Discriminative=posDetectPerImg./(posDetectPerImg+negDetectPerImg+0.0000000000000001);
[~,list]=sort(Discriminative,'descend');
TheModel=TheModel(list(1:min(Learning.TopCluNum,length(list))));
size(TheModel,2)
load(sprintf('./mat/PatchSet_%s%s.mat',PreName,Name_batch),'fset','TheImgHWScale');
LargesetImgNo=max(TheImgHWScale(1,:));
Learning.ThreadNum=min(Learning.ThreadNum,LargesetImgNo);
tmp=round(linspace(1,LargesetImgNo+1,Learning.ThreadNum+1));
list1=tmp(1:end-1);
list2=tmp(2:end)-1;
Division(Learning.ThreadNum).TheFset=[];
Division(Learning.ThreadNum).TheImgHWScale=[];
for i=1:Learning.ThreadNum
    len=list2(i)-list1(i)+1;
    Division(Learning.ThreadNum).TheFset(len).matrix=[];
    Division(Learning.ThreadNum).TheImgHWScale(len).matrix=[];
    Division(Learning.ThreadNum).output(len).HWScaleVal=[];
    Division(Learning.ThreadNum).output(len).HOGFeature=[];
    c=0;
    for ImgNo=list1(i):list2(i)
        c=c+1;
        list=find(TheImgHWScale(1,:)==ImgNo);
        Division(i).TheFset(c).matrix=fset(:,list)';
        Division(i).TheImgHWScale(c).matrix=TheImgHWScale(:,list);
    end    
end
clear fset TheImgHWScale
parfor par=1:Learning.ThreadNum
    TheMinScore=Learning.MinCollectScore;
    c=0;
    for ImgNo=list1(par):list2(par)
        c=c+1;
        parFset=Division(par).TheFset(c).matrix;
        parImgHWScale=Division(par).TheImgHWScale(c).matrix;
        patchnum=size(parImgHWScale,2);
        Value=ones(1,patchnum).*(-10000000000);
        ValidIndex=1:patchnum;
        parLabel=ones(size(parFset,1),1);
        parHWScaleVal=[];
        parHOGFeature=[];
        for i=1:size(TheModel,2)
            if(isempty(parLabel))
                break;
            end
            [~,~,decision_values]=svmpredict(parLabel,parFset,TheModel(i));
            TheTmp=find(decision_values.*PosNeg-TheMinScore>0);
            if(isempty(TheTmp))
                [~,TheTmp]=max(decision_values);
            else
                if(length(TheTmp)>Learning.MaxCollectNumPerModel)
                    [~,t]=sort(decision_values(TheTmp),'descend');
                    TheTmp=TheTmp(t(1:Learning.MaxCollectNumPerModel));
                end
            end
            t=find(Value(ValidIndex(TheTmp))-decision_values(TheTmp)'<0);
            if(~isempty(t))
                Value(ValidIndex(TheTmp(t)))=decision_values(TheTmp(t));
            end
            
            parLabel=parLabel(1:end-length(TheTmp));
            tmp_list=setdiff(1:size(parFset,1),TheTmp);
            ValidIndex=ValidIndex(tmp_list);
            parFset=parFset(tmp_list,:);
            parImgHWScale=parImgHWScale(:,tmp_list);
        end
        [~,t]=sort(Value,'descend');
        if(length(t)>Learning.CollectNum)
            t=t(1:Learning.CollectNum);
        end
        Division(par).output(c).HWScaleVal=[Division(par).TheImgHWScale(c).matrix(2:4,t);Value(t)];
        Division(par).output(c).HOGFeature=Division(par).TheFset(c).matrix(t,:)';
    end
end
FinalPatches(LargesetImgNo).HWScaleVal=zeros(4,0);
FinalPatches(LargesetImgNo).HOGFeature=zeros(100,0);
for i=1:Learning.ThreadNum
    c=0;
    for ImgNo=list1(i):list2(i)
        c=c+1;
        FinalPatches(ImgNo).HWScaleVal=Division(i).output(c).HWScaleVal;
        FinalPatches(ImgNo).HOGFeature=Division(i).output(c).HOGFeature;
    end
end
save(sprintf('./mat/FinalPatches_%s%s.mat',PreName,Name_batch),'FinalPatches');


function viewFinalPatches(ImgRoot,Name_batch)
load(sprintf('./mat/FinalPatches_%s.mat',Name_batch),'FinalPatches');
for ImgNo=1:size(FinalPatches,2)
    close all;
    figure(ImgNo);
    filename=sprintf('%s%s/%03d.jpg',ImgRoot,Name_batch,ImgNo);
    imshow(imread(filename));
    hold on
    PatchNum=size(FinalPatches(ImgNo).HWScaleVal,2);
    for pNo=1:PatchNum
        H=FinalPatches(ImgNo).HWScaleVal(1,pNo);
        W=FinalPatches(ImgNo).HWScaleVal(2,pNo);
        Scale=FinalPatches(ImgNo).HWScaleVal(3,pNo)/2;
        Y=[H-Scale,H+Scale,H+Scale,H-Scale,H-Scale];
        X=[W-Scale,W-Scale,W+Scale,W+Scale,W-Scale];
        plot(X,Y,'r-');
    end
    disp(PatchNum);
    pause;
end


% function getFinalPatches(Name_batch,Learning,IsShown)
% load(sprintf('./mat/Final_PatchSet_%s.mat',Name_batch),'TopSamples','TheModel','posValPerImg','posDetectPerImg','negValPerImg','negDetectPerImg');
% Discriminative_1=posValPerImg;
% Discriminative_2=posDetectPerImg./(posDetectPerImg+negDetectPerImg+0.0000000000000001);
% [~,list_1]=sort(Discriminative_1,'descend');
% [~,list_2]=sort(Discriminative_2,'descend');
% list=sort(unique([list_1(1:Learning.TopCluNum);list_2(1:Learning.TopCluNum)]));
% TheModel=TheModel(list);
% size(TheModel)
% load(sprintf('./mat/PatchSet_%s.mat',Name_batch),'fset','TheImgHWScale');
% LargesetImgNo=max(TheImgHWScale(1,:));
% for ImgNo=1:LargesetImgNo
%     list=find(TheImgHWScale(1,:)==ImgNo);
%     TheLabel=ones(size(list,2),1);
%     TheFset=fset(:,list)';
%     if(IsShown)
%         close all;
%         figure;
%         filename=sprintf('./ImgSet/%s/%03d.jpg',Name_batch,ImgNo);
%         imshow(imread(filename));
%         hold on
%     end
%     c=0;
%     for i=1:size(TheModel,2)
%         [~,~,decision_values]=svmpredict(TheLabel,TheFset,TheModel(i));
%         [val,tmp]=max(decision_values);
%         if(val>Learning.MinScore)
%             c=c+1;
%             tar=list(tmp);
%             H=TheImgHWScale(2,tar);
%             W=TheImgHWScale(3,tar);
%             Scale=TheImgHWScale(4,tar)/2;
%             Y=[H-Scale,H+Scale,H+Scale,H-Scale,H-Scale];
%             X=[W-Scale,W-Scale,W+Scale,W+Scale,W-Scale];
%             plot(X,Y,'r-');
%             fprintf('%d/%d\n',i,size(TheModel,2));
%             list=[list(1:tmp-1),list(tmp+1:end)];
%             TheLabel=TheLabel(1:end-1);
%             TheFset=[TheFset(1:tmp-1,:);TheFset(tmp+1:end,:)];
%         end
%     end
%     disp(c);
%     pause
% end


function SelectBestClusterAndSamples(Name_batch,Learning)
load(sprintf('./mat/PatchSet_%s.mat',Name_batch),'TheImgHWScale');
TopK=Learning.TopK;Learning.TopK=max(Learning.CluCollectK*max(TheImgHWScale(1,:)),Learning.TopK);
clear TheImgHWScale fset
load(sprintf('./mat/Clu_%s.mat',Name_batch),'TopSamples','TheModel');
Domain_target=[];
PreName='';
[TopSamples,TopValues,~,posValPerImg,posDetectPerImg]=getCluClassification(Name_batch,Learning,TheModel,Domain_target,PreName);
%posMVal=mean(TopValues(1:Learning.TopR,:),1)';
PreName='neg_';
[~,~,~,negValPerImg,negDetectPerImg]=getCluClassification(Name_batch,Learning,TheModel,Domain_target,PreName);
save(sprintf('./mat/Final_PatchSet_%s.mat',Name_batch),'TopSamples','TheModel','posValPerImg','posDetectPerImg','negValPerImg','negDetectPerImg');
Learning.TopK=TopK;


function TrainClusters(Name_batch,Learning)
load(sprintf('./mat/Kmeans_%s.mat',Name_batch),'Clu','Clu_large');
[TopSamples,~]=getTopSamples(Clu_large,Learning);
SampleNum=size(Clu.list,2);
tmp=randperm(SampleNum);
Domain_1=sort(tmp(1:floor(SampleNum/2)));
Domain_2=sort(tmp(floor(SampleNum/2)+1:SampleNum));
for iter=1:Learning.IterNum
    TheModel=TrainClassifier(Name_batch,Learning,TopSamples);
    if(mod(iter,2)==1)
        Domain_target=Domain_1;
    else
        Domain_target=Domain_2;
    end
    [TopSamples,~,SelectCluList,~,~]=getCluClassification(Name_batch,Learning,TheModel,Domain_target,'');
    TopSamples=TopSamples(:,SelectCluList);
    TheModel=TheModel(SelectCluList);
    save(sprintf('./mat/Clu_%s.mat',Name_batch),'');
    save(sprintf('./mat/Clu_%s.mat',Name_batch),'TopSamples','TheModel','iter');
end


function [TopSamples,TopValues,SelectCluList,ValPerImg,DetectPerImg]=getCluClassification(Name_batch,Learning,TheModel,Domain_target,PreName)
load(sprintf('./mat/PatchSet_%s%s.mat',PreName,Name_batch),'fset','TheImgHWScale');
if(size(Domain_target,2)==0)
    Domain_target=1:size(fset,2);
end
ori_fset=fset(:,Domain_target)';
ori_TheImgHWScale=TheImgHWScale(:,Domain_target);
ori_Domain_target=Domain_target;
clear fset TheImgHWScale Domain_target

CluNum=size(TheModel,2);
Learning.ThreadNum=min(Learning.ThreadNum,CluNum);
list=round(linspace(1,CluNum+1,Learning.ThreadNum+1));
list1=list(1:end-1);
list2=list(2:end)-1;
ModelSet(Learning.ThreadNum).model=[];
for i=1:Learning.ThreadNum
    ModelSet(i).model=TheModel(list1(i):list2(i));
end
clear TheModel

ImgNo_max=max(ori_TheImgHWScale(1,:));
TopSamples=zeros(Learning.TopK,CluNum);
TopValues=ones(Learning.TopK,CluNum).*(-1000);
ValPerImg=zeros(CluNum,1);
DetectPerImg=zeros(CluNum,1);
count_ValPerImg=zeros(CluNum,1);
count_DetectPerImg=zeros(CluNum,1);
for ImgNo_start=1:Learning.ThreadBuffer_ImgNum:ImgNo_max
    ImgNo_end=min(ImgNo_start+Learning.ThreadBuffer_ImgNum-1,ImgNo_max);
    ImgNum=ImgNo_end-ImgNo_start+1;
    tmplist=find((ori_TheImgHWScale(1,:)>=ImgNo_start).*(ori_TheImgHWScale(1,:)<=ImgNo_end)>0);
    TheImgHWScale=ori_TheImgHWScale(:,tmplist);
    fset=ori_fset(tmplist,:);
    Domain_target=ori_Domain_target(tmplist);
    ImgIndex=zeros(ImgNum,2);
    c=0;
    for i=ImgNo_start:ImgNo_end
        c=c+1;
        tmp=find(TheImgHWScale(1,:)==i);
        if(size(tmp,2)>0)
            ImgIndex(c,:)=[tmp(1),tmp(end)];
        end
    end
    clear TheImgHWScale
    PtNum=size(fset,1);
    flabel=ones(PtNum,1);
    decision(Learning.ThreadNum).val=[];
    decision(Learning.ThreadNum).id=[];
    decision(Learning.ThreadNum).rsp=[];
    decision(Learning.ThreadNum).mpi=[];
    parfor par=1:Learning.ThreadNum
        c=0;
        TheImgNum=ImgNum;
        TheImgIndex=ImgIndex;
        TopK=min(Learning.TopK,TheImgNum);
        MinScore=Learning.MinScore;
        TheDomain_target=Domain_target;
        len=list2(par)-list1(par)+1;
        decision(par).val=zeros(TopK,len);
        decision(par).id=zeros(TopK,len);
        decision(par).rsp=zeros(2,len);
        decision(par).mpi=zeros(2,len);
        for Clu_ID=list1(par):list2(par)
            c=c+1;
            [~,~,decision_values]=svmpredict(flabel,fset,ModelSet(par).model(c));

            %[decision_values,tmplist]=sort(decision_values,'descend');
            %decision(par).val(:,c)=decision_values(1:TopK);
            %decision(par).id(:,c)=TheDomain_target(tmplist(1:TopK));

            %decision(par).rsp(:,c)=[sum(decision_values>MinScore);size(decision_values,1)];
            rsp=0;
            mpi=0;
            ValList=ones(TheImgNum,1).*(-100000000000000);
            IDList=zeros(TheImgNum,1);
            for i=1:TheImgNum
                if(TheImgIndex(i,1)>0)
                    %size(decision_values)
                    %disp(TheImgIndex(i,:));
                    [ValList(i),tmp]=max(decision_values(TheImgIndex(i,1):TheImgIndex(i,2)));
                    mpi=mpi+(ValList(i)>MinScore);
                    rsp=rsp+ValList(i);
                    IDList(i)=TheDomain_target(tmp+TheImgIndex(i,1)-1);
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            decision(par).mpi(:,c)=[mpi;sum(TheImgIndex(:,1)>0)];
            decision(par).rsp(:,c)=[rsp;sum(TheImgIndex(:,1)>0)];
            [val,tmplist]=sort(ValList,'descend');
            decision(par).val(:,c)=val(1:TopK);
            decision(par).id(:,c)=IDList(tmplist(1:TopK));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            fprintf('             ImgNo_start %d  Par %d  CluNo %d  %%%.3f\n',ImgNo_start,par,Clu_ID,c/len);
        end
    end
    c=0;
    for i=1:Learning.ThreadNum
        num=size(decision(i).id,2);
        tmp_val=[TopValues(:,c+1:c+num);decision(i).val];
        tmp_id=[TopSamples(:,c+1:c+num);decision(i).id];
        [val,tmplist]=sort(tmp_val,1,'descend');
        TopValues(:,c+1:c+num)=val(1:Learning.TopK,:);
        TopSamples(:,c+1:c+num)=tmp_id(repmat((0:num-1),[Learning.TopK,1]).*size(tmp_id,1)+tmplist(1:Learning.TopK,:));
        ValPerImg(c+1:c+num)=ValPerImg(c+1:c+num)+decision(i).rsp(1,:)';
        DetectPerImg(c+1:c+num)=decision(i).mpi(1,:)';
        count_ValPerImg(c+1:c+num)=count_ValPerImg(c+1:c+num)+decision(i).rsp(2,:)';
        count_DetectPerImg(c+1:c+num)=count_DetectPerImg(c+1:c+num)+decision(i).mpi(2,:)';
        c=c+num;
    end
end
count_ValPerImg=max(count_ValPerImg,1);
count_DetectPerImg=max(count_DetectPerImg,1);
ValPerImg=ValPerImg./count_ValPerImg;
DetectPerImg=DetectPerImg./count_DetectPerImg;
SelectCluList=find(TopValues(Learning.TopK,:)>Learning.MinScore);
fprintf('Obtain %d clusters\n',size(SelectCluList,2));


function TheModel=TrainClassifier(Name_batch,Learning,TopSamples)
load(sprintf('./mat/PatchSet_neg_%s.mat',Name_batch),'fset','TheImgHWScale');
Learning.NegSampleNum=min(Learning.NegSampleNum,size(fset,2));
tmp=randperm(size(fset,2));
negFset=fset(:,tmp(1:Learning.NegSampleNum));
clear fset TheImgHWScale
CluNum=size(TopSamples,2);
Learning.ThreadNum=min(Learning.ThreadNum,CluNum);
list=round(linspace(1,CluNum+1,Learning.ThreadNum+1));
list1=list(1:end-1);
list2=list(2:end)-1;
load(sprintf('./mat/PatchSet_%s.mat',Name_batch),'fset','TheImgHWScale');
Fset(Learning.ThreadNum).fset=[];
for i=1:Learning.ThreadNum
    Fset(i).fset=reshape(fset(:,TopSamples(:,list1(i):list2(i))),[size(fset,1),Learning.TopK,list2(i)-list1(i)+1]);
end
clear fset TheImgHWScale





ErrorWeight=Learning.NegSampleNum/Learning.TopK;
%ErrorWeight=1;




classifier(Learning.ThreadNum).model=[];
parfor par=1:Learning.ThreadNum
    TopK=Learning.TopK;
    TheLabel=[ones(TopK,1);-ones(Learning.NegSampleNum,1)];
    c=0;
    for CluNo=list1(par):list2(par)
        c=c+1;
        f=reshape(Fset(par).fset(:,:,c),[size(Fset(par).fset,1),TopK]);
        TheStr=sprintf('-s 0 -h 0 -t 0 -c 1000 -w1 %f -w-1 1',ErrorWeight);
        TheMatrix=[f,negFset]';
        model=svmtrain(TheLabel,TheMatrix,TheStr);
        classifier(par).model=[classifier(par).model,model];
        fprintf('                      Par %d  CluNo %d\n',par,CluNo);
    end
end
TheModel=[];
for i=1:Learning.ThreadNum
    TheModel=[TheModel,classifier(i).model];
end


function ViewPatches(ImgRoot,Name_batch,HOG,Learning)
load(sprintf('./mat/Kmeans_%s.mat',Name_batch),'Clu','Clu_large');
load(sprintf('./mat/PatchSet_%s.mat',Name_batch),'fset','TheImgHWScale');
[TopSamples,validClu]=getTopSamples(Clu_large,Learning);
root=sprintf('%s%s/',ImgRoot,Name_batch);
for i=1:size(TopSamples,2)
    close all;
    for j=1:Learning.TopK
        target=TopSamples(j,i);
        filename=sprintf('%s%03d.jpg',root,TheImgHWScale(1,target));
        I=double(imread(filename));
        SampleOri=0;
        [Patch,~,~]=segmentPatch(I,TheImgHWScale(2,target),TheImgHWScale(3,target),SampleOri,TheImgHWScale(4,target),HOG);
        figure;
        imshow(Patch);
        [var(fset(:,target)),min(fset(:,target)),max(fset(:,target))]
        %showHOG(fset(:,target),Patch,HOG);
    end
    pause;
end


function [TopSamples,validClu]=getTopSamples(Clu_large,Learning)
validClu=CluClearUp(Clu_large,Learning.TopK);
CluNum=size(validClu.CluSize,1);
TopSamples=zeros(Learning.TopK,CluNum);
for i=1:CluNum
    tmp=find(validClu.owner==i);
    [~,order]=sort(Clu_large.PtDist(tmp));
    TopSamples(:,i)=validClu.list(tmp(order(1:Learning.TopK)));
end


function KmeansPatches(Name_batch,Kmeans,Learning)
load(sprintf('./mat/PatchSet_%s.mat',Name_batch),'fset','TheImgHWScale');
%%% disallowing patches without significant gradients
SampleNum=size(fset,2);
disp(SampleNum);
%%% K-means
fset=[(1:SampleNum);fset];
tmp=randperm(SampleNum);
TheK=ceil(Kmeans.KRate*SampleNum);
kcentre_initial=fset(:,tmp(1:TheK));
%[kcentre,owner,sqrerr,CluMaxDist,PtDist,CluSize]=kmeans_mex_Linux(fset,kcentre_initial,2,Kmeans.MaxIterNum,Learning.ThreadNum);
%[kcentre,owner,sqrerr,CluMaxDist,PtDist,CluSize]=kmeans_mex(fset,kcentre_initial,2,Kmeans.MaxIterNum,Learning.ThreadNum);
[kcentre,owner,sqrerr,CluMaxDist,PtDist,CluSize]=kmeans_cluster(fset,kcentre_initial,2,Kmeans.MaxIterNum,Learning);
Clu.list=1:SampleNum;
Clu.kcentre=kcentre;
Clu.owner=owner;
Clu.sqrerr=sqrerr;
Clu.CluMaxDist=CluMaxDist;
Clu.PtDist=PtDist;
Clu.CluSize=CluSize;
Clu_large=CluClearUp(Clu,Kmeans.MinSize);
save(sprintf('./mat/Kmeans_%s.mat',Name_batch),'');
save(sprintf('./mat/Kmeans_%s.mat',Name_batch),'Clu','Clu_large');


function Clu_large=CluClearUp(Clu,CluMinSize)
CluNum=size(Clu.CluSize,1);
list_LargeClu=find(Clu.CluSize>=CluMinSize);
Clu_large.kcentre=Clu.kcentre(:,list_LargeClu);
Clu_large.CluMaxDist=Clu.CluMaxDist(list_LargeClu);
Clu_large.CluSize=Clu.CluSize(list_LargeClu);
tmp=zeros(CluNum,1);
tmp(list_LargeClu)=(1:size(list_LargeClu,1))';
Pt_LargeClu=find(tmp(Clu.owner)>0);
Clu_large.list=Clu.list(Pt_LargeClu);
Clu_large.owner=tmp(Clu.owner(Pt_LargeClu));
Clu_large.PtDist=Clu.PtDist(Pt_LargeClu);


function ProducePatchSet(ImgRoot,Name_batch,HOG,PreName)
root=sprintf('%s%s/%s',ImgRoot,Name_batch,PreName);
filter=fspecial('gaussian',[HOG.GausianFilterScale,HOG.GausianFilterScale],HOG.GausianFilterScale/2);
ImgNo=0;
fset=[];
TheImgHWScale=[];
while(true)
    filename=sprintf('%s%03d.jpg',root,ImgNo+1);
    if(exist(filename,'file')~=2)
        break;
    end
    ImgNo=ImgNo+1;
    I=double(imread(filename));
    I=imfilter(I,filter);
    [MaxH,MaxW,~]=size(I);
    ori=getPixelOri(I);
    ImgHW=[MaxH,MaxW];
    LargestScale=min(ImgHW)*HOG.ScaleLargest;
    [f,TheHWScale,fnorm]=HOG_mex(LargestScale,HOG.ScaleLevelNum,HOG.ScaleDecrease,HOG.CellNum,HOG.MinCellScale,HOG.SampleWindowStep,HOG.VoteOriNum,ImgHW,I,ori.Ori,ori.Val,ImgNo);
    [fnorm,list]=sort(fnorm,'descend');
    SampleNum=min(floor(length(list)*HOG.MinAvgGradientRate),sum(fnorm>HOG.MinAvgGradient));
    list=sort(list(1:SampleNum),'ascend');
    fset=[fset,f(:,list)];
    TheImgHWScale=[TheImgHWScale,[ones(1,SampleNum).*ImgNo;TheHWScale(:,list)]];
%     for i=1:size(f,2)
%         SampleOri=0;
%         TheHWScale(:,i)
%         [Patch,ori,rect]=segmentPatch(I,TheHWScale(1,i),TheHWScale(2,i),SampleOri,TheHWScale(3,i),HOG);
%         close all;
%         showHOG(f(:,i),Patch,HOG);
%         pause;
%     end
    disp(ImgNo);
end
save(sprintf('./mat/PatchSet_%s%s.mat',PreName,Name_batch),'');
save(sprintf('./mat/PatchSet_%s%s.mat',PreName,Name_batch),'fset','TheImgHWScale');
clear fset TheImgHWScale


function ProduceBackgroundPatchSet(ImgRoot,Name_batch,HOG)
PreName='neg_';
ProducePatchSet(ImgRoot,Name_batch,HOG,PreName);
