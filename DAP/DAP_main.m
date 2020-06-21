function DAP_main(model_set,Name_batch,ARGSet,Para,TargetDir,ARGSet_neg)
Para.LearningItems.IsWeightTraining_.IsTrainAttributeWeight=Para.LearningItems.IsWeightTraining_.IsTrainAttributeWeight&&Para.LearningItems.IsWeightTraining;
Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri=Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri&&Para.LearningItems.IsWeightTraining;
Para.LearningItems.IsWeightTraining_.IsProtectLastPairwiseWFromTraining=Para.LearningItems.IsWeightTraining_.IsProtectLastPairwiseWFromTraining&&Para.LearningItems.IsWeightTraining;
Para.LearningItems.IsWeightTraining_.IsTrainCoreSVMForFirstLocalAtri=Para.LearningItems.IsWeightTraining_.IsTrainCoreSVMForFirstLocalAtri&&Para.LearningItems.IsWeightTraining;

filename=sprintf('%s%s_model.mat',TargetDir,Name_batch);
modelNum=length(model_set);

for i=1:modelNum
    Log(i).Insert=0;
    Log(i).Delete=0;
end
Accuracy.curve=[];
Accuracy.AP=0;
Accuracy.energy_pos=[];
Accuracy.label_pos=[];
Accuracy.energy_neg=[];
Accuracy.label_neg=[];
record.model_set=model_set;
record.GoodARGList(modelNum).list=[];
record.Log=Log;
record.Accuracy=Accuracy;
save(filename,'record');
clear record
StartNo=1;


% load(filename,'record');
% record=record(1:end-1);
% model_set=record(end).model_set;
% Log=record(end).Log;
% Accuracy=record(end).Accuracy;
% StartNo=length(record);


tic;
for Iter=StartNo:Para.MaxIterationNum+Para.IterationNum_NoStructureModification
    GoodARGList_iter=selectGoodARGs(ARGSet,model_set,Para,Iter);
    IsReDivideTerminals=(mod(Iter-1,Para.ReDivideTerminalsInterval)==0);
    for model_No=1:modelNum
        ARGList=GoodARGList_iter(model_No).list;
        if(length(ARGList)<3)
            if(Iter>1)
                load(filename,'record');
                ARGList=record(end).GoodARGList(model_No).list;
                if(length(ARGList)<3)
                    clear record
                    continue;
                end
                GoodARGList_iter(model_No)=record(end).GoodARGList(model_No);
                clear record
            else
                continue;
            end
        end
        model=model_set(model_No);
        if((Para.LearningItems.IsStructureModification&&(Iter<=Para.MaxIterationNum))||(Para.LearningItems.IsWeightTraining&&(Iter>Para.MaxIterationNum))||Para.LearningItems.IsTrainingPenalty||IsReDivideTerminals)
            GoodARGList_neg=getARGList_neg(model,ARGSet_neg,Para);
            ARGList_neg=GoodARGList_neg.list;
            matchList_neg=GoodARGList_neg.match;
        else
            ARGList_neg=[];
            matchList_neg=[];
        end
        if(IsReDivideTerminals)
            model=ReDivideTerminals(model,ARGSet(ARGList),ARGSet_neg(ARGList_neg),GoodARGList_iter(model_No).match,matchList_neg,Para);
            GoodARGList_iter(model_No).psi=[];
        end
        model=ModelAttributeEstimation(model,ARGSet(ARGList),GoodARGList_iter(model_No).match,Para,GoodARGList_iter(model_No).psi,IsReDivideTerminals);
        if((Para.LearningItems.IsWeightTraining&&(Iter>Para.MaxIterationNum))||Para.LearningItems.IsTrainingPenalty)
            Para_tmp=Para.LearningItems.IsWeightTraining;
            Para.LearningItems.IsWeightTraining=(Para.LearningItems.IsWeightTraining&&(Iter>Para.MaxIterationNum));
            model=WeightTraining_discriminative(model,ARGSet(ARGList),ARGSet_neg(ARGList_neg),Para,modelNum,IsReDivideTerminals);
            Para.LearningItems.IsWeightTraining=Para_tmp;
        end
        if(modelNum>1)
            b=getB(model,ARGSet(ARGList),ARGSet_neg,Para,0);
            model.b=b;
        end
        if(Para.LearningItems.IsStructureModification&&(Iter<=Para.MaxIterationNum))
            [model,IsChange1,argFeature_pos,argFeature_neg,MaxUnmatchNum_pos,MaxUnmatchNum_neg]=EliminateNode(model,ARGSet(ARGList),ARGSet_neg(ARGList_neg),Para);
            [model,IsChange2]=AddNode(argFeature_pos,argFeature_neg,MaxUnmatchNum_pos,MaxUnmatchNum_neg,model,ARGSet(ARGList),ARGSet_neg(ARGList_neg),Para); % HERE
        else
            IsChange1=false;
            IsChange2=false;
        end
        model_set(model_No)=model;
        if(IsChange1)
            Log(model_No).Delete=Log(model_No).Delete+1;
        end
        if(IsChange2)
           Log(model_No).Insert=Log(model_No).Insert+1;
        end
    end

    load(filename,'record');
    record(Iter+1).model_set=model_set;
    record(Iter+1).Log=Log;
    record(Iter+1).GoodARGList=GoodARGList_iter;
    record(Iter+1).Accuracy=Accuracy;
    save(filename,'record');
    clear record
end
load(filename,'record');
time=toc;
save(filename,'record','time');


function [unaryF,valid]=ReDivideTerminals_newNode(DummyModel,ARGSet_pos,ARGSet_neg,match_pos,match_neg,typeNo,Para)
argNum_pos=length(ARGSet_pos);
argNum_neg=length(ARGSet_neg);
nodeNum=DummyModel.nodeNum;
unaryNum=length(DummyModel.NodeType(typeNo).unaryW);
NodeInfo.count=0;
NodeInfo.unaryF(unaryNum).f=[];
for atri=1:unaryNum
    NodeInfo.unaryF(atri).f=zeros(size(DummyModel.NodeInfo(nodeNum).unaryF(atri).f,1),argNum_pos);
end
for argNo=1:argNum_pos
    TheX=match_pos(nodeNum,argNo);
    if(TheX<=ARGSet_pos(argNo).arg.nodeNum)
        Attribute=getAttribute(TheX,TheX,ARGSet_pos(argNo).arg);
        argTypeIdx=getNodeType(DummyModel.NodeType(typeNo).type,Attribute.NodeType);
        NodeInfo.count=NodeInfo.count+1;
        for atri=1:unaryNum
            NodeInfo.unaryF(atri).f(:,NodeInfo.count)=Attribute.NodeTypeInfo(argTypeIdx).unaryF(atri).f;
        end
    end
end
for atri=1:unaryNum
    NodeInfo.unaryF(atri).f=NodeInfo.unaryF(atri).f(:,1:NodeInfo.count);
end
count_neg=0;
for argNo=1:argNum_neg
    count_neg=count_neg+(match_neg(nodeNum,argNo)<=ARGSet_neg(argNo).arg.nodeNum);
end
if(NodeInfo.count>0)
    unaryF=getTerminalDivision(NodeInfo.unaryF,argNum_pos,argNum_neg,NodeInfo.count,count_neg,DummyModel.NodeType(typeNo).unaryW,Para);
    valid=true;
else
    unaryF=[];
    valid=false;
end


function model=ReDivideTerminals(model,ARGSet_pos,ARGSet_neg,match_pos,match_neg,Para)
typeNum=length(model.NodeType);
argNum_pos=length(ARGSet_pos);
argNum_neg=length(ARGSet_neg);
NodeInfo(model.nodeNum).unaryF=[];
NodeInfo(model.nodeNum).count=[];
for typeNo=1:typeNum
    unaryNum=length(model.NodeType(typeNo).unaryW);
    nodeList=model.NodeType(typeNo).nodeList;
    if(~isempty(nodeList))
        for i=nodeList
            NodeInfo(i).unaryF(unaryNum).f=[];
            NodeInfo(i).count=0;
            for atri=1:unaryNum
                NodeInfo(i).unaryF(atri).f=zeros(size(model.NodeInfo(i).unaryF(atri).f,1),argNum_pos);
            end
        end
    end
end
for argNo=1:argNum_pos
    x=match_pos(:,argNo)';
    argNodeNum=ARGSet_pos(argNo).arg.nodeNum;
    list=find(x<=argNodeNum);
    TheX=x(list);
    if(size(list,2)>0)
        Attribute=getAttribute(TheX,TheX,ARGSet_pos(argNo).arg);
        for typeNo=1:typeNum
            unaryNum=length(model.NodeType(typeNo).unaryW);
            [nodeList,~]=getIntersectNodeList(model.NodeType(typeNo).nodeList,list);
            argTypeIdx=getNodeType(model.NodeType(typeNo).type,Attribute.NodeType);
            if(~isempty(nodeList))
                i=0;
                for nodeNo=nodeList
                    i=i+1;
                    NodeInfo(nodeNo).count=NodeInfo(nodeNo).count+1;
                    for atri=1:unaryNum
                        NodeInfo(nodeNo).unaryF(atri).f(:,NodeInfo(nodeNo).count)=Attribute.NodeTypeInfo(argTypeIdx).unaryF(atri).f(:,i);
                    end
                end
            end
        end
    end
end
for typeNo=1:typeNum
    unaryNum=length(model.NodeType(typeNo).unaryW);
    nodeList=model.NodeType(typeNo).nodeList;
    if(~isempty(nodeList))
        for nodeNo=nodeList
            for atri=1:unaryNum
                NodeInfo(nodeNo).unaryF(atri).f=NodeInfo(nodeNo).unaryF(atri).f(:,1:NodeInfo(nodeNo).count);
            end
        end
    end
end
count_neg=zeros(model.nodeNum,1);
for argNo=1:argNum_neg
    x=match_neg(:,argNo)';
    argNodeNum=ARGSet_neg(argNo).arg.nodeNum;
    list=find(x<=argNodeNum);
    count_neg(list)=count_neg(list)+1;
end
for typeNo=1:typeNum
    nodeList=model.NodeType(typeNo).nodeList;
    if(~isempty(nodeList))
        for nodeNo=nodeList
            if(NodeInfo(nodeNo).count>0)
                model.NodeInfo(nodeNo).unaryF=getTerminalDivision(NodeInfo(nodeNo).unaryF,argNum_pos,argNum_neg,NodeInfo(nodeNo).count,count_neg(nodeNo),model.NodeType(typeNo).unaryW,Para);
            end
        end
    end
end


function GoodARGList_neg=getARGList_neg(model_neg,ARGSet_neg,Para)
Para.GoodMatchRate=-1;
Iter=Para.MaxIterationNum;
Para.StartMaxARGNumPerIteration=1.0;
Para.MaxARGNumPerIteration=1.0;
GoodARGList_neg=selectGoodARGs(ARGSet_neg,model_neg,Para,Iter);


function b=getB(model,ARGSet,ARGSet_neg,Para,IsFlip)
LargeNum=100000000000000000000000000;
IsShown=0;
[match,TheF1,TheF2,~]=getMatch(model,ARGSet,IsShown,Para);
[energy_pos,MatchRate]=getNormalizedMatchEnergy_perARG(ARGSet,match,TheF1,TheF2,Para,model.nodeNum,model.link,model.b,[]);
energy_pos(MatchRate<Para.GoodMatchRate)=LargeNum;
energy_pos=energy_pos(getFlipIndex(energy_pos,ARGSet,IsFlip)); % deal with flip
[match,TheF1,TheF2,~]=getMatch(model,ARGSet_neg,IsShown,Para);
[energy_neg,MatchRate]=getNormalizedMatchEnergy_perARG(ARGSet_neg,match,TheF1,TheF2,Para,model.nodeNum,model.link,model.b,[]);
energy_neg(MatchRate<Para.GoodMatchRate)=LargeNum;
num_pos=length(energy_pos);
num_neg=length(energy_neg);
energy_list=[energy_pos;energy_neg];
label_list=[ones(num_pos,1);zeros(num_neg,1)];
[energy_list,order]=sort(energy_list,'ascend');
label_list=label_list(order);
ListLen=length(label_list);
label_list_tmp=label_list(energy_list<LargeNum);
num_pos_tmp=sum(label_list_tmp==1);
num_neg_tmp=sum(label_list_tmp==0);
TheB=0;
MaxAccuracy=-1;
count=0;
for i=1:ListLen
    if(label_list(i)==1)
        count=count+1;
        if((i<ListLen)&&(energy_list(i+1)<LargeNum)&&(MaxAccuracy<(count/num_pos_tmp+(count+num_neg_tmp-i)/num_neg_tmp)/2))
            MaxAccuracy=(count/num_pos_tmp+(count+num_neg_tmp-i)/num_neg_tmp)/2;
            TheB=-(energy_list(i)+energy_list(i+1))/2;
        end
        if(count==num_pos)
            break;
        end
    end
end
if(num_pos_tmp*num_neg_tmp==0)
    TheB=0;
end
b=model.b+TheB;


function model=WeightTraining_discriminative(model,ARGSet_pos,ARGSet_neg,Para,TotalModelNum,IsReDivideTerminals)
getAccuracyWeight=@(a,Para)(min(exp(-Para.AW.Gradient*(Para.AW.Roof-(a))),1)*Para.AW.MaxStep);  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
getUpdate=@(mean_pos,mean_neg,PenaltyTrainingBias)((mean_pos+(mean_neg-mean_pos)*PenaltyTrainingBias));
pairwiseNum=size(model.pairwiseF,2);
IsShown=0;
[TheMatch.match_pos,~,~,TheMatch.psi_pos]=getMatch(model,ARGSet_pos,IsShown,Para);
Psi=TheMatch.psi_pos;
model_neg=model;
model_neg.penalty.unmatch=model_neg.penalty.largelarge;
model_neg.penalty.unmatchPair=model_neg.penalty.largelarge;
[TheMatch.match_neg,~,~,TheMatch.psi_neg]=getMatch(model_neg,ARGSet_neg,IsShown,Para);
clear model_neg
[feature_pos,MajorLocalFeature_pos]=getAttributeSquareDistFeature_discriminative(TheMatch.match_pos,TheMatch.psi_pos,model,ARGSet_pos,Para);
[feature_neg,MajorLocalFeature_neg]=getAttributeSquareDistFeature_discriminative(TheMatch.match_neg,TheMatch.psi_neg,model,ARGSet_neg,Para);
%save('DAP_dis.mat','feature_pos','feature_neg','MajorLocalFeature_pos','MajorLocalFeature_neg');
argNum_pos=length(ARGSet_pos);
argNum_neg=length(ARGSet_neg);
feature_pos.pairwiseType.unaryResponse.feature=zeros(1,argNum_pos);
feature_neg.pairwiseType.unaryResponse.feature=zeros(1,argNum_neg);
feature_pos.pairwiseType.unaryResponse.valid=zeros(argNum_pos,1);
feature_neg.pairwiseType.unaryResponse.valid=zeros(argNum_neg,1);
typeNum=length(model.NodeType);
for typeNo=1:typeNum
    unaryNum=length(model.NodeType(typeNo).unaryW);
    nodeList=model.NodeType(typeNo).nodeList;
    f_pos=feature_pos.unaryType(typeNo).feature(:,feature_pos.unaryType(typeNo).valid==1);
    f_neg=feature_neg.unaryType(typeNo).feature(:,feature_neg.unaryType(typeNo).valid==1);
    num_pos=size(f_pos,2);
    num_neg=size(f_neg,2);
    if((num_pos>0)&&(num_neg>0))
        [w,accuracy,TheMatrix]=TrainSVMClassifier(num_pos,num_neg,f_pos,f_neg,1,0);
        TheW=max(-w,0);
        if(sum(TheW)<1e-15)
            TheW=ones(size(TheW));
        end
        NormTerm=norm(TheW'*f_pos);
        if(NormTerm>0)
            TheW=TheW./NormTerm;
        end
        NormTerm=norm(model.NodeType(typeNo).unaryW'*f_pos);
        if(Para.LearningItems.IsWeightTraining&&(NormTerm>0))
            ThePreW=model.NodeType(typeNo).unaryW./NormTerm;
        else
            ThePreW=model.NodeType(typeNo).unaryW;
        end
        if(Para.LearningItems.IsWeightTraining_.IsTrainAttributeWeight)
            AW=getAccuracyWeight(accuracy,Para);
            for atri=1:unaryNum
                model.NodeType(typeNo).unaryW(atri)=getUpdate(ThePreW(atri),TheW(atri),AW);
            end
        end
        TheW=model.NodeType(typeNo).unaryW;
        feature_pos.pairwiseType.unaryResponse.feature(feature_pos.unaryType(typeNo).valid==1)=TheW'*f_pos;
        feature_neg.pairwiseType.unaryResponse.feature(feature_neg.unaryType(typeNo).valid==1)=TheW'*f_neg;
        feature_pos.pairwiseType.unaryResponse.valid=max(feature_pos.pairwiseType.unaryResponse.valid,feature_pos.unaryType(typeNo).valid);
        feature_neg.pairwiseType.unaryResponse.valid=max(feature_neg.pairwiseType.unaryResponse.valid,feature_neg.unaryType(typeNo).valid);
        if(Para.LearningItems.IsTrainingPenalty)
            value_unary=TheMatrix*TheW;
            mean_pos=mean(value_unary(1:num_pos));
            mean_neg=max(mean_pos,mean(value_unary(num_pos+(1:num_neg))));
            model.NodeType(typeNo).penalty.unmatch=getUpdate(mean_pos,mean_neg,Para.PenaltyTrainingBias);
        end
        if(Para.LearningItems.IsWeightTraining&&(Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri||Para.LearningItems.IsWeightTraining_.IsTrainCoreSVMForFirstLocalAtri))
            for nodeNo=nodeList
                for p=1:max(MajorLocalFeature_pos(nodeNo).psi)
                    list_pos=find(MajorLocalFeature_pos(nodeNo).psi==p);
                    list_neg=find(MajorLocalFeature_neg(nodeNo).psi==p);
                    num_pos=length(list_pos);
                    num_neg=length(list_neg);
                    if((num_pos>0)&&(num_neg>0))
                        if(Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri)
                            TheF_pos=MajorLocalFeature_pos(nodeNo).matrix(:,list_pos);
                            TheF_neg=MajorLocalFeature_neg(nodeNo).matrix(:,list_neg);
                            [w,accuracy,~]=TrainSVMClassifier(num_pos,num_neg,TheF_pos,TheF_neg,1,0);
                            TheW=-w./mean(abs(w));
                            if(IsReDivideTerminals)
                                model.NodeInfo(nodeNo).unaryF(1).DiscriminativeW(:,p)=TheW;
                            else
                                AW=getAccuracyWeight(accuracy,Para);
                                model.NodeInfo(nodeNo).unaryF(1).DiscriminativeW(:,p)=getUpdate(model.NodeInfo(nodeNo).unaryF(1).DiscriminativeW(:,p),TheW,AW);
                                model.NodeInfo(nodeNo).unaryF(1).DiscriminativeW(:,p)=model.NodeInfo(nodeNo).unaryF(1).DiscriminativeW(:,p)./mean(abs(model.NodeInfo(nodeNo).unaryF(1).DiscriminativeW(:,p)));
                            end
                        end
                        if(Para.LearningItems.IsWeightTraining_.IsTrainCoreSVMForFirstLocalAtri)
                            atri_pos=MajorLocalFeature_pos(nodeNo).atri(:,list_pos);
                            atri_neg=MajorLocalFeature_neg(nodeNo).atri(:,list_neg);
                            svmmodel=TrainSVMClassifier(num_pos,num_neg,atri_pos,atri_neg,1,1);
                            if(isempty(svmmodel))
                                error('abc\n');
                            end
                            model.NodeInfo(nodeNo).unaryF(1).CoreSVM(p).model=svmmodel;
                        end
                    end
                end
            end
        end
    end
end
list_pos=find(feature_pos.pairwiseType.valid.*feature_pos.pairwiseType.unaryResponse.valid==1);
list_neg=find(feature_neg.pairwiseType.valid.*feature_neg.pairwiseType.unaryResponse.valid==1);
f_pos=[feature_pos.pairwiseType.unaryResponse.feature(list_pos);feature_pos.pairwiseType.feature(:,list_pos)];
f_neg=[feature_neg.pairwiseType.unaryResponse.feature(list_neg);feature_neg.pairwiseType.feature(:,list_neg)];
num_pos=size(f_pos,2);
num_neg=size(f_neg,2);
if((num_pos>0)&&(num_neg>0))
    if(Para.LearningItems.IsWeightTraining_.IsProtectLastPairwiseWFromTraining)
        [w,accuracy,~]=TrainSVMClassifier(num_pos,num_neg,f_pos(1:end-1,:),f_neg(1:end-1,:),1,0);
        TheMatrix=[f_pos';f_neg'];
        TheW=max(-w,0);
        if(sum(TheW)<1e-15)
            TheW=ones(size(TheW));
        end
        TheW=TheW.*(length(TheW)/sum(TheW));
        TheW=[TheW;1];
    else
        [w,accuracy,TheMatrix]=TrainSVMClassifier(num_pos,num_neg,f_pos,f_neg,1,0);
        TheW=max(-w,0);
        if(sum(TheW)<1e-15)
            TheW=ones(size(TheW));
        end
        TheW=TheW.*(length(TheW)/sum(TheW));
    end
    %TheW
    %pause
    if(Para.LearningItems.IsWeightTraining_.IsTrainAttributeWeight&&(TheW(1)>0))
        AW=getAccuracyWeight(accuracy,Para);
        for typeNo=1:typeNum
            if(~isempty(model.NodeType(typeNo).nodeList))
                model.NodeType(typeNo).unaryW=model.NodeType(typeNo).unaryW.*TheW(1);
                model.NodeType(typeNo).penalty.unmatch=model.NodeType(typeNo).penalty.unmatch*TheW(1);
            else
                model.NodeType(typeNo).unaryW=ones(size(model.NodeType(typeNo).unaryW));
                model.NodeType(typeNo).penalty.unmatch=model.penalty.large;
            end
        end
        for atri=1:pairwiseNum
            fNo=atri+1;
            model.pairwiseF(atri).w=getUpdate(model.pairwiseF(atri).w,TheW(fNo),AW);
        end
    else
        TheW=zeros(pairwiseNum+1,1);
        for atri=1:pairwiseNum
            fNo=atri+1;
            TheW(fNo)=model.pairwiseF(atri).w;
        end
    end
    if(Para.LearningItems.IsTrainingPenalty)
        value_pairwise=TheMatrix(:,1+(1:pairwiseNum))*TheW(1+(1:pairwiseNum));
        mean_pos=mean(value_pairwise(1:num_pos));
        mean_neg=max(mean_pos,mean(value_pairwise(num_pos+(1:num_neg))));
        model.penalty.unmatchPair=getUpdate(mean_pos,mean_neg,Para.PenaltyTrainingBias);
    end
end


function [w,accuracy,TheMatrix]=TrainSVMClassifier(num_pos,num_neg,feature_pos,feature_neg,TotalModelNum,IsTrainCoreSVMForFirstLocalAtri)
% if(TotalModelNum==1)
%     NegWeight=num_pos/num_neg;
% else
%     NegWeight=1;
% end
% PosWeight=1;

PosWeight=1/num_pos;
NegWeight=1/num_neg;

if(IsTrainCoreSVMForFirstLocalAtri)
    TheStr=sprintf('-s 0 -t 2 -w1 %f -w-1 %f -c 100',PosWeight,NegWeight);
else
    TheStr=sprintf('-s 0 -t 0 -w1 %f -w-1 %f -c 0.01',PosWeight,NegWeight);
end
TheLabel=[ones(num_pos,1);-ones(num_neg,1)];
TheMatrix=[feature_pos';feature_neg'];
tmp=[ones(num_pos,1);ones(num_neg,1).*NegWeight];
accuracy=-1;
for i=1:10
    SVMModel_tmp=svmtrain(TheLabel,TheMatrix,TheStr);
    [predict_label,~,~]=svmpredict(TheLabel,TheMatrix,SVMModel_tmp);
    accuracy_tmp=sum((TheLabel==predict_label).*tmp)/sum(tmp);
    if(accuracy_tmp>accuracy)
        accuracy=accuracy_tmp;
        SVMModel=SVMModel_tmp;
    end
end
if(IsTrainCoreSVMForFirstLocalAtri)
    w=SVMModel;
else
    w=SVMModel.SVs'*SVMModel.sv_coef;
end


function [feature,MajorLocalFeature]=getAttributeSquareDistFeature_discriminative(match,psi,model,ARGSet,Para)
arg_num=size(match,2);
MajorLocalFeature(model.nodeNum).matrix=[];
MajorLocalFeature(model.nodeNum).atri=[];
MajorLocalFeature(model.nodeNum).valid=[];
typeNum=length(model.NodeType);
feature.unaryType(typeNum).feature=[];
for typeNo=1:typeNum
    unaryNum=length(model.NodeType(typeNo).unaryW);
    nodeList=model.NodeType(typeNo).nodeList;
    if(isempty(nodeList))
        continue;
    end
    local_dim=size(model.NodeInfo(nodeList(1)).unaryF(1).f,1);
    for node=nodeList
        MajorLocalFeature(node).matrix=zeros(local_dim,arg_num);
        MajorLocalFeature(node).atri=zeros(local_dim,arg_num);
        MajorLocalFeature(node).psi=zeros(arg_num,1);
    end
    feature_unary=zeros(unaryNum,arg_num);
    valid_unary=zeros(arg_num,1);
    for argNo=1:arg_num
        TheARG=ARGSet(argNo).arg;
        x=match(nodeList,argNo)';
        p=psi(nodeList,argNo)';
        list=find(x<=TheARG.nodeNum);
        TheX=x(list);
        TheP=p(list);
        validNum=size(list,2);
        if(validNum<2)
            continue;
        end
        for atri=1:unaryNum
            fNo=atri;
            feature_unary(fNo,argNo)=0;
            for nodeID=1:validNum
                node=nodeList(list(nodeID));
                theAtri=TheARG.NodeTypeInfo(typeNo).unaryF(atri).f(:,TheARG.NodeType(typeNo).nodeList==TheX(nodeID));
                tmptmp=(model.NodeInfo(node).unaryF(atri).f(:,TheP(nodeID))-theAtri).^2;
                if((atri==1)&&(Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri))
                    feature_unary(fNo,argNo)=feature_unary(fNo,argNo)+sum(tmptmp.*model.NodeInfo(node).unaryF(atri).DiscriminativeW(:,TheP(nodeID)),1);
                else
                    feature_unary(fNo,argNo)=feature_unary(fNo,argNo)+sum(tmptmp,1);
                end
                if(atri==1)
                    MajorLocalFeature(node).matrix(:,argNo)=tmptmp;
                    MajorLocalFeature(node).atri(:,argNo)=theAtri;
                    MajorLocalFeature(node).psi(argNo)=TheP(nodeID);
                end
            end
            feature_unary(fNo,argNo)=feature_unary(fNo,argNo)/validNum;
        end
        valid_unary(argNo)=1;
    end
    feature.unaryType(typeNo).feature=feature_unary;
    feature.unaryType(typeNo).valid=valid_unary;
end
pairwiseNum=length(model.pairwiseF);
feature_pairwise=zeros(pairwiseNum,arg_num);
valid_pairwise=zeros(arg_num,1);
for argNo=1:arg_num
    TheARG=ARGSet(argNo).arg;
    x=match(:,argNo)';
    list=find(x<=TheARG.nodeNum);
    TheX=x(list);
    validNum=size(list,2);
    if(validNum<2)
        continue;
    end
    list_2DIndex=zeros(validNum*(validNum-1)/2,2);
    weight_2DIndex=zeros(validNum*(validNum-1)/2,1);
    tmp_c=0;
    for i=1:validNum-1
        for j=i+1:validNum
            tmp_c=tmp_c+1;
            list_2DIndex(tmp_c,1)=model.nodeNum*(list(j)-1)+list(i);
            list_2DIndex(tmp_c,2)=TheARG.nodeNum*(TheX(j)-1)+TheX(i);
            %weight_2DIndex(tmp_c)=(model.link(list(i)).to(list(j))/sum(model.link(list(i)).to(list))+model.link(list(j)).to(list(i))/sum(model.link(list(j)).to(list)))/2;
            weight_2DIndex(tmp_c)=model.link(list(i)).to(list(j))/sum(model.link(list(i)).to(list))+model.link(list(j)).to(list(i))/sum(model.link(list(j)).to(list));  % edges (s,t) and (t,s) are counted as two different edges
        end
    end
    for atri=1:pairwiseNum
        fNo=atri;
        model_f=reshape(model.pairwiseF(atri).f,[size(model.pairwiseF(atri).f,1),model.nodeNum^2]);
        arg_f=reshape(TheARG.pairwiseF(atri).f,[size(TheARG.pairwiseF(atri).f,1),TheARG.nodeNum^2]);
        feature_pairwise(fNo,argNo)=(sum((model_f(:,list_2DIndex(:,1))-arg_f(:,list_2DIndex(:,2))).^2,1)*weight_2DIndex)/validNum;
    end
    valid_pairwise(argNo)=1;
end
feature.pairwiseType.feature=feature_pairwise;
feature.pairwiseType.valid=valid_pairwise;


function model=ModelAttributeEstimation(model,ARGSet,match,Para,Psi,IsReDivideTerminals)
TheModel=model;
nodeNum=TheModel.nodeNum;
if(~IsReDivideTerminals)
    typeNum=length(TheModel.NodeType);
    for typeNo=1:typeNum
        unaryNum=length(TheModel.NodeType(typeNo).unaryW);
        nodeList=TheModel.NodeType(typeNo).nodeList;
        if(~isempty(nodeList))
            for i=nodeList
                for atri=1:unaryNum
                    model.NodeInfo(i).unaryF(atri).f=zeros(size(TheModel.NodeInfo(i).unaryF(atri).f));
                end
            end
        end
    end
    TerminalNum=getTerminalNum(TheModel);
    UnaryCount=zeros(nodeNum,max(TerminalNum));
end
pairwiseNum=length(model.pairwiseF);
for atri=1:pairwiseNum
    model.pairwiseF(atri).f=zeros(size(TheModel.pairwiseF(atri).f));
end
PairwiseCount=zeros(1,nodeNum,nodeNum);
argNum=size(ARGSet,2);
for argNo=1:argNum
    x=match(:,argNo)';
    argNodeNum=ARGSet(argNo).arg.nodeNum;
    list=find(x<=argNodeNum);
    TheX=x(list);
    if(size(list,2)>0)
        Attribute=getAttribute(TheX,TheX,ARGSet(argNo).arg);
        for atri=1:pairwiseNum
            model.pairwiseF(atri).f(:,list,list)=model.pairwiseF(atri).f(:,list,list)+Attribute.pairwiseF(atri).f;
        end
        PairwiseCount(:,list,list)=PairwiseCount(:,list,list)+1;
        if(~IsReDivideTerminals)
            the_psi=Psi(:,argNo)';
            for typeNo=1:typeNum
                unaryNum=length(TheModel.NodeType(typeNo).unaryW);
                [nodeList,~]=getIntersectNodeList(TheModel.NodeType(typeNo).nodeList,list);
                argTypeIdx=getNodeType(TheModel.NodeType(typeNo).type,Attribute.NodeType);
                if(size(Attribute.NodeTypeInfo(argTypeIdx).unaryF(1).f,2)~=length(nodeList))
                    error('here_1');
                end
                if(length(nodeList)>1)
                    if(sum(nodeList(2:end)-nodeList(1:end-1)<0)>0)
                        error('here_2');
                    end
                end
                if(~isempty(nodeList))
                    i=0;
                    for nodeNo=nodeList
                        i=i+1;
                        for atri=1:unaryNum
                            model.NodeInfo(nodeNo).unaryF(atri).f(:,the_psi(nodeNo))=model.NodeInfo(nodeNo).unaryF(atri).f(:,the_psi(nodeNo))+Attribute.NodeTypeInfo(argTypeIdx).unaryF(atri).f(:,i);
                        end
                        UnaryCount(nodeNo,the_psi(nodeNo))=UnaryCount(nodeNo,the_psi(nodeNo))+1;
                    end
                end
            end
        end
    end
end
if(~IsReDivideTerminals)
    UnaryCount_ori=UnaryCount;
    UnaryCount=max(UnaryCount,1);
    for typeNo=1:typeNum
        unaryNum=length(TheModel.NodeType(typeNo).unaryW);
        nodeList=TheModel.NodeType(typeNo).nodeList;
        if(~isempty(nodeList))
            for i=nodeList
                for atri=1:unaryNum
                    model.NodeInfo(i).unaryF(atri).f=model.NodeInfo(i).unaryF(atri).f./repmat(UnaryCount(i,1:TerminalNum(i)),[size(model.NodeInfo(i).unaryF(atri).f,1),1]);
                end
                % clear empty terminals
                tmp=find(UnaryCount_ori(i,1:TerminalNum(i))>0);
                if(isempty(tmp))
                    tmp=1;
                end
                for atri=1:unaryNum
                    model.NodeInfo(i).unaryF(atri).f=model.NodeInfo(i).unaryF(atri).f(:,tmp);
                end
                if(Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri)
                    model.NodeInfo(i).unaryF(1).DiscriminativeW=model.NodeInfo(i).unaryF(1).DiscriminativeW(:,tmp);
                end
            end
        end
    end
end
PairwiseCount=max(PairwiseCount,1);
for atri=1:pairwiseNum
    model.pairwiseF(atri).f=model.pairwiseF(atri).f./repmat(PairwiseCount,[size(model.pairwiseF(atri).f,1),1,1]);
end


function [TheModel,IsChange,argFeature_pos,argFeature_neg,MaxUnmatchNum_pos,MaxUnmatchNum_neg]=EliminateNode(TheModel,ARGSet_pos,ARGSet_neg,Para)
% compute the energy of positive matches
N_pos=size(ARGSet_pos,2);
argFeature_pos.TheX=zeros(TheModel.nodeNum,N_pos);
argFeature_pos.argNodeNum=zeros(N_pos,1);
MaxUnmatchNum_pos=0;
IsShown=0;
[match_pos,TheF1_pos,TheF2_pos]=getMatch(TheModel,ARGSet_pos,IsShown,Para);
[UnaryPenalty_pos,AvgPairwisePenalty_pos,RawPairwisePenalty_pos]=getPenaltyFromMatch(ARGSet_pos,match_pos,TheF1_pos,TheF2_pos,Para,TheModel.nodeNum,[]);
for argNo=1:N_pos
    x=match_pos(:,argNo)';
    argNodeNum=ARGSet_pos(argNo).arg.nodeNum;
    TheX=x(x<=argNodeNum);
    x(x>argNodeNum)=-1;
    argFeature_pos.TheX(:,argNo)=x';
    argFeature_pos.argNodeNum(argNo)=argNodeNum;
    MaxUnmatchNum_pos=max(MaxUnmatchNum_pos,argNodeNum-length(TheX));
end
TotalPenalty_pos=UnaryPenalty_pos+AvgPairwisePenalty_pos;
disp([TotalPenalty_pos,UnaryPenalty_pos,AvgPairwisePenalty_pos]);
% modify the connections
TheModel.link=SetLinkage(Para,TheModel.nodeNum,RawPairwisePenalty_pos);
% compute the energy of negative matches
N_neg=size(ARGSet_neg,2);
argFeature_neg.TheX=zeros(TheModel.nodeNum,N_neg);
argFeature_neg.argNodeNum=zeros(N_neg,1);
MaxUnmatchNum_neg=0;
IsShown=0;
[match_neg,TheF1_neg,TheF2_neg]=getMatch(TheModel,ARGSet_neg,IsShown,Para); % negative matches with the original parameters; used for node delete
[UnaryPenalty_neg,AvgPairwisePenalty_neg,~]=getPenaltyFromMatch(ARGSet_neg,match_neg,TheF1_neg,TheF2_neg,Para,TheModel.nodeNum,TheModel.link);
TotalPenalty_neg=UnaryPenalty_neg+AvgPairwisePenalty_neg;
disp([TotalPenalty_neg,UnaryPenalty_neg,AvgPairwisePenalty_neg]);
% negative matches with the infinite penalties for matching to none; used for further node disoovery
TheModel_neg=TheModel;
TheModel_neg.penalty.unmatchPair=TheModel_neg.penalty.large;
typeNum=length(TheModel_neg.NodeType);
for typeNo=1:typeNum
    TheModel_neg.NodeType(typeNo).penalty.unmatch=TheModel_neg.penalty.large;
end
[match_neg_infinite,~,~]=getMatch(TheModel_neg,ARGSet_neg,IsShown,Para);
clear TheModel_neg
for argNo=1:N_neg
    x=match_neg_infinite(:,argNo)';
    argNodeNum=ARGSet_neg(argNo).arg.nodeNum;
    TheX=x(x<=argNodeNum);
    x(x>argNodeNum)=-1;
    argFeature_neg.TheX(:,argNo)=x';
    argFeature_neg.argNodeNum(argNo)=argNodeNum;
    MaxUnmatchNum_neg=max(MaxUnmatchNum_neg,argNodeNum-length(TheX));    
end
% delete the worst node based on the objective function
ObjectiveFunction=TotalPenalty_pos-TotalPenalty_neg+Para.Lambda.*getTerminalNum(TheModel);
ObjectiveFunction
%pause
[Value,NonMatchNode]=max(ObjectiveFunction);
if((Value>Para.PenaltyThreshold)&&(TheModel.nodeNum>3))
    IsChange=true;
    Remain=setdiff((1:TheModel.nodeNum)',NonMatchNode);
    argFeature_pos.TheX=argFeature_pos.TheX(Remain,:);
    argFeature_neg.TheX=argFeature_neg.TheX(Remain,:);
    MaxUnmatchNum_pos=MaxUnmatchNum_pos+1;
    MaxUnmatchNum_neg=MaxUnmatchNum_neg+1;
    TheModel=NodeEliminate(NonMatchNode,TheModel);
    % modify the connections
    RawPairwisePenalty_pos=RawPairwisePenalty_pos(Remain,Remain);
    TheModel.link=SetLinkage(Para,TheModel.nodeNum,RawPairwisePenalty_pos);
else
    IsChange=false;
end


function [TheModel,IsChange]=AddNode(argFeature_pos,argFeature_neg,MaxUnmatchNum_pos,MaxUnmatchNum_neg,TheModel,ARGSet_pos,ARGSet_neg,Para)
argFeature_pos=getArgFeature(argFeature_pos,TheModel,ARGSet_pos,MaxUnmatchNum_pos);
argFeature_neg=getArgFeature(argFeature_neg,TheModel,ARGSet_neg,MaxUnmatchNum_neg);
%get the attributes and penalty of the new node (the average attribute of the predicted matched nodes in N images)
[Add,x]=MRF_ForNewNode(argFeature_pos,argFeature_neg,TheModel,Para);


% % show the new node detected by function MRF_ForNewNode
% for i=1:size(argFeature_pos.TheX,2)
%     list=find(argFeature_pos.TheX(:,i)~=-1);
%     list2=argFeature_pos.TheX(list,i)';
%     list1=setdiff(1:argFeature_pos.argNodeNum(i),list2);
%     %show_MatchingResult(TheModel.ARGInfo.name,ARGSet_pos(i).arg.ARGInfo.name,list1(x(i)),Para,[]);
%     %show_MatchingResult_animal(ARGSet_pos(i).arg.ARGInfo.name,[list2,list1(x(i))],Para,[],[repmat('m',[1,length(list2)]),'c']);
%     %show_MatchingResult_SIFTImg(ARGSet_pos(i).arg.ARGInfo.name,[list2,list1(x(i))],Para,[],[repmat('m',[1,length(list2)]),'c']);
%     show_MatchingResult_SIVAL(TheModel.ARGInfo.name,ARGSet_pos(i).arg.ARGInfo.name,[list2,list1(x(i))],Para,TheModel.link);
% end


[DummyModel,ObjectiveFunction]=getEnlargedDummyModelAndPenalty(TheModel,Add,ARGSet_pos,ARGSet_neg,Para);
disp('before add node');
TotalPenalty=ObjectiveFunction(end);
disp(ObjectiveFunction);
%pause;
% % add the new node
if(TotalPenalty<=Para.PenaltyThreshold*Para.AddNodeAlpha)
    IsChange=true;
    TheModel=DummyModel;
else
    IsChange=false;
end


function argFeature=getArgFeature(argFeature,TheModel,ARGSet,MaxUnmatchNum)
N=length(ARGSet);
argFeature.ModelNodeNum=TheModel.nodeNum;
argFeature.LabelNum=MaxUnmatchNum;
typeNum=length(TheModel.NodeType);
pairwiseNum=size(TheModel.pairwiseF,2);
argFeature.pairwiseF(pairwiseNum).f=[];
argFeature.pairwiseF_symmetry(pairwiseNum).f=[];
for atri=1:pairwiseNum
    AtriDim=size(TheModel.pairwiseF(atri).f,1);
    argFeature.pairwiseF(atri).f=zeros(AtriDim,TheModel.nodeNum,MaxUnmatchNum,N);
    argFeature.pairwiseF_symmetry(atri).f=zeros(AtriDim,MaxUnmatchNum,TheModel.nodeNum,N);
end
for typeNo=1:typeNum
    unaryNum=length(TheModel.NodeType(typeNo).unaryW);
    argFeature.NodeTypeInfo(typeNo).unaryF(unaryNum).arg(N).f=[];
    argFeature.NodeType(typeNo).arg(N).RemainList=[];
end
argFeature.TargetNodeNum=zeros(N,1);
for argNo=1:N
    list=find(argFeature.TheX(:,argNo)~=-1);
    list2=argFeature.TheX(list,argNo)';
    list1=setdiff(1:argFeature.argNodeNum(argNo),list2);
    Attribute=getAttribute(list1,list2,ARGSet(argNo).arg);
    targetNum=length(list1);
    argFeature.TargetNodeNum(argNo)=targetNum;
    if(~isempty(list))
        Attribute_symmetry=getAttribute(list2,list1,ARGSet(argNo).arg);
        for atri=1:pairwiseNum
            argFeature.pairwiseF(atri).f(:,list,1:targetNum,argNo)=Attribute.pairwiseF(atri).f;
            argFeature.pairwiseF_symmetry(atri).f(:,1:targetNum,list,argNo)=Attribute_symmetry.pairwiseF(atri).f;
        end
        clear Attribute_symmetry
    end
    for typeNo=1:typeNum
        unaryNum=length(TheModel.NodeType(typeNo).unaryW);
        argTypeIdx=getNodeType(TheModel.NodeType(typeNo).type,Attribute.NodeType);
        nodeList=Attribute.NodeType(argTypeIdx).nodeList;
        if(~isempty(nodeList))
            for atri=1:unaryNum
                argFeature.NodeTypeInfo(typeNo).unaryF(atri).arg(argNo).f=Attribute.NodeTypeInfo(argTypeIdx).unaryF(atri).f;
            end
        end
        % find positions of the nodes in the nodeList w.r.t. the list
        % of the unmatched nodes in the ARGs
        [~,tmp]=getIntersectNodeList(list1,nodeList);
        argFeature.NodeType(typeNo).arg(argNo).RemainList=tmp;
    end
end


function [DummyModel,ObjectiveFunction]=getEnlargedDummyModelAndPenalty(TheModel,Add,ARGSet_pos,ARGSet_neg,Para)
DummyModel=TheModel;
clear TheModel
nodeNum=DummyModel.nodeNum+1;
DummyModel.nodeNum=nodeNum;
typeNo=Add.targetType;
DummyModel.NodeType(typeNo).nodeList(end+1)=nodeNum;
DummyModel.NodeInfo(nodeNum).type=DummyModel.NodeType(typeNo).type;
unaryNum=length(DummyModel.NodeType(typeNo).unaryW);
N_pos=length(ARGSet_pos);
N_neg=length(ARGSet_neg);
% first get a new node with a single terminal
dummyPara=Para;
dummyPara.Lambda=0;
DummyModel.NodeInfo(nodeNum).unaryF=getTerminalDivision(Add.NodeTypeInfo(typeNo).unaryF,N_pos,N_neg,N_pos,N_neg,DummyModel.NodeType(typeNo).unaryW,dummyPara);
clear dummyPara
pairwiseNum=size(DummyModel.pairwiseF,2);
for atri=1:pairwiseNum
    tmp=size(DummyModel.pairwiseF(atri).f,1);
    DummyModel.pairwiseF(atri).f(:,nodeNum,1:nodeNum-1)=reshape(Add.pairwiseF_symmetry(atri).f,[tmp,nodeNum-1]);
    DummyModel.pairwiseF(atri).f(:,1:nodeNum-1,nodeNum)=reshape(Add.pairwiseF(atri).f,[tmp,nodeNum-1]);
    DummyModel.pairwiseF(atri).f(:,nodeNum,nodeNum)=0;
end
% estimate the linkage
avgLinkNum=0;
for i=1:DummyModel.nodeNum-1
    avgLinkNum=avgLinkNum+sum(DummyModel.link(i).to);
    DummyModel.link(i).to=[DummyModel.link(i).to;0];
end
avgLinkNum=round(avgLinkNum/(DummyModel.nodeNum-1));
[~,list]=sort(Add.pairwiseF_var+Add.pairwiseF_symmetry_var);
DummyModel.link(nodeNum).to=zeros(nodeNum,1);
DummyModel.link(nodeNum).to(list(1:avgLinkNum))=1;
DummyModel.link(nodeNum).to(nodeNum)=0;
IsShown=0;
[match_pos,TheF1_pos,TheF2_pos]=getMatch(DummyModel,ARGSet_pos,IsShown,Para);
[~,~,RawPairwisePenalty_pos]=getPenaltyFromMatch(ARGSet_pos,match_pos,TheF1_pos,TheF2_pos,Para,DummyModel.nodeNum,DummyModel.link);
DummyModel.link=SetLinkage(Para,nodeNum,RawPairwisePenalty_pos);
% divide terminals for the new node
[match_neg,TheF1_neg,TheF2_neg]=getMatch(DummyModel,ARGSet_neg,IsShown,Para); % negative matches with the original parameters;
[ttmmpp,valid]=ReDivideTerminals_newNode(DummyModel,ARGSet_pos,ARGSet_neg,match_pos,match_neg,typeNo,Para);
if(valid)
    DummyModel.NodeInfo(nodeNum).unaryF=ttmmpp;
end
% compute the final ObjectiveFunction
[match_pos,TheF1_pos,TheF2_pos]=getMatch(DummyModel,ARGSet_pos,IsShown,Para);
[UnaryPenalty_pos,AvgPairwisePenalty_pos,~]=getPenaltyFromMatch(ARGSet_pos,match_pos,TheF1_pos,TheF2_pos,Para,DummyModel.nodeNum,DummyModel.link);
[UnaryPenalty_neg,AvgPairwisePenalty_neg,~]=getPenaltyFromMatch(ARGSet_neg,match_neg,TheF1_neg,TheF2_neg,Para,DummyModel.nodeNum,DummyModel.link);
TotalPenalty_pos=UnaryPenalty_pos+AvgPairwisePenalty_pos;
TotalPenalty_neg=UnaryPenalty_neg+AvgPairwisePenalty_neg;
disp('add node: pos');
disp([TotalPenalty_pos,UnaryPenalty_pos,AvgPairwisePenalty_pos]);
disp('add node: neg');
disp([TotalPenalty_neg,UnaryPenalty_neg,AvgPairwisePenalty_neg]);
ObjectiveFunction=TotalPenalty_pos-TotalPenalty_neg+Para.Lambda.*getTerminalNum(DummyModel);

% % show the matching results with the detected node
% IsShown=1;Para.ThreadNum=1;
% [match,TheF1,TheF2]=getMatch(TheModel,ARGSet,IsShown,Para);
% xg=zeros(1,30);
% for i=1:30
%     list1=setdiff(1:ARGSet(i).arg.nodeNum,match(1:end-1,i));
%     xg(i)=find(list1==match(end,i));
% end
% disp('get good matching');
% pause


function [Add,x]=MRF_ForNewNode(argFeature_pos,argFeature_neg,TheModel,Para)
DiscriminativeWDimMean=1;
typeNum=length(TheModel.NodeType);
N=size(argFeature_pos.pairwiseF(1).f,4);
pairwiseNum=size(argFeature_pos.pairwiseF,2);
TheLarge=TheModel.penalty.large;
TheLargeLarge=TheModel.penalty.largelarge;
LabelNum=argFeature_pos.LabelNum;
V=0:N-1;
Enum=N*(N-1)/2;
E=zeros(2,Enum);
t=0;
for i=0:N-2
    len=N-1-i;
    E(:,t+1:t+len)=[ones(1,len).*i;i+1:i+len];
    t=t+len;
end
f1=zeros(LabelNum,N);
f2=zeros(LabelNum,LabelNum,Enum);
% filling f1
N_neg=size(argFeature_neg.pairwiseF(1).f,4);
for argNo_pos=1:N
    len_pos=argFeature_pos.TargetNodeNum(argNo_pos);
    TheF1Penalty=zeros(len_pos,1);
    for argNo_neg=1:N_neg
        len_neg=argFeature_neg.TargetNodeNum(argNo_neg);
        if(len_neg>0)
            % the unary-attribute terms in f1
            UnaryPenalty=ones(len_pos,len_neg).*(TheLargeLarge);
            for typeNo=1:typeNum
                unaryNum=length(TheModel.NodeType(typeNo).unaryW);
                RemainList_pos=argFeature_pos.NodeType(typeNo).arg(argNo_pos).RemainList;
                RemainList_neg=argFeature_neg.NodeType(typeNo).arg(argNo_neg).RemainList;
                sub_len_pos=length(RemainList_pos);
                sub_len_neg=length(RemainList_neg);
                if(sub_len_pos*sub_len_neg>0)
                    TheTmp=zeros(sub_len_pos,sub_len_neg);
                    for atri=1:unaryNum
                        tmpSize=size(argFeature_pos.NodeTypeInfo(typeNo).unaryF(atri).arg(argNo_pos).f,1);
                        tmpDiff=repmat(reshape(argFeature_pos.NodeTypeInfo(typeNo).unaryF(atri).arg(argNo_pos).f,[tmpSize,sub_len_pos,1]),[1,1,sub_len_neg])-repmat(reshape(argFeature_neg.NodeTypeInfo(typeNo).unaryF(atri).arg(argNo_neg).f,[tmpSize,1,sub_len_neg]),[1,sub_len_pos,1]);
                        if((atri==1)&&(Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri))
                            TheTmp=TheTmp+reshape(sum(tmpDiff.^2,1),[sub_len_pos,sub_len_neg]).*(DiscriminativeWDimMean*TheModel.NodeType(typeNo).unaryW(atri)/(N*N_neg));
                        else
                            TheTmp=TheTmp+reshape(sum(tmpDiff.^2,1),[sub_len_pos,sub_len_neg]).*(TheModel.NodeType(typeNo).unaryW(atri)/(N*N_neg));
                        end
                    end
                    UnaryPenalty(RemainList_pos,RemainList_neg)=TheTmp;
                end
            end
            % the pairwise-attribute terms in f1
            ModelNodeSet_pos=find(argFeature_pos.TheX(:,argNo_pos)'~=-1);
            ModelNodeSet_neg=find(argFeature_neg.TheX(:,argNo_neg)'~=-1);
            ValidModelNodeSet=intersect(ModelNodeSet_pos,ModelNodeSet_neg);
            PairwisePenalty=zeros(len_pos,len_neg,length(ValidModelNodeSet));
            count=0;
            for k=ValidModelNodeSet
                count=count+1;
                CountValidARG=sum(argFeature_pos.TheX(k,:)'~=-1);
                term_NonNone=zeros(len_pos,len_neg);
                for atri=1:pairwiseNum
                    tmpSize=size(argFeature_pos.pairwiseF(atri).f,1);
                    tmpDiff=repmat(reshape(argFeature_pos.pairwiseF(atri).f(:,k,1:len_pos,argNo_pos),[tmpSize,len_pos,1]),[1,1,len_neg])-repmat(reshape(argFeature_neg.pairwiseF(atri).f(:,k,1:len_neg,argNo_neg),[tmpSize,1,len_neg]),[1,len_pos,1]);
                    term_NonNone=term_NonNone+reshape(sum(tmpDiff.^2,1),[len_pos,len_neg]).*(TheModel.pairwiseF(atri).w/(TheModel.nodeNum*N_neg*CountValidARG));
                end
                PairwisePenalty(:,:,count)=term_NonNone;
            end
            TheF1Penalty=TheF1Penalty+min(UnaryPenalty+sum(PairwisePenalty,3),[],2);
        end
    end
    f1(1:len_pos,argNo_pos)=-TheF1Penalty;
    f1(len_pos+1:end,argNo_pos)=TheLargeLarge;
end
% filling f2
t=0;
for i=1:N-1
    for j=i+1:N
        % the unary-attribute terms in f2
        UnaryPenalty=ones(LabelNum,LabelNum).*(TheLargeLarge);
        for typeNo=1:typeNum
            unaryNum=length(TheModel.NodeType(typeNo).unaryW);
            RemainList_i=argFeature_pos.NodeType(typeNo).arg(i).RemainList;
            RemainList_j=argFeature_pos.NodeType(typeNo).arg(j).RemainList;
            sub_len_i=length(RemainList_i);
            sub_len_j=length(RemainList_j);
            if(sub_len_i*sub_len_j>0)
                TheTmp=zeros(sub_len_i,sub_len_j);
                for atri=1:unaryNum
                    tmpSize=size(argFeature_pos.NodeTypeInfo(typeNo).unaryF(atri).arg(i).f,1);
                    tmpDiff=repmat(reshape(argFeature_pos.NodeTypeInfo(typeNo).unaryF(atri).arg(i).f,[tmpSize,sub_len_i,1]),[1,1,sub_len_j])-repmat(reshape(argFeature_pos.NodeTypeInfo(typeNo).unaryF(atri).arg(j).f,[tmpSize,1,sub_len_j]),[1,sub_len_i,1]);
                    if((atri==1)&&(Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri))
                        TheTmp=TheTmp+reshape(sum(tmpDiff.^2,1),[sub_len_i,sub_len_j]).*(DiscriminativeWDimMean*TheModel.NodeType(typeNo).unaryW(atri)/(N^2));
                    else
                        TheTmp=TheTmp+reshape(sum(tmpDiff.^2,1),[sub_len_i,sub_len_j]).*(TheModel.NodeType(typeNo).unaryW(atri)/(N^2));
                    end
                end
                UnaryPenalty(RemainList_i,RemainList_j)=TheTmp;
            end
        end
        % the pairwise-attribute terms in f2
        ModelNodeSet1=find(argFeature_pos.TheX(:,i)'~=-1);
        ModelNodeSet2=find(argFeature_pos.TheX(:,j)'~=-1);
        ValidModelNodeSet=intersect(ModelNodeSet1,ModelNodeSet2);
        PairwisePenalty=ones(LabelNum,LabelNum,TheModel.nodeNum).*TheLarge;
        len_i=argFeature_pos.TargetNodeNum(i);
        len_j=argFeature_pos.TargetNodeNum(j);
        if(~isempty(ValidModelNodeSet))
            for k=ValidModelNodeSet
                CountValidARG=sum(argFeature_pos.TheX(k,:)'~=-1);
                term_NonNone=zeros(len_i,len_j);
                for atri=1:pairwiseNum
                    tmpSize=size(argFeature_pos.pairwiseF(atri).f,1);
                    tmpDiff=repmat(reshape(argFeature_pos.pairwiseF(atri).f(:,k,1:len_i,i),[tmpSize,len_i,1]),[1,1,len_j])-repmat(reshape(argFeature_pos.pairwiseF(atri).f(:,k,1:len_j,j),[tmpSize,1,len_j]),[1,len_i,1]);
                    term_NonNone=term_NonNone+reshape(sum(tmpDiff.^2,1),[len_i,len_j]).*(TheModel.pairwiseF(atri).w*(N+CountValidARG)/(2*N*(CountValidARG^2)));
                end
                PairwisePenalty(1:len_i,1:len_j,k)=term_NonNone;
            end
        end
        nodeset=setdiff(1:TheModel.nodeNum,ValidModelNodeSet);
        if(~isempty(nodeset))
            for k=nodeset
                CountValidARG=sum(argFeature_pos.TheX(k,:)'~=-1);
                term_None=TheModel.penalty.unmatchPair/(N*(N+CountValidARG));
                PairwisePenalty(1:len_i,1:len_j,k)=term_None;
            end
        end
        Transfer=UnaryPenalty+sum(PairwisePenalty,3)./TheModel.nodeNum;
        t=t+1;
        f2(:,:,t)=Transfer;
    end
end
% doing TRW-S
f1=f1-min(min(f1));
x=zeros(N,1);
if(matlabpool('size')>0)
    parfor par=1:1
        tmp=TRWSProcess(V,E,f1,f2,Para);
        x(:,par)=tmp';
    end
else
    x=TRWSProcess(V,E,f1,f2,Para)';
end
x=double(x');
Add=getInfo_NewNode(x,argFeature_pos,TheModel,Para);


function Add=getInfo_NewNode(x,argFeature_pos,TheModel,Para)
[~,~,MaxUnmatchNum_pos,N_pos]=size(argFeature_pos.pairwiseF(1).f);
typeNum=length(TheModel.NodeType);
% get matched unary attributes and their type
Add.NodeTypeInfo(typeNum).unaryF=[];
for typeNo=1:typeNum
    unaryNum=length(TheModel.NodeType(typeNo).unaryW);
    Add.NodeTypeInfo(typeNo).unaryF(unaryNum).f=[];
end
for argNo=1:N_pos
    for tmp_typeNo=1:typeNum
        tar=find(argFeature_pos.NodeType(tmp_typeNo).arg(argNo).RemainList==x(argNo));
        if(~isempty(tar))
            typeNo=tmp_typeNo;
            break;
        end
    end
    unaryNum=length(TheModel.NodeType(typeNo).unaryW);
    for atri=1:unaryNum
        Add.NodeTypeInfo(typeNo).unaryF(atri).f=[Add.NodeTypeInfo(typeNo).unaryF(atri).f,argFeature_pos.NodeTypeInfo(typeNo).unaryF(atri).arg(argNo).f(:,tar)];
    end
end
targetType=-1;
count=0;
for typeNo=1:typeNum
    if(isempty(Add.NodeTypeInfo(typeNo).unaryF))
        continue;
    end
    tmp=size(Add.NodeTypeInfo(typeNo).unaryF(1).f,2);
    if(count<tmp)
        count=tmp;
        targetType=typeNo;
    end
end
Add.targetType=targetType;
for typeNo=1:typeNum
    if(typeNo==targetType)
        continue;
    end
    Add.NodeTypeInfo(typeNo).unaryF=[];
end
% get mean values of the matched pairwise attributes and estimate linkages
Add.pairwiseF_var=zeros(TheModel.nodeNum,1);
Add.pairwiseF_symmetry_var=zeros(TheModel.nodeNum,1);
pairwiseNum=size(argFeature_pos.pairwiseF,2);
for atri=1:pairwiseNum
    theSize=size(TheModel.pairwiseF(atri).f,1);
    Add.pairwiseF(atri).f=zeros(theSize,TheModel.nodeNum);
    Add.pairwiseF_symmetry(atri).f=zeros(theSize,TheModel.nodeNum);
end
NodeRelatedPairPenalty=zeros(TheModel.nodeNum,1);
for i=1:TheModel.nodeNum
    ValidModelNodeSet=find(argFeature_pos.TheX(i,:)~=-1);
    ValidSize=length(ValidModelNodeSet);
    tmp=i+TheModel.nodeNum.*(x(ValidModelNodeSet)-1)+(TheModel.nodeNum*MaxUnmatchNum_pos).*(ValidModelNodeSet-1);
    tmp_symmetry=x(ValidModelNodeSet)+MaxUnmatchNum_pos.*(i-1)+(TheModel.nodeNum*MaxUnmatchNum_pos).*(ValidModelNodeSet-1);
    for atri=1:pairwiseNum
        theSize=size(argFeature_pos.pairwiseF(atri).f);
        tmpF=reshape(argFeature_pos.pairwiseF(atri).f,[theSize(1),theSize(2)*theSize(3)*theSize(4)]);
        Add.pairwiseF(atri).f(:,i)=mean(tmpF(:,tmp),2);
        Add.pairwiseF_var(i)=Add.pairwiseF_var(i)+sum(var(tmpF(:,tmp),0,2),1);
        if(isempty(tmp))
            Add.pairwiseF(atri).f(:,i)=0;
            Add.pairwiseF_var(i)=100000;
        end
        tmpDiff=tmpF(:,tmp)-repmat(Add.pairwiseF(atri).f(:,i),[1,ValidSize]);
        NodeRelatedPairPenalty(i)=NodeRelatedPairPenalty(i)+sum(sum(tmpDiff.^2,1),2)*TheModel.pairwiseF(atri).w/N_pos;
        tmpF=reshape(argFeature_pos.pairwiseF_symmetry(atri).f,[theSize(1),theSize(2)*theSize(3)*theSize(4)]);
        Add.pairwiseF_symmetry(atri).f(:,i)=mean(tmpF(:,tmp_symmetry),2);
        Add.pairwiseF_symmetry_var(i)=Add.pairwiseF_symmetry_var(i)+sum(var(tmpF(:,tmp_symmetry),0,2),1);
        if(isempty(tmp_symmetry))
            Add.pairwiseF_symmetry(atri).f(:,i)=0;
            Add.pairwiseF_symmetry_var(i)=100000;
        end
    end
    NodeRelatedPairPenalty(i)=NodeRelatedPairPenalty(i)+TheModel.penalty.unmatchPair*(N_pos-ValidSize)/N_pos;
end
Add.TheLink=zeros(TheModel.nodeNum+1,1);
[~,list]=sort(NodeRelatedPairPenalty);
LinkNum=getLinkNum(Para,TheModel.nodeNum+1);
Add.TheLink(list(1:LinkNum))=1;


function TheModel=NodeEliminate(Nodes,TheModel)
typeNum=length(TheModel.NodeType);
for typeNo=1:typeNum
    if(~isempty(TheModel.NodeType(typeNo).nodeList))
        [nodeList,list]=setdiff(TheModel.NodeType(typeNo).nodeList,Nodes);
        [~,tmp]=sort(list);
        TheModel.NodeType(typeNo).nodeList=nodeList(tmp);
        tmp2=find(TheModel.NodeType(typeNo).nodeList>Nodes);
        TheModel.NodeType(typeNo).nodeList(tmp2)=TheModel.NodeType(typeNo).nodeList(tmp2)-1;
    end
end
Remain=setdiff((1:TheModel.nodeNum)',Nodes);
TheModel.nodeNum=length(Remain);
TheModel.NodeInfo=TheModel.NodeInfo(Remain);
pairwiseNum=length(TheModel.pairwiseF);
for atri=1:pairwiseNum
    TheModel.pairwiseF(atri).f=TheModel.pairwiseF(atri).f(:,Remain,Remain);
end
TheModel.link=TheModel.link(Remain);
for i=1:TheModel.nodeNum
    TheModel.link(i).to=TheModel.link(i).to(Remain);
end


function Attribute=getAttribute(xList1,xList2,TheARG)
typeNum=length(TheARG.NodeType);
Attribute.NodeTypeInfo(typeNum).unaryF=[];
for argTypeIdx=1:typeNum
    Attribute.NodeType(argTypeIdx).type=TheARG.NodeType(argTypeIdx).type;
    [nodeList,list]=getIntersectNodeList(TheARG.NodeType(argTypeIdx).nodeList,xList1);
    Attribute.NodeType(argTypeIdx).nodeList=nodeList;
    unaryNum=length(TheARG.NodeTypeInfo(argTypeIdx).unaryF);
    Attribute.NodeTypeInfo(argTypeIdx).unaryF(unaryNum).f=[];
    for atri=1:unaryNum
        Attribute.NodeTypeInfo(argTypeIdx).unaryF(atri).f=TheARG.NodeTypeInfo(argTypeIdx).unaryF(atri).f(:,list);
    end
end
pairwiseNum=size(TheARG.pairwiseF,2);
Attribute.pairwiseF(pairwiseNum).f=[];
for atri=1:pairwiseNum
    Attribute.pairwiseF(atri).f=TheARG.pairwiseF(atri).f(:,xList2,xList1);
end


function Evaluation(ModelName,ARGSet,Para,TargetDir)
load(sprintf('%s%s_model.mat',TargetDir,ModelName),'record');
TheModel=record(end).model_set;
clear record
N=size(ARGSet,2);
MatchingRate=zeros(N,1);
DetectionRate=zeros(N,1);
ErrorRate=zeros(N,1);
IsShown=1;
match=getMatch(TheModel,ARGSet,IsShown,Para);
for argNo=1:N
    x=match(:,argNo)';
    
    %GroundTruth=ones(ARGSet(argNo).arg.nodeNum,1);
    %load(sprintf('./mat/GroundTruth/%s_GroundTruth.mat',ARGSet(argNo).arg.ARGInfo.name),'GroundTruth');
    load(sprintf('./DAP_produce_ETHZ/SpatialTexture_edges/mat/%s/%s_GroundTruth.mat',ARGSet(argNo).arg.ARGInfo.name(1:end-3),ARGSet(argNo).arg.ARGInfo.name(end-2:end)),'GroundTruth');
    
    PositiveNodeNum=sum(GroundTruth);
    TargetNodeNum=min(TheModel.nodeNum,PositiveNodeNum);
    x=x(x<=ARGSet(argNo).arg.nodeNum);
    AllDetectNum=size(x,2);
    TrueDetectNum=sum(GroundTruth(x));
    FalseDetectNum=AllDetectNum-sum(GroundTruth(x));
    MatchingRate(argNo)=TrueDetectNum/TheModel.nodeNum;
    DetectionRate(argNo)=sum(GroundTruth(unique(x)))/TargetNodeNum;
    ErrorRate(argNo)=FalseDetectNum/TheModel.nodeNum;
end
AvgMatchingRate=mean(MatchingRate);
AvgDetectionRate=mean(DetectionRate);
AvgErrorRate=mean(ErrorRate);
disp([AvgDetectionRate,AvgErrorRate]);
save(sprintf('%s%s_MatchingPerform.mat',TargetDir,TheModel.ARGInfo.name),'DetectionRate','ErrorRate','AvgDetectionRate','AvgErrorRate','MatchingRate','AvgMatchingRate');
