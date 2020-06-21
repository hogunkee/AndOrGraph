function Accuracy=getRecallPrecision(model_set,ARGSet,ARGSet_neg,Para,IsFlip,IsTrainCoreSVMForFirstLocalAtri,IsShown,GroundTruth)

IsLoadKnowledge=false;

LargeNum=100000000000000000000000000;
modelNum=length(model_set);
firstTime=true;
for model_No=1:modelNum
    if(IsTrainCoreSVMForFirstLocalAtri)
        CoreSVM(model_set(model_No).nodeNum).model=[];
        for i=1:model_set(model_No).nodeNum
            CoreSVM(i).model=model_set(model_No).NodeInfo(i).unaryF(1).CoreSVM.model;
        end
    else
        CoreSVM=[];
    end
    clear match_neg TheF1_neg TheF2_neg match_pos TheF1_pos TheF2_pos
    if(~IsLoadKnowledge)
        IsShown=0;
        [match_neg,TheF1_neg,TheF2_neg]=getMatch(model_set(model_No),ARGSet_neg,IsShown,Para);
        [match_pos,TheF1_pos,TheF2_pos]=getMatch(model_set(model_No),ARGSet,IsShown,Para);
        %save(sprintf('knowledge_%d.mat',model_No),'match_neg','TheF1_neg','TheF2_neg','match_pos','TheF1_pos','TheF2_pos');
    else
        load(sprintf('knowledge_%d.mat',model_No),'match_neg','TheF1_neg','TheF2_neg','match_pos','TheF1_pos','TheF2_pos');
    end
    [energy_neg,MatchRate]=getNormalizedMatchEnergy_perARG(ARGSet_neg,match_neg,TheF1_neg,TheF2_neg,Para,model_set(model_No).nodeNum,model_set(model_No).link,model_set(model_No).b,CoreSVM);
    if(IsTrainCoreSVMForFirstLocalAtri)
        energy_neg=energy_neg+MatchRate.*Para.MatchRateCost;
    else
        %energy_neg(MatchRate<Para.GoodMatchRate)=LargeNum;
        energy_neg=energy_neg+MatchRate.*Para.MatchRateCost;
    end
    MatchRate_neg=MatchRate;
    [energy_pos,MatchRate]=getNormalizedMatchEnergy_perARG(ARGSet,match_pos,TheF1_pos,TheF2_pos,Para,model_set(model_No).nodeNum,model_set(model_No).link,model_set(model_No).b,CoreSVM);
    FlipList=getFlipIndex(energy_pos,ARGSet,IsFlip); % deal with flip
    if(IsTrainCoreSVMForFirstLocalAtri)
        energy_pos=energy_pos+MatchRate.*Para.MatchRateCost;
    else
        %energy_pos(MatchRate<Para.GoodMatchRate)=LargeNum;
        energy_pos=energy_pos+MatchRate.*Para.MatchRateCost;
    end
    energy_pos=energy_pos(FlipList);
    MatchRate_pos=MatchRate(FlipList);
    if(firstTime)
        firstTime=false;
        Accuracy.energy_pos=energy_pos;
        Accuracy.energy_neg=energy_neg;
        Accuracy.label_pos=ones(size(energy_pos));
        Accuracy.label_neg=ones(size(energy_neg));
        Accuracy.MatchRate_pos=MatchRate_pos;
        Accuracy.MatchRate_neg=MatchRate_neg;
        checkGroundTruth=getCheckResults(match_pos,GroundTruth,FlipList);
        matchedFlip=FlipList;
    else
        list_pos=find(Accuracy.energy_pos>energy_pos);
        Accuracy.energy_pos(list_pos)=energy_pos(list_pos);
        Accuracy.label_pos(list_pos)=model_No;
        Accuracy.MatchRate_pos(list_pos)=MatchRate_pos(list_pos);
        list_neg=find(Accuracy.energy_neg>energy_neg);
        Accuracy.energy_neg(list_neg)=energy_neg(list_neg);
        Accuracy.label_neg(list_neg)=model_No;
        Accuracy.MatchRate_neg(list_neg)=MatchRate_neg(list_neg);
        checkGroundTruth(list_pos)=getCheckResults(match_pos,GroundTruth,FlipList(list_pos));
        matchedFlip(list_pos)=FlipList(list_pos);
    end
end
Accuracy.checkGroundTruth=checkGroundTruth;
num_pos=sum(checkGroundTruth);
energy_list=[Accuracy.energy_pos;Accuracy.energy_neg];
label_list=[checkGroundTruth;zeros(length(Accuracy.energy_neg),1)];
[~,order]=sort(energy_list,'ascend');
label_list=label_list(order);
ListLen=length(label_list);
Accuracy.curve=zeros(sum(checkGroundTruth),2);
Accuracy.AP=0;
Accuracy.ClassificationAccuracy=0;
ClassificationAccuracy=0;
count=0;
for i=1:ListLen
    if(label_list(i)==1)
        count=count+1;
        Accuracy.curve(count,:)=[count/num_pos,count/i];
        ClassificationAccuracy=max(ClassificationAccuracy,1-(i+num_pos-2*count)/ListLen);
        if(count==num_pos)
            break;
        end
    end
end
Accuracy.AP=mean(Accuracy.curve(:,2));
Accuracy.ClassificationAccuracy=ClassificationAccuracy;
if(IsShown)
    showDetection(model_set,ARGSet,ARGSet_neg,Accuracy,matchedFlip,Para);
end


function checkGroundTruth=getCheckResults(match,GroundTruth,FlipList)
len=length(FlipList);
checkGroundTruth=ones(len,1);
if(isempty(GroundTruth))
    return;
end
match=match(:,FlipList);
GroundTruth.image

image=GroundTruth.image(FlipList);
MinOverlap=GroundTruth.MinOverlap;
clear GroundTruth FlipList
for i=1:len
    tar=match(:,i);
    tar=tar(tar<=size(image(i).hws,2));
    H=image(i).hws(1,tar);
    W=image(i).hws(2,tar);
    halfS=image(i).hws(3,tar);
    tmp=mean((H+halfS>=image(i).bb(2)).*(H-halfS<=image(i).bb(4)).*(W+halfS>=image(i).bb(1)).*(W-halfS<=image(i).bb(3)));
    checkGroundTruth(i)=(tmp>=MinOverlap);
end


function showDetection(model_set,ARGSet,ARGSet_neg,Accuracy,matchedFlip,Para)
ARGSet=ARGSet(matchedFlip);
modelNum=length(model_set);
Para.ThreadNum=1;
for model_No=1:modelNum
    getMatch(model_set(model_No),ARGSet(Accuracy.label_pos==model_No),1,Para);
end
for model_No=1:modelNum
    getMatch(model_set(model_No),ARGSet_neg(Accuracy.label_neg==model_No),1,Para);
end
