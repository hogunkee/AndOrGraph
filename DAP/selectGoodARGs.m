function GoodARGList=selectGoodARGs(ARGSet,model_set,Para,Iter)
InftyEnergy=10000000000000000000000;
N=length(ARGSet);
TheMaxARGNumPerIteration=round(linspace(Para.StartMaxARGNumPerIteration*N,Para.MaxARGNumPerIteration*N,Para.MaxIterationNum));
TheMaxARGNumPerIteration=TheMaxARGNumPerIteration(min(Iter,Para.MaxIterationNum));
modelNum=length(model_set);
GoodARGList(modelNum).list=[];
GoodARGList(modelNum).match=[];
GoodARGList(modelNum).psi=[];
correspond=getFlipCorrespond(ARGSet,Para.IsFlip);
energy_normalized=ones(modelNum,N).*InftyEnergy;
matchRecord(modelNum).match=[];
matchRecord(modelNum).psi=[];
IsShown=0;
for model_No=1:modelNum
    model=model_set(model_No);
    [match,TheF1,TheF2,psi]=getMatch(model,ARGSet,IsShown,Para);
    [energy_tmp,MatchRate]=getNormalizedMatchEnergy_perARG(ARGSet,match,TheF1,TheF2,Para,model.nodeNum,model.link,model.b,[]);
    energy_tmp(MatchRate<Para.GoodMatchRate)=InftyEnergy;
    list=getFlipIndex(energy_tmp,ARGSet,Para.IsFlip);
    energy_normalized(model_No,list)=energy_tmp(list)';
    matchRecord(model_No).match=match;
    matchRecord(model_No).psi=psi;
end
count=zeros(modelNum,1);
ARG_Assigned=zeros(N,1);
energy_normalized=reshape(energy_normalized,[modelNum*N,1]);
model_list=repmat((1:modelNum)',[N,1]);
ARG_list=reshape(repmat((1:N),[modelNum,1]),[modelNum*N,1]);
[energy_normalized,order]=sort(energy_normalized,'ascend');
model_list=model_list(order);
ARG_list=ARG_list(order);
for i=1:length(ARG_list)
    if(energy_normalized(i)>=InftyEnergy)
        break;
    end
    if(ARG_Assigned(ARG_list(i)))
        continue;
    end
    if(count(model_list(i))==TheMaxARGNumPerIteration)
        ARG_Assigned(ARG_list(i))=true; %prevent the small sub-categories from shifting to large sub-categories
        continue;
    end
    GoodARGList(model_list(i)).list=[GoodARGList(model_list(i)).list,ARG_list(i)];
    GoodARGList(model_list(i)).match=[GoodARGList(model_list(i)).match,matchRecord(model_list(i)).match(:,ARG_list(i))];
    GoodARGList(model_list(i)).psi=[GoodARGList(model_list(i)).psi,matchRecord(model_list(i)).psi(:,ARG_list(i))];
    ARG_Assigned(ARG_list(i))=true;
    if(correspond(ARG_list(i))>0)
        ARG_Assigned(correspond(ARG_list(i)))=true; % prevent the same ARG from being represented by two sub-categories
    end
    count(model_list(i))=count(model_list(i))+1;
end

