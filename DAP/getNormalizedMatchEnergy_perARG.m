function [energy,MatchRate]=getNormalizedMatchEnergy_perARG(ARGSet,match,TheF1,TheF2,Para,ModelNodeNum,TheLink,b,CoreSVM)
N=size(ARGSet,2);
energy=zeros(N,1);
MatchRate=zeros(N,1);
for argNo=1:N
    argNodeNum=ARGSet(argNo).arg.nodeNum;
    RawX=match(:,argNo)';
    ValidList=find(RawX<=argNodeNum);
    MatchRate(argNo)=length(ValidList)/ModelNodeNum;
    if(isempty(CoreSVM))
        tmp=(0:ModelNodeNum-1).*(argNodeNum+1)+RawX;
        UnaryEnergy=sum(TheF1(argNo).matrix(tmp));
        PairwisePenalty=zeros(ModelNodeNum,ModelNodeNum);
        t=0;
        for i=1:ModelNodeNum-1
            for j=i+1:ModelNodeNum
                t=t+1;
                tmp=TheF2(argNo).matrix(RawX(i),RawX(j),t);
                PairwisePenalty(i,j)=tmp;
                PairwisePenalty(j,i)=tmp;
            end
        end
        TotalPairwisePenalty=0;
        for i=1:ModelNodeNum
            TotalPairwisePenalty=TotalPairwisePenalty+(PairwisePenalty(i,:)*TheLink(i).to)/sum(TheLink(i).to);
        end
        energy(argNo)=(UnaryEnergy+TotalPairwisePenalty)/ModelNodeNum+b;
    else
        response=getCoreSVMResponse(ARGSet(argNo).arg.NodeTypeInfo.unaryF(1).f(:,RawX(ValidList)),CoreSVM(ValidList));
        energy(argNo)=(sum(response)+0.5*(ModelNodeNum-length(ValidList)))/ModelNodeNum;
    end
end


function response=getCoreSVMResponse(TheARGAtri,CoreSVMList)
getSigmoid=@(a)(1/(1+exp(-(a))));
n=length(CoreSVMList);
response=zeros(1,n);
for i=1:n
    if(isempty(CoreSVMList(i).model))
        response(i)=getSigmoid(0);
    else
        [~,~,estimate]=lib_svmpredict(1,TheARGAtri(:,i)',CoreSVMList(i).model);
        response(i)=getSigmoid(-estimate);
    end
end
