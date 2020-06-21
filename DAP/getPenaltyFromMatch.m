function [UnaryPenalty,AvgPairwisePenalty,ThePairwisePenalty]=getPenaltyFromMatch(ARGSet,match,TheF1,TheF2,Para,ModelNodeNum,TheLink)
N=size(ARGSet,2);
UnaryPenalty=zeros(ModelNodeNum,1);
AvgPairwisePenalty=zeros(ModelNodeNum,1);
ThePairwisePenalty=zeros(ModelNodeNum,ModelNodeNum);
for argNo=1:N
    RawX=match(:,argNo)';
    f1=TheF1(argNo).matrix;
    f2=TheF2(argNo).matrix;
    argNodeNum=ARGSet(argNo).arg.nodeNum;
    tmp=(0:ModelNodeNum-1).*(argNodeNum+1)+RawX;
    UnaryPenalty=UnaryPenalty+f1(tmp)';
    t=0;
    for i=1:ModelNodeNum-1
        for j=i+1:ModelNodeNum
            t=t+1;
            tmp=f2(RawX(i),RawX(j),t);
            ThePairwisePenalty(i,j)=ThePairwisePenalty(i,j)+tmp;
            ThePairwisePenalty(j,i)=ThePairwisePenalty(j,i)+tmp;
        end
    end
end
UnaryPenalty=UnaryPenalty./N;
ThePairwisePenalty=ThePairwisePenalty./N;
if(isempty(TheLink))
    TheLink=SetLinkage(Para,ModelNodeNum,ThePairwisePenalty);
end
for i=1:ModelNodeNum
    AvgPairwisePenalty(i)=(ThePairwisePenalty(i,:)*TheLink(i).to)/sum(TheLink(i).to);
end
