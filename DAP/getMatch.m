function [match,TheF1,TheF2,Psi]=getMatch(TheModel,ARGSet,IsShown,Para)
num=size(ARGSet,2);
Para.ThreadNum=min(Para.ThreadNum,num);
nodeNum=TheModel.nodeNum;
match=zeros(nodeNum,num);
TheF1(num).matrix=[];
TheF2(num).matrix=[];
Psi=zeros(nodeNum,num);
if(Para.ThreadNum>1)
    tmp=round(linspace(1,num+1,Para.ThreadNum+1));
    list1=tmp(1:end-1);
    list2=tmp(2:end)-1;
    Division(Para.ThreadNum).subARGSet=[];
    Division(Para.ThreadNum).match=[];
    Division(Para.ThreadNum).TheF1=[];
    Division(Para.ThreadNum).TheF2=[];
    Division(Para.ThreadNum).psi=[];
    Division(Para.ThreadNum).TheModel=[];
    Division(Para.ThreadNum).IsShown=[];
    Division(Para.ThreadNum).Para=[];
    for i=1:Para.ThreadNum
        len=list2(i)-list1(i)+1;
        Division(i).subARGSet=ARGSet(list1(i):list2(i));
        Division(i).match=zeros(nodeNum,len);
        Division(i).TheF1(len).matrix=[];
        Division(i).TheF2(len).matrix=[];
        Division(i).psi=zeros(nodeNum,len);
        Division(i).TheModel=TheModel;
        Division(i).IsShown=IsShown;
        Division(i).Para=Para;
    end
    clear ARGSet
    parfor par=1:Para.ThreadNum
        len=size(Division(par).subARGSet,2);
        for i=1:len
            [x,f1,f2,psi]=MRF(Division(par).TheModel,Division(par).subARGSet(i).arg,Division(par).IsShown,Division(par).Para);
            Division(par).match(:,i)=x';
            Division(par).TheF1(i).matrix=f1;
            Division(par).TheF2(i).matrix=f2;
            Division(par).psi(:,i)=psi';
            Division(par).subARGSet(i).arg=[];
        end
    end
    for i=1:Para.ThreadNum
        match(:,list1(i):list2(i))=Division(i).match;
        Psi(:,list1(i):list2(i))=Division(i).psi;
        c=0;
        for j=list1(i):list2(i)
            c=c+1;
            TheF1(j).matrix=Division(i).TheF1(c).matrix;
            TheF2(j).matrix=Division(i).TheF2(c).matrix;
        end
    end
else
    for argNo=1:num
        [x,f1,f2,psi]=MRF(TheModel,ARGSet(argNo).arg,IsShown,Para);
        match(:,argNo)=x';
        TheF1(argNo).matrix=f1;
        TheF2(argNo).matrix=f2;
        Psi(:,argNo)=psi';
    end
end


function [x,f1,f2,psi]=MRF(TheModel,TheARG,IsShown,Para)
typeNum=length(TheModel.NodeType);
UnaryPenalty=zeros(TheModel.nodeNum,TheARG.nodeNum);
AllPsi=zeros(TheModel.nodeNum,TheARG.nodeNum);
TheLarge=TheModel.penalty.large;
TheLargeLarge=TheModel.penalty.largelarge;
for typeNo=1:typeNum
    unaryNum=length(TheModel.NodeType(typeNo).unaryW);
    nodeList=TheModel.NodeType(typeNo).nodeList;
    argTypeIdx=getNodeType(TheModel.NodeType(typeNo).type,TheARG.NodeType);
    targetList=TheARG.NodeType(argTypeIdx).nodeList;
    targetNodeNum=length(targetList);
    if((targetNodeNum>0)&&(~isempty(nodeList)))
        for j=nodeList
            terminalNum=size(TheModel.NodeInfo(j).unaryF(1).f,2);
            tmp=ones(1,targetNodeNum).*TheLarge;
            tmp_terminal=zeros(1,targetNodeNum);
            for terminal=1:terminalNum
                tt_sum=zeros(1,targetNodeNum);
                for atri=1:unaryNum
                    if((atri==1)&&(Para.LearningItems.IsWeightTraining_.IsTrainDetailedWForFirstLocalAtri))
                        try
                            tt=sum(((repmat(TheModel.NodeInfo(j).unaryF(atri).f(:,terminal),[1,targetNodeNum])-TheARG.NodeTypeInfo(argTypeIdx).unaryF(atri).f).^2).*repmat(TheModel.NodeInfo(j).unaryF(atri).DiscriminativeW(:,terminal),[1,targetNodeNum]),1);
                        catch
                            TheModel.NodeInfo(j).unaryF(atri).DiscriminativeW=ones(size(TheModel.NodeInfo(j).unaryF(atri).f,1),terminalNum);
                            tt=sum(((repmat(TheModel.NodeInfo(j).unaryF(atri).f(:,terminal),[1,targetNodeNum])-TheARG.NodeTypeInfo(argTypeIdx).unaryF(atri).f).^2).*repmat(TheModel.NodeInfo(j).unaryF(atri).DiscriminativeW(:,terminal),[1,targetNodeNum]),1);
                        end
                    else
                        tt=sum((repmat(TheModel.NodeInfo(j).unaryF(atri).f(:,terminal),[1,targetNodeNum])-TheARG.NodeTypeInfo(argTypeIdx).unaryF(atri).f).^2,1);
                    end
                    
                    tt_sum=tt_sum+tt.*TheModel.NodeType(typeNo).unaryW(atri);
                end
                tmp_list=find(tmp>tt_sum);
                tmp(tmp_list)=tt_sum(tmp_list);
                tmp_terminal(tmp_list)=terminal;
            end
            UnaryPenalty(j,targetList)=UnaryPenalty(j,targetList)+tmp;
            AllPsi(j,targetList)=tmp_terminal;
        end
    end
    UnaryPenalty(nodeList,setdiff(1:TheARG.nodeNum,targetList))=TheLargeLarge;
end
pairwiseNum=length(TheModel.pairwiseF);
for i=1:TheModel.nodeNum
    for j=1:TheModel.nodeNum
        if(j==i)
            continue;
        end
        PairwisePenalty=zeros(TheARG.nodeNum,TheARG.nodeNum);
        if(TheARG.nodeNum>1)
            for atri=1:pairwiseNum
                PairwisePenalty=PairwisePenalty+reshape(sum((repmat(TheModel.pairwiseF(atri).f(:,i,j),[1,TheARG.nodeNum,TheARG.nodeNum])-TheARG.pairwiseF(atri).f).^2,1),size(PairwisePenalty)).*TheModel.pairwiseF(atri).w;
            end
        end
        MRF(i,j).Transfer=PairwisePenalty;
    end
end
V=0:TheModel.nodeNum-1;
Enum=TheModel.nodeNum*(TheModel.nodeNum-1)/2;
E=zeros(2,Enum);
t=0;
for i=0:TheModel.nodeNum-2
    len=TheModel.nodeNum-1-i;
    E(:,t+1:t+len)=[ones(1,len).*i;i+1:i+len];
    t=t+len;
end
unmatchPenalty=zeros(1,TheModel.nodeNum);
for typeNo=1:typeNum
    unmatchPenalty(TheModel.NodeType(typeNo).nodeList)=TheModel.NodeType(typeNo).penalty.unmatch;
end
f1=[UnaryPenalty';unmatchPenalty];
f2=ones(TheARG.nodeNum+1,TheARG.nodeNum+1,Enum).*TheModel.penalty.unmatchPair;
f2_normalized=ones(TheARG.nodeNum+1,TheARG.nodeNum+1,Enum).*TheModel.penalty.unmatchPair;
t=0;
TheDiag=zeros(TheARG.nodeNum,TheARG.nodeNum);
TheDiag(linspace(1,TheARG.nodeNum^2,TheARG.nodeNum))=TheModel.penalty.large;
for i=1:TheModel.nodeNum-1
    for j=i+1:TheModel.nodeNum
        t=t+1;
        %tmp=(TheModel.link(i).to(j)/sum(TheModel.link(i).to)+TheModel.link(j).to(i)/sum(TheModel.link(j).to))/2;
        tmp=TheModel.link(i).to(j)/sum(TheModel.link(i).to)+TheModel.link(j).to(i)/sum(TheModel.link(j).to); % edges (s,t) and (t,s) are counted as two different edges
        f2(1:TheARG.nodeNum,1:TheARG.nodeNum,t)=MRF(i,j).Transfer;
        f2_normalized(:,:,t)=f2(:,:,t).*tmp;
        ManyToOneIndex=linspace(1,(TheARG.nodeNum+1)^2,TheARG.nodeNum+1);
        ManyToOneIndex=ManyToOneIndex(1:TheARG.nodeNum)+((TheARG.nodeNum+1)^2)*(t-1);
        f2(ManyToOneIndex)=TheModel.penalty.large;
        f2_normalized(ManyToOneIndex)=TheModel.penalty.large;
    end
end
f1_normalized=f1-repmat(min(f1,[],1),[TheARG.nodeNum+1,1]);
x=TRWSProcess(V,E,f1_normalized,f2_normalized,Para);
AllPsi_enlarge=[AllPsi,zeros(TheModel.nodeNum,1)]';
psi=AllPsi_enlarge((0:TheModel.nodeNum-1).*(TheARG.nodeNum+1)+x);
TheX=x;TheX(TheX>TheARG.nodeNum)=-1;disp([TheX;psi]);
if((IsShown==1)&&(Para.ThreadNum<=1))
    show_MatchingResult_SIVAL(TheModel.ARGInfo.name,TheARG.ARGInfo.name,x,Para,TheModel.link);
    %show_MatchingResult_SIVAL_neg(TheModel.ARGInfo.name,TheARG.ARGInfo.name,x,Para,TheModel.link);
    %show_MatchingResult(TheModel.ARGInfo.name,TheARG.ARGInfo.name,x,Para,TheModel.link);
    %show_MatchingResult_ETHZ(TheModel.ARGInfo.name,TheARG.ARGInfo.name,x,Para,TheModel.link);
    %show_MatchingResult_animal(TheARG.ARGInfo.name,x,Para,TheModel.link);
    %show_MatchingResult_SIFTImg(TheARG.ARGInfo.name,x,Para,TheModel.link);
end
