function link=SetLinkage(Para,nodeNum,RawPairwisePenalty_pos)
link(nodeNum).to=[];
if(Para.LearningItems.IsLinkLearning)
    TheLinkNum=getLinkNum(Para,nodeNum);
    for i=1:nodeNum
        RawPairwisePenalty_pos(i,i)=100000000000000000000000000000000;
        [~,order]=sort(RawPairwisePenalty_pos(i,:),'ascend');
        link(i).to=zeros(nodeNum,1);
        link(i).to(order(1:TheLinkNum))=1;
    end
else
    for i=1:nodeNum
        link(i).to=ones(nodeNum,1);
        link(i).to(i)=0;
    end
end
