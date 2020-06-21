function LinkNum=getLinkNum(Para,modelNodeNum)
if(Para.LearningItems.IsLinkLearning)
    LinkNum=max(ceil(Para.LinkNumRate*(modelNodeNum-1)),min(Para.MinLinkNum,modelNodeNum-1));
else
    LinkNum=modelNodeNum-1;
end
