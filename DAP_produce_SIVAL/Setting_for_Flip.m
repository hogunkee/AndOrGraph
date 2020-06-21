function ARGSet=Setting_for_Flip(ARGSet,Para)
HOG.VoteOriNum=8;
HOG.CellNum=8;
if(~Para.IsFlip)
    return;
end
ARGSet=repmat(ARGSet,[2,1]);
for argno=1:size(ARGSet,2)
    ARGSet(1,argno).flip=false;
    ARGSet(2,argno).flip=true;
    for nodeNo=1:ARGSet(2,argno).arg.nodeNum
        tmp_f=reshape(ARGSet(2,argno).arg.NodeTypeInfo.unaryF(1).f(:,nodeNo),[HOG.VoteOriNum,HOG.CellNum,HOG.CellNum]);
        for ori=1:floor((HOG.VoteOriNum-1)/2)
            t=tmp_f(ori,:,:);
            tmp_f(ori,:,:)=tmp_f(HOG.VoteOriNum-ori,:,:);
            tmp_f(HOG.VoteOriNum-ori,:,:)=t;
        end
        for w=1:floor(HOG.CellNum/2)
            t=tmp_f(:,:,w);
            tmp_f(:,:,w)=tmp_f(:,:,HOG.CellNum+1-w);
            tmp_f(:,:,HOG.CellNum+1-w)=t;
        end
        ARGSet(2,argno).arg.NodeTypeInfo.unaryF(1).f(:,nodeNo)=reshape(tmp_f,[length(tmp_f(1:end)),1]);
    end
    ARGSet(2,argno).arg.pairwiseF(3).f(2,:,:)=-ARGSet(2,argno).arg.pairwiseF(3).f(2,:,:);    
end
ARGSet=ARGSet(1:end);
