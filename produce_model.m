function produce_model(Name_batch)

if(exist(sprintf('./mat/model_%s.mat',Name_batch),'file')==2)
    while(true)
        disp('The system found that the initial graph template had been labeled before.');
        s=input('Do you want to label a new template?  [y/n]  ','s');
        if(strcmp(s,'n'))
            return;
        end
        if(strcmp(s,'y'))
            break;
        end
    end
end
load(sprintf('./mat/FinalPatches_%s.mat',Name_batch),'FinalPatches');
load(sprintf('./mat/ARGs_%s.mat',Name_batch),'ARGSet');
MaxImgNum=size(ARGSet,2);
count=0;
for ImgNo=1:MaxImgNum
    close all;
    figure(ImgNo);
    filename=sprintf('./DAP_produce_SIVAL/ImgSet/%s/%03d.jpg',Name_batch,ImgNo);
    I=imread(filename);
    [h,w,~]=size(I);
    imshow(I);
    hold on
    PatchNum=size(FinalPatches(ImgNo).HWScaleVal,2);
    TheRect=zeros(4,PatchNum);
    for pNo=1:PatchNum
        H=FinalPatches(ImgNo).HWScaleVal(1,pNo);
        W=FinalPatches(ImgNo).HWScaleVal(2,pNo);
        Scale=FinalPatches(ImgNo).HWScaleVal(3,pNo)/2;
        Y=[H-Scale,H+Scale,H+Scale,H-Scale,H-Scale];
        X=[W-Scale,W-Scale,W+Scale,W+Scale,W-Scale];
        plot(X,Y,'r-');
        TheRect(:,pNo)=[H-Scale;H+Scale;W-Scale;W+Scale];
    end
    fprintf('%s  PatchNum %d\n',filename,PatchNum);
    A=input('Input ''y'' to lable this image or input others to continue:','s');
    if((strcmp(A,'y')==1)||(strcmp(A,'Y')==1))
        count=count+1;
        model_set(count)=ModelInteraction(ARGSet(ImgNo).arg,TheRect,h,w);
        close gcf
        break;
    end
end
if(count==0)
    error('Have not selected the model.');
end;
save(sprintf('./mat/model_%s.mat',Name_batch),'model_set');


function model=ModelInteraction(TheARG,TheRect,h,w)
num=size(TheRect,2);
list=[];
while(1)
    waitforbuttonpress;
    point1=get(gca,'CurrentPoint');
    rbbox;
    point2=get(gca,'CurrentPoint');
    point1=point1(1,1:2);
    point2=point2(1,1:2);
    MinH=min(point1(2),point2(2));
    MaxH=max(point1(2),point2(2));
    MinW=min(point1(1),point2(1));
    MaxW=max(point1(1),point2(1));
    if((MinH>h)||(MaxH<0)||(MinW>w)||(MaxW<0))
        model.nodeNum=size(list,2);
        model.ARGInfo.name=TheARG.ARGInfo.name;
        model.NodeInfo(model.nodeNum).type=[];
        model.NodeInfo(model.nodeNum).unaryF=[];
        num_unary=length(TheARG.NodeTypeInfo.unaryF);
        for nodeID=1:model.nodeNum
            model.NodeInfo(nodeID).type='MidLevel';
            model.NodeInfo(nodeID).unaryF(num_unary).f=[];
            model.NodeInfo(nodeID).unaryF(num_unary).DiscriminativeW=[];
            for atri=1:num_unary
                model.NodeInfo(nodeID).unaryF(atri).f=TheARG.NodeTypeInfo.unaryF(atri).f(:,list(nodeID));
                if(atri==1)
                    model.NodeInfo(nodeID).unaryF(atri).DiscriminativeW=ones(size(TheARG.NodeTypeInfo.unaryF(atri).f,1),1);
                end
            end
        end
        for atri=1:length(TheARG.pairwiseF)
            model.pairwiseF(atri).f=TheARG.pairwiseF(atri).f(:,list,list);
            model.pairwiseF(atri).w=1;
        end
        for i=1:model.nodeNum
            model.link(i).to=ones(model.nodeNum,1);
            model.link(i).to(i)=0;
        end
        model.NodeType.type='MidLevel';
        model.NodeType.nodeList=1:model.nodeNum;
        model.NodeType.penalty.unmatch=20;
        model.NodeType.unaryW=0.3;
        model.penalty.unmatchPair=100;
        model.penalty.large=20;
        model.penalty.largelarge=30;
        model.b=0;
        break;
    end
    rect=[MinH;MaxH;MinW;MaxW];
    [~,TheTar]=min(sum((TheRect-repmat(rect,[1,num])).^2,1));
    list=unique([list,TheTar]);
    Y=[TheRect(1,TheTar),TheRect(2,TheTar),TheRect(2,TheTar),TheRect(1,TheTar),TheRect(1,TheTar)];
    X=[TheRect(3,TheTar),TheRect(3,TheTar),TheRect(4,TheTar),TheRect(4,TheTar),TheRect(3,TheTar)];
    plot(X,Y,'c-','LineWidth',3);
    fprintf('nodeNum: %d\n',size(list,2));
end
