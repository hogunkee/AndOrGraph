function show_MatchingResult(ModelName,CldName,x,Para,TheLink,col)
str=CldName(1:end-1);
str(end-1)='%';
str(end-0)='d';
num=sscanf(CldName,str);
CldName=CldName(1:end-3);
img=imread(sprintf('./DAP_produce_SIVAL/ImgSet/%s/%03d.jpg',CldName,num));
figure;
%img=img.*(3/4)+63;
imshow(img);
hold on;
load(sprintf('./mat/FinalPatches_%s.mat',CldName),'FinalPatches');
n=size(FinalPatches(num).HWScaleVal,2);
if(nargin<6)
    col=repmat('m',[1,length(x)]);
end
list=find((x<=n).*(x>=1)>0);
col=col(list);
TheX=x(list);
if(size(TheX,2)>0)
    HWScale=FinalPatches(num).HWScaleVal(1:3,TheX);
    %HWScale=FinalPatches(num).HWScaleVal(1:3,:);
else
    HWScale=[];
end
clear FinalPatches
for i=1:size(HWScale,2)
    H=HWScale(1,i);
    W=HWScale(2,i);
    Scale=HWScale(3,i)/2;
    Y=[H-Scale,H+Scale,H+Scale,H-Scale,H-Scale];
    X=[W-Scale,W-Scale,W+Scale,W+Scale,W-Scale];
    plot(X,Y,'Color',col(i),'LineWidth',3);
end
% if(Para.LearningItems.IsLinkLearning)
%     num=size(TheLink,2);
%     for i=1:num
%         if(x(i)>n)
%             continue;
%         end
%         for j=find(TheLink(i).to==1)'
%             if(x(j)>n)
%                 continue;
%             end
%             if(TheLink(j).to(i)==1)
%                 
%                 line([HWScale(2,i),HWScale(2,j)],[HWScale(1,i),HWScale(1,j)],'Color','m','LineWidth',1);
%             else
%                 line([HWScale(2,i),HWScale(2,j)],[HWScale(1,i),HWScale(1,j)],'Color','r','LineWidth',1);
%             end
%         end
%     end
% end
