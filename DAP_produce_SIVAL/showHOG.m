function showHOG(f,Patch,HOG)
if(size(Patch,1)>0)
    figure;
    imshow(Patch);
end
k=sqrt(size(f,1)/HOG.VoteOriNum);
figure;
bar=colormap;
close;
figure;
hold on;
f=max(f./max(f),0.000001);
c=0;
for w=1:k
    for h=1:k
        for o=1:HOG.VoteOriNum
            c=c+1;
            l=f(c);
            angle=pi*o/HOG.VoteOriNum;
            line([w-0.5*cos(angle),w+0.5*cos(angle)],-[h+0.5*sin(angle),h-0.5*sin(angle)],'LineWidth',f(c)./2,'Color',bar(ceil(l*64),:));
        end
    end
end
