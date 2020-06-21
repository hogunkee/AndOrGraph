function try_kmeans
addpath('..//');
Num=10000;
tmp=rand(Num,3);
tmp=tmp./repmat(sqrt(sum(tmp.^2,2)),[1,size(tmp,2)]);
data=[(1:Num)',tmp];
K=100;
start=2;
tmp=randperm(Num);
kcentre_initial=data(tmp(1:K),:);
ThreadNum=55;
tic
%[kcentre2,owner2,sqrerr,kdist2,PtDist,CluSize]=kmeans_mex_Linux(data',kcentre_initial',start,100,ThreadNum);
%[kcentre2,owner2,sqrerr,kdist2,PtDist,CluSize]=kmeans_mex(data',kcentre_initial',start,100,ThreadNum);
Learning.ThreadNum=4;[kcentre2,owner2,sqrerr,kdist2,PtDist,CluSize]=kmeans_cluster(data',kcentre_initial',start,100,Learning);
toc
col=[1,0,0;0,1,0;0,0,1;1,1,0;1,0,1;0,1,1];

figure;
hold on;
for i=1:Num
   plot(data(i,2),data(i,3),'.','Color',col(mod(owner2(i),6)+1,:));
end
