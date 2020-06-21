function [Patch,ori,rect]=segmentPatch(I,pos_h,pos_w,SampleOri,Scale,HOG)
PatchSize_NormalizedTo=30;
ImgHW=[pos_h,pos_w];
D_Vertical=[cos(SampleOri),sin(SampleOri)];
D_Parallel=[-sin(SampleOri),cos(SampleOri)];
Index=linspace(-Scale/2,Scale/2,PatchSize_NormalizedTo)'*ones(1,PatchSize_NormalizedTo);
[h,w,d]=size(I);
H_Index=(ImgHW(1)+Index.*D_Vertical(1)+Index'.*D_Parallel(1));
W_Index=(ImgHW(2)+Index.*D_Vertical(2)+Index'.*D_Parallel(2));
H_Index(H_Index>h)=h;H_Index(H_Index<1)=1;W_Index(W_Index>w)=w;W_Index(W_Index<1)=1;
W_res=W_Index-floor(W_Index);H_res=H_Index-floor(H_Index);
tb1=reshape((1-W_res).*(1-H_res),[PatchSize_NormalizedTo^2,1]);
tb2=reshape((1-W_res).*H_res,[PatchSize_NormalizedTo^2,1]);
tb3=reshape(W_res.*(1-H_res),[PatchSize_NormalizedTo^2,1]);
tb4=reshape(W_res.*H_res,[PatchSize_NormalizedTo^2,1]);
I=reshape(I,[h*w,3]);

Fig1=double(I(floor(H_Index)+(floor(W_Index)-1).*h,:));
Fig2=double(I(ceil(H_Index)+(floor(W_Index)-1).*h,:));
Fig3=double(I(floor(H_Index)+(ceil(W_Index)-1).*h,:));
Fig4=double(I(ceil(H_Index)+(ceil(W_Index)-1).*h,:));
Patch=uint8(reshape(repmat(tb1,[1,3]).*Fig1+repmat(tb2,[1,3]).*Fig2+repmat(tb3,[1,3]).*Fig3+repmat(tb4,[1,3]).*Fig4,[PatchSize_NormalizedTo,PatchSize_NormalizedTo,3]));
ori=getPixelOri(Patch);
rect=[-D_Parallel-D_Vertical;-D_Parallel+D_Vertical;+D_Parallel+D_Vertical;+D_Parallel-D_Vertical].*Scale+ones(4,1)*ImgHW;
