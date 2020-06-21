function ori=getPixelOri(I)
template=[-1,0,1,]'*ones(1,3);
[m,n,d]=size(I);
Value=ones(m,n).*(-1);
Ori=zeros(m,n);
I=double(I);
for i=1:3
    Ver=imfilter(I(:,:,i),template,'replicate');
    Hor=imfilter(I(:,:,i),template','replicate');
    O=double(Ver)./(Hor+0.001);
    V=sqrt(double(Ver.^2+Hor.^2));
    Ori(Value<V)=O(Value<V);
    Value(Value<V)=V(Value<V);
end
ori.Ori=atan(Ori);
ori.Val=Value;
