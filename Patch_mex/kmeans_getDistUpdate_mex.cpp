#include "mex.h"
#include "stdio.h"
#include "math.h"
#include "string.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define sqr(a) ((a)*(a))

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if(nrhs!=3)
		mexErrMsgTxt("\nErrors in input.\n");

	int PtNum,K,W,start,*LabelList,cs,ps,i,j,k;
	double *data,*kcenter,target,dist,*DistList,sqrerr,*y1,*y2,*y3,*y4,*y5;
	
	start=int(*(double*)mxGetPr(prhs[0]));
	kcenter=(double*)mxGetPr(prhs[1]);
	K=int(mxGetN(prhs[1]));
	data=(double*)mxGetPr(prhs[2]);
	W=int(mxGetM(prhs[2]));
	PtNum=int(mxGetN(prhs[2]));
	
	DistList=0;
	LabelList=0;
	DistList=new double[PtNum];
	LabelList=new int[PtNum];
	if((DistList==0)||(LabelList==0))
		mexErrMsgTxt("\nCannot alloc enough memory.\n");

	for(i=0;i<K;i++){
		cs=i*W;
		for(j=0;j<PtNum;j++){
			ps=j*W;
			if(i==0)
				target=9999999999999;
			else
				target=sqr(DistList[j]);
			dist=0;
			for(k=start-1;((dist<target)&&(k<W));k++)
				dist+=sqr(data[ps+k]-kcenter[cs+k]);
			if(target>dist){
				DistList[j]=sqrt(dist);
				LabelList[j]=i;
			}
		}
	}

	plhs[0]=mxCreateDoubleMatrix(PtNum,1,mxREAL);
	y1=mxGetPr(plhs[0]);
	plhs[1]=mxCreateDoubleMatrix(PtNum,1,mxREAL);
	y2=mxGetPr(plhs[1]);
	plhs[2]=mxCreateDoubleMatrix(K,1,mxREAL);
	y3=mxGetPr(plhs[2]);
	plhs[3]=mxCreateDoubleMatrix(K,1,mxREAL);
	y4=mxGetPr(plhs[3]);
	plhs[4]=mxCreateDoubleMatrix(1,1,mxREAL);
	y5=mxGetPr(plhs[4]);

	sqrerr=0;
	memset(y3,0,K*sizeof(mxREAL));
	memset(y4,0,K*sizeof(mxREAL));
	for(i=0;i<PtNum;i++){
		sqrerr+=sqr(DistList[i]);
		y1[i]=DistList[i];
		y2[i]=LabelList[i]+1;
		y3[LabelList[i]]++;
		y4[LabelList[i]]=max(y4[LabelList[i]],DistList[i]);
	}
	*y5=sqrerr;

	delete[] DistList;
	delete[] LabelList;
}
