#include<iostream>
#include "mex.h"
#include "stdio.h"
#include "math.h"
#include<string.h>

#define sqr(A) ((A)*(A))
#define max(A,B) ((A)>(B)?(A):(B))
#define ROUND(X) (int)(X+0.5)
#define MAXCELLNUM 300
#define MAXORINUM 20
#define PI 3.14159265359

double CellSpace[MAXCELLNUM][MAXCELLNUM][MAXORINUM];


void getCellVoting(int cellW,int cellH,int cellScale,double* ori_Val,double* ori_Ori,double interval,int H,int HOG_VoteOriNum){
	double tmp,val;
	int w,h,k,WLen,ori;
	memset(CellSpace,0,sizeof(double)*MAXCELLNUM*MAXCELLNUM*MAXORINUM);
	for(w=0;w<cellW*cellScale;w++){
		WLen=w*H;
		for(h=0;h<cellH*cellScale;h++){
			val=ori_Val[WLen+h];
			ori=int(fmod(ori_Ori[WLen+h]+PI*2-interval/2,PI)/interval);
			CellSpace[h/cellScale][w/cellScale][ori]+=val;
		}
	}
}

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if(nrhs!=12)
		mexErrMsgTxt("\nErrors in input.\n");
    
	//readinfo
    double LargestScale,HOG_ScaleDecrease,*ImgHW,*I,*ori_Ori,*ori_Val;
    int HOG_ScaleLevelNum,HOG_CellNum,HOG_MinCellScale,HOG_SampleWindowStep,HOG_VoteOriNum,ImgNo,H,W;
    LargestScale=*((double*)mxGetPr(prhs[0]));
    HOG_ScaleLevelNum=int(*((double*)mxGetPr(prhs[1])));
    HOG_ScaleDecrease=*((double*)mxGetPr(prhs[2]));
	HOG_CellNum=int(*((double*)mxGetPr(prhs[3])));
	HOG_MinCellScale=int(*((double*)mxGetPr(prhs[4])));
	HOG_SampleWindowStep=int(*((double*)mxGetPr(prhs[5])));
	HOG_VoteOriNum=int(*((double*)mxGetPr(prhs[6])));
	ImgHW=(double*)mxGetPr(prhs[7]);
	I=(double*)mxGetPr(prhs[8]);
	ori_Ori=(double*)mxGetPr(prhs[9]);
	ori_Val=(double*)mxGetPr(prhs[10]);
	ImgNo=int(*((double*)mxGetPr(prhs[11])));
	H=int(ImgHW[0]);W=int(ImgHW[1]);

	if(HOG_VoteOriNum>=MAXORINUM)
		mexErrMsgTxt("\nMAXORINUM over stack.\n");
	//count FeatureNum
	double scale,interval,tmp,*f,*TheHWScale,*norm,sum;
	int i,h,w,k1,k2,k3,dim,cellH,cellW,cellScale,FeatureDim;
	long FeatureNum,count;
	interval=PI/HOG_VoteOriNum;

	FeatureNum=0;
	for(i=0;i<HOG_ScaleLevelNum;i++){
		scale=LargestScale*pow(HOG_ScaleDecrease,i);
		cellScale=int(ceil(scale/HOG_CellNum));
		scale=cellScale*HOG_CellNum;
		if(cellScale<HOG_MinCellScale)
			continue;
		cellH=H/cellScale;cellW=W/cellScale;
		FeatureNum+=((cellH-(HOG_CellNum-HOG_SampleWindowStep))/HOG_SampleWindowStep)*((cellW-(HOG_CellNum-HOG_SampleWindowStep))/HOG_SampleWindowStep);
	}
	FeatureDim=HOG_CellNum*HOG_CellNum*HOG_VoteOriNum;
	plhs[0]=mxCreateDoubleMatrix(FeatureDim,FeatureNum,mxREAL);
	f=mxGetPr(plhs[0]);
	plhs[1]=mxCreateDoubleMatrix(3,FeatureNum,mxREAL);
	TheHWScale=mxGetPr(plhs[1]);
	plhs[2]=mxCreateDoubleMatrix(1,FeatureNum,mxREAL);
	norm=mxGetPr(plhs[2]);

	//processing
	count=0;
	for(i=0;i<HOG_ScaleLevelNum;i++){
		scale=LargestScale*pow(HOG_ScaleDecrease,i);
		cellScale=int(ceil(scale/HOG_CellNum));
		scale=cellScale*HOG_CellNum;
		if(cellScale<HOG_MinCellScale)
			continue;
		cellH=H/cellScale;cellW=W/cellScale;
		if((cellH>=MAXCELLNUM)||(cellH>=MAXCELLNUM))
			mexErrMsgTxt("\nMAXCELLNUM over stack.\n");
		getCellVoting(cellW,cellH,cellScale,ori_Val,ori_Ori,interval,H,HOG_VoteOriNum);
		//mexPrintf("\n%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",CellSpace[1][1][1],CellSpace[2][1][1],CellSpace[3][1][1],CellSpace[1][1][2],CellSpace[2][1][2],CellSpace[3][1][2],CellSpace[1][1][3],CellSpace[2][1][3],CellSpace[3][1][3]);
		FeatureNum=((cellH-(HOG_CellNum-HOG_SampleWindowStep))/HOG_SampleWindowStep)*((cellW-(HOG_CellNum-HOG_SampleWindowStep))/HOG_SampleWindowStep);
		FeatureDim=HOG_CellNum*HOG_CellNum*HOG_VoteOriNum;
		for(w=0;w<cellW-HOG_CellNum+1;w+=HOG_SampleWindowStep){
			for(h=0;h<cellH-HOG_CellNum+1;h+=HOG_SampleWindowStep){
				TheHWScale[count*3]=h*cellScale+scale/2;
				TheHWScale[count*3+1]=w*cellScale+scale/2;
				TheHWScale[count*3+2]=scale;
				dim=count*FeatureDim;
				sum=0;
				for(k1=0;k1<HOG_CellNum;k1++){
					for(k2=0;k2<HOG_CellNum;k2++){
						for(k3=0;k3<HOG_VoteOriNum;k3++){
							f[dim]=CellSpace[k2+h][k1+w][k3];
							sum+=f[dim]*f[dim];
							dim++;
						}
					}
				}
				sum=sqrt(sum);
				for(dim=count*FeatureDim;dim<(count+1)*FeatureDim;dim++)
					f[dim]/=max(sum,0.001);
				norm[count]=sqrt(sqr(sum)/(sqr(sqr(cellScale))*sqr(HOG_CellNum)*HOG_VoteOriNum));
				count++;
			}
		}
	}
}
