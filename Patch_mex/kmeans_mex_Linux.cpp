#include "mex.h"
#include "stdio.h"
#include "math.h"
#include "string.h"
#include "pthread.h"
#include "unistd.h"

#define sqr(a) ((a)*(a))
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

typedef struct{
	int s,t,K,W,start,ThreadNo,*LabelList;
	double *data,*kcenter,*DistList;
}THEThread;


void* getThread(void* p){
    THEThread para;
    para=*((THEThread*)p);
	double target,dist;
	int cs,ps,i,j,k;
	for(i=0;i<para.K;i++){
        cs=i*para.W;
		for(j=para.s;j<para.t;j++){
			ps=j*para.W;
			if(i==0)
				target=9999999999999;
			else
				target=para.DistList[j]*para.DistList[j];
			dist=0;
			for(k=para.start-1;((dist<target)&&(k<para.W));k++)
				dist+=sqr(para.data[ps+k]-para.kcenter[cs+k]);
			if(target>dist){
				para.DistList[j]=sqrt(dist);
				para.LabelList[j]=i;
			}
		}
	}
	mexPrintf("Thread %d end\n",para.ThreadNo);
    pthread_exit(NULL);
}

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if((nrhs!=4)&&(nrhs!=5))
		mexErrMsgTxt("\nErrors in input.\n");
    
	char str[256];
	double *data,*kcenter,dist,*DistList,sqrerr,*y1,*y2,*y3,*y4,*y5,*y6;
	int ThreadNum,ThreadLen,MaxIterNum,*OldLabelList,*LabelList,*Count,K,start,PtNum,W,iter,ps,cs,i,j,k,c,ret,stacksize;
	bool IsChange;
	THEThread trd_ori,*trd;
	pthread_t *hThread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
    stacksize=20480;
    ret=pthread_attr_setstacksize(&attr,stacksize);
    if(ret)
		mexErrMsgTxt("\nCannot alloc enough memory.\n");
    
	PtNum=mxGetN(prhs[0]);
	W=mxGetM(prhs[0]);
	data=(double*)mxGetPr(prhs[0]);
	K=mxGetN(prhs[1]);
	kcenter=(double*)mxGetPr(prhs[1]);
	start=int(*(double*)mxGetPr(prhs[2]));
	MaxIterNum=int(*(double*)mxGetPr(prhs[3]));
    if(nrhs==5)
        ThreadNum=int(*(double*)mxGetPr(prhs[4]));
    else
        ThreadNum=1;
    
	DistList=0;
	LabelList=0;
	OldLabelList=0;
	Count=0;
	hThread=0;
	trd=0;
	DistList=new double[PtNum];
	LabelList=new int[PtNum];
	OldLabelList=new int[PtNum];
	Count=new int[K];
	hThread=new pthread_t[ThreadNum];
	trd=new THEThread[ThreadNum];
	if((DistList==0)||(LabelList==0)||(OldLabelList==0)||(Count==0)||(hThread==0)||(trd==0))
		mexErrMsgTxt("\nCannot alloc enough memory.\n");

	trd_ori.W=W;
	trd_ori.start=start;
	trd_ori.LabelList=LabelList;
	trd_ori.DistList=DistList;
	trd_ori.data=data;
	trd_ori.kcenter=kcenter;
	for(iter=0;iter<MaxIterNum;iter++){
		///////////////////////////////////////////////////
		//Multiple threads
		ThreadNum=min(K,ThreadNum);
		ThreadLen=PtNum/ThreadNum;
		for(i=0;i<ThreadNum;i++){
			trd[i]=trd_ori;
			trd[i].K=K;
			trd[i].s=i*ThreadLen;
			if(i<ThreadNum-1)
				trd[i].t=trd[i].s+ThreadLen;
			else
				trd[i].t=PtNum;
			trd[i].ThreadNo=i;
            usleep(1000);
            ret=pthread_create(&hThread[i],&attr,getThread,(void *)&trd[i]);
            sleep(0.1);
            if(ret){
                mexErrMsgTxt("\nCannot create thread.\n");
                sleep(10);
            }
		}
		for(i=0;i<ThreadNum;i++){
            ret=pthread_join(hThread[i],NULL);
            if(ret){
                mexErrMsgTxt("\nCannot stop thread.\n");
                sleep(10);
            }
		}
		///////////////////////////////////////////////////

		IsChange=false;
		mexPrintf("%d %d\n",OldLabelList[1],LabelList[1]);
		for(k=0;k<W;k++)
			mexPrintf(" %lf",kcenter[k]);
		mexPrintf("\n");
		for(j=0;j<PtNum;j++){
			if((iter==0)||(OldLabelList[j]!=LabelList[j]))
				IsChange=true;
			OldLabelList[j]=LabelList[j];
		}
		if(!IsChange)
			break;
		memset(kcenter,0,K*W*sizeof(double));
		memset(Count,0,K*sizeof(int));
		for(j=0;j<PtNum;j++){
			cs=LabelList[j]*W;
			ps=j*W;
			for(k=0;k<W;k++)
				kcenter[cs+k]+=data[ps+k];
			Count[LabelList[j]]++;
		}
		for(i=0;i<K;i++){
			cs=i*W;
			if(Count[i]!=0){
				for(k=0;k<W;k++)
					kcenter[cs+k]/=Count[i];
			}
		}
		c=0;
		for(i=0;i<K;i++){
			cs=i*W;
			if(Count[i]!=0){
				DistList[c]=DistList[i];
				LabelList[c]=LabelList[i];
				Count[c]=Count[i];
				for(k=0;k<W;k++)
					kcenter[c*W+k]=kcenter[i*W+k];
				c++;
			}
		}
		K=c;
		sqrerr=0;
		for(i=0;i<PtNum;i++){
			sqrerr+=sqr(DistList[i]);
		}
		mexPrintf("Iteration %d/%d, K=%d, Sqrerr=%lf.\n",iter,MaxIterNum,K,sqrerr);
	}
    
    ThreadNum=min(K,ThreadNum);
	ThreadLen=PtNum/ThreadNum;
	for(i=0;i<ThreadNum;i++){
		trd[i]=trd_ori;
		trd[i].K=K;
		trd[i].s=i*ThreadLen;
		if(i<ThreadNum-1)
			trd[i].t=trd[i].s+ThreadLen;
		else
			trd[i].t=PtNum;
		trd[i].ThreadNo=i;
        usleep(1000);
        ret=pthread_create(&hThread[i],&attr,getThread,(void *)&trd[i]);
        if(ret){
            mexErrMsgTxt("\nCannot create thread.\n");
            sleep(10);
        }
	}
	for(i=0;i<ThreadNum;i++){
        sleep(0.1);
        ret=pthread_join(hThread[i],NULL);
        if(ret){
            mexErrMsgTxt("\nCannot stop thread.\n");
            sleep(10);
        }
	}

	plhs[0]=mxCreateDoubleMatrix(W,K,mxREAL);
    y1=mxGetPr(plhs[0]);
	plhs[1]=mxCreateDoubleMatrix(PtNum,1,mxREAL);
	y2=mxGetPr(plhs[1]);
	plhs[2]=mxCreateDoubleMatrix(1,1,mxREAL);
	y3=mxGetPr(plhs[2]);
	plhs[3]=mxCreateDoubleMatrix(K,1,mxREAL);
	y4=mxGetPr(plhs[3]);
	plhs[4]=mxCreateDoubleMatrix(PtNum,1,mxREAL);
	y5=mxGetPr(plhs[4]);
	plhs[5]=mxCreateDoubleMatrix(K,1,mxREAL);
	y6=mxGetPr(plhs[5]);
	for(i=0;i<K*W;i++)
		y1[i]=kcenter[i];
	memset(y4,0,K*sizeof(double));
	memset(Count,0,K*sizeof(int));
	sqrerr=0;
	for(i=0;i<PtNum;i++){
		sqrerr+=sqr(DistList[i]);
		y2[i]=LabelList[i]+1;
		y4[LabelList[i]]=max(y4[LabelList[i]],DistList[i]);
		Count[LabelList[i]]++;
		y5[i]=DistList[i];
	}
	for(i=0;i<K;i++){
		if(Count[i]==0)
			mexErrMsgTxt("\nErrors in empty clusters.\n");
		y6[i]=Count[i];
	}
	*y3=sqrerr;
    ret=pthread_attr_destroy(&attr);
	delete[] DistList;
	delete[] LabelList;
	delete[] OldLabelList;
	delete[] Count;
}
