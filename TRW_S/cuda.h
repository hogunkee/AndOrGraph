#include "instances.h"

//#define MaxThreadNum 1024
#define MaxThreadNum 65536
#define CUDAThreadNum 1024
#define MaxPatternNodeNum 100
#define getCUDABlockNum(num) (int((num+CUDAThreadNum-1)/CUDAThreadNum))

TypeGeneral::REAL TheTmp[MaxThreadNum],shown[MaxThreadNum],Gamma[MaxThreadNum];
int NodeIdx[MaxPatternNodeNum],ForwardIdx[MaxPatternNodeNum],BackwardIdx[MaxPatternNodeNum];
int forward_backward_tmp[MaxPatternNodeNum][MaxPatternNodeNum],forward_backward[MaxThreadNum],backward_forward[MaxThreadNum];

inline TypeGeneral::REAL* TypeGeneral::Vector::GetArrayHead(){
	return m_data;
}

inline int TypeGeneral::LocalSize::GetM_K(){
	return m_K;
}

template<class T> void MRFEnergy<T>::InitializeCudaProcess(Node* m_nodeFirst,Node* m_nodeLast){
    Node *i,*j;
    MRFEdge* e;
    REAL tmp;
    int count,edgeNum_forward,edgeNum_backward,m_K,Kdest_K,m_dir,iter,k1,k2,k3;
    cudaError_t cudaStatus;
    cudaStatus=cudaMalloc((void**)&cu.Di,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.e_forward,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.e_backward,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.buf,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.this_data,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.forward_backward,MaxThreadNum*sizeof(int));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.backward_forward,MaxThreadNum*sizeof(int));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.NodeMessage,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.EdgeMessage_forward,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.EdgeMessage_backward,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.gamma_forward,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    cudaStatus=cudaMalloc((void**)&cu.gamma_backward,MaxThreadNum*sizeof(TypeGeneral::REAL));
    if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMalloc failed!\n");
    }
    //////////////////////////////////////////////////////////////
    // cudaMemcpy
    count=0;
    NodeIdx[count]=0;
    ForwardIdx[count]=0;
    BackwardIdx[count]=0;
    for(i=m_nodeFirst;i;i=i->m_next){
        m_K=(i->m_K).GetM_K();
        assert(NodeIdx[count]+m_K<MaxThreadNum);
        cudaStatus=cudaMemcpy(&cu.NodeMessage[NodeIdx[count]],(i->m_D).GetArrayHead(),m_K*sizeof(REAL),cudaMemcpyHostToDevice);
        /*if(cudaStatus!=cudaSuccess){
            ReleaseCudaProcess();
            mexErrMsgTxt("\ncudaMemcpy failed! cu.NodeMessage\n");
        }*/
        for(e=i->m_firstForward,edgeNum_forward=0;e;e=e->m_nextForward,edgeNum_forward+=m_K){
            assert(ForwardIdx[count]+edgeNum_forward+m_K<MaxThreadNum);
            cudaStatus=cudaMemcpy(&cu.e_forward[ForwardIdx[count]+edgeNum_forward],(e->m_message.GetMessagePtr())->GetArrayHead(),m_K*sizeof(REAL),cudaMemcpyHostToDevice);
            /*if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed! cu.e_forward\n");
            }*/
            j=e->m_head;
            Kdest_K=(j->m_K).GetM_K();
            assert((ForwardIdx[count]+edgeNum_forward+m_K)*Kdest_K<MaxThreadNum);
            cudaStatus=cudaMemcpy(&cu.EdgeMessage_forward[(ForwardIdx[count]+edgeNum_forward)*Kdest_K],e->m_message.getGeneralEdgeData(&m_dir),Kdest_K*m_K*sizeof(REAL),cudaMemcpyHostToDevice);
            /*if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed! cu.e_forward\n");
            }*/
        }
        for(e=i->m_firstBackward,edgeNum_backward=0;e;e=e->m_nextBackward,edgeNum_backward+=m_K){
            assert(BackwardIdx[count]+edgeNum_backward+m_K<MaxThreadNum);
            cudaStatus=cudaMemcpy(&cu.e_backward[BackwardIdx[count]+edgeNum_backward],(e->m_message.GetMessagePtr())->GetArrayHead(),m_K*sizeof(REAL),cudaMemcpyHostToDevice);
            /*if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed! cu.backward\n");
            }*/
            j=e->m_tail;
            Kdest_K=(j->m_K).GetM_K();
            assert((BackwardIdx[count]+edgeNum_backward+m_K)*Kdest_K<MaxThreadNum);
            cudaStatus=cudaMemcpy(&cu.EdgeMessage_backward[(BackwardIdx[count]+edgeNum_backward)*Kdest_K],e->m_message.getGeneralEdgeData(&m_dir),Kdest_K*m_K*sizeof(REAL),cudaMemcpyHostToDevice);
            /*if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed! cu.e_forward\n");
            }*/
        }
        count++;
        NodeIdx[count]=NodeIdx[count-1]+m_K;
        ForwardIdx[count]=ForwardIdx[count-1]+edgeNum_forward;
        BackwardIdx[count]=BackwardIdx[count-1]+edgeNum_backward;
    }
    iter=0;
    for(k1=count-1;k1>0;k1--){
        tmp=1.0/((k1>count-1-k1)?k1:(count-1-k1));
        for(k2=0;k2<m_K*k1;k2++,iter++)
            Gamma[iter]=tmp;
    }
    cudaStatus=cudaMemcpy(cu.gamma_forward,Gamma,iter*sizeof(REAL),cudaMemcpyHostToDevice);
    /*if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMemcpy failed! cu.e_forward\n");
    }*/
    iter=0;
    for(k1=0;k1<count;k1++){
        tmp=1.0/((k1>count-1-k1)?k1:(count-1-k1));
        for(k2=0;k2<m_K*k1;k2++,iter++)
            Gamma[iter]=tmp;
    }
    cudaStatus=cudaMemcpy(cu.gamma_backward,Gamma,iter*sizeof(REAL),cudaMemcpyHostToDevice);
    /*if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMemcpy failed! cu.e_forward\n");
    }*/
    iter=0;
    for(k1=0;k1<count-1;k1++){
        for(k2=0;k2<=k1;k2++){
            forward_backward_tmp[k1][k2]=iter;
            iter++;
        }
    }
    iter=0;
    for(k1=count-2;k1>=0;k1--){
        for(k2=0;k2<=k1;k2++){
            for(k3=0;k3<m_K;k3++){
                forward_backward[iter*m_K+k3]=forward_backward_tmp[count-2-k2][k1-k2]*m_K+k3;
                backward_forward[forward_backward_tmp[count-2-k2][k1-k2]*m_K+k3]=iter*m_K+k3;
            }
            iter++;
        }
    }
    assert(edgeNum_forward==iter*m_K);
    cudaStatus=cudaMemcpy(cu.forward_backward,forward_backward,iter*m_K*sizeof(int),cudaMemcpyHostToDevice);
    /*if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMemcpy failed! cu.e_forward\n");
    }*/
    cudaStatus=cudaMemcpy(cu.backward_forward,backward_forward,iter*m_K*sizeof(int),cudaMemcpyHostToDevice);
    /*if(cudaStatus!=cudaSuccess){
        ReleaseCudaProcess();
        mexErrMsgTxt("\ncudaMemcpy failed! cu.e_forward\n");
    }*/
    //////////////////////////////////////////////////////////////
}

template<class T> void MRFEnergy<T>::ReleaseCudaProcess(){
    cudaFree(cu.Di);
    cudaFree(cu.e_forward);
    cudaFree(cu.e_backward);
    cudaFree(cu.buf);
    cudaFree(cu.this_data);
    cudaFree(cu.forward_backward);
    cudaFree(cu.backward_forward);
    cudaFree(cu.NodeMessage);
    cudaFree(cu.EdgeMessage_forward);
    cudaFree(cu.EdgeMessage_backward);
    cudaFree(cu.gamma_forward);
    cudaFree(cu.gamma_backward);
}

__global__ void addKernel(TypeGeneral::REAL* Di,TypeGeneral::REAL* e_forward,TypeGeneral::REAL* e_backward,TypeGeneral::REAL* NodeMessage,int edgeNum_forward,int edgeNum_backward,int m_K){
    int i,idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<m_K){
        Di[idx]=NodeMessage[idx];
        for(i=idx;i<edgeNum_forward;i+=m_K)
            Di[idx]=Di[idx]+e_forward[i];
        for(i=idx;i<edgeNum_backward;i+=m_K)
            Di[idx]=Di[idx]+e_backward[i];
    }
}

__global__ void setBuf(TypeGeneral::REAL* buf,TypeGeneral::REAL* gamma,TypeGeneral::REAL* Di,TypeGeneral::REAL* message,int m_K){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<m_K)
        buf[idx]=gamma[idx]*Di[idx]-message[idx];
}

__global__ void setMinMessage1(TypeGeneral::REAL* message,TypeGeneral::REAL* buf,TypeGeneral::REAL* this_data,int m_K,int Kdest_K){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<Kdest_K){
        TypeGeneral::REAL vMin=buf[0]+this_data[0+idx*m_K],tmp;
        for(int i=1;i<m_K;i++){
            tmp=buf[i]+this_data[i+idx*m_K];
            if(vMin>tmp)
                vMin=tmp;
        }
        message[idx]=vMin;
    }
}

__global__ void setMinMessage2(TypeGeneral::REAL* message,TypeGeneral::REAL* buf,TypeGeneral::REAL* this_data,int m_K,int Kdest_K){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<Kdest_K){
        TypeGeneral::REAL vMin=buf[0]+this_data[idx+0*Kdest_K],tmp;
        for(int i=1;i<m_K;i++){
            tmp=buf[i]+this_data[idx+i*Kdest_K];
            if(vMin>tmp)
                vMin=tmp;
        }
        message[idx]=vMin;
    }
}

__global__ void updateMessage(TypeGeneral::REAL* message,int Kdest_K,TypeGeneral::REAL vMin){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<Kdest_K){
        message[idx]-=vMin;
    }
}

__global__ void linkForwardBackward(TypeGeneral::REAL* a,TypeGeneral::REAL* b,int* a_b,int start,int size){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<size)
        b[a_b[start+idx]]=a[start+idx];
}

template<class T> void MRFEnergy<T>::doCudaProcess(Node* m_nodeFirst,Node* m_nodeLast,REAL& lowerBound,bool lastIter,REAL* min_marginals,REAL** min_marginals_ptr){
    cudaError_t cudaStatus;
    MRFEdge* e;
    Node *i,*j;
    int m_K,Kdest_K,edgeNum_forward,edgeNum_backward,m_dir,iter,theGivenDir,count;
    REAL vMin,gamma;
    
    count=0;
    for(i=m_nodeFirst;i;i=i->m_next){
        m_K=NodeIdx[count+1]-NodeIdx[count];
        edgeNum_forward=ForwardIdx[count+1]-ForwardIdx[count];
        edgeNum_backward=BackwardIdx[count+1]-BackwardIdx[count];
        
        //mexPrintf("count    %d    NodeIdx   %d\n",count,NodeIdx[count]);
        /*cudaStatus=cudaMemcpy(shown,&cu.NodeMessage[NodeIdx[count]],m_K*sizeof(REAL),cudaMemcpyDeviceToHost);
        if(cudaStatus!=cudaSuccess){
            ReleaseCudaProcess();
            mexErrMsgTxt("\ncudaMemcpy failed! cu.NodeMessage\n");
        }
        mexPrintf("aaaa  %lf %lf %lf %lf %lf\n",shown[1], shown[3], shown[5], shown[7], shown[9]); //////////////////////////////////////
        */
        
        addKernel<<<getCUDABlockNum(m_K),CUDAThreadNum>>>(&cu.Di[NodeIdx[count]],&cu.e_forward[ForwardIdx[count]],&cu.e_backward[BackwardIdx[count]],&cu.NodeMessage[NodeIdx[count]],edgeNum_forward,edgeNum_backward,m_K);
        /*if((cudaGetLastError()!=cudaSuccess)||(cudaDeviceSynchronize()!=cudaSuccess)){
            ReleaseCudaProcess();
            mexErrMsgTxt("\naddKernel launch failed: addKernel\n");
        }*/
        
        /*cudaStatus=cudaMemcpy(shown,&cu.e_forward[ForwardIdx[count]],m_K*sizeof(REAL),cudaMemcpyDeviceToHost);
        if(cudaStatus!=cudaSuccess){
            ReleaseCudaProcess();
            mexErrMsgTxt("\ncudaMemcpy failed! cu.Di\n");
        }
        mexPrintf("bbbb  %lf %lf %lf %lf %lf\n",shown[1], shown[3], shown[5], shown[7], shown[9]); ///////////////////////////////////////
        cudaStatus=cudaMemcpy(shown,&cu.e_backward[BackwardIdx[count]],m_K*sizeof(REAL),cudaMemcpyDeviceToHost);
        if(cudaStatus!=cudaSuccess){
            ReleaseCudaProcess();
            mexErrMsgTxt("\ncudaMemcpy failed! cu.Di\n");
        }
        mexPrintf("cccc  %lf %lf %lf %lf %lf\n",shown[1], shown[3], shown[5], shown[7], shown[9]); ///////////////////////////////////////
        */
        
        /*cudaStatus=cudaMemcpy(shown,&cu.Di[NodeIdx[count]],m_K*sizeof(REAL),cudaMemcpyDeviceToHost);
        if(cudaStatus!=cudaSuccess){
            ReleaseCudaProcess();
            mexErrMsgTxt("\ncudaMemcpy failed! cu.Di\n");
        }
        mexPrintf("a  %lf %lf %lf %lf %lf\n",shown[1], shown[3], shown[5], shown[7], shown[9]); ///////////////////////////////////////
        */
        
        for(e=i->m_firstForward,edgeNum_forward=0;e;e=e->m_nextForward,edgeNum_forward+=m_K){
            assert(ForwardIdx[count]+edgeNum_forward+m_K<MaxThreadNum);
            j=e->m_head;
            //m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, e->m_gammaForward, 0, buf);
            gamma=e->m_gammaForward;
            setBuf<<<getCUDABlockNum(m_K),CUDAThreadNum>>>(cu.buf,&cu.gamma_forward[ForwardIdx[count]+edgeNum_forward],&cu.Di[NodeIdx[count]],&cu.e_forward[ForwardIdx[count]+edgeNum_forward],m_K);
            /*if((cudaGetLastError()!=cudaSuccess)||(cudaDeviceSynchronize()!=cudaSuccess)){
                ReleaseCudaProcess();
                mexErrMsgTxt("\naddKernel launch failed: setBuf\n");
            }*/
            Kdest_K=(j->m_K).GetM_K();
                        
            setMinMessage1<<<getCUDABlockNum(Kdest_K),CUDAThreadNum>>>(&cu.e_forward[ForwardIdx[count]+edgeNum_forward],cu.buf,&cu.EdgeMessage_forward[(ForwardIdx[count]+edgeNum_forward)*Kdest_K],m_K,Kdest_K);
            /*if((cudaGetLastError()!=cudaSuccess)||(cudaDeviceSynchronize()!=cudaSuccess)){
                ReleaseCudaProcess();
                mexErrMsgTxt("\naddKernel launch failed: setMinMessage\n");
            }*/
                        
            cudaStatus=cudaMemcpy(TheTmp,&cu.e_forward[ForwardIdx[count]+edgeNum_forward],Kdest_K*sizeof(REAL),cudaMemcpyDeviceToHost);
            /*if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed!\n");
            }*/
            vMin=TheTmp[0];
            for(iter=0;iter<Kdest_K;iter++){
                if(vMin>TheTmp[iter])
                    vMin=TheTmp[iter];
            }
            
            //mexPrintf("d  %d\n",vMin); ///////////////////////////////////
            
            updateMessage<<<getCUDABlockNum(Kdest_K),CUDAThreadNum>>>(&cu.e_forward[ForwardIdx[count]+edgeNum_forward],Kdest_K,vMin);
            /*if((cudaGetLastError()!=cudaSuccess)||(cudaDeviceSynchronize()!=cudaSuccess)){
                ReleaseCudaProcess();
                mexErrMsgTxt("\naddKernel launch failed: updateMessage\n");
            }*/
            /////////////////////////////////////////////
        }
        edgeNum_forward=ForwardIdx[count+1]-ForwardIdx[count];
        linkForwardBackward<<<getCUDABlockNum(edgeNum_forward),CUDAThreadNum>>>(cu.e_forward,cu.e_backward,cu.forward_backward,ForwardIdx[count],edgeNum_forward);
        
        for(e=i->m_firstForward,edgeNum_forward=0;e;e=e->m_nextForward,edgeNum_forward+=m_K){
            assert(ForwardIdx[count]+edgeNum_forward+m_K<MaxThreadNum);
            cudaStatus=cudaMemcpy((e->m_message.GetMessagePtr())->GetArrayHead(),&cu.e_forward[ForwardIdx[count]+edgeNum_forward],m_K*sizeof(REAL),cudaMemcpyDeviceToHost);
            /*if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed! cu.e_forward\n");
            }*/
        }
        if(lastIter&&min_marginals)
            *min_marginals_ptr+=m_K;
        count++;
    }
    lowerBound=0;
    count--;
    for(i=m_nodeLast;i;i=i->m_prev){
        m_K=NodeIdx[count+1]-NodeIdx[count];
        edgeNum_forward=ForwardIdx[count+1]-ForwardIdx[count];
        edgeNum_backward=BackwardIdx[count+1]-BackwardIdx[count];
        addKernel<<<getCUDABlockNum(m_K),CUDAThreadNum>>>(&cu.Di[NodeIdx[count]],&cu.e_forward[ForwardIdx[count]],&cu.e_backward[BackwardIdx[count]],&cu.NodeMessage[NodeIdx[count]],edgeNum_forward,edgeNum_backward,m_K);
        cudaStatus=cudaMemcpy(TheTmp,&cu.Di[NodeIdx[count]],m_K*sizeof(REAL),cudaMemcpyDeviceToHost);
        /*if(cudaStatus!=cudaSuccess){
            ReleaseCudaProcess();
            mexErrMsgTxt("\ncudaMemcpy failed!\n");
        }*/
        vMin=TheTmp[0];
        for(iter=0;iter<m_K;iter++){
            if(vMin>TheTmp[iter])
                vMin=TheTmp[iter];
        }
        updateMessage<<<getCUDABlockNum(m_K),CUDAThreadNum>>>(&cu.Di[NodeIdx[count]],m_K,vMin);
        /*if((cudaGetLastError()!=cudaSuccess)||(cudaDeviceSynchronize()!=cudaSuccess)){
            ReleaseCudaProcess();
            mexErrMsgTxt("\naddKernel launch failed: updateMessage\n");
        }*/
        
        lowerBound+=vMin;
        for(e=i->m_firstBackward,edgeNum_backward=0;e;e=e->m_nextBackward,edgeNum_backward+=m_K){
            j=e->m_tail;
            //m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, e->m_gammaBackward, 1, buf);
            gamma=e->m_gammaBackward;
            theGivenDir=1;
            setBuf<<<getCUDABlockNum(m_K),CUDAThreadNum>>>(cu.buf,&cu.gamma_backward[BackwardIdx[count]+edgeNum_backward],&cu.Di[NodeIdx[count]],&cu.e_backward[BackwardIdx[count]+edgeNum_backward],m_K);
            /*if((cudaGetLastError()!=cudaSuccess)||(cudaDeviceSynchronize()!=cudaSuccess)){
                ReleaseCudaProcess();
                mexErrMsgTxt("\naddKernel launch failed: setBuf\n");
            }*/
            Kdest_K=(j->m_K).GetM_K();
            /*assert(m_K*Kdest_K<MaxThreadNum);
            cudaStatus=cudaMemcpy(cu.this_data,this_data,m_K*Kdest_K*sizeof(REAL),cudaMemcpyHostToDevice);
            if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed! cu.this_data 2\n");
            }*/
            
            if(m_dir==theGivenDir)
                setMinMessage1<<<getCUDABlockNum(Kdest_K),CUDAThreadNum>>>(&cu.e_backward[BackwardIdx[count]+edgeNum_backward],cu.buf,&cu.EdgeMessage_backward[(BackwardIdx[count]+edgeNum_backward)*Kdest_K],m_K,Kdest_K);
            else
                setMinMessage2<<<getCUDABlockNum(Kdest_K),CUDAThreadNum>>>(&cu.e_backward[BackwardIdx[count]+edgeNum_backward],cu.buf,&cu.EdgeMessage_backward[(BackwardIdx[count]+edgeNum_backward)*Kdest_K],m_K,Kdest_K);
            /*if((cudaGetLastError()!=cudaSuccess)||(cudaDeviceSynchronize()!=cudaSuccess)){
                ReleaseCudaProcess();
                mexErrMsgTxt("\naddKernel launch failed: setMinMessage\n");
            }*/
            cudaStatus=cudaMemcpy(TheTmp,&cu.e_backward[BackwardIdx[count]+edgeNum_backward],Kdest_K*sizeof(REAL),cudaMemcpyDeviceToHost);
            /*if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed!\n");
            }*/
            vMin=TheTmp[0];
            for(iter=0;iter<Kdest_K;iter++){
                if(vMin>TheTmp[iter])
                    vMin=TheTmp[iter];
            }
            updateMessage<<<getCUDABlockNum(Kdest_K),CUDAThreadNum>>>(&cu.e_backward[BackwardIdx[count]+edgeNum_backward],Kdest_K,vMin);
            /*if((cudaGetLastError()!=cudaSuccess)||(cudaDeviceSynchronize()!=cudaSuccess)){
                ReleaseCudaProcess();
                mexErrMsgTxt("\naddKernel launch failed: updateMessage\n");
            }*/
            ////////////////////////////////////////////////////
            lowerBound+=vMin;
        }
        edgeNum_backward=BackwardIdx[count+1]-BackwardIdx[count];
        linkForwardBackward<<<getCUDABlockNum(edgeNum_backward),CUDAThreadNum>>>(cu.e_backward,cu.e_forward,cu.backward_forward,BackwardIdx[count],edgeNum_backward);
        for(e=i->m_firstBackward,edgeNum_backward=0;e;e=e->m_nextBackward,edgeNum_backward+=m_K){
            assert(BackwardIdx[count]+edgeNum_backward+m_K<MaxThreadNum);
            cudaStatus=cudaMemcpy((e->m_message.GetMessagePtr())->GetArrayHead(),&cu.e_backward[BackwardIdx[count]+edgeNum_backward],m_K*sizeof(REAL),cudaMemcpyDeviceToHost);
            /*if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed! cu.e_backward\n");
            }*/
        }
        if(lastIter&&min_marginals){
            *min_marginals_ptr-=m_K;
            cudaStatus=cudaMemcpy(*min_marginals_ptr,&cu.Di[NodeIdx[count]],m_K*sizeof(REAL),cudaMemcpyDeviceToHost);
            /*if(cudaStatus!=cudaSuccess){
                ReleaseCudaProcess();
                mexErrMsgTxt("\ncudaMemcpy failed!\n");
            }*/
        }
        count--;
    }
}
