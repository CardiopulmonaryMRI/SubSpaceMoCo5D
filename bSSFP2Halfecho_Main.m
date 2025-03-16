
% main program of 5D Subspace-Moco reconstruction

% A five-dimensional cardiopulmonary MRI technique:
% Free-running simultaneous imaging of the entire lung and heart with
% isotropic resolution

%%
clc
clear 
close all
addpath(genpath('Code'));
addpath(genpath('5DMRI_GPUAccel'));
addpath(genpath('DynamicLung'));
addpath(genpath('5DMRI'));
setenv('CUDA_VISIBLE_DEVICES','2');
cardIdx = 1;
gpuDevice(cardIdx);

%% before recon, edit the recon parameters!!!!%%%%%%%%%%%

ThrowSeg = 0;
ratio = 0.7;%for MostMoCo,0.695 128,0.77 144
nframe = 5;%CS:   2584/8=323
ncardiac = 20;
prin = 12;
UsingGroupReg = 0;


pathname1='recon result/20240803/';%data saving path
pathname2=pathname1;
mkdir(pathname1);
mkdir(pathname2);

traidx = 110;%
coridx = 90;%
sagidx = 155;
idxGroup = round([traidx,coridx,sagidx]);

reconParam.idxGroup = idxGroup;
reconParam.UsingGroupReg = UsingGroupReg;
reconParam.nframe = nframe;
reconParam.ncardiac = ncardiac;
reconParam.cardIdx = cardIdx;
reconParam.prin = prin;

%% load Imaging data

load('Echo1Data.mat');load('Echo2Data.mat');%k-space data of dual-echo bSSFP
load('Echo1Traj.mat');load('Echo2Traj.mat');%k-space trajectory of two echoes;
load('Echo1DCF.mat');load('Echo2DCF.mat');%density compensation factor of two echoes;
load('SI_Navigator.mat');% superior-inferior navigator signal used for respiratory and cardiac motion estimation
load('ZGradientTraj.mat');% gardient waveform of physical Z gradient used for interpolation of ramp-sampled SI navigator;
load('seqParam.mat')% sequence parameters used for reconstruction
% These data are publicly avaiable at zenodo.org by searching "simultaneous 5D cardiopulmonary MRI" 

reconParam.seqParam = seqParam;

numPoint = round(ratio*size(kdata1_,1)); 
Lecho1Idx = 1:numPoint;
[imgNufft1,imgLNufft1,sens1,sensL1,k,w]=bartReconSens_New(kdata1_,DCF1r,Crds1r,idxGroup,ratio);
[imgNufft3,imgLNufft3,sens3,sensL3,k,w]=bartReconSens_New(kdata3_(end:-1:1,:,:),DCF3r(end:-1:1,:),-Crds3r(:,end:-1:1,:),idxGroup,ratio);

showDyImg(imgNufft1,idxGroup,107)
showDyImg(cat(4,imgNufft1,imgNufft3),idxGroup,108)
figure,plot(abs(FIDrawdata_temp(:,100,10)))
% showDyImg(imgNufftAll,idxGroup,109)
playImg = imgNufft1;
playImg = playImg/max(abs(playImg(:)));
% implay(mat2gray(abs((permute(playImg(:,:,:),[1,3,2]))),[0,0.1]));

[InterpSI1,InterpSI3] = InterpolationSI(kdata_SI(:,:,:),ZTraj,seqParam);
InterpSI1 = InterpSI1(1:208,:,:);
InterpSI3 = InterpSI3(1:208,:,:);

%% 4D MostMoCo Recon
if UsingSINav
    [TrueIndex,Res_Signal,cardiacSig,fResp,fCard] =...
        RespCardiacMotionEsti(cat(4,InterpSI1(:,:,:),InterpSI3(:,:,:)),ThrowSeg,seqParam);    
end
save([pathname1,'TrueIndex.mat'],'TrueIndex','-v7.3')
save([pathname1,'Res_Signal.mat'],'Res_Signal','-v7.3')
save([pathname1,'cardiacSig.mat'],'cardiacSig','-v7.3')
save([pathname1,'Freq.mat'],'fCard','fResp','-v7.3')

[imgNufft1,imgLNufft1,sens1,sensL1,k,w]=bartReconSens_New(kdata1_,DCF1r,Crds1r,idxGroup,ratio);

[recon_nufft1,DyImage1]=MostMoCoRecon4D(kdata1_,imgNufft1,imgLNufft1,sens1,sensL1,k,w,TrueIndex,Lecho1Idx,reconParam);
save([pathname1,'imgNufft1.mat'],'imgNufft1','-v7.3');
save([pathname1,'recon_nufft1.mat'],'recon_nufft1','-v7.3');
save([pathname1,'MostMoCo5frame1.mat'],'DyImage1','-v7.3');
  
[imgNufft3,imgLNufft3,sens3,sensL3,k,w]=bartReconSens_New(kdata3_(end:-1:1,:,:),DCF3r(end:-1:1,:),-Crds3r(:,end:-1:1,:),idxGroup,ratio);
numPoint = round(ratio*size(kdata3_,1)); 
Lecho3Idx = 1:numPoint;
[recon_nufft3,DyImage3]=MostMoCoRecon4D(kdata3_(end:-1:1,:,:),imgNufft3,imgLNufft3,sens3,sensL3,k,w,TrueIndex,Lecho3Idx,reconParam);
save([pathname2,'imgNufft3.mat'],'imgNufft3','-v7.3');
save([pathname2,'recon_nufft3.mat'],'recon_nufft3','-v7.3');
save([pathname2,'MostMoCo5frame3.mat'],'DyImage3','-v7.3');

%% 5D Subspace motion compensation Recon
if UsingSINav
    [TrueIndex,Res_Signal,cardiacSig,fResp,fCard] =...
        RespCardiacMotionEsti(cat(4,InterpSI1(:,:,:),InterpSI3(:,:,:)),ThrowSeg,seqParam);    
end
ratio = 0.7;
numPoint = round(ratio*size(kdata1_,1)); 
Lecho1Idx = 1:numPoint;

% echo1
[imgNufft1,imgLNufft1,sens1,sensL1,k,w]=bartReconSens_New(kdata1_,DCF1r,Crds1r,idxGroup,ratio);
img_size = size(imgNufft1);
img_sizeL = size(imgLNufft1);
downsampling = img_size./img_sizeL;
Imgratio = img_sizeL(1)/img_size(1);

[kdata_HalfSIu] = SI_sorting6D(cat(4,InterpSI1,InterpSI3),nframe,ncardiac,seqParam,Res_Signal,cardiacSig);
LSI1Idx = 1:size(sens1,1);
Vk = subspaceEsti_HalfSI(kdata_HalfSIu(1:end,:,:,:,:,:),prin);

[recon_nufft_5D1,recon_LR1,recon_RLR_2Ref1,recon_RLR_2Ref_AlignCardiac1]=...
    subspace5DRecon(kdata1_,k,w,reconParam,ratio,imgNufft1,imgLNufft1,Res_Signal,cardiacSig,Vk,sens1,sensL1);
save([pathname1,'recon_nufft_5D1.mat'],'recon_nufft_5D1','-v7.3');
save([pathname1,'recon_LR1.mat'],'recon_LR1','-v7.3');
save([pathname1,'recon_RLR_2Ref1.mat'],'recon_RLR_2Ref1','-v7.3');
save([pathname1,'recon_RLR_2Ref_AlignCardiac1.mat'],'recon_RLR_2Ref_AlignCardiac1','-v7.3');

% echo2
[imgNufft3,imgLNufft3,sens3,sensL3,k,w]=bartReconSens_New(kdata3_(end:-1:1,:,:),DCF3r(end:-1:1,:),-Crds3r(:,end:-1:1,:),idxGroup,ratio);
img_size = size(imgNufft3);
img_sizeL = size(imgLNufft3);
downsampling = img_size./img_sizeL;
Imgratio = img_sizeL(1)/img_size(1);
[recon_nufft_5D3,recon_LR3,recon_RLR_2Ref3,recon_RLR_2Ref_AlignCardiac3]=...
    subspace5DRecon(kdata3_(end:-1:1,:,:),k,w,reconParam,ratio,imgNufft3,imgLNufft3,Res_Signal,cardiacSig,Vk,sens3,sensL3);
save([pathname2,'recon_nufft_5D3.mat'],'recon_nufft_5D3','-v7.3');
save([pathname2,'recon_LR3.mat'],'recon_LR3','-v7.3');
save([pathname2,'recon_RLR_2Ref3.mat'],'recon_RLR_2Ref3','-v7.3');
save([pathname2,'recon_RLR_2Ref_AlignCardiac3.mat'],'recon_RLR_2Ref_AlignCardiac3','-v7.3');
