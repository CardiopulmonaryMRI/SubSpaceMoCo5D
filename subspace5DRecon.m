function [recon_nufft_5D1,recon_LR1,recon_RLR_2Ref1,recon_RLR_2Ref_AlignCardiac1]=...
    subspace5DRecon(kdata1_,k,w,reconParam,ratio,imgNufft1,imgLNufft1,Res_Signal,cardiacSig,Vk,sens1,sensL1)

idxGroup = reconParam.idxGroup;
nframe = reconParam.nframe;
ncardiac = reconParam.ncardiac;
prin = reconParam.prin;
cardIdx = reconParam.cardIdx;
seqParam = reconParam.seqParam;

numPoint = round(ratio*size(kdata1_,1)); 
Lecho1Idx = 1:numPoint;
img_size = size(imgNufft1);
img_sizeL = size(imgLNufft1);
downsampling = img_size./img_sizeL;
Imgratio = img_sizeL(1)/img_size(1);

%% Low resolution
kdata_bart = kdata1_/max(abs(kdata1_(:)));
kdata_bart=kdata_bart(Lecho1Idx,:,:);
k_bart=k(:,Lecho1Idx,:);
w_bart=w(Lecho1Idx,:);
[k_bart,w_bart,kdata_cs] = data_sorting5D(k_bart,w_bart,kdata_bart,nframe,ncardiac,seqParam,Res_Signal,cardiacSig);
kdata_cs = permute(kdata_cs,[7,1,2,3,6,4,5]);% 1 286 12895 6c 1 5r 8c
k_bart = permute(k_bart,[1,2,3,7,6,4,5]);%3 286 12895 1 1 5 8
w_bart = permute(w_bart,[7,1,2,6,5,3,4]);%1 286 12895 1 1 5 8
w_bart = w_bart/max(abs(w_bart(:)));
w_cs2 = sqrt(w_bart);

[recon_nufft_Low,param,k_cs,w_cs] = ...
    Dynamic_NUFFT5D(squeeze(kdata_cs.*w_cs2),squeeze(k_bart/max(k_bart(:))/2),squeeze(w_bart),img_sizeL,sensL1,8);
recon_nufft_Low = recon_nufft_Low/max(abs(recon_nufft_Low(:)));
showDyImg(squeeze(recon_nufft_Low(:,:,:,1,[1,end])),round(Imgratio*idxGroup),605)
% showDyImg(squeeze(recon_nufft_Low(:,:,:,[1,end],[1])),round(Imgratio*idxGroup),606)
% implay(mat2gray(abs(permute(squeeze(recon_nufft_5D(:,idxGroup(2),:,1,:)),[2,1,3])),[0,0.25]));
clear kdata_bart FIDrawdata img kdatadcf

% [recon_LR1]=LowRankRecon5D(recon_nufft_Low,param,0.01,0.01,0.04,idxGroup,prin,Vk,1,Imgratio);gpuDevice(cardIdx);%recon1_nufft,param,lambdaTR,lambdaTC,lambdaS,idxGroup,prin,Vk
[recon_LR1]=LowRankRecon5D(recon_nufft_Low,param,0.005,0.01,0.04,idxGroup,prin,Vk,1,Imgratio);gpuDevice(cardIdx);%recon1_nufft,param,lambdaTR,lambdaTC,lambdaS,idxGroup,prin,Vk
recon_LR1 = recon_LR1/max(abs(recon_LR1(:)));
showDyImg(squeeze(recon_LR1(:,:,:,1,[1,end])),round(Imgratio*idxGroup),207)
% showDyImg(squeeze(recon_LR1(:,:,:,[1,end],1)),round(Imgratio*idxGroup),257)

%  2Ref: respiratoty and cardiac
Image4Reg = abs(recon_LR1/max(abs(recon_LR1(:))));
[BrL,FrL,BcL,FcL] = registrationP_5D_2direct(Image4Reg);
gpuDevice(cardIdx);

% Br = BrL;Fr = FrL;Bc = BcL;Fc = FcL;

Br = zeros([img_size,3,nframe,ncardiac]);
Fr = zeros(size(Br));Bc = zeros(size(Br));Fc = zeros(size(Br));
for i = 1:3
   for j = 1:nframe
      for c = 1:ncardiac
          Br(:,:,:,i,j,c) = downsampling(i)*imresize3(BrL(:,:,:,i,j,c),img_size);
          Fr(:,:,:,i,j,c) = downsampling(i)*imresize3(FrL(:,:,:,i,j,c),img_size);
          Bc(:,:,:,i,j,c) = downsampling(i)*imresize3(BcL(:,:,:,i,j,c),img_size);
          Fc(:,:,:,i,j,c) = downsampling(i)*imresize3(FcL(:,:,:,i,j,c),img_size);
      end
   end
end

showDyImgMF(Br(:,:,:,3,1,1),idxGroup,300)

%% high resolution
kdata_bart = kdata1_/max(abs(kdata1_(:)));
kdata_bart=kdata_bart(:,:,:);
k_bart=k(:,:,:);
w_bart=w(:,:);
[k_bart,w_bart,kdata_cs] = data_sorting5D(k_bart,w_bart,kdata_bart,nframe,ncardiac,seqParam,Res_Signal,cardiacSig);
kdata_cs = permute(kdata_cs,[7,1,2,3,6,4,5]);% 1 286 12895 6c 1 5r 8c
k_bart = permute(k_bart,[1,2,3,7,6,4,5]);%3 286 12895 1 1 5 8
w_bart = permute(w_bart,[7,1,2,6,5,3,4]);%1 286 12895 1 1 5 8
w_bart = w_bart/max(abs(w_bart(:)));
w_cs2 = sqrt(w_bart);

[recon_nufft_5D1,param,k_cs,w_cs] =...
    Dynamic_NUFFT5D(squeeze(kdata_cs.*w_cs2),squeeze(k_bart/max(k_bart(:))/2),squeeze(w_bart),img_size,sens1,8);
recon_nufft_5D1 = recon_nufft_5D1/max(abs(recon_nufft_5D1(:)));
showDyImg(squeeze(recon_nufft_5D1(:,:,:,1,[1,end])),round(idxGroup),605)
% showDyImg(squeeze(recon_nufft_5D1(:,:,:,[1,end],[1])),round(idxGroup),606)
clear kdata_bart FIDrawdata img kdatadcf

param.TransR = Registra5D(nframe,ncardiac,Br,Fr);
param.TransC = Registra5D(nframe,ncardiac,Bc,Fc);
% [recon_RLR_2Ref1,param]=RegisLowRankRecon5D(recon_nufft_5D1,param,0.045,0.045,0.01,idxGroup,prin,Vk,seqParam,cardIdx);%recon1_nufft,param,lambdaTR,lambdaTC,lambdaS,idxGroup,prin,Vk
[recon_RLR_2Ref1,param]=RegisLowRankRecon5D(recon_nufft_5D1,param,0.02,0.03,0.01,idxGroup,prin,Vk,seqParam,cardIdx);%recon1_nufft,param,lambdaTR,lambdaTC,lambdaS,idxGroup,prin,Vk
recon_RLR_2Ref1 = recon_RLR_2Ref1/max(abs(recon_RLR_2Ref1(:)));
showDyImg(squeeze(recon_RLR_2Ref1(:,:,:,1,[1,end])),round(idxGroup),157)
% showDyImg(squeeze(recon_RLR_2Ref1(:,:,:,[1,end],1)),round(idxGroup),167)
recon_RLR_2Ref_AlignCardiac1 = gather(param.TransC*(gpuArray(recon_RLR_2Ref1)));

