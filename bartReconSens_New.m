function [imgNufft,imgLNufft,sens,sensL,k,w]=bartReconSens_New(kdata1_,DCF1,Crds1,idxGroup,ratio)

kdata1 = kdata1_/max(abs(kdata1_(:)));
kdata1 = permute(kdata1,[4,1,2,3]);
DCF1 = permute(DCF1,[4,1,2,3]);

% high resolution sensitivity map
kdatadcf0 = bart('fmac',kdata1,DCF1);
kdatadcf = bart('slice 5 0',kdatadcf0);
traj = bart('slice 5 0',Crds1);
img = bart('nufft -a',traj,kdatadcf);%69.5
% showDyImg(img,[145 95 90],104)
% showDyImg(img(:,:,:,1:8),idxGroup,104)
% showDyImg(img(:,:,:,9:16),idxGroup,105)
% showDyImg(img(:,:,:,17:end),idxGroup,106)

% imgss = bart('rss 8',img);
% showDyImg(imgss,[145 95 90],105)
ksp = bart('fft 7',img);

siz = size(img,1,2,3);
sizeRatio = siz/max(siz);
cpoint1 = max(round(sizeRatio*24),round(32*sizeRatio));
strtmp1 = [num2str(cpoint1(1)),':',num2str(cpoint1(2)),':',num2str(cpoint1(3))];

sens = bart(['caldir ',strtmp1],ksp);
% showDyImg(sens,[145 95 90],106)
imgNufft = bart('fmac -C -s 8',img,sens);%96.2
showDyImg(imgNufft,idxGroup,107)

% low resolution sensitivity map
% lowimg_size = round(size(imgNufft)*ratio);
% imgL = bart(['nufft -a -d ',num2str(lowimg_size(1)),':',num2str(lowimg_size(2)),':',num2str(lowimg_size(3))],traj,kdatadcf);

numPoint = round(ratio*size(kdata1,2));
imgL = bart('nufft -a',traj(:,1:numPoint,:),kdatadcf(:,1:numPoint,:,:));
% showDyImg(imgL,[145 95 90],108)
ksp = bart('fft 7',imgL);

cpoint2 = max(round(sizeRatio*24),round(32*ratio*sizeRatio));
strtmp2 = [num2str(cpoint2(1)),':',num2str(cpoint2(2)),':',num2str(cpoint2(3))];

sensL = bart(['caldir ',strtmp2],ksp);
% showDyImg(sensL,[145 95 90],109)
imgLNufft = bart('fmac -C -s 8',imgL,sensL);
Imgratio = size(imgLNufft,1)/size(imgNufft,1);
showDyImg(imgLNufft,round(idxGroup*Imgratio),110)
k = squeeze(Crds1);
w = squeeze(DCF1);
sens = sens/max(abs(sens(:)));
sensL = sensL/max(abs(sensL(:)));
