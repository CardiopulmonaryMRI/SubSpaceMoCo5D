function [recon1_nufft,param,k_cs,w_cs] = Dynamic_NUFFT(kdata_cs,k_cs,w_cs,img_size,sens1,nframe,sw)

SpkIdx = 1:round(size(k_cs,3));

k_cs = k_cs(:,:,SpkIdx,:);
w_cs = w_cs(:,SpkIdx,:);
kdata_cs = kdata_cs(:,SpkIdx,:,:);

FOVRatio = img_size/max(img_size(:));
k_cs(1,:,:,:) = k_cs(1,:,:,:)/FOVRatio(1);
k_cs(2,:,:,:) = k_cs(2,:,:,:)/FOVRatio(2);
k_cs(3,:,:,:) = k_cs(3,:,:,:)/FOVRatio(3);

param.E = GeneDygpu_NUFFTOperator(k_cs,w_cs,sens1,sw);
param.y = kdata_cs;
% max(abs(kdata_cs(:)))
recon1_nufft=AdjNUFFT_GPU(param,kdata_cs,img_size);%%%%%%%nufft recon

% dy_image=recon1_nufft/max(abs(recon1_nufft(:)));
% p=110;
% nt=size(dy_image,4);
% dy_nufft85=zeros(img_size,img_size,2*nt-1);
% for i=1:nt
%     dy_nufft85(:,:,i)=fliplr(abs(squeeze(dy_image(:,p,:,i)))');
% end
% for i=nt+1:2*nt-1
%     dy_nufft85(:,:,i)=fliplr(abs(squeeze(dy_image(:,p,:,2*nt-i)))');
% end
% implay(mat2gray(dy_nufft85));
% figure,imshow(fliplr(abs(squeeze(recon1_nufft(:,110,:,1)))'),[])