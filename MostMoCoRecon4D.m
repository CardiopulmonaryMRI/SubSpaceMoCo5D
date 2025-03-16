function [recon_nufft,DyImage]=MostMoCoRecon4D(kdata1_,imgNufft1,imgLNufft1,sens1,sensL1,k,w,TrueIndex,LechoIdx,reconParam)

idxGroup = reconParam.idxGroup;
UsingGroupReg = reconParam.UsingGroupReg;
nframe = reconParam.nframe;
cardIdx = reconParam.cardIdx;
seqParam = reconParam.seqParam;

img_size = size(imgNufft1);
lowimg_size = size(imgLNufft1);
Imgratio = lowimg_size(1)/img_size(1);

kdata_bart = kdata1_/max(abs(kdata1_(:)));
kdata_bart=kdata_bart(:,TrueIndex,:);
k_bart=k(:,:,TrueIndex);
w_bart=w(:,TrueIndex);
[k_bart,w_bart,kdata_cs] = data_sorting(k_bart,w_bart,kdata_bart,nframe);
kdata_cs = permute(kdata_cs,[7,1,2,3,6,4,5]);% 1 286 12895 6c 1 5t 1
k_bart = permute(k_bart,[1,2,3,7,6,4,5]);%3 286 12895 1 1 5 1
w_bart = permute(w_bart,[7,1,2,6,5,3,4]);%1 286 12895 1 1 5 1
w_bart = w_bart/max(abs(w_bart(:)));
w_cs2 = sqrt(w_bart);

% numPoint = round(ratio*size(kdata1_,1));
% % kdata_cs = kdata_cs.*w_cs2;%wrong, not need
writecfl('k_cs',k_bart);
writecfl('w_cs',w_bart);
writecfl('w_cs2',w_cs2);
writecfl('k_csL',k_bart(:,LechoIdx,:,:,:,:,:));
writecfl('w_cs2L',w_cs2(:,LechoIdx,:,:,:,:,:));

[recon_nufft,param,k_cs,w_cs] = Dynamic_NUFFT(squeeze(kdata_cs.*w_cs2),squeeze(k_bart/max(k_bart(:))/2),squeeze(w_bart),img_size,sens1,nframe,8);
showDyImg(recon_nufft(:,:,:,[1,nframe]),round(idxGroup),604)
DyImage = 1;

%%
kdatadcf = bart('fmac',kdata_cs,w_bart);
img = bart('nufft -a',k_bart(:,LechoIdx,:,:,:,:,:),kdatadcf(:,LechoIdx,:,:,:,:,:));%69.5
lowres_nufft = squeeze(bart('fmac -C -s 8',img,sensL1));%96.2
showDyImg(squeeze(lowres_nufft(:,:,:,[1:nframe])),round(idxGroup*Imgratio),700)

% lowres_CS = squeeze(bart('pics -C 20 -i 80 -R T:32:0:0.01 -R T:7:0:0.005 -p w_cs2L -t k_csL',kdata_cs(:,1:numPoint,:,:,:,:,:),sensL));%
lowres_CS = squeeze(bart('pics -C 20 -i 20 -R T:7:0:0.005 -p w_cs2L -t k_csL',kdata_cs(:,LechoIdx,:,:,:,:,:),sensL1));%0.0001
showDyImg(squeeze(lowres_CS(:,:,:,[1,nframe])),round(idxGroup*Imgratio),701)
% implay(mat2gray(abs((permute(lowres_CS(:,:,:),[1,2,3])))));

clear kdata_bart FIDrawdata img kdatadcf

% save([pathname,'recon_nufft.mat'],'recon_nufft','-v7.3');

%SENSE recon
% recon_SENSE = XDGRASPRecon(recon1_nufft,param,0,0.01,idxGroup);
% showDyImg(recon_SENSE(:,:,:,[1,nframe]),idxGroup,450)
% save([pathname1,'SENSE8frame_2timesCoarserResolu.mat'],'recon_SENSE','-v7.3');

% [recon_XDGRASP]=XDGRASPRecon(recon_nufft,param,0.05,0.005,idxGroup);
% save([pathname,'XDGRASP5frame.mat'],'recon_XDGRASP','-v7.3');

% recon of dynamic image starts
ref = round(nframe/2);

%Image Registration
Img_L = abs(lowres_CS)./max(abs(lowres_CS(:)))+eps;
nbin = size(Img_L,4);
sizeH = size(recon_nufft(:,:,:,1));
sizeL = size(Img_L(:,:,:,1));
downsampling = sizeH./sizeL;

[B_L,F_L] = registrationPandG(ref,nbin,Img_L,sizeL,UsingGroupReg,seqParam);
gpuDevice(cardIdx);

B = zeros([sizeH,3,nbin,nbin]);
F = zeros([sizeH,3,nbin]);
[X1,Y1,Z1] = meshgrid(linspace(1,sizeL(2),sizeL(2)),...
    linspace(1,sizeL(1),sizeL(1)),...
    linspace(1,sizeL(3),sizeL(3)));
[Xq,Yq,Zq] = meshgrid(linspace(1,sizeL(2),sizeH(2)),...
    linspace(1,sizeL(1),sizeH(1)),...
    linspace(1,sizeL(3),sizeH(3)));
for ref = 1:nbin
for i = 1:nbin
    for j = 1:3       
        B(:,:,:,j,i,ref) = downsampling(j)*interp3(X1,Y1,Z1,B_L(:,:,:,j,i,ref),Xq,Yq,Zq,'cubic');
%         F(:,:,:,j,i) = downsampling(j)*interp3(X1,Y1,Z1,F_L(:,:,:,j,i),Xq,Yq,Zq,'cubic');
    end
end
end
showDyImgMF(B(:,:,:,3,1,3),idxGroup,200)
clear X1 Y1 Z1 Xq Yq Zq
% save([pathname1,'B_0.mat'],'B','-v7.3');
% 
vec = @(z)z(:);
x = recon_nufft;
scale = 1;
E = @(z)ForNUFFT_GPU(param,z)*scale;
Et = @(z)AdjNUFFT_GPU(param,z,[size(x,1),size(x,2),size(x,3)])*scale;
scale = sqrt(1/(eps+abs(mean(vec(Et(E(ones(size(x)))))))));
E = @(z)ForNUFFT_GPU(param,z)*scale;
Et = @(z)AdjNUFFT_GPU(param,z,[size(x,1),size(x,2),size(x,3)])*scale;
%param.y%0.0155
recon_CS = Et(param.y);%0.0063
normal = 1/max(abs(recon_CS(:)));%157
param.y = param.y*normal;%2.4499
% recon1_CS = Et(param.y);%1.0
clear recon_CS

mTVs = TVs(3);
sizeI2 = size(recon_nufft);

rho = 1;
rho1 = 1;

A = E;At = Et;
d = param.y;
Trans = Registra(nbin,B,F);
weightingCoef = 8;
mTVx = TVma(nbin,B,F,weightingCoef);

OutIter = 3;
Iter = 0;

% init
% sizeL = sizeH;
downsampling = sizeH./sizeL;

x_k0 = At(d);
z_k0 = zeros(size(mTVx*x_k0));
u_k0 = zeros(size(z_k0));
z1_k0 = zeros(size(mTVs*x_k0));
u1_k0 = zeros(size(z1_k0));

lambda1 = 0.02*max(abs(x_k0(:)));%0.02
lambda2 = 0.005*max(abs(x_k0(:)));%0.003

while(Iter<OutIter)
    
%     Ad = vec(Ad);
    mTVx = TVma(nbin,B,F,weightingCoef);
    Trans = Registra(nbin,B,F);
    
    iter = 0;
    tic;
    while(iter<5)
        Iter,iter
        iter = iter+1;
        
        % temporal L1 normalization
        if rho
            if 1 %TVt of T_theta(x)
          
                z_k1 = wthresh(mTVx*(x_k0)+u_k0/rho,'s',lambda1/rho);
               
            else %TVt of operation(x)
                
            end
            test = squeeze(reshape(z_k1,sizeI2));
            showDyImg(test(:,:,:,[1,nframe]),idxGroup,900)
        end

        % spatial L1 normalization
        if rho1 %spatial regularization
            if 1 %spatial TV
                
                z1_k1 = wthresh(mTVs*(x_k0)+u1_k0/rho1,'s',lambda2/rho1);
          
            else %spatial wavelet
                
            end
%             test = squeeze(reshape(z1_k1,sizeI2));
%             showDyImg(test(:,:,:,[1,nframe]),idxGroup,901)
        end

        % L2 opt conjugate gradient descent
        cg_iterM = 30;
        tol = 1e-3;
        x_k1 = conj_grad_x_MotionAverage(A,At,x_k0,d,rho,mTVx,z_k1,u_k0,rho1,mTVs,z1_k1,u1_k0,tol,cg_iterM);
        
        test = squeeze(reshape(x_k1,sizeI2));
        showDyImg(test(:,:,:,[1,nframe]),idxGroup,902)
        
        % dual update
        if rho, u_k1 = u_k0 + rho*(mTVx*(x_k1) -z_k1); end
        if rho1, u1_k1 = u1_k0 + rho1*(mTVs*(x_k1) -z1_k1); end

        converg = norm(x_k1(:)-x_k0(:))/norm(x_k0(:))
        if converg <= 1e-5, break; end
        
        % all update
        z_k0 = z_k1; 
        z1_k0 = z1_k1;
        x_k0 = x_k1;
        u_k0 = u_k1;
        u1_k0 = u1_k1;
    %     writecfl('temp_3D_mc1',x_k0);
        rho = rho*1.0;
        rho1 = rho1*1.0;
        
    end
    % L1 wavelet
    % data consistancy
    toc
    
    tmpImg = squeeze(reshape(x_k0,sizeI2));
    showDyImg(tmpImg(:,:,:,[1,nframe]),idxGroup,1000+Iter)
    
    Img = x_k0;
    Img = squeeze(abs(Img)./max(abs(Img(:))))+eps;
    
    if Iter<OutIter-1      
        
        [X1,Y1,Z1] = meshgrid(linspace(1,sizeH(2),sizeH(2)),...
            linspace(1,sizeH(1),sizeH(1)),...
            linspace(1,sizeH(3),sizeH(3)));
        [Xq,Yq,Zq] = meshgrid(linspace(1,sizeH(2),sizeL(2)),...
            linspace(1,sizeH(1),sizeL(1)),...
            linspace(1,sizeH(3),sizeL(3)));
        for i = 1:nbin                   
            Img_tmp(:,:,:,i) = interp3(X1,Y1,Z1,Img(:,:,:,i),Xq,Yq,Zq,'cubic');            
        end
%         showDyImg(Img_tmp(:,:,:,[1,nframe]),round(idxGroup),3000)
        [B_L,F_L] = registrationPandG(ref,nbin,Img_tmp,sizeL,UsingGroupReg,seqParam);
        gpuDevice(cardIdx);
        
        [X1,Y1,Z1] = meshgrid(linspace(1,sizeL(2),sizeL(2)),...
            linspace(1,sizeL(1),sizeL(1)),...
            linspace(1,sizeL(3),sizeL(3)));
        [Xq,Yq,Zq] = meshgrid(linspace(1,sizeL(2),sizeH(2)),...
            linspace(1,sizeL(1),sizeH(1)),...
            linspace(1,sizeL(3),sizeH(3)));
        for ref = 1:nbin
        for i = 1:nbin
            for j = 1:3       
                B(:,:,:,j,i,ref) = downsampling(j)*interp3(X1,Y1,Z1,B_L(:,:,:,j,i,ref),Xq,Yq,Zq,'cubic');
%                 F(:,:,:,j,i) = downsampling(j)*interp3(X1,Y1,Z1,F_L(:,:,:,j,i),Xq,Yq,Zq,'cubic');
            end
        end
        end
        
        showDyImgMF(B(:,:,:,3,1,3),idxGroup,200),
%         save([pathname1,'B_',num2str(Iter+1),'.mat'],'B','-v7.3');
    end
    clear X1 Y1 Z1 Xq Yq Zq
    
    Iter = Iter + 1;    
end
DyImage = squeeze(x_k0);
showDyImg(DyImage(:,:,:,[1,nframe]),idxGroup,400)
% save([pathname,'Echo1_MostMoCo5frame_c',num2str(weightingCoef),'_1l0_05_2l0_002_all0_75.mat'],'DyImage','-v7.3');