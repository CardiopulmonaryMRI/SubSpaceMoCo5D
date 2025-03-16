function [Br,Fr,Bc,Fc] = registrationP_5D_2direct(Imgo)

traidx = 109;%75?
coridx = 60;%85,110
sagidx = 135;
idxGroup = round([traidx,coridx,sagidx]);
% [Img] = normalDifSta(Img);

imgsize = size(Imgo);
nframe = imgsize(4);
ncardiac = imgsize(5);

Img = zeros(imgsize);
for i = 1:ncardiac
   Img(:,:,:,:,i) = abs(imgauss4d(Imgo(:,:,:,:,i),.5));
end
% Img = abs(imgauss4d(Img,.5));

%     showDyImg(squeeze(dImg(:,:,:,[1:nframe])),round(idxGroup*Imgratio),701)
Br = (single(zeros([size(Img(:,:,:,1)),3,nframe,ncardiac])));
Fr = (single(zeros([size(Img(:,:,:,1)),3,nframe,ncardiac])));
Bc = (single(zeros([size(Img(:,:,:,1)),3,nframe,ncardiac])));
Fc = (single(zeros([size(Img(:,:,:,1)),3,nframe,ncardiac])));

% Img = double(Img);
Img = single(Img/max(Img(:)));

refr = round(nframe/2);
refc = round(ncardiac/2);

Smo = 1.5;
for j = 1:ncardiac
        fixedGPU = gpuArray(Img(:,:,:,refr,j));
%         fixedHist = imhist(fixedGPU);
        for i = 1:nframe
            movingGPU = gpuArray(Img(:,:,:,i,j));   
%             movingGPU = histeq(movingGPU,fixedHist);% bring background noise in motion field
            Br(:,:,:,:,i,j) = gather(imregdemons(movingGPU,fixedGPU,[100,50,25]*2,'PyramidLevels',3,'DisplayWaitbar',false,'AccumulatedFieldSmoothing',Smo));
            Fr(:,:,:,:,i,j) = gather(imregdemons(fixedGPU,movingGPU, [100,50,25]*2,'PyramidLevels',3,'DisplayWaitbar',false,'AccumulatedFieldSmoothing',Smo));
%             Br(:,:,:,:,i,j) = (B_GPU);
%             Fr(:,:,:,:,i,j) = (F_GPU);
        end  
end

for i = 1:nframe
        fixedGPU = gpuArray(Img(:,:,:,i,refc));
%         fixedHist = imhist(fixedGPU);
        for j = 1:ncardiac
            movingGPU = gpuArray(Img(:,:,:,i,j));   
%             movingGPU = histeq(movingGPU,fixedHist);
            Bc(:,:,:,:,i,j) = gather(imregdemons(movingGPU,fixedGPU,[100,50,25]*2,'PyramidLevels',3,'DisplayWaitbar',false,'AccumulatedFieldSmoothing',Smo));
            Fc(:,:,:,:,i,j) = gather(imregdemons(fixedGPU,movingGPU, [100,50,25]*2,'PyramidLevels',3,'DisplayWaitbar',false,'AccumulatedFieldSmoothing',Smo));
%             Bc(:,:,:,:,i,j) = (B_GPU);
%             Fc(:,:,:,:,i,j) = (F_GPU);
        end  
end





