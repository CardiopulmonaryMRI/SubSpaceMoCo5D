function [B,F] = registrationPandG(ref,nbin,Img,imgsize,UsingGroupReg,seqParam)

Img = abs(imgauss4d(Img,.5));
[Img] = normalDifSta(Img);

dImg = squeeze(Img);
dImg = abs(dImg/max(abs(dImg(:))))*32767;
%     showDyImg(squeeze(dImg(:,:,:,[1:nframe])),round(idxGroup*Imgratio),701)
[status, ~, ~] = rmdir('mcFile','s');
mkdir mcFile;
sElastixPath = '/home/dzk/DingZekang/elastix/bin';
mcPath = '/home/dzk/DingZekang/DynamicLung/mcFile';
mcParaPath ='/home/dzk/DingZekang/DynamicLung/mcParaFile/';
sElastixParamFile = cell(1,2);
sElastixParamFile{1} = 'par000.forward.txt';
sElastixParamFile{2} = 'par001.inverse.txt';
voxelSize =  [seqParam.FOV(1)/imgsize(1), seqParam.FOV(2)/imgsize(2),seqParam.FOV(3)/imgsize(3)];
save_nii(make_nii(dImg,voxelSize,[0,0,0],16),[mcPath,filesep,'Image01.nii']);

if UsingGroupReg
    [status2,~] = system([sElastixPath, filesep, 'elastix -f ',mcPath,filesep,'Image01.nii -m ',mcPath,filesep,'Image01.nii', ' -out ',mcPath,filesep, ' -p ', mcParaPath,sElastixParamFile{1},...
      ' -p ', mcParaPath,sElastixParamFile{2}]); % call elastix forward
end

B = zeros([size(Img(:,:,:,1)),3,nbin,nbin]);
F = 0;
if ~ UsingGroupReg
    traidx = 250;
    coridx = 95;
    sagidx = 160;
    idxGroup = round([traidx,coridx,sagidx]);
    Img = double(Img);
    Img = Img/max(Img(:));
%     Img = uint16((Img*65535));
    Imgeq = zeros(size(Img));
    ImgReg = zeros(size(Img));
    for ref = 1:nbin
        fixedGPU = gpuArray(Img(:,:,:,ref));
    %     fixedGPU = histeq(fixedGPU);
        fixedHist = imhist(fixedGPU);
        for i = 1:nbin

            movingGPU = gpuArray(Img(:,:,:,i));   
%             movingGPU = histeq(movingGPU,fixedHist);
%             Imgeq(:,:,:,i) = gather(movingGPU);


%             [B_GPU,movingRegGPU] = imregdemons(movingGPU,fixedGPU,[200,100,50,25]*4,'PyramidLevels',4,'DisplayWaitbar',false,'AccumulatedFieldSmoothing',1.0);
            [B_GPU,movingRegGPU] = imregdemons(movingGPU,fixedGPU,      [100,50,25]*2,'PyramidLevels',3,'DisplayWaitbar',false,'AccumulatedFieldSmoothing',1.5);
%             F_GPU = imregdemons(fixedGPU,movingGPU,[200,100,50,25]*4,'PyramidLevels',4,'DisplayWaitbar',false,'AccumulatedFieldSmoothing',1.5);
            B(:,:,:,:,i,ref) = gather(B_GPU);
%             F(:,:,:,:,i) = gather(F_GPU);
%             ImgReg(:,:,:,i) = gather(movingRegGPU);
        end
    end
%     F = inv_field(B);
    % writecfl('Ireg',Ireg);
%     showDyImg(Img,round(idxGroup*0.75),2000)
%     showDyImg(Imgeq,round(idxGroup*0.75),2001)
%     showDyImg(ImgReg,round(idxGroup*0.75),2002)
    a= 1;
else
    
 
        [status3,~] = system([sElastixPath, filesep, 'combine.py point ',mcPath,filesep,'TransformParameters.0.txt ',...
                mcPath,filesep,'TransformParameters.1.txt ',mcPath,filesep,'Combined.0.txt ',mcPath,filesep,'Combined.1.txt ',num2str(ref-1)]); 
        [status4,~] = system([sElastixPath, filesep, 'transformix -tp ',mcPath,filesep,'Combined.1.txt -def ','all ','-out ',mcPath,filesep]);

        tmpPath = pwd;
        cd(mcPath);
        % ImageData = read_mhd('result.mhd');
        SData = read_mhd('deformationField.mhd');
        B(:,:,:,1,:) = SData.datax./voxelSize(1); % in [pixel]
        B(:,:,:,2,:) = SData.datay./voxelSize(2); % in [pixel]
        B(:,:,:,3,:) = SData.dataz./voxelSize(3); % in [pixel]
        cd(tmpPath);
    %     showDyImgMF(squeeze(B_L(:,:,:,1,[1:nframe])),round(idxGroup),401)
    %     showDyImgMF(squeeze(B_L(:,:,:,2,[1:nframe])),round(idxGroup),402)
    %     showDyImgMF(squeeze(B(:,:,:,3,[1,end])),round(idxGroup),403)
    %     for i = 1:size(dImg,4)
    %         reg_Img(:,:,:,i) = imwarp(recon_CS(:,:,:,i),reg_field(:,:,:,:,i)); 
    %     end
    %     showDyImg(reg_Img,idxGroup,1000)
        for groupRef = 1:nbin  
            [status3,~] = system([sElastixPath, filesep, 'combine.py point ',mcPath,filesep,'TransformParameters.0.txt ',...
                mcPath,filesep,'TransformParameters.1.txt ',mcPath,filesep,'Combined.0.txt ',mcPath,filesep,'Combined.1.txt ',num2str(groupRef-1)]); 
            [status4,~] = system([sElastixPath, filesep, 'transformix -tp ',mcPath,filesep,'Combined.1.txt -def ','all ','-out ',mcPath,filesep]);
            tmpPath = pwd;
            cd(mcPath);
            % ImageData = read_mhd('result.mhd');
            SData = read_mhd('deformationField.mhd');
            F(:,:,:,1,groupRef) = SData.datax(:,:,:,ref)./voxelSize(1); % in [pixel]
            F(:,:,:,2,groupRef) = SData.datay(:,:,:,ref)./voxelSize(2); % in [pixel]
            F(:,:,:,3,groupRef) = SData.dataz(:,:,:,ref)./voxelSize(3); % in [pixel]
            cd(tmpPath);
        end
    %         F = inv_field(B);
end