function [InterpSI1,InterpSI3] = InterpolationSI(kdata_HalfSI,ZTraj,seqParam)

ramp = seqParam.ramp;
flattop1 = seqParam.flattop1;
Dwell = seqParam.Dwell;
UTEDataBuffer = seqParam.UTEDataBuffer;

[nx,np,nc] = size(kdata_HalfSI);
index1=1:(round((2*ramp+flattop1)/Dwell) + UTEDataBuffer);

ZTraj1 = abs(ZTraj(index1));
kdata_HalfSI1 = kdata_HalfSI(index1,:,:);
minZ = round(min(ZTraj1(:)));%0
maxZ = round(max(ZTraj1(:)));%320
pointIdx = 0;
InterpSI1 = zeros(maxZ-minZ+1,np,nc);

for i = minZ:maxZ
    pointIdx = pointIdx+1;
    idx = find(ZTraj1>=i-2 & ZTraj1<=i+2);
    distan = abs(ZTraj1(idx)-i);
    distan(distan ==0) = 1e-5;
    weightcoef = 1./distan;
    weightcoef = weightcoef/sum(weightcoef(:));
    InterpSI1(pointIdx,:,:) = sum(kdata_HalfSI1(idx,:,:).*(weightcoef'),1); 
end

index3=(round((2*ramp+flattop1)/Dwell) + UTEDataBuffer + 1):(nx);
ZTraj3 = abs(ZTraj(index3(end:-1:1)));
kdata_HalfSI3 = kdata_HalfSI(index3(end:-1:1),:,:);
minZ = round(min(ZTraj3(:)));%0
maxZ = round(max(ZTraj3(:)));%320
pointIdx = 0;
InterpSI3 = zeros(maxZ-minZ+1,np,nc);

for i = minZ:maxZ
    pointIdx = pointIdx+1;
    idx = find(ZTraj3>=i-2 & ZTraj3<=i+2);
    distan = abs(ZTraj3(idx)-i);
    distan(distan ==0) = 1e-5;
    weightcoef = 1./distan;
    weightcoef = weightcoef/sum(weightcoef(:));
    InterpSI3(pointIdx,:,:) = sum(kdata_HalfSI3(idx,:,:).*(weightcoef'),1); 
end