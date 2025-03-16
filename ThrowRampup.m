function [rawdata_temp]=ThrowRampup(seqParam,rawdata,ThrowSeg)

spk = seqParam.spk;
ThrowLine = ThrowSeg * spk;
rawdata_temp=rawdata(ThrowLine+1:end,:,:);
rawdata_temp = permute(rawdata_temp,[2,1,3]);