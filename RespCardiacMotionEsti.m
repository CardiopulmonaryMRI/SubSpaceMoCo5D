function [TrueIndex,Res_Signal,cardiacSig,fResp,fCard] = RespCardiacMotionEsti(kdata_SI,ThrowSeg,seqParam)
ntviews=size(kdata_SI,2) - ThrowSeg;
LCH =size(kdata_SI,3);
Necho=size(kdata_SI,4);
spk = seqParam.spk;
seg = seqParam.seg;
LPE = spk*seg;
ThrowLine=ThrowSeg*spk;
TR = seqParam.TR;
%nz: number of slice
%ntviews: number of acquired spokes
%nc: number of coil elements
%nx: readout point of each spoke (2x oversampling included)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Respiratory motion detection
kc = kdata_SI(1:4:end,:,:,:);
nx = size(kc,1);

%figure,plot(abs(kc(:,1,1)))
% Generate the z-projection profiles
%ZIP: Projection profiles along the Z dimension with interpolation.

% ZIP=abs(ifftshift(ifft(ifftshift(kc,1),size(kc,1),1),1)); %full echo
ZIP_tmp=abs((ifft(kc,size(kc,1),1))); %half echo
ZIP = ZIP_tmp(:,:,:,1);
if Necho>1
    for i = 1:Necho-1
        ZIP = cat(1,ZIP,ZIP_tmp(:,:,:,i+1));
    end
end
% ZIP=abs((ifft(kc,size(kc,1),1))); %half echo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        400*400*20   233*233*28
% figure,imshow(ZIP(:,:,1),[])
% figure,mesh(ZIP(:,:,1))
% Normalization of each projefCardction in each coil element
for ii=1:LCH
    for jj=1:ntviews
        maxprof=max(ZIP(:,jj,ii));
        minprof=min(ZIP(:,jj,ii));
        ZIP(:,jj,ii)=(ZIP(:,jj,ii)-minprof)./(maxprof-minprof);
    end
end
showPoint = 1000;%20s
% figure,imagesc(abs(ZIP(:,1:showPoint,5))),axis image,colormap(gray), axis off,%title('Respiratory Motion')

% for ii=1:LCH
%     for jj=1:nx  
%         ZIP(jj,:,ii)=filt1d(squeeze(ZIP(jj,:,ii)),TR*(spk+1)/1000,[0.6,2.5],100);
%     end
% end

% figure,imagesc(abs(ZIP(:,1:500,5))),axis image,colormap(gray), axis off;
%figure,plot(abs(ZIP(:,2,1)))
% Perform PCA on each coil element
% close all
kk=1;clear PCs

tmp=permute(ZIP(:,:,:),[1,3,2]);
tmp=abs(reshape(tmp,[size(tmp,1)*size(tmp,2),ntviews])');
[tmp2, V] = eig(tmp*tmp');
V = diag(V);
[junk, rindices] = sort(-1*V);
V = V(rindices);
tmp2 = tmp2(:,rindices);
PC = tmp2;

idx1 = ceil(0.15/(1/(TR*(spk+1)/1000))*ntviews)+1;
idx2 = ceil(0.5/(1/(TR*(spk+1)/1000))*ntviews)+1;
idx3 = ceil(0.8/(1/(TR*(spk+1)/1000))*ntviews)+1;%origin 0.8
idx4 = ceil(1.8/(1/(TR*(spk+1)/1000))*ntviews)+1;%origin2.0

freqcomp = fft(PC,[],1);
P2 = abs(freqcomp/ntviews);
P1 = P2(1:round(ntviews/2)+1,:);
P1(2:end-1,:) = 2*P1(2:end-1,:);
for i = 1:20
    P1(:,i)=smooth(P1(:,i),25,'lowess'); % do some moving average smoothing,origin 15
end
f = (1/(TR*(spk+1)/1000))*(0:round(ntviews/2))/ntviews;

[fResp1,iResp1] = max(P1(idx1:idx2,1:4),[],1);%1:4
iResp1 = iResp1+idx1-1;
[fRespMax,iResp2] = max(fResp1);
ridx = [iResp1(iResp2),iResp2];
fResp = f(ridx(1));

[fCard1,iCard1] = max(P1(idx3:idx4,1:14),[],1);%1:12 is important
iCard1 = iCard1+idx3-1;
[fCardMax,iCard2] = max(fCard1);
cidx = [iCard1(iCard2),iCard2];
fCard = f(cidx(1));

t = TR*(spk+1)/1000*[0:showPoint-1];
figure,plot(f(idx1:end),P1(idx1:end,ridx(2)),'k','LineWidth',1),%63:0.1Hz
hold on,plot(f(idx1:end),P1(idx1:end,cidx(2)),'r','LineWidth',1),xlabel('f(Hz)');legend('Respi','Cardi')%63:0.1Hz
% figure,plot(t,PC(1:showPoint,ridx(2)),'r');
% hold on,plot(t,PC(1:showPoint,cidx(2))+2*max(PC(1:showPoint,ridx(2))),'g'),xlabel('time:s');legend('Respi','Cardi')

fcutoff = (fResp+fCard)/2;
% Res_Signal = filt1d2(PC(:,ridx(2)),TR*(spk+1)/1000,[fcutoff],200);
Res_Signal = filt1d(PC(:,ridx(2)),TR*(spk+1)/1000,[0.005,fResp+0.3],200);
cardiacSig = filt1d(PC(:,cidx(2)),TR*(spk+1)/1000,[fCard-0.3,1.8],200);
    
% Res_Signal=smooth(Res_Signal,6,'lowess'); % do some moving average smoothing
% cardiacSig=smooth(cardiacSig,6,'lowess'); % do some moving average smoothing

%Normalize the signal for display
Res_Signal=Res_Signal-min(Res_Signal(:));
Res_Signal=Res_Signal./max(Res_Signal(:));
%Normalize the signal for display
cardiacSig=cardiacSig-min(cardiacSig(:));
cardiacSig=cardiacSig./max(cardiacSig(:));
% figure,plot(Res_Signal(1:end),'k'),legend('Respi')
% figure,plot(cardiacSig(1:end)+1,'r'),legend('Cardi')
% figure,plot(t,Res_Signal(1:showPoint),'k','LineWidth',1.5);
% hold on,plot(t,cardiacSig(1:showPoint)+0.8,'r','LineWidth',1.5),xlabel('time:s');legend('Respi','Cardi')

% figure,plot(t,Res_Signal(1:showPoint),'r');
% figure,plot(t,cardiacSig(1:showPoint)+0.8,'g')

figure,imagesc(abs(ZIP(:,1:showPoint,15))),axis image,colormap(gray), axis off,%title('Respiratory Motion')
hold on
plot(Res_Signal(1:showPoint)*100+220,'k','LineWidth',1) 
hold on
plot(cardiacSig(1:showPoint)*100+120,'r','LineWidth',1)        
hold off,legend('Respi','Cardi')
figure,subplot(211),plot(Res_Signal(1:2:end),'k')
subplot(212),plot(cardiacSig(1:2:end),'r')

% figure,imagesc(abs(ZIP(:,1:showPoint,5))),axis image,colormap(gray), axis off,
% figure,plot(Res_Signal(1:end))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sort the k-space data and trajectory according to respiratory motion 
[~,index]=sort(Res_Signal,'descend');
%figure,plot(Res_Signal,'r');
TrueIndex=1:LPE-ThrowLine;
A=1:spk;
for i=1:seg-ThrowSeg
    TrueIndex((i-1)*spk+1:i*spk) =(index(i)-1)*spk+A;
end

fResp = [fResp,iResp2];
fCard = [fCard,iCard2];
end

function k0f = filt1d(k0,TR,f0,N)
% 1d low pass filter
% Inputs:
%   k0: signal
%   TR: repetition time(s)
%   f0: motion freq(Hz)
%   N:  filter length
% Outputs:
%   k0f:filtered k0

if nargin<3
    f0 = 1;
end
if nargin<4
    N = 100;
end
N = ceil(max(2/(f0(1)*TR),N)/2)*2;
win = fir1(N,2*f0*TR,'bandpass');%'high' 'low' 'bandpass' 'stop'
k0e = zeros(length(k0(:))+N+1,1);
k0e(N/2+2:end-N/2) = k0(:);
k0e(1:N/2+1)=k0(1);
k0e(end-N/2+1:end)=k0(end);
k0f = conv(k0e(:),win,'same');
k0f = k0f(N/2+2:end-N/2);
end

function k0f = filt1d2(k0,TR,f0,N)
% 1d low pass filter
% Inputs:
%   k0: signal
%   TR: repetition time(s)
%   f0: motion freq(Hz)
%   N:  filter length
% Outputs:
%   k0f:filtered k0

if nargin<3
    f0 = 1;
end
if nargin<4
    N = 100;
end
N = ceil(max(2/(f0(1)*TR),N)/2)*2;
win = fir1(N,2*f0*TR,'low');%'high' 'low' 'bandpass' 'stop'
k0e = zeros(length(k0(:))+N+1,1);
k0e(N/2+2:end-N/2) = k0(:);
k0e(1:N/2+1)=k0(1);
k0e(end-N/2+1:end)=k0(end);
k0f = conv(k0e(:),win,'same');
k0f = k0f(N/2+2:end-N/2);
end