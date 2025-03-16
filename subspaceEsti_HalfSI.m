function Vk = subspaceEsti_HalfSI(kdata_SIu,prin)

kdata_SIu = kdata_SIu(1:2:end,:,:,:,:,:);
[nx,ntviews,nc,nframe,ncardi,ne] = size(kdata_SIu);

ZIP =abs(ifft(kdata_SIu,nx,1)); %full echo, abs is right; complex is wrong.
ZIP_tmp = ZIP;
ZIP = ZIP_tmp(:,:,:,:,:,1);
if ne>1
   for i = 1:ne-1
      ZIP = cat(1,ZIP,ZIP_tmp(:,:,:,:,:,i+1)); 
   end
end
absZIP = abs(ZIP);
% ZIP=abs((ifft(kc,size(kc,1),1))); %half echo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        400*400*20   233*233*28
% figure,imshow(ZIP(:,:,1),[])
% figure,mesh(ZIP(:,:,1))
% Normalization of each projection in each coil element

    for cc = 1:ncardi
        for rr = 1:nframe
            for ii=1:nc
                for jj=1:ntviews
                    maxprof=max(absZIP(:,jj,ii,rr,cc,1));
                    minprof=min(absZIP(:,jj,ii,rr,cc,1));
                    ZIP(:,jj,ii,rr,cc,1)=(ZIP(:,jj,ii,rr,cc,1)-minprof)./(maxprof-minprof);
                end
            end
        end
    end


ZIP = squeeze(ZIP);

ZIP = squeeze(mean(ZIP,2));
figure,imshow(abs(squeeze(ZIP(:,5,:))),[])
figure,imshow(cat(1,squeeze(ZIP(1:end/2,13,:)),squeeze(ZIP(1:end/2,14,:))),[],'Border','tight')% Resp
figure,imshow((squeeze(ZIP(1:end/2,14,:))),[],'Border','tight')% Resp

m = reshape(ZIP,nx*nc*ne,nframe*ncardi);
[U,S,V] = svd(double(m),'econ');
figure,plot(diag(S));

% Uk = U*S(:,1:prin);%wrong
Uk = U(:,1:prin)*S(1:prin,1:prin);% coefficient
Vk = V(:,1:prin);
Vk = Vk';%temporal basis function
x = 1:size(Vk,2);
% figure,plot(x,abs(Vk(1,:)),'r',x,abs(Vk(2,:)),'y',x,abs(Vk(3,:)),'g'),legend;
figure,
for i = 1:prin
   subplot(3,5,i),plot(x,(Vk(i,:))),title(num2str(i)),set(gca,'YLim',[-0.2,0.2]); 
end
