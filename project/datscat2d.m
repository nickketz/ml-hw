function datscat2d(f1,f2,ft,cfg)
%
% scatter plot of MRI data along two MR image types with target voxels
% sized larger than non-target, and color is different for each patient
% Plots are done one 'z' slice at a time
%
%input:
%   f1: x1 nii file name and path
%   f2: x2 nii file name and path
%   ft: y nii file name and path
%   cfg: config struc
%       cfg.dozscore: bool, zscore x vars (def: 1)
%       cfg.npoints: num of rand points to sample from each sub (def:500)
%       cfg.slices: vector of which z slices to view (def:all)
%       cfg.dosubplot: bool, plot slices in subplots (def:0)
%
% example:
% datscat2d('MR_DWI_28subs_3mm.nii','MR_Flair_28subs_3mm.nii','OT_28subs_3mm.nii');


X1 = load_nii(f1);
X2 = load_nii(f2);
Y = load_nii(ft);

close all
figure('color','white');
hold on

npoints = 500;
dozscore = 1;
slices = 'all';
dosubplot = 0;
if exist('cfg','var')
    if isfield(cfg, 'npoints');     npoints = cfg.npoints;      end
    if isfield(cfg, 'dozscore');    dozscore = cfg.dozscore;    end
    if isfield(cfg, 'slices');      slices = cfg.slices;        end
    if isfield(cfg, 'dosubplot');   dosubplot = cfg.dosubplot;  end
end


maxx1 = max(max(max(max(X1.img))));
maxx2 = max(max(max(max(X2.img))));
nsubs = size(Y.img,4);

dims = size(Y.img);

if ischar(slices)
    slices = 1:dims(3);
end

m = floor(length(slices)^.5);
n = ceil(length(slices)^.5);
if m*n<length(slices); m = ceil(length(slices)^.5); end
zcnt = 0;
for iz = slices
    %fprintf('\n\nslice %i:\n\t',iz);
    if ~dosubplot
        clf;
    else
        zcnt = zcnt+1;
        subplot(m,n,zcnt);
    end
    
    for isub = 1:nsubs
        %fprintf('sub%i, ',isub);
        
        x1 = X1.img(:,:,iz,isub);
        x2 = X2.img(:,:,iz,isub);
        x1 = reshape(x1,[prod(dims(1:2)) 1]);
        x2 = reshape(x2,[prod(dims(1:2)) 1]);       
        
        if dozscore
            tmpx1 = reshape(X1.img(:,:,:,isub),[prod(dims(1:3)) 1]);
            mux1 = nanmean(tmpx1);
            stdx1 = nanstd(tmpx1);
            tmpx2 = reshape(X2.img(:,:,:,isub),[prod(dims(1:3)) 1]);
            mux2 = nanmean(tmpx2);
            stdx2 = nanstd(tmpx2);
            
            x1 = (x1-mux1)/stdx1;
            x2 = (x2-mux2)/stdx2;
            maxx1 = 7;
            maxx2 = 7;
        end
        
        y = Y.img(:,:,iz,isub);
        logy = reshape(y,[prod(dims(1:2)), 1]);
        rndind = randperm(length(x1));
        scatter(x1(rndind(1:npoints)),x2(rndind(1:npoints)), (logy(rndind(1:npoints))*30)+5,'filled');
        xlim([0,maxx1]);
        ylim([0,maxx2]);
        set(gca,'fontsize',15);
        ylabel(strrep(f2,'_',' '));
        xlabel(strrep(f1,'_',' '));
        hold on
        ntargs = nansum(nansum(nansum((squeeze(Y.img(:,:,iz,:))))));
        title(sprintf('Slice %i, %i subs, %i targets',iz,nsubs,ntargs));        
        
    end
    drawnow
    
end

















