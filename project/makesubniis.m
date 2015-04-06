basedir = '/Users/nketz/Documents/Documents/boulder/CSCI5622/git/project';
datadir = fullfile(basedir,'SISS2015','Training');

subsstr = cellfun(@(x)(num2str(x,'%02d')),num2cell(1:28),'uniformoutput',0);
subs = 1:28;
seqs = {'MR_DWI','MR_Flair','MR_T1','MR_T2','OT'};

mindims = [230   230   153];
dims = [];
for iseq = 2:length(seqs)
    fprintf('sequence %s:\n\tSubject:',seqs{iseq});
    nii = cell(1,length(subs));
    seqimgs = nan([mindims length(subs)]);
    for isub = 1:length(subs)
        sub = num2str(subs(isub),'%02d');
        fprintf('%s,',sub);
        %find dir name
        dirs = dir(fullfile(datadir,sub,'VSD*'));
        dirs = {dirs.name};
        dirind = ~cellfun(@isempty,regexp(dirs,['VSD\.Brain\.XX\.O\.' seqs{iseq} '\.[0-9]{5}']));
        if sum(dirind)~=1; error('nonunique folder name for seq %s, sub %s',seqs{iseq},sub); end
        dirname = dirs{dirind};
        
        fname = fullfile(datadir,sub,dirname,[dirname '.nii']);
        nii{isub} = load_nii(fname);
        if isub ==1
            dims = size(nii{1}.img);
            seqimgs = nan([dims length(subs)]);
        end
        seqimgs(:,:,1:mindims(3),isub) = nii{isub}.img(:,:,1:mindims(3));
    end
    fprintf('\nSaving...');
    newnii.hdr = nii{1}.hdr;
    newnii.hdr.dime.dim = [4 size(seqimgs) 1 1 1];
    newnii.img = seqimgs;
    fname = fullfile(datadir,'compositimgs',[seqs{iseq} '_' num2str(length(subs)) 'subs.nii']);
    save_nii(newnii,fname);
    fprintf('done\n\n');
end
        
        %create nii file for each seq type with the 4th dim being subs
        
    
