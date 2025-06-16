%test_mWidarHMMtracker_perfectDA.m
%%% Test HMM multi-target tracker with perfect data association using
%%% mWidar simulation data provided by Wavesens.
%%% Note that squared image peaks data and data association results
%%% were already pre-processed and tagged to ground truth target trajectories in a
%%% different script (tag_mWidarDataTruth.m), and HMM model parameters for
%%% STM and likelihood observation functions were also pre-generated in
%%% other scripts (hmm_2DTruncGaussSTM.m and test_mWidarLikelihoodModel.m).

clc, clear, close all

%% load simulated mWidar imaging data
load data.mat -mat data
%% load true target trajectory data for validation:
load truetargtrajs.mat pttraj Xbounds Ybounds Xblind Yblind
%% load pre-generated squared image peak extraction and Data Association results
load precalc_imagepeaktargetdataassoc.mat target_peakmatches impeaks_xy
%% load HMM model params
load precalc_imagegridHMMSTMn15.mat A
A_slow = A; clear A
load precalc_imagegridHMMSTMn30.mat A
A_fast = A; clear A

% load precalc_imagegrid
load precalc_imagegridHMMEmLike.mat pointlikelihood_image

%% Define true target entry and exit time steps from surveillance area
targetEntryTimes = [4 5 1]; %target 2 actually enters at k=4 but is not seen until k=5
targetExitTimes = [19 19 22]; %set target 3 exit time to 1+ final stop time

%% Define grid
Lscene = 4; %physical length of scene in m (square shape)
npx = 128; %number of pixels in image (same in x&y dims)
npx2 = npx ^ 2;
c = physconst('LightSpeed'); %speed of light in m/s
dtsamp = 0.5 * c * 667e-12; %image frame subsampling step size for each Tx

xgrid = linspace(-2, 2, npx);
ygrid = linspace(0, Lscene, npx);
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);
pxyvec = [pxgrid(:), pygrid(:)];
dx = xgrid(2) - xgrid(1);
dy = ygrid(2) - ygrid(1);

%% Set up sparse target probability vectors
ptargetkk = sparse(zeros(npx2, 1));
ptarget_Hist = cell(3, 22);
%%Intialize target probs near first detection points
%%%Target 1
for tt = 1:3
    entry_time = targetEntryTimes(tt);
    peak_match_time = target_peakmatches(targetEntryTimes(tt), tt);
    
    % t0xy = impeaks_xy{targetEntryTimes(tt)}(target_peakmatches(targetEntryTimes(tt),tt),:);
    t0xy = impeaks_xy{entry_time}(peak_match_time, :);
    val1 = pxyvec(:, 1) == t0xy(1);
    val2 = pxyvec(:, 2) == t0xy(2);
    t0xyind = find(val1 & val2); % %find index in XY grid
    fprintf("Target %d, Entry Time %d, Peak Match Time %d, Peak Match Location\n", tt, entry_time, peak_match_time);
    disp(t0xy)
    fprintf("Target %d, Index:\n", tt);
    disp(t0xyind)
    ptarget0 = sparse(zeros(npx2, 1));
    ptarget0(t0xyind) = 1;
    % %%propagate once to spread out probability:
    % ptarget0 = A_slow*ptarget0;
    ptarget_Hist{tt, targetEntryTimes(tt)} = ptarget0;
    %%figure(40+tt), surf(xgrid,ygrid,reshape(ptarget0,[npx npx]),'EdgeColor','none'), view(2)
end
% return

figure(84), hold on
fig = figure(84);
%%Specify the output data and file name for gif
imMov = cell(1, 21);
filename = "HMMBayesFilter_3targets_Animated.gif";

%%Apply recursive Bayes filter updates
sig2imnoise = 0.1; %image noise variance
mutarget_Hist = nan(2, 22, 3);
sig2target_Hist = nan(2, 2, 22, 3);
mmse_errtarget_Hist = nan(2, 22, 3);
map_errtarget_Hist = nan(2, 22, 3);

predict_history = nan(22, 3, npx, npx);
likeframe_num_hist = nan(22, 3, 1);
likeframe_raw_hist = nan(22, 3, npx, npx);

for kk = 1:21
    %%get mWidar data image
    imframekk = squeeze(data(kk, :, :));
    imframekk_abs = sqrt(imframekk .^ 2); % %get rid of negative peaks
    size(imframekk_abs)
    for tt = 1:3

        if kk >= targetEntryTimes(tt) && kk < targetExitTimes(tt)
            %%Time update
            ptargetkkm1 = ptarget_Hist{tt, kk};

            if tt == 1
                ptargetkk_tprop = A_fast * ptargetkkm1;
            else
                ptargetkk_tprop = A_slow * ptargetkkm1;
            end

            exp = ['figure(', num2str(fig.Number), '),subplot(33', num2str(tt), ')'];
            eval(exp), cla,
            surf(xgrid, ygrid, reshape(ptargetkk_tprop, [npx, npx]), 'EdgeColor', 'none'), view(2)
            title(['time update predicted location at kk=', num2str(kk)])

            %%Measurement update
            %%%Pick off true image peak for this target
            if target_peakmatches(kk, tt) > 0
                tt_imkk_xy = impeaks_xy{kk}(target_peakmatches(kk, tt), :);
                indtt_imkk_xy = find(pxyvec(:, 1) == tt_imkk_xy(1) & pxyvec(:, 2) == tt_imkk_xy(2)); %find index in XY grid
                %%Compute likelihood of corresponding target peak pixel -
                %%%get values from corresponding from row of
                %%%pointlikelihood_image, should be the npx by npx grid of
                %%%image likelihood at indtt_imkk_xy pixel location for all npx^2
                %%%target location hypotheses according to their resulting
                %%%ellipse models with Gaussian noise.
                likeframekk_tt_raw = pointlikelihood_image(indtt_imkk_xy, :)';

                likeframe_num_hist(kk, tt) = indtt_imkk_xy;
                likeframe_raw_hist(kk, tt, :, :) = reshape(likeframekk_tt_raw, [npx, npx]);


                %%TUNING HACK: rescale via Gaussian mask around observed peak
                sf = 0.1; %scaling factor for mask
                gaussmask_tt_kk = mvnpdf(pxyvec, tt_imkk_xy, sf * eye(2));
                gaussmask_tt_kk(gaussmask_tt_kk < 0.1 * max(gaussmask_tt_kk)) = 0;
                likeframekk_tt = likeframekk_tt_raw .* gaussmask_tt_kk;

                exp = ['figure(', num2str(fig.Number), '),subplot(33', num2str(3 + tt), ')'];
                eval(exp), cla,
                surf(xgrid, ygrid, reshape(likeframekk_tt, [npx, npx]), 'EdgeColor', 'none'), view(2)
                title(['meas likelihood at kk=', num2str(kk)])

                ptargetkk_mupdate = ptargetkk_tprop .* likeframekk_tt;
                ptargetkk_mupdate = ptargetkk_mupdate / sum(sum(ptargetkk_mupdate));
            else
                %%target not detected in image frame, no actual meas update
                ptargetkk_mupdate = ptargetkk_tprop;
            end

            %%Store and update for next recursion
            ptarget_Hist{tt, kk + 1} = ptargetkk_mupdate;

            exp = ['figure(', num2str(fig.Number), '),subplot(33', num2str(6 + tt), ')'];
            eval(exp), cla,
            surf(xgrid, ygrid, reshape(ptargetkk_mupdate, [npx npx]), 'EdgeColor', 'none'), view(2)
            hold on
            plot3(pttraj(1, kk, tt), pttraj(2, kk, tt), 1, 'md', 'MarkerSize', 5, 'LineWidth', 1)
            title(['Posterior at kk =', num2str(kk)]), hold off

            %%For paper: generate separate posterior figures at specific kk
            %             if kk==20 %%ismember(kk,[6,10,12,17])
            %                 figure(36+tt)
            %                 surf(xgrid,ygrid,reshape(ptargetkk_mupdate,[npx npx]),'EdgeColor','none'), view(2)
            %                 hold on
            %                 plot3(pttraj(1,kk,tt),pttraj(2,kk,tt),1,'md','MarkerSize',5,'LineWidth',1)
            %                 title(['Target ',num2str(tt),' Posterior at Time k =',num2str(kk)]),
            %                 colorbar
            %                 hold off
            %             end

            %%compute mean and covariance from grid
            mutarget_Hist(:, kk, tt) = sum(pxyvec .* repmat(ptarget_Hist{tt, kk + 1}, [1, 2]), 1)';
            sig2target_Hist(:, :, kk, tt) = reshape(sum([pxyvec(:, 1) .^ 2, pxyvec(:, 1) .* pxyvec(:, 2), ...
                                                          pxyvec(:, 2) .* pxyvec(:, 1), pxyvec(:, 2) .^ 2] .* repmat(ptarget_Hist{tt, kk + 1}, [1, 4]), 1), [2 2]) ...
                - mutarget_Hist(:, kk, tt) * (mutarget_Hist(:, kk, tt))';
            %%compute MMSE and MAP errors w.r.t. truth
            mmse_errtarget_Hist(:, kk, tt) = mutarget_Hist(:, kk, tt) - pttraj(:, kk, tt);
            [~, indmapxyttkk] = max(ptarget_Hist{tt, kk + 1});
            map_errtarget_Hist(:, kk, tt) = pxyvec(indmapxyttkk, :)' - pttraj(:, kk, tt);

        end

    end

    % pause

    %%save to gif
    %     frame = getframe(fig);
    %     imMov{kk} = frame2im(frame);
    %     [F,map] = rgb2ind(imMov{kk},256);
    %     if kk == 1
    %         imwrite(F,map,filename,"gif","LoopCount",Inf,"DelayTime",1);
    %     else
    %         imwrite(F,map,filename,"gif","WriteMode","append","DelayTime",1);
    %     end

end

%%plot MMSE errors vs. time
figure(90)
cc = 0;

for tt = 1:3
    tvec = 1:22;

    for dd = 1:2
        cc = cc + 1;
        eval(['subplot(32', num2str(cc), ')']);
        plot(tvec, squeeze(mmse_errtarget_Hist(dd, :, tt)), 'r-o'), hold on
        plot(tvec, 2 * sqrt(squeeze(sig2target_Hist(dd, dd, :, tt))), 'r--')
        plot(tvec, -2 * sqrt(squeeze(sig2target_Hist(dd, dd, :, tt))), 'r--'), hold off
        xlabel('Time step, k')

        if dd == 1
            ylabel('X posn error, m')
        else
            ylabel('Y posn error, m')
        end

        title(['MMSE estimation error, Target ', num2str(tt)])
        xlim([1 21])
    end

end

%%plot MAP errors vs. time
figure(91)
cc = 0;

for tt = 1:3
    tvec = 1:22;

    for dd = 1:2
        cc = cc + 1;
        eval(['subplot(32', num2str(cc), ')']);
        plot(tvec, squeeze(map_errtarget_Hist(dd, :, tt)), 'r-o'), hold on
        plot(tvec, 2 * sqrt(squeeze(sig2target_Hist(dd, dd, :, tt))), 'r--')
        plot(tvec, -2 * sqrt(squeeze(sig2target_Hist(dd, dd, :, tt))), 'r--'), hold off
        xlabel('Time step, k')

        if dd == 1
            ylabel('X posn error, m')
        else
            ylabel('Y posn error, m')
        end

        title(['MAP estimation error, Target ', num2str(tt)])
        xlim([1 21])
    end

end
