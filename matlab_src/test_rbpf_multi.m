% TEST_RBPF_MULTI
% Quick test script for multi-target RBPF (KF_RBPF_multi)
%
% DESCRIPTION:
%   This script demonstrates how to run the KF_RBPF_multi algorithm
%   on the multi-object tracking datasets and compare with JPDA.
%
% ALGORITHM:
%   KF_RBPF_multi implements a Rao-Blackwellized Particle Filter for
%   multi-target tracking with:
%   - Joint data association with exclusivity constraints
%   - Optimal importance distribution sampling
%   - One Kalman filter per target per particle
%
% USAGE:
%   1. Uncomment the test cases you want to run
%   2. Run the script: test_rbpf_multi
%   3. Results will be saved as GIF files in the current directory
%
% OUTPUT FILES:
%   - JPDAF_test_traj_RBPF.gif  (RBPF results)
%   - JPDAF_test_traj_JPDA.gif  (JPDA results for comparison)
%
% PARAMETERS:
%   - NumParticles: Number of particles (default: 1000)
%     * More particles = better accuracy but slower
%     * Recommended: 500-2000 for 2-target scenarios
%   - Debug: Enable debug output (default: false)
%
% DATASETS:
%   - JPDAF_test_traj    : Default 2-target scenario
%   - JPDAF_test_traj_2  : Alternative 2-target scenario
%   - JPDAF_test_traj_3  : Another 2-target scenario
%
% SEE ALSO:
%   main_multiObj, KF_RBPF_multi, JPDA_KF

%% Test Case 1: Default dataset with RBPF (1000 particles)
fprintf('\n========================================\n');
fprintf('Test Case 1: RBPF on JPDAF_test_traj\n');
fprintf('========================================\n');
main_multiObj('Algorithm', 'RBPF', 'NumParticles', 1000, 'Debug', false);

%% Test Case 2: Different dataset
% fprintf('\n========================================\n');
% fprintf('Test Case 2: RBPF on JPDAF_test_traj_2\n');
% fprintf('========================================\n');
% main_multiObj('Algorithm', 'RBPF', 'Dataset', 'JPDAF_test_traj_2', 'NumParticles', 500);

%% Test Case 3: Comparison with JPDA
% fprintf('\n========================================\n');
% fprintf('Test Case 3: JPDA on JPDAF_test_traj\n');
% fprintf('========================================\n');
% main_multiObj('Algorithm', 'JPDA', 'Debug', false);

fprintf('\n========================================\n');
fprintf('All tests complete!\n');
fprintf('========================================\n');
