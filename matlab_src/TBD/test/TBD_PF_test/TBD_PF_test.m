%%% Interactive entry point. Lists the tests in config/ and runs the one you
%%% pick. From code, call run_test directly:
%%%
%%%   out = run_test('T5');
%%%
%%% See TEST_FRAMEWORK.md for how tests are defined.

addpath(fileparts(mfilename('fullpath')));
out = run_test();
