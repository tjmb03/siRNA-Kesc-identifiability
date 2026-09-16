function p = reference_params()
%REFERENCE_PARAMS Read the shared parameter file, so MATLAB and Python cannot
% drift apart. Single source of truth is ../config/reference_parameters.json.
here = fileparts(mfilename('fullpath'));
cfg  = jsondecode(fileread(fullfile(here, '..', 'config', ...
                                    'reference_parameters.json')));
p = cfg.parameters;
end
