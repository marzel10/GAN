% which_path: bidirectional mapping between sensors and paths
% Usage:
%   path_id = which_path(sensor1, sensor2)         % sensors -> path
%   [sensor1, sensor2] = which_path(path_id)       % path -> sensors (two outputs)
%   sensors = which_path(path_id)                  % path -> [sensor1 sensor2] (one output)

function varargout = which_path(varargin)
    % Load mapping table of sensor pairs to path IDs
    try
        data = load('data\SP.mat');
        pair_table = data.SP; % Nx2, each row is [sensor1 sensor2]
    catch ME
        error('which_path:LoadError', 'Failed to load data/SP.mat: %s', ME.message);
    end

    if nargin == 2
        % sensors -> path
        sensor1 = varargin{1};
        sensor2 = varargin{2};

        % Basic input validation
        if ~isscalar(sensor1) || ~isscalar(sensor2) || ~isnumeric(sensor1) || ~isnumeric(sensor2)
            error('which_path:InvalidInput', 'sensor1 and sensor2 must be numeric scalars.');
        end
        sensor1 = round(sensor1);
        sensor2 = round(sensor2);

        % Find matching path (order-insensitive)
        idx = find( (pair_table(:,1) == sensor1 & pair_table(:,2) == sensor2) | ...
                    (pair_table(:,1) == sensor2 & pair_table(:,2) == sensor1), 1);

        if isempty(idx)
            error('which_path:NoMatch', 'Invalid sensor pair: (%d, %d)', sensor1, sensor2);
        end

        varargout{1} = idx; % path_id

    elseif nargin == 1
        % path -> sensors
        path_id = varargin{1};
        if ~isscalar(path_id) || ~isnumeric(path_id)
            error('which_path:InvalidInput', 'path_id must be a numeric scalar.');
        end
        path_id = round(path_id);
        if path_id < 1 || path_id > size(pair_table,1)
            error('which_path:OutOfRange', 'path_id must be in the range [1, %d].', size(pair_table,1));
        end

        sensors = pair_table(path_id, 1:2);

        if nargout <= 1
            % Return as a 1x2 vector when a single output is requested
            varargout{1} = sensors;
        else
            % Return as two separate outputs when two outputs are requested
            varargout{1} = sensors(1);
            varargout{2} = sensors(2);
        end

    else
        error('which_path:Arity', 'Usage: path_id = which_path(sensor1, sensor2) or [s1,s2] = which_path(path_id).');
    end
end
