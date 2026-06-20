classdef tiny_architectures_container
    methods(Static)
        function fc_net = tiny_fully_connected_network(params)
            
            if ~isfield(params, 'input_size'), params.input_size = 15; end
            if ~isfield(params, 'latent_size'), params.latent_size = 1; end
            
            input_size = params.input_size;
            output_size = params.latent_size;
            
            fc_net = dlnetwork;
            input_layer = inputLayer([input_size, NaN],'CB', "Name","input");
            
            fc_layer_2 = fullyConnectedLayer( ...
                  output_size*5, ...
                  'Name','tiny_fc_output_1');
            relu_layer = reluLayer('Name', 'relu_output_1');
            fc_layer = fullyConnectedLayer( ...
                output_size, ...
                'Name','tiny_fc_output', 'WeightsInitializer', 'glorot');
           % relu_layer = sigmoidLayer('Name', 'relu_output');

            fc_net = addLayers(fc_net, input_layer);
            fc_net = addLayers(fc_net, fc_layer_2);
            fc_net = addLayers(fc_net, relu_layer);
            fc_net = addLayers(fc_net, fc_layer);
         %   fc_net = addLayers(fc_net, relu_layer);
            fc_net = connectLayers(fc_net, "input", "tiny_fc_output_1");
            fc_net = connectLayers(fc_net, "tiny_fc_output_1", "relu_output_1");
            fc_net = connectLayers(fc_net, "relu_output_1", "tiny_fc_output");
            fc_net = initialize(fc_net);
        end


        function new_net = network_connecter(path_to_net1, net2, learn_weight_net1)

            % 1. Load the pre-trained network
            data = load(path_to_net1);
            if isfield(data,'trained_net')
                net1 = data.trained_net;
            else
                net1 = data.net;
            end
            
            % 2. Extract the LayerGraph to PRESERVE TOPOLOGY
            if isa(net1, 'dlnetwork')
                lgraph = layerGraph(net1);
            else
                % If it's a SeriesNetwork or DAGNetwork
                lgraph = layerGraph(net1.Layers); 
            end
            
            % 3. Freeze weights of the old network
            layers = lgraph.Layers;
            for i = 1:length(layers)
                
                    disp("Freezing layer: " + layers(i).Name+ " with learn rate factor: " + learn_weight_net1);
                    
                    lgraph = replaceLayer(lgraph, layers(i).Name, ...
                        tiny_architectures_container.setLearnRates(layers(i), learn_weight_net1));
                
            end
            
            % 4. Add the new Tiny Network layers
            % We assume net2 is a dlnetwork or layer array. 
            if isa(net2, 'dlnetwork')
                tiny_layers = net2.Layers;
            else
                tiny_layers = net2;
            end

            % Exclude input layer of tiny net 
            tiny_layers = tiny_layers(~arrayfun(@(l) isa(l, 'nnet.cnn.layer.InputLayer'), tiny_layers));

          
            lgraph = addLayers(lgraph, tiny_layers);
       
            % 5. Connect Latent Space to Tiny Network (chek what is the first layer after input in tiny net and connect to it)
            % This creates a fork: Latent goes to Decoder (existing) AND Tiny Net (new)
            % Find the first layer in tiny_layers that is not an input layer
            first_tiny_layer = tiny_layers(1);
            lgraph = connectLayers(lgraph, 'fc_latent_1', first_tiny_layer.Name);
            
            % 6. Create the final dlnetwork
            new_net = dlnetwork(lgraph);

        end

        function layer = setLearnRates(layer, rate)
            % Helper to set rates cleanly
            if isprop(layer, 'WeightLearnRateFactor')
                layer.WeightLearnRateFactor = rate;
                layer.BiasLearnRateFactor = rate;
                layer.WeightL2Factor = rate;
                layer.BiasL2Factor = 0; % Preserving your logic
            end
            if isprop(layer, 'OffsetLearnRateFactor')
                layer.OffsetLearnRateFactor = rate;
                layer.ScaleLearnRateFactor = rate;
                layer.OffsetL2Factor = rate;
                layer.ScaleL2Factor = 0;
            end
        end


        %% Old networks, probably not used anymore but keeping for now for reference and potential reuse of components in future architectures
        
        function GAN_net = tiny_graph_network(params)

            if ~isfield(params, 'input_size'), params.input_size = 15; end
            if ~isfield(params, 'latent_size'), params.latent_size = 1; end
            if ~isfield(params, 'adjacency_matrix'), params.adjacency_matrix = eye(28); end

            input_size = params.input_size;
            latent_size = params.latent_size;
            norm_attention = params.adjacency_matrix;

            GAN_net = dlnetwork;
            input_layer = inputLayer([input_size,28, NaN],'SCB', "Name","input");
            GAN_net = addLayers(GAN_net,input_layer);

            layers_enc = [
                GAN("G1", input_size, latent_size,norm_attention,0);
                reluLayer("Name","relu_latent");
            ];
            GAN_net = addLayers(GAN_net,layers_enc);

            GAN_net = connectLayers(GAN_net,"input","G1/in");
      
            GAN_net = initialize(GAN_net);
        end

        function new_net = path_convolution(varargin)

            % Parse input size if provided if not use default
            p = inputParser;
            addParameter(p, 'input_size', 15);
            parse(p, varargin{:});
            input_size = p.Results.input_size;

            net = dlnetwork;
            tempNet = [
                inputLayer([input_size, 1, 28, NaN],"SSCB","Name","input")
                groupedConvolution2dLayer([input_size 1],1,"channel-wise","Name","groupedconv","Padding","same","Stride",[input_size 1])
                reluLayer("Name","relu")];
            net = addLayers(net,tempNet);

            % clean up helper variable
            clear tempNet;

            new_net = initialize(net);
            
        end

        function new_net = path_convolution_and_GNN(varargin)

            % Parse input size if provided if not use default
            p = inputParser;
            addParameter(p, 'input_size', 15);
            addParameter(p, 'adjacency_matrix', eye(28));
            addParameter(p, 'latent_size', 1);
            addParameter(p, 'conv_mode', true);
            
            parse(p, varargin{:});
            input_size = p.Results.input_size;
            norm_attention = p.Results.adjacency_matrix;
            conv_mode = p.Results.conv_mode;

            net = dlnetwork;
            tempNet = [
                inputLayer([input_size, 1, 28, NaN],"SSCB","Name","input")
                groupedConvolution2dLayer([input_size 1],1,"channel-wise","Name","groupedconv","Padding","same","Stride",[input_size 1],'WeightsInitializer', 'ones')
                reluLayer("Name","relu")
                GAN("G1", 1, 1, norm_attention,0, 'conv_mode', conv_mode)
                reluLayer("Name","relu_latent")];
            net = addLayers(net,tempNet);

            % clean up helper variable
            clear tempNet;

            new_net = initialize(net);
            
        end

        
        function layer = changename(layer, new_name)
            layer.Name = new_name;
        end

        function BIG_net = AE_GAN_connector(AE_path, GAN_params, learn_rate_AE)

            if ~isfield(GAN_params, 'input_size'), GAN_params.input_size = 15; end
            if ~isfield(GAN_params, 'latent_size'), GAN_params.latent_size = 1; end
            input_size = GAN_params.input_size;
            latent_size = GAN_params.latent_size;

            files = dir(fullfile(AE_path, '*.mat')); % or '*.txt', '*' for all files
            % Cell with all AE
            BIG_net = dlnetwork;
            networks_count = 0;    
            for k = 1:length(files)

                filePath = fullfile(AE_path, files(k).name);
                data = load(filePath);
                disp(fieldnames(data));
                if isfield(data, 'trained_net')
                    net = data.trained_net;
                    networks_count = networks_count + 1;
                elseif isfield(data, 'net')
                    net = data.net;
                    networks_count = networks_count + 1;
                else 
                    fieldnames(data);
                    continue; % Skip files without a network
                end
                %net = data.net; % or use fopen/fread for non-MAT files
                % 2. Extract the LayerGraph to PRESERVE TOPOLOGY
                if isa(net, 'dlnetwork')
                    lgraph = layerGraph(net);
                else
                    % If it's a SeriesNetwork or DAGNetwork
                    lgraph = layerGraph(net.Layers); 
                end
                
                % 3. Freeze weights of the old network
                layers = lgraph.Layers;
                for i = 1:length(layers)
                    if isprop(layers(i), 'WeightLearnRateFactor') 
                        lgraph = replaceLayer(lgraph, layers(i).Name, ...
                            tiny_architectures_container.setLearnRates(layers(i), learn_rate_AE));
                    end
                    
                    lgraph = replaceLayer(lgraph, layers(i).Name, tiny_architectures_container.changename(layers(i), sprintf("%s_%d", layers(i).Name, k)));
                   
                end

                BIG_net = addLayers(BIG_net, lgraph.Layers);
                disp("Network " + k + " layers added.");
                change_format_layer = change_format(sprintf('change_format_AE_%d', k), 'SCB', [input_size,1]);
                BIG_net = addLayers(BIG_net, change_format_layer);
                BIG_net = connectLayers(BIG_net, sprintf('fc_latent_1_%d', k), sprintf('change_format_AE_%d', k));

            end

            if ~isfield(GAN_params, 'adjacency_matrix'), 
                GAN_params.adjacency_matrix = eye(networks_count); 
                disp("No adjacency matrix provided, using identity.");
            end
            norm_attention = GAN_params.adjacency_matrix;
            
            % Create GAN with concatenation layer
            GAN_net = [ concatenationLayer(2,networks_count, 'Name', 'concat_latent'),
                        GAN("G1", input_size, latent_size, norm_attention,0),
                        reluLayer('Name', 'relu_latent'),
                        change_format('change_format_GAN_output', 'CB', [latent_size*networks_count])];

            BIG_net = addLayers(BIG_net, GAN_net);
            disp("GAN layers added.");
            disp(networks_count+"AE connected.");
            for i = 1:networks_count
                BIG_net = connectLayers(BIG_net, sprintf('change_format_AE_%d', i), sprintf('concat_latent/in%d', i));
            end

        end
    end
end