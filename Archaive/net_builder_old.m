classdef net_builder_old

    methods(Static)

        function layer = helperDeconcatenation(name,output_number)
            layer = deconcatenation(name,output_number);
        end

        function layer = helperGAN(name,adj,input_size,output_size,type)
            layer = GAN(name,input_size,output_size,adj,type);
        end

        function layer = helperprint(name)
            layer = print(name);
        end

        function net = build_net_with_28_inputs()

            net = dlnetwork;

            num_in = 28;
            input_x = 4000;
            input_y = 2;
            input_z = 6;

            kernel_conv1 = [2 2];
            channels_conv1 = 10;
            kernel_pool1 = [4 2];
            stride_pool1 = [4 2];

            dropout_rate = 0.5;

            kernel_conv2 = [3 1];
            channels_conv2 = 1;
            kernel_pool2 = [2 2];
            stride_pool2 = [2 2];

            for i=1:num_in

                tempNet = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    %net_builder.helperprint(sprintf("start_%d",i))
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same")
                    maxPooling2dLayer(kernel_pool1,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool1)
                    reluLayer("Name",sprintf("Relu_1_%d",i))
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_%d",i))
                    convolution2dLayer(kernel_conv2,channels_conv2,"Name",sprintf("conv_2_%d",i),"Padding","same")
                    %net_builder.helperprint(sprintf("2ndconv_%d",i))
                    %net_builder.helperprint(sprintf("Before_2ndpooling_%d",i))
                    maxPooling2dLayer(kernel_pool2,"Name",sprintf("maxpoolForUnpool_2_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool2)
                    %net_builder.helperprint(sprintf("2ndpool_%d",i))
                    reluLayer("Name",sprintf("Relu_2_%d",i))];       
                net = addLayers(net,tempNet);

            end
            
            adjacency_matrix = attention_matrix.build_attention();
            input_size = 500;
            hidden_size = 100;
            latent_size = 10;


            tempNet = [
                concatenationLayer(2,num_in,"Name","concat")
                net_builder.helperGAN("G1", adjacency_matrix,input_size,hidden_size,0)
                reluLayer("Name","reluG1")
                net_builder.helperGAN("G2",adjacency_matrix,hidden_size,latent_size,1)];
            net = addLayers(net,tempNet);

            tempNet = [
                reluLayer("Name","reluG2")
                %net_builder.helperprint("After_G2")
                net_builder.helperGAN("G3",adjacency_matrix,latent_size,hidden_size,0)
                reluLayer("Name","reluG3")
                net_builder.helperGAN("G4",adjacency_matrix,hidden_size,input_size,0)
                reluLayer("Name","reluG4")
                net_builder.helperDeconcatenation("d1",num_in)];
            net = addLayers(net,tempNet);


            for i=1:num_in
                tempNet = [
                    %net_builder.helperprint(sprintf("Before_unpooling_%d",i))
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))
                    %net_builder.helperprint(sprintf("1stunpool_%d",i))
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_1_%d",i),"Cropping","same")
                    %net_builder.helperprint(sprintf("1stdeconv_%d",i))
                    reluLayer("Name",sprintf("Relu_3_%d",i))
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_2_%d",i))
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_2_%d",i),"Cropping","same")
                    %net_builder.helperprint(sprintf("end_%d",i))
                    ];
                net = addLayers(net,tempNet);
            end

            % clean up helper variable
            clear tempNet;

            for i=1:num_in
                
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/indices",i),sprintf("maxunpool_2_%d/indices",i));
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/size",i),sprintf("maxunpool_2_%d/size",i));
                
                net = connectLayers(net,sprintf("maxpoolForUnpool_2_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,sprintf("maxpoolForUnpool_2_%d/size",i),sprintf("maxunpool_1_%d/size",i));
                
                net = connectLayers(net,sprintf("Relu_2_%d",i),sprintf("concat/in%d",i));
                
                net = connectLayers(net,sprintf("d1/out%d",i),sprintf("maxunpool_1_%d/in%d",i));
                % if you debug uncoment the print layers and the following line
                %net = connectLayers(net,sprintf("d1/out%d",i),sprintf("Before_unpooling_%d",i));

            end

            net = connectLayers(net, "G2/graph_out", "reluG2");

            net = initialize(net);
        end
    end
end
