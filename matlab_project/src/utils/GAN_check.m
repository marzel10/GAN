GAN_net = dlnetwork

input_size = 15;
latent_size = 1;
[attention, norm_attention] = attention_matrix.build_attention();
% Calculate the determinant of the attention matrix to ensure it's not singular
det_attention = det(attention);
det_attention_normalized = det(norm_attention);
disp(['Determinant of attention matrix: ', num2str(det_attention)]);
disp(['Determinant of normalized attention matrix: ', num2str(det_attention_normalized)]);
input_layer = inputLayer([input_size,28, NaN],'SCB', "Name","input");
GAN_net = addLayers(GAN_net,input_layer);

layers_enc = [
    GAN("G1", input_size, latent_size,norm_attention,0);
    reluLayer("Name","relu_latent");
];
GAN_net = addLayers(GAN_net,layers_enc);

layers_dec = [
    GAN("G3", latent_size, input_size ,norm_attention,0);
   
];

GAN_net = addLayers(GAN_net,layers_dec);
GAN_net = connectLayers(GAN_net,"input","G1/in");
GAN_net = connectLayers(GAN_net,"relu_latent","G3/in");

GAN_net = initialize(GAN_net);

l = GAN("trial",15,1,rand(28,28),0);
X = l.predict(rand(15,28));
disp(size(X));

X1 = GAN_net.predict(rand(input_size,28,1));
disp(size(X1));