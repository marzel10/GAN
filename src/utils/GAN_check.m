GAN_net = dlnetwork

input_size = 500;
out_G1 = 100;
latent_size = 10;

layers_enc = [
    GAN("G1", input_size, out_G1,attention);
    reluLayer;
    GAN("G2", out_G1, latent_size,attention);
];
GAN_net = addLayers(GAN_net,layers_enc);

layers_dec = [
    reluLayer("Name","reluG3");
    GAN("G3", latent_size, out_G1,attention);
    reluLayer;
    GAN("G4", out_G1, input_size,attention);
];

GAN_net = addLayers(GAN_net,layers_dec);


l = GAN("trial",500,100,rand(28,28));
X = l.predict(rand(500,28));
disp(size(X));
