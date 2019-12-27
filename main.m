
image = 1;
cluster_num = 5;
rng_seed = 10;
hyper_params = [1e-5, 1e-4];

[centroids, objective] = kkmeans(image, cluster_num, rng_seed, hyper_params);

disp(['KKmeans resulting cluster centers are ', num2str(centroids')]);

figure(4);
plot(1:length(objective), objective);
xlabel('iterations');
ylabel('Objective');
title('Objective of KKmeans');