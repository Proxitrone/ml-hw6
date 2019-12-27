
images = [1, 2];
cluster_num = [2, 3, 5, 10];
rng_seed = 10;
hyper_params = [1e-6, 1e-4];
% initialize with rng_seed or kmeans++
init_type = [1, 2];


for i=1:length(images)
    for j=1:length(init_type)
        for c=1:length(cluster_num)
            [centroids, objective] = kkmeans(images(i), cluster_num(c), init_type(j), rng_seed, hyper_params);
        end
    end
end

disp(['KKmeans resulting cluster centers are ', num2str(centroids')]);
figure(4);
plot(1:length(objective), objective);
xlabel('iterations');
ylabel('Objective');
title('Objective of KKmeans');