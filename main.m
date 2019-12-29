
images = [1, 2];
cluster_num = [2, 3, 5, 10];
rng_seed = 10;
hyper_params = [1e-6, 1e-4];
% initialize with rng_seed or kmeans++
init_type = [1, 2];
% do RatioCut or NormalizedCut
cut_type = [1, 2];



 for i=1:length(images)
    %% Extract image matrix
    if i ==1
        image_mat = imread('image1.png');
    elseif i == 2
        image_mat = imread('image2.png');
    end
    %% Compute the Gram matrix first, we'll use it's elements in the
    % distance computations. Regular for-loop implementation is too
    % slow, use vectorized version for improved preformance
    [Gram, Coord, Color] = compute_Gram(image_mat, hyper_params(1), hyper_params(2));
    figure(1);
    imshow(Gram);
    title('Gram matrix');
    %% Degree matrix
    D = diag(sum(Gram, 1));
    %% Laplacian for ratio cut
    L_ratio = D - Gram;
    %% Laplacian for normalized cut
    D_sqrt= diag(1./sqrt(sum(Gram, 1)));
    L_norm = D_sqrt*L_ratio*D_sqrt;
    %% Get the eigenvalues and vectors of our graph Laplacian (Ratio)
    [eigVec_ratio, eigVal] = eig(L_ratio); 
    [d, ind] = sort(diag(eigVal));
    eigVal = eigVal(:, ind);
    eigVec_ratio = eigVec_ratio(:, ind);
    figure(2);
    scatter(1:numel(d), d);
    title('Eigenvalues of Graph Laplacian (Ratio)');
    ylabel('eigenvalue');
    xlabel('soreted order');

    %% Get the eigenvalues and vectors of our graph Laplacian (Normal)
    [eigVec_norm, eigVal] = eig(L_norm); 
    [d, ind] = sort(diag(eigVal));
    eigVal = real(eigVal(:, ind));
    eigVec_norm = real(eigVec_norm(:, ind));
    % Normalize the rows with norm 1
    eigVec_norm = eigVec_norm./sqrt(sum(eigVec_norm.^2, 2));
    figure(5);
    scatter(1:numel(d), real(d));
    title('Eigenvalues of Graph Laplacian (Norma)');
    ylabel('eigenvalue');
    xlabel('soreted order');

    %% Start simulations
    for j=1:length(init_type)
        for c=1:length(cluster_num)
            %% KKmeans
            [centroids_KK, objective_KK] = kkmeans(images(i), image_mat, cluster_num(c), init_type(j), rng_seed, Gram);
            %% Spectral Clustering RatioCut
            [centroids_SC, objective_DC] = spectral(images(i), image_mat, cluster_num(c), init_type(j), rng_seed, Gram, eigVec_ratio, 1);
            %% Spectral Clustering NormalCut
            [centroids_SC, objective_DC] = spectral(images(i), image_mat, cluster_num(c), init_type(j), rng_seed, Gram, eigVec_norm, 2);
        end
        
    end
    
end

% disp(['KKmeans resulting cluster centers are ', num2str(centroids')]);
% figure(4);
% plot(1:length(objective), objective);
% xlabel('iterations');
% ylabel('Objective');
% title('Objective of KKmeans');





