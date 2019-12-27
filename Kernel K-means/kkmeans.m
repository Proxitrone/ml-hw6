function [means_new, objective] = kkmeans(image_num, cluster_num, rngseed, hyper)
%KKMEANS Perform kernel K-means
%   Standard k-means, but we use our kernel for the distance metric between
%   datapoints in kernel space
    rng(rngseed);
    %% Initialize 
    K_max = 100;
    objective = [];
    if image_num ==1
        image_mat = imread('image1.png');
    elseif image_num == 2
        image_mat = imread('image2.png');
    end
    
    % Total number of data_points
    datapoints_num = size(image_mat, 1)*size(image_mat, 2);
    % Matrix to store assignment of datapoints to clusters (1 if in cluster, 0 if not)
    % clusters = zeros(cluster_num, datapoints_num);
    cluster_distances = zeros(cluster_num, datapoints_num);
    % Choose random datapoints as initial cluster centers
    means_old = round(rand(cluster_num, 1)*datapoints_num+1);
    means_new = means_old;
    % Kernel hyperparameters
    gamma_s = hyper(1);
    gamma_c = hyper(2);
    % Kluster colors 
    cluster_colors = zeros(size(image_mat,3), cluster_num);
    for k=1:cluster_num
        cluster_colors(:, k) = color_vec(means_old(k), image_mat)/256;
    end
    clustered_image = zeros(size(image_mat));
    clustered_image_gray = zeros(100, 100);
    % Klustered GIF filename
    filename = ['Kernel K-means/KKMeansImage',num2str(image_num), 'RNG', num2str(rngseed), 'Klusters', num2str(cluster_num),'.gif'];
    

    %% Compute the Gram matrix first, we'll use it's elements in the
    % distance computations. Regular for-loop implementation is too
    % slow, use vectorized version for improved preformance
    [Gram, Coord, Color] = compute_Gram(image_mat, gamma_s, gamma_c);
    figure(1);
    imshow(Gram);
    %% Start K-means
    for i=1:K_max
        disp(['--KKmeans iteration ', num2str(i), '--']);
        clusters = zeros(cluster_num, datapoints_num);
        % E-step, assign points to clusters
        % Compute distances from datapoints to cluster means
%         for n=1:datapoints_num
%             
%             for k=1:cluster_num
%                 cluster_distances_slow(k, n) = exp(-gamma_s*norm(Coord(:, n)-Coord(:, means_old(k, 1)))^2)* exp(-gamma_c*norm(Color(:, n)-Color(:, means_old(k, 1)))^2);
%             end
%         end
        % Basically extract the approapriate rows of our Gram matrix
        cluster_distances = Gram(means_old, :);
        % Assignm minmum distance points to appropriate clusters, use
        % linear indexing
        [~, index] = max(cluster_distances, [], 1, 'linear');
        clusters(index) = 1;
        N_k = sum(clusters==1, 2);
        % Visualize klusters
        for n=1:datapoints_num
            [~, k] = max(clusters(:,n), [], 1);
            [x, y] =ind2sub([100, 100], n);
            clustered_image(x, y, :) = cluster_colors(:,k);
            clustered_image_gray(x, y) = sum(cluster_colors(:,k))/3;
        end

        figure(2);
        imshow(clustered_image);
        figure(3);
        imshow(clustered_image_gray);

        [imind,cm] = rgb2ind(clustered_image, 255);
        if i ==1 
            imwrite(imind,cm, filename, 'DelayTime', 1, 'Loopcount', inf);
        else
            imwrite(imind,cm, filename, 'DelayTime',0.1, 'WriteMode', 'Append');
        end
        
        L = sqrt(diag(1./N_k));
        % M-step, find new means  
        
        objective = [objective, trace(L*clusters*Gram*clusters'*L)];
        for k=1:cluster_num
            % For each datapoint in a cluster, find a point, that minimizes
            % the overall distance to every other point in a cluster, that
            % will be our new mean
            % We 
            data_distances = zeros(datapoints_num, 1);
            for n=1:datapoints_num
                if clusters(k, n)
                    % Compute distances from datapoint n to all
                    % other datapoints m in cluster k
                    data_distances(n, 1) = Gram(n, :)*clusters(k, :)';
                end
            end
            [~, index] = max(data_distances);
            means_new(k, 1) = index;
        end
        for k=1:cluster_num
            cluster_colors(:, k) = color_vec(means_new(k), image_mat)/256;
        end
        % Termination condition: Our cluster centers don't change from last
        % iteration
        if (sum(means_old - means_new ==0) == cluster_num )
            disp(['Number of iterations spend for KKmeans is ', num2str(i)]);
            break;
        end
        means_old = means_new;
    end

end

