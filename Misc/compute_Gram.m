function [Gram, Coordinates, Color] = compute_Gram(image_mat, gamma_s, gamma_c)
%COMPUTE_GRAM Summary of this function goes here
%   Detailed explanation goes here
    datapoints_num = size(image_mat, 1) * size(image_mat, 2);
    datapoints = 1:datapoints_num;
    Coordinates = spatial(datapoints, image_mat);

    Color = zeros(3, datapoints_num);
    for n=1:size(image_mat, 1)
        for m=1:size(image_mat, 2)
            Color(:, (n-1)*size(image_mat, 2)+m) = double(image_mat(m, n,:));
        end
    end
%     Coordinates([1, 2], :) = Coordinates([2, 1], :);
    %Pdist form
    % Space
%     K = squareform(pdist(Coordinates', 'euclidean'));
%     K1 = exp(-gamma_s*K.^2);
% 
%     % Color
%     K = squareform(pdist(Color', 'euclidean'));
%     K2 = exp(-gamma_s*K.^2);

    % Vectorized form
    % spactial_rbf
    A = dot(Coordinates, Coordinates, 1);
    B = -2* (Coordinates'* Coordinates);
    K = A+B+A';
    spatial_rbf = exp(-gamma_s*K);
    
    % color_rbf
    A = dot(Color, Color, 1);
    B = -2* (Color'* Color);
    K = A+B+A';
    color_rbf = exp(-gamma_c*K);

    Gram_vec = spatial_rbf.*color_rbf;
    
%     Gram_pdist = K1.*K2;

    Gram = Gram_vec;
end

