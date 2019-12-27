function [data_vector] = spatial(datapoint_num, image_mat)
%SPATIAL Return the x and y coordinates of datapoint of an image
%   Our datapoint is basically the 1D coordinate (y*image_col_num+x) of an image, we translate
%   it back to 2D
    [mat_dim_1 , mat_dim_2, ~] = size(image_mat);
    data_vector = zeros(2, numel(datapoint_num));
    [data_vector(1,:) , data_vector(2,:)] = ind2sub([mat_dim_1 , mat_dim_2], datapoint_num);
end

