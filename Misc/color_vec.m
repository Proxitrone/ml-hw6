function [data_vector] = color_vec(datapoint_num, image_mat)
%COLOR_VEC Return the RGB values of datapoint of an image
%   Detailed explanation goes here

    [x, y]=ind2sub(size(image_mat), datapoint_num);
    data_vector(:, 1) = double(image_mat(x, y, :));
end

