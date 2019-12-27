function [K] = Kernel(x, y, gamma_s, gamma_c, image_mat)
%KERNEL Kernel for k-means and spectral clustering
%   Multiplying two RBF kernels in order to consider spatial similarity and
%   color similarity
    spactial_rbf = exp(-gamma_s*norm(spatial(x, image_mat)-spatial(y, image_mat))^2);
    color_rbf = exp(-gamma_c*norm(color_vec(x, image_mat)-color_vec(y, image_mat))^2);
    K =  spactial_rbf*color_rbf;
end

