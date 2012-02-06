function [ result ] = warp_backward( h, source, dest, warping_engine )
%WARP_BACKWARD Summary of this function goes here
%   Detailed explanation goes here
    [nrows, ncols, nbands] = size(dest);
    result = zeros(nrows, ncols);
    h_inv = inv(h);
    for x = 1:ncols
        for y = 1:nrows
            dest_pixel = [x ; y ; 1];
            source_pixel = h_inv * dest_pixel;
            source_x = round( source_pixel(1) / source_pixel(3) );
            source_y = round( source_pixel(2) / source_pixel(3) );
            if source_x < 1 || source_x > ncols || ...
               source_y < 1 || source_y > nrows
                 continue;
            end
            result(y, x) = source(source_y, source_x);
        end
    end
end
