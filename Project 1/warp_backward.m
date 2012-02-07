function [ result, offset_x, offset_y ] = warp_backward( h, source, dest, warping_engine )
%WARP_BACKWARD Summary of this function goes here
    [nrows, ncols, nbands] = size(source);
    
    % find the corners
    [size_x, size_y, offset_x, offset_y] = find_corners( h, ncols, nrows );
    result = zeros(size_y, size_x, nbands);
    h_inv = inv(h);
    for x = 1:size_x
        for y = 1:size_y
            [source_x source_y] = straighten(h_inv * [x ; y ; 1]);
            source_x = source_x + offset_x + 1;
            source_y = source_y + offset_y + 1;
            if source_x < 1 || source_x > ncols || ...
               source_y < 1 || source_y > nrows
                 continue;
            end
            [ul, ur, ll, lr] = ...
                get_neighbor_values( source, source_x, source_y );
            dx = source_x - floor(source_x);
            dy = source_y - floor(source_y);
            result( y, x, : ) = my_bilinear( dx, dy, ul, ur, ll, lr );
        end
    end
end
