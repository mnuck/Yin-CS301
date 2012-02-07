function [ result ] = warp_backward( h, source, dest, warping_engine )
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

function [ ul, ur, ll, lr ] = get_neighbor_values( source, x, y )
% concept is that x and y are fractional numbers, and we need
% the values for the surrounding pixels
    [nrows, ncols, ~] = size(source);
    fx = max( [floor(x) 1] );
    fy = max( [floor(y) 1] );
    cx = min( [ceil(x) ncols] );
    cy = min( [ceil(y) nrows] );
    
    ul = source(fy, fx, :);
    ur = source(fy, cx, :);
    ll = source(cy, fx, :);
    lr = source(cy, cx, :);    
end


function [ result ] = my_bilinear( delta_x, delta_y, ...
                                   ul_value, ur_value, ll_value, lr_value )
    [~, ~, nbands] = size(ul_value);
    virtual_pix1 = zeros(1,1,nbands);
    virtual_pix2 = zeros(1,1,nbands);
    result = zeros(1,1,nbands);
    
    for b = 1:nbands
        virtual_pix1(1,1,b) = delta_x       * ur_value(1,1,b) + ...
                              (1 - delta_x) * ul_value(1,1,b);
        virtual_pix2(1,1,b) = delta_x       * lr_value(1,1,b) + ...
                              (1 - delta_x) * ll_value(1,1,b);
        result(1,1,b) = delta_y       * virtual_pix2(1,1,b) + ...
                        (1 - delta_y) * virtual_pix1(1,1,b);
    end
    
end
