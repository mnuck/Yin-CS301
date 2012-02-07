function [ result ] = warp_forward( h, source, dest, warping_engine )
%WARP_FORWARD Summary of this function goes here
    [nrows, ncols, nbands] = size(source);
    
    % find the corners
    [size_x, size_y, offset_x, offset_y] = find_corners( h, ncols, nrows );
    result = zeros(size_y, size_x, nbands);
    
    for x = 1:ncols
        for y = 1:nrows
            [dest_x dest_y] = straighten(h * [x ; y ; 1]);
            result( round(dest_y) - offset_y + 1, ...
                    round(dest_x) - offset_x + 1, : ) = source(y, x, :); 
        end
    end
end
