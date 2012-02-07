function [ result ] = warp_forward( h, source, dest, warping_engine )
%WARP_FORWARD Summary of this function goes here
%   Detailed explanation goes here
    [nrows, ncols, nbands] = size(source);
    result = zeros(nrows, ncols, nbands);
    for x = 1:ncols
        for y = 1:nrows
            source_pixel = [x ; y ; 1];
            dest_pixel = h * source_pixel;
            dest_x = round( dest_pixel(1) / dest_pixel(3) ); % doing nearest interpolation, FIXME
            dest_y = round( dest_pixel(2) / dest_pixel(3) );
            if dest_x < 1 || dest_x > ncols || ...
               dest_y < 1 || dest_y > nrows
                continue;
            end
            for b = 1:nbands
                result(dest_y, dest_x, b) = source(y, x, b);
            end
        end
    end
 end
