function [ result, offset_x, offset_y, mask ] = warp_interp2( h, source )
%WARP_INTERP2 Summary of this function goes here
    [nrows, ncols, nbands] = size(source);
    
    % find the corners
    [size_x, size_y, offset_x, offset_y] = find_corners( h, ncols, nrows );
    mask_src = ones(ncols, nrows);
    h = inv(h);    
    
    [xi, yi] = meshgrid( 1:ncols, 1:nrows );
    xx = (h(1,1)*xi + h(1,2)*yi + h(1,3)) ./ ...
         (h(3,1)*xi + h(3,2)*yi + h(3,3));
    yy = (h(2,1)*xi + h(2,2)*yi + h(2,3)) ./ ...
         (h(3,1)*xi + h(3,2)*yi + h(3,3));
    mask = interp2(mask_src, xx, yy);
    for b = 1:nbands
        result(:,:,b) = uint8(interp2(double(source(:,:,b)), xx, yy, 'cubic'));
    end
end
