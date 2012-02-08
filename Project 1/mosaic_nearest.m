function [ result, middle] = mosaic_nearest( warped_src, dest, offset_x, offset_y )
%MOSAIC_NEAREST Summary of this function goes here
%   Detailed explanation goes here
    [warped_src_rows warped_src_cols ~] = size (warped_src);
    [dest_rows       dest_cols       ~] = size (dest);
    
    x_min = min([offset_x 1]);
    x_max = max([offset_x+warped_src_cols dest_cols]); 
    
    y_min = min([offset_y 1]);
    y_max = max([offset_y+warped_src_rows dest_rows]); 
    
    mosaic = zeros((y_max-y_min), (x_max-x_min), 3);
    
    % if the source image is on the left
    if offset_x < 0
        middle = ((1-offset_x) + warped_src_cols)/2;
        mosaic(1-offset_y:dest_rows-offset_y, middle:dest_cols-offset_x, :) = dest(:, middle+offset_x:end, :);
        mosaic(1:warped_src_rows, 1:middle, :) = warped_src(:, 1:middle, :);
    else
        middle = (dest_cols + offset_x)/2;
    end
    
    result = mosaic;
end

