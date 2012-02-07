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
