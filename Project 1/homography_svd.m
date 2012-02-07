function [ result ] = homography_svd( source_x, source_y, dest_x, dest_y )
%HOMOGRAPHY_SVD Compute the homography matrix via singular value decomp
    vec1 = ones( length(source_x), 1 );
    vec0 = zeros( length(source_x), 1 );
    
    first   = -1 * source_x .* dest_x; % column vectors
    second  = -1 * source_x .* dest_y; % built up here
    third   = -1 * source_y .* dest_x; % for clarity
    fourth  = -1 * source_y .* dest_y;
    fifth   = -1 * dest_x;
    sixth   = -1 * dest_y;
    
    A = [ source_x source_y vec1 vec0     vec0     vec0 first  third  fifth ; ...
          vec0     vec0     vec0 source_x source_y vec1 second fourth sixth ];
    [~, ~, V] = svd(transpose(A) * A);
    result = transpose( [V(1:3,end) V(4:6,end) V(7:9,end)] );           
end
