function [ result ] = homography_pseudo_inverse( source_x, source_y, dest_x, dest_y )
%HOMOGRAPHY_PSEUDO_INVERSE Calculate homography matrix assuming h_33 = 1
    vec1 = ones( length(source_x), 1 );
    vec0 = zeros( length(source_x), 1 );
    
    first   = -1 * source_x .* dest_x; % column vectors
    second  = -1 * source_x .* dest_y; % built up here for clarity
    third   = -1 * source_y .* dest_x;
    fourth  = -1 * source_y .* dest_y;
    
    A = [ source_x source_y vec1 vec0     vec0     vec0 first  third ; ...
          vec0     vec0     vec0 source_x source_y vec1 second fourth ];
    b = [ dest_x ; dest_y ];
    h = A \ b;
    result = transpose([h(1:3) h(4:6) [h(7:8) ; 1]]);
end
