function [ size_x, size_y, offset_x, offset_y ] = find_corners( h, ncols, nrows )
%FIND_CORNERS determine the extents of a warped array
    [ulx uly] = straighten(h * [1 ; 1 ; 1]);
    [urx ury] = straighten(h * [ncols ; 1 ; 1]);
    [llx lly] = straighten(h * [1 ; nrows ; 1]);
    [lrx lry] = straighten(h * [ncols ; nrows ; 1]);
    
    offset_x = floor(min([ulx urx llx lrx]));
    offset_y = floor(min([uly ury lly lry]));
    brx = ceil(max([ulx urx llx lrx]));
    bry = ceil(max([uly ury lly lry]));
    size_x = brx - offset_x;
    size_y = bry - offset_y;
end
