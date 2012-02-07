% CS301 Computational Perception and Cognition
% Matthew Nuckolls <matthew.nuckolls@gmail.com>
% Michael Wisely <michaelwisely@gmail.com>
% Project 1

clear all; clc; close all;

% Configuration
click_new_points = 'no';
homography_method = 'svd'; % ('pseudo_inverse', 'svd')
warping_direction = 'backward';       % ('forward', 'backward')
warping_engine    = 'for_loop';       % ('for_loop, 'interp2')
interpolator      = 'nearest';        % ('nearest', 'bilinear')

% 1. get source and destination images
%source_filename      = uigetfile('','First Image File');
%destination_filename = uigetfile('','Second Image File');
%source = imread(source_filename);
%dest   = imread(destination_filename);

source = imread('img1.tif');
dest   = imread('img2.tif');

% 2. manually select correspondence points
% This block borrowed from Dr Yin
figure(1); imshow(source,[]); title('source');
figure(2); imshow(dest,[]); title('destination');

figure(1);
[source_x, source_y] = get_points(click_new_points, 'src');
hold on; plot(source_x, source_y, 'rs','Markersize',12);
text(source_x, source_y, num2str((1:length(source_x))'),'Color','r')
hold off;

figure(2);
[dest_x, dest_y] = get_points(click_new_points, 'dest');
hold on; plot(dest_x, dest_y, 'gs','Markersize',12);    
text(dest_x, dest_y, num2str((1:length(source_x))'),'Color','g');  
hold off;

% 3. compute homography matrix
switch homography_method
    case 'pseudo_inverse'
        h = homography_pseudo_inverse( source_x, source_y, dest_x, dest_y );
    case 'svd'
        h = homography_svd( source_x, source_y, dest_x, dest_y );
    otherwise
        msgbox('Unknown homography method selected [' ...
               + homography_method + '] Now exiting.', ...
               'Unknown homography', 'error', 'modal');
        exit();
end

% 4. warp source to destination
switch warping_direction
    case 'forward'
        [warped_src, offset_x, offset_y] = ...
            warp_forward( h, source, dest, warping_engine );
    case 'backward'
        [warped_src, offset_x, offset_y] = ...
            warp_backward( h, source, dest, warping_engine );
    otherwise
        msgbox('Unknown warping method selected [' ...
               + warping_direction + '] Now exiting.', ...
               'Unknown warping', 'error', 'modal');
        exit();
end

figure(3); imshow(uint8(warped_src), []);

% 5. mosaic images together
switch interpolator
    case 'nearest'
        result = mosaic_nearest( warped_src, dest );
    case 'bilinear'
        result = mosaic_bilinear( warped_src, dest );
    otherwise
        msgbox('Unknown warping method selected [' ...
               + warping_direction + '] Now exiting.', ...
               'Unknown warping', 'error', 'modal');
        exit();
end

%. 6. Display result
%msgbox('Imagine a pretty picture here!');
