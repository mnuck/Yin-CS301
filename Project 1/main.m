% CS301 Computational Perception and Cognition
% Matthew Nuckolls <matthew.nuckolls@gmail.com>
% Michael Wisely <michaelwisely@gmail.com>
% Project 1

clear all; clc; close all;

% Configuration
click_new_points  = 'yes';
homography_method = 'pseudo_inverse';        % ('pseudo_inverse', 'svd')
warping_method    = 'backward';    % ('forward', 'backward', 'interp2')
interpolator      = 'nearest';    % ('nearest', 'blended')

% 1. get source and destination images


if strcmp(click_new_points, 'yes')
    source_filename      = uigetfile('','First Image File');
    destination_filename = uigetfile('','Second Image File');
else
    source_filename = 'img1.tif';
    destination_filename = 'img2.tif';
end

source = imread(source_filename);
dest   = imread(destination_filename);


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

%h1 = homography_pseudo_inverse( source_x, source_y, dest_x, dest_y )
%h2 = homography_svd( source_x, source_y, dest_x, dest_y )


% 4. warp source to destination
switch warping_method
    case 'forward'
        [warped_src, offset_x, offset_y, src_mask] = ...
            warp_forward( h, source );
    case 'backward'
        [warped_src, offset_x, offset_y, src_mask] = ...
            warp_backward( h, source );
    case 'interp2'
        [warped_src, offset_x, offset_y, src_mask] = ...
            warp_interp2( h, source );        
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
        result = mosaic_combined( warped_src, dest, src_mask, ... 
                                  [offset_y offset_x], [0, 0], interpolator );
    case 'blended'
        result = mosaic_combined( warped_src, dest, src_mask, ... 
                                  [offset_y offset_x], [0, 0], interpolator );
    otherwise
        msgbox('Unknown warping method selected [' ...
               + warping_direction + '] Now exiting.', ...
               'Unknown warping', 'error', 'modal');
        exit();
end

%. 6. Display result
figure(4); imshow(uint8(result), []);
% plot the offset point
hold on; plot(-offset_x, -offset_y, 'rs','Markersize',12);
hold off;
display 'Done! :D'
