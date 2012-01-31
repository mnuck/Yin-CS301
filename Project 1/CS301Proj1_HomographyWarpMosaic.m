%plane warp demo using homographies
%Zhaozheng Yin, Computer Science, MST
%Spring 2012

clear all; clc; close all;

% read source and dest images
source = imread('img1.tif');
dest = imread('img2.tif');

% source = imread('Jacob.jpg');
% dest = imread('time_square.jpg');

[destnr,destnc,destnb] = size(dest);
[srcnr,srcnc,srcnb] = size(source);
figure(1); imshow(source,[]); title('source');
figure(2); imshow(dest,[]); title('destination');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 1: manually select correpsonding points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fLoad = false;
if ~fLoad    
    %click points in source
    figure(1);
    [xpts,ypts] = ginput;
    hold on; plot(xpts, ypts, 'rs','Markersize',12);
    text(xpts, ypts, num2str((1:length(xpts))'),'Color','r')
    hold off;

    %click points in destination
    figure(2);
    [xprimes,yprimes] = ginput;
    hold on; plot(xprimes, yprimes, 'gs','Markersize',12);    
    text(xprimes,yprimes,num2str((1:length(xpts))'),'Color','g');  
    hold off;

    %save the points
    save('CorrespondingPoints.mat','xpts','ypts','xprimes','yprimes');
else
    %load the point correspondece
    load('CorrespondingPoints.mat','xpts','ypts','xprimes','yprimes');
    
    %show points in source
    figure(1);
    hold on; plot(xpts, ypts, 'rs','Markersize',12);
    text(xpts, ypts, num2str((1:length(xpts))'),'Color','r')
    hold off;
    
    %show points in destination
    figure(2);
    hold on; plot(xprimes, yprimes, 'gs','Markersize',12);
    text(xprimes,yprimes,num2str((1:length(xpts))'),'Color','g');    
    hold off;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 2: compute homography (from source to dest coord system)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Method 1. compute h assuming h_33 = 1

%Method 2. compute h with constraint ||h|| = 1


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 3: warp source image onto dest coord system
% try forward and backward warping (nearest neighbor and bilinear
% interpolation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 4: stitch two images together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
