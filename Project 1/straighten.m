function [ x, y ] = straighten( homo_pixel )
%STRAIGHTEN takes a homogeneous coordinate and normalizes it
    x = homo_pixel(1) / homo_pixel(3);
    y = homo_pixel(2) / homo_pixel(3);
end

