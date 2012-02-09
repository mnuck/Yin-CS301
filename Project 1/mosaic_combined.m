function [ mosaic ] = mosaic_combined( image1, image2, ...
                                      i1_offset, i2_offset, mode )
%MOSAIC_COMBINED Summary of this function goes here
    i1_size = size(image1);
    i2_size = size(image2);
        
    y_min = min([i1_offset(1) i2_offset(2)]);
    x_min = min([i1_offset(2) i2_offset(2)]);
    
    y_max = max([i1_offset(1)+i1_size(1) ...
                 i2_offset(1)+i2_size(1)]); 
    x_max = max([i1_offset(2)+i1_size(2) ...
                 i2_offset(2)+i2_size(2)]);                   
    
    mosaic = zeros((y_max-y_min), (x_max-x_min), 3);
    mask   = zeros((y_max-y_min), (x_max-x_min));
    
    %first, blindly paint the two inputs into the result
    a1 = 1 - y_min + i1_offset(1);
    b1 = 1 - x_min + i1_offset(2);
    mosaic(a1:a1+i1_size(1)-1, b1:b1+i1_size(2)-1,:) = image1(:,:,:);
    mask  (a1:a1+i1_size(1)-1, b1:b1+i1_size(2)-1,:) = ...
      mask(a1:a1+i1_size(1)-1, b1:b1+i1_size(2)-1,:) + 1;
    
    a2 = 1 - y_min + i2_offset(1)
    b2 = 1 - x_min + i2_offset(2)
    mosaic(a2:a2+i2_size(1)-1, b2:b2+i2_size(2)-1,:) = image2(:,:,:);
    mask  (a2:a2+i2_size(1)-1, b2:b2+i2_size(2)-1,:) = ...
      mask(a2:a2+i2_size(1)-1, b2:b2+i2_size(2)-1,:) + 1;

    image1_center = [ round( i1_size(1)/2 + a1 ) ...
                      round( i1_size(2)/2 + b1 ) ];
    image2_center = [ round( i2_size(1)/2 + a2 ) ...
                      round( i2_size(2)/2 + b2 ) ];
  
    %then, anywhere mask == 2, blend those pixels
    for y = 1:y_max-y_min
        for x = 1:x_max-x_min
            if mask(y,x,1) == 2
                weight = calc_weight([y x], image1_center, image2_center);
                pixel1 = image1(1+(y-a1), 1+(x-b1), :);
                pixel2 = image2(1+(y-a2), 1+(x-b2), :);
                switch mode
                    case 'nearest'
                        if weight > 0.5
                            mosaic(y,x,:) = pixel1;
                        else
                            mosaic(y,x,:) = pixel2;
                        end
                    case 'blended'
                        mosaic(y,x,:) = blend_pixel(weight, pixel1, pixel2);
                end
            end
        end
    end
end

function [ result ] = distance( a, b )
  delta_y = a(1) - b(1);
  delta_x = a(2) - b(2);
  result = sqrt( delta_y^2 + delta_x^2 );
end

function [ alpha ] = calc_weight( here, a, b )
    dist_a = distance( here, a );
    dist_b = distance( here, b );
    dist_total = dist_a + dist_b;
    alpha = dist_b / dist_total;
end

function [ result ] = blend_pixel( alpha, pixel1, pixel2 )
%BLEND_PIXEL weighted blend of two pixels
%   Constructs a result pixels based on a weighted
%   average of two input pixels. pixel1 and pixel2 should be
%   3d matrixes with color bands in the 3rd dimension.
%   alpha is between 0 and 1, with 1 meaning full weight
%   to pixel1 and 0 meaning full weight to pixel2
    [~, ~, nbands] = size(pixel1);
    result = zeros(1,1,nbands);
    for b = 1:nbands
        result(1,1,b) = alpha       * pixel1(1,1,b) + ...
                        (1 - alpha) * pixel2(1,1,b); 
    end
end
