function [ result ] = my_bilinear( delta_x, delta_y, ...
                                   ul_value, ur_value, ll_value, lr_value )
    [~, ~, nbands] = size(ul_value);
    virtual_pix1 = zeros(1,1,nbands);
    virtual_pix2 = zeros(1,1,nbands);
    result = zeros(1,1,nbands);
    
    for b = 1:nbands
        virtual_pix1(1,1,b) = delta_x       * ur_value(1,1,b) + ...
                              (1 - delta_x) * ul_value(1,1,b);
        virtual_pix2(1,1,b) = delta_x       * lr_value(1,1,b) + ...
                              (1 - delta_x) * ll_value(1,1,b);
        result(1,1,b) = delta_y       * virtual_pix2(1,1,b) + ...
                        (1 - delta_y) * virtual_pix1(1,1,b);
    end  
end
