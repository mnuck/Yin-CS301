function [ x_pts, y_pts ] = get_points( click_new_points, src_or_dest )
%GET_POINTS Gets click points, either saved or new

source_x = [ 294.2500;
             575.9167;
             319.2500;
             440.9167 ];
source_y = [ 115.0833;
             135.9167;
             240.0833;
             413.4167 ];

dest_x = [ 137.5833;
           414.2500;
           161.7500;
           281.7500 ];
dest_y = [ 116.7500;
           144.2500;
           245.0833;
           415.9167 ];

if strcmp(click_new_points, 'yes')
    [x_pts, y_pts] = ginput;
else
    if strcmp(src_or_dest, 'src')
        x_pts = source_x;
        y_pts = source_y;
    else
        x_pts = dest_x;
        y_pts = dest_y;
    end
end
end
