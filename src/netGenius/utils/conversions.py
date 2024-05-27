

def convert_pixel_distance_to_meters(pixel_dist,ref_height_in_m,ref_height_in_pxl):

    return (pixel_dist*ref_height_in_m) / ref_height_in_pxl

def convert_meters_to_pixel_distance(meters, ref_height_in_m,ref_height_in_pxl):
    
    return (meters*ref_height_in_pxl) / ref_height_in_m

