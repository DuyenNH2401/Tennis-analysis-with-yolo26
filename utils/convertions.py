def covert_pixel_to_meters(
        pixel_dis:float,
        refernce_height_in_meters:float,
        refernce_height_in_pixels:float):
    return (pixel_dis * refernce_height_in_meters) / refernce_height_in_pixels

def convert_meters_to_pixels(
        meters:float,
        refernce_height_in_meters:float,
        refernce_height_in_pixels:float
):
    return (meters * refernce_height_in_pixels) / refernce_height_in_meters