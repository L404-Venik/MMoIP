# MMoIP-Task1
Task 1 of Modern Methods of Image Processing (https://imaging.cs.msu.ru/ru/seminars/2024/tasks/task1)

Program provides 5 simple methods of 24-bit bitmap image manipulation"
mirroring;
rotation;
fragment extraction;
autocontrast;
deinterlasing.
For more information check the link.
 
# Console interface:
#### python main.py (command) (parameters...) (input_file) (output_file)

## Commands
__mirror__ { h | v | d | cd }

Reflection relative to the horizontal axis (_h_), vertical axis (_v_), main diagonal (_d_), side diagonal (_cd_).

__extract__ (left_x) (top_y) (width) (height)

Extracting an image fragment with parameters :\
left indentation (_left_x_, integer),\
top indentation (_top_y_, integer),\
fragment width (_width_, positive),\
fragment height (_height_, positive).
  
__rotate__ { cw | ccw } (angle)

Turn clockwise (_cw_) or counterclockwise (_ccw_) by a specified number of degrees, a multiple of 90.
  
__autocontrast__

Bring the range of image brightness values to the range [0,255].
  
__fixinterlace__

Interlaced scan artifact detection and correction.
