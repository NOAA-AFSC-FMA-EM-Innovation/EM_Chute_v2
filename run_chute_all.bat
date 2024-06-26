set config_path="chuteConfig.yml"

call conda activate chute_env
call python calibrate/undistort_videos.py --config_path %config_path%

cd detection
set new_path=../%config_path%
call python track.py --config_path %new_path%

cd ../classification
call conda activate classify
call python test.py --config_path %new_path%

cd ../segmentation
call conda activate segmentation
call python segment.py --config_path %new_path%

cd ..