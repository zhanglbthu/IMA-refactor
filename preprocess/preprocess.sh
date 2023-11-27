#######################################################
# Things you need to modify
subject_name='gray'
path='/bufferhdd/zhanglibo/project/IMavatar/data/retarget'
video_folder=$path/$subject_name
video_names='220700191.mp4, 221501007.mp4, 222200042.mp4, 222200049.mp4'
fps=30
# Center crop
# crop="1334:1334:0:200"
resize=512
# fx, fy, cx, cy in pixels, need to adjust with resizing and cropping
# fx=1268.14515
# fy=1268.057125
# cx=285.939975
# cy=244.142425
########################################################
pwd=$(pwd)
path_modnet=$(pwd)'/submodules/MODNet'
path_deca=$(pwd)'/submodules/DECA'
path_parser=$(pwd)'/submodules/face-parsing.PyTorch'
########################################################
set -e
# echo "crop and resize video"
# cd $pwd
# for video in $video_names
# do
#   video_path=$video_folder/$video
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   echo $video_folder/$subject_name/"${array[0]}"/"image"
#   ffmpeg -y -i $video_path -vf "fps=$fps, crop=$crop, scale=$resize:$resize" -c:v libx264 $video_folder/"${array[0]}_cropped.mp4"
# done
# echo "background/foreground segmentation"
# cd $path_modnet
# for video in $video_names
# do
#   video_path=$video_folder/$video
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   mkdir -p $video_folder/$subject_name/"${array[0]}"
#   python -m demo.video_matting.custom.run --video $video_folder/"${array[0]}_cropped.mp4" --result-type matte --fps $fps
# done
# echo "save the images and masks with ffmpeg"
# # sudo apt install ffmpeg
# cd $pwd
# for video in $video_names
# do
#   video_path=$video_folder/$video
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   echo $video_folder/$subject_name/"${array[0]}"/"image"
#   mkdir -p $video_folder/$subject_name/"${array[0]}"/"image"
#   ffmpeg -i $video_folder/"${array[0]}_cropped.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"image"/"%d.png"
#   mkdir -p $video_folder/$subject_name/"${array[0]}"/"mask"
#   ffmpeg -i $video_folder/"${array[0]}_cropped_matte.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"mask"/"%d.png"
# done

# change : get image paths
combined_image_path=""
for video in $video_names; do
  IFS='.' read -r -a array <<< $video
  image_path=$video_folder/$subject_name/"${array[0]}"/"image"
  combined_image_path=$combined_image_path","$image_path
done
combined_image_path=${combined_image_path:1}

combined_save_path=""
for video in $video_names; do
  IFS='.' read -r -a array <<< $video
  save_path=$video_folder/$subject_name/"${array[0]}"/"deca"
  combined_save_path=$combined_save_path","$save_path
done
combined_save_path=${combined_save_path:1}

echo "DECA FLAME parameter estimation"
cd $path_deca
python demos/demo_reconstruct.py -images $combined_image_path -saves $combined_save_path --saveCode True --saveVis False --sample_step 1  --render_orig False
# for video in $video_names
# do
#   echo $video
#   IFS='.' read -r -a array <<< $video
#   mkdir -p $video_folder/$subject_name/"${array[0]}"/"deca"
#   python demos/demo_reconstruct.py -i $video_folder/$subject_name/"${array[0]}"/image --savefolder $video_folder/$subject_name/"${array[0]}"/"deca" --saveCode True --saveVis False --sample_step 1  --render_orig False
# done

echo "face alignment landmark detector"
cd $pwd
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  python keypoint_detector.py --path $video_folder/$subject_name/"${array[0]}"
done
echo "iris segmentation with fdlite"
cd $pwd
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python iris.py --path $video_folder/$subject_name/"${array}"
done

# change
camera_idx="220700191,221501007,222200042,222200049"

combined_path=""
for video in $video_names; do
  IFS='.' read -r -a array <<< $video
  path=$video_folder/$subject_name/"${array[0]}"
  combined_path=$combined_path","$path
done
combined_path=${combined_path:1}
echo "fit FLAME parameter begin"
cd $path_deca
python optimize.py --paths $combined_path --size $resize --camera_idx $camera_idx
echo "fit FLAME parameter finish"

# cd $path_deca
# for video in $video_names
# do
#   if [ "$shape_video" == "$video" ];
#   then
#     continue
#   fi
#   IFS='.' read -r -a array <<< $(basename $shape_video)
#   shape_from=$video_folder/$subject_name/"${array}"
#   IFS='.' read -r -a array <<< $(basename $video)
#   echo $video
#   python optimize.py --path $video_folder/$subject_name/"${array}" --shape_from $shape_from  --cx $cx --cy $cy --fx $fx --fy $fy --size $resize
# done
echo "semantic segmentation with face parsing"
cd $path_parser
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python test.py --dspth $video_folder/$subject_name/"${array}"/image --respth $video_folder/$subject_name/"${array}"/semantic
done