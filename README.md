# pipeline_IR
分辨率 需要对应 内参-相机 更改后需要重新测定

cd ~/FS/FoundationStereo \
conda activate foundationpose

python fs_depth_to_rgb_alignment.py --save_calib  先运行得到具体的相机参数 -- 仅需执行一次

以上没必要

////////////////

cd ~/FS \
conda activate foundationpose

python realsense_IR_v1.py   （也无所谓 这里也得到了并更新了内参）

python make_K_txt_from_json.py 


cd FoundationStereo \
conda activate foundation_stereo


python scripts/run_demo.py --left_file ./shared_fs_test/ir_left_0000.png --right_file ./shared_fs_test/ir_right_0000.png --ckpt_dir ./pretrained_models/model_best_bp2.pth --out_dir ./outputs_test --intrinsic_file ./shared_fs_test/K_ir_fs.txt  

stereo demo的 运行 得到 depth_meter.npy	

cp /home/match/FS/FoundationStereo/outputs_test/depth_meter.npy shared_fs_test/    拷贝npy深度信息 

cd ~/FS

python IR-RGB-clp.py 


然后align -- 在posen环境

python fs_depth_to_rgb_alignment.py \
  --depth_fs   ./outputs_test/depth_meter.npy \
  --calib_json ./rs_calib_d435.json \
  --out        ./outputs_test/depth_fs_aligned.png
  
得到新的 epth_fs_aligned.png

然后就是最后一步

输入 color depth 内参 。。 mesh mask 最后确定物体的6d posen 位置



