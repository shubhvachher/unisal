import os
import ffmpeg
import h5py
from time import time
import multiprocessing as mp
from functools import partial

def get_folders_to_process(base_path, return_sorted=False):
   """
   Given the base path of eve dataset, returns all the 2nd level sub-folders that
   contain the file screen.mp4. i.e. all the folders that need to be pre-processed
   """

   if "eve" not in base_path or not os.path.isdir(base_path):
      raise RuntimeError("Base path must be a directory with the word 'eve' in it. Check docs to understand this function.")

   paths_to_process = []
   first_level_files = os.listdir(base_path)
   print(first_level_files)

   for flfile in first_level_files:
      if os.path.isdir(os.path.join(base_path, flfile)):
         second_level_files = os.listdir(os.path.join(base_path, flfile))
         for slfile in second_level_files:
            if os.path.isfile(os.path.join(base_path, flfile, slfile, "screen.mp4")):
               paths_to_process.append(os.path.join(base_path, flfile, slfile))
   
   if return_sorted:
      paths_to_process.sort()
   
   return paths_to_process

def read_timestamps(timestamps_file_path):
    timestamps = []  # Read all timestamps as list of strings
    with open(timestamps_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                timestamps.append(line)
    return timestamps

def process_one_folder(folder_path, images_save_path, output_resolution='640x360'):
   """
   Process given folder's screen.mp4 file to save frames as well as PoG related metadata
   """
   startTime = time()

   first_level_path, folder_name = os.path.split(folder_path)
   base_eve_folder_path, user_folder_name = os.path.split(first_level_path)

   images_save_prefix = "{}_{}".format(user_folder_name, folder_name)
   print("Processing {}".format(images_save_prefix))

   # Save frames as images and their timestamps 
   if "image" in folder_name:
      """
      Since images are static we will only save one of them to calculate saliency on.

      Most have 90 frames. Some of them have 150 frames for some reason. In the eve_data_subset:
      ./test08/step065_image_Wikimeda-ilya-repin-unexpected-visitors/screen.timestamps.txt
      ./train34/step094_image_Wikimeda-ilya-repin-unexpected-visitors/screen.timestamps.txt

      The 80th frame should still give the correct image.
      """
      ffmpeg.input(
         os.path.join(folder_path, "screen.mp4"), vsync=0
         ).output(
            os.path.join(images_save_path, images_save_prefix+"_80.jpg"), 
            s=output_resolution, 
            vf=r"select='between(n\,79\,79)'",
            loglevel="warning"
            ).run()  # Save the 80th frame in images_save_path with name eg. train17_step17_image_abc_80.jpg
      
      screen_timestamps = [read_timestamps(os.path.join(folder_path, "screen.timestamps.txt"))[79]]
   else:  # Video data, so, save all frames
      ffmpeg.input(
         os.path.join(folder_path, "screen.mp4"), vsync=0
         ).output(
            os.path.join(images_save_path, images_save_prefix+r"_%d.jpg"), 
            s=output_resolution,
            loglevel="warning"
            ).run()
      
      screen_timestamps = read_timestamps(os.path.join(folder_path, "screen.timestamps.txt"))
      
   # Get POG data
   basler_timestamps = read_timestamps(os.path.join(folder_path, "basler.timestamps.txt"))
   
   h5f = h5py.File(os.path.join(folder_path, "basler.h5"), 'r')
   left_PoG_tobii = (h5f['left_PoG_tobii/data'] if 'left_PoG_tobii' in h5f else None)
   right_PoG_tobii = (h5f['right_PoG_tobii/data'] if 'right_PoG_tobii' in h5f else None)

   if left_PoG_tobii is None:
      left_PoG_tobii = [["None", "None"]*len(basler_timestamps)]
   if right_PoG_tobii is None:
      right_PoG_tobii = [["None", "None"]*len(basler_timestamps)]
   
   assert len(basler_timestamps)==len(left_PoG_tobii)

   # Save metadata in file
   with open(os.path.join(images_save_path, images_save_prefix+"_metadata.txt"), "w") as fw:
      fw.write("Image Timestamp Begin\n")
      for timestamp in screen_timestamps:
         fw.write(timestamp + "\n")
      fw.write("PoG Metadata Begin\n")
      for timestamp, left_PoG, right_PoG in zip(basler_timestamps, left_PoG_tobii, right_PoG_tobii):
         fw.write("{}\t{}\t{}\n".format(timestamp, left_PoG, right_PoG))

   print(time()-startTime, " seconds. DONE {}".format(images_save_prefix))

if __name__=="__main__":
   eve_data_folder_path = r"..\..\..\..\ass_masters_tsinghua_sem3\deep_learning\NEW\project\eve_data_subset"  # need to change this
   images_save_path = r".\examples\SALICON\images"  # need to change this

   # eve_data_folder_path = r"./data/eve_data_subset"  # need to change this
   # images_save_path = r"./data/eve_processed"  # need to change this

   output_resolution = '640x360'

   folders_to_process = get_folders_to_process(eve_data_folder_path, return_sorted=True)

   # for folder_path in folders_to_process:
   #    process_one_folder(folder_path, images_save_path, output_resolution)

   pool = mp.Pool(processes=4)
   partial_process_one_folder = partial(process_one_folder, images_save_path=images_save_path, output_resolution=output_resolution)

   pool.map(partial_process_one_folder, folders_to_process)