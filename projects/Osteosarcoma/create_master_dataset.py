# Copyright 2022 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


#
# create_master_dataset.py
#
#
# 2022/09/19 Copyright (C) antillia.com

# image name format:
# Case 48 - P5 C16-53844-23083.jpg

import os
import sys
import shutil
import glob
import csv
import traceback


def create_master(base_dir, output_dir):

  #training_sets = ["./Training-Set-1", "./Training-Set-2"]
  training_sets = os.listdir(base_dir)
  print("--- training_sets {}".format(training_sets))
  
  #validation_csv = "./PathologistValidation.csv"
  
  for training_set in training_sets:
    training_dir     = os.path.join(base_dir, training_set) 
    subsets = os.listdir(training_dir)
    print("--- subsets {}".format(subsets))
    
    for set in subsets:
      subset_dir  = base_dir + "/" + training_set + "/" + set
      print("--- subset_dir {}".format(subset_dir))
      
      csvs      = glob.glob(subset_dir + "/*.csv")
      validation_file = None
      if len(csvs) == 1:
        validation_file = csvs[0]
      print("--- PathologistValidation {}".format(validation_file))
      
      if not os.path.exists(validation_file):
        raise Exception("Not found PathologistValidatio.csv file")
        
      with open(validation_file, encoding='utf8', newline='') as f:
        #This validation_file is a headless csv file.
        csvreader = csv.reader(f)
        for row in csvreader:      
          image_name = row[0]
          image_name = image_name.replace(" - ", "-").replace(" ", "-")

          label      = row[1]
          if ":" in label:
            #for example, label:viable: non-viable
            print("--- Skip invalid label: row:{} label:{}".format(row, label))
            continue

          output_subdir = os.path.join(output_dir, label)
            
          if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
          image_file = os.path.join(subset_dir, image_name)
          print("--- image_file {}  label {}".format(image_file, label))
          
          if os.path.exists(image_file):    
            shutil.copy2(image_file, output_subdir)
            print("--- Copied {} to {}".format(image_file, output_subdir))
      
if __name__ == "__main__":
  try:
    base_dir   = "./Osteosarcoma-UT"
    output_dir = "./Osteosarcoma-master"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      
    create_master(base_dir, output_dir)
    
  
  except:
    traceback.print_exc()
    


