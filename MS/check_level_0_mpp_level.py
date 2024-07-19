# find all the ndpi files in the folder and then compile a csv file of the level_0 mpp of the slides
import os
import csv
import openslide
from openslide import OpenSlideError
from tqdm import tqdm

def list_ndpi_files(directory):
    """ List all .ndpi files in the directory """
    return [f for f in os.listdir(directory) if f.endswith('.ndpi')]

def get_mpp(file_path):
    """ Get the mpp at level 0, return 'NA' if openslide fails """
    try:
        slide = openslide.OpenSlide(file_path)
        mpp_x = slide.properties.get('openslide.mpp-x', 'NA')
        return mpp_x
    except OpenSlideError:
        return 'NA'

def compile_mpp_data(directory):
    """ Compile mpp data into a CSV file """
    ndpi_files = list_ndpi_files(directory)
    csv_file_path = os.path.join(directory, 'mpp_data.csv')
    
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Level_0_MPP'])
        
        for ndpi_file in tqdm(ndpi_files, desc='Compiling MPP Data'):
            file_path = os.path.join(directory, ndpi_file)
            mpp_x = get_mpp(file_path)
            writer.writerow([ndpi_file, mpp_x])
    
    print(f'Data compiled into {csv_file_path}')

# Replace '/path/to/dir' with your actual directory path
ndpi_dir = '/media/hdd1/neo/BMA_AML_lite'
compile_mpp_data(ndpi_dir)
