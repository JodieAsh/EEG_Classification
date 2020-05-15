"""
A quick tutorial.

## Version history:
May 2020:
   Original script by Jodie Ashford [ashfojsm], Aston University
"""
import EEG_generate_training_matrix

# To create a training matrix from the files provided
# (See "A Guide to Machine Learning with EEG with a Muse Headband" for more details)

# TODO change this to the file path where your training data is
file_path = r"D:\Documents\University\Code Projects\EEG_Classification\training_data"

EEG_generate_training_matrix.gen_training_matrix(file_path, cols_to_ignore=-1, output_file="training_matrix_generated.csv")
# When this script is running you should see an output like this:
# "Using file x-concentrating-1.csv - resulting vector shape for the file (116, 989)"
