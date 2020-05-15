"""
A quick tutorial.

## Version history:
May 2020:
   Original script by Jodie Ashford [ashfojsm], Aston University
"""
import EEG_generate_training_matrix
import Build_and_test_classifier

# To create a training matrix from the files provided
# (See "A Guide to Machine Learning with EEG with a Muse Headband" for more details)

# TODO change this to the file path where your training data is
# File path to raw EEG data from Muse headset
file_path = r"D:\Documents\University\Code Projects\EEG_Classification\training_data"

# Generate training matrix / calculate all features for data
# TODO name the output file whatever you like
EEG_generate_training_matrix.gen_training_matrix(file_path, cols_to_ignore=-1, output_file="example_training_matrix.csv")
# When this script is running you should see an output like this:
# "Using file x-concentrating-1.csv - resulting vector shape for the file (116, 989)"
# Your output training matrix csv should look like the example one provided "example_training_matrix.csv"

# Building and saving a (Sklearn) random forest classifier trained on the features we just extracted
# TODO change this to the file path where your training matrix is and name clf_output_file
training_path = r"D:\Documents\University\Code Projects\EEG_Classification\example_training_matrix.csv"
Build_and_test_classifier.build_classifier(training_path, test_size=0.2, clf_output_file="Random_Forest_Classifier")
# Note accuracy is output as it is calculated in
# 'Build_and_test_classifier.build_classifier() - # Predict on the testing data'
