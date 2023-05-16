-------------------------------
-- COSC2963 Assignment 2 Code
-- Group 42
--   Nelson Cheng (S2105802)
--   Kah Hie Toh (S3936897)
-------------------------------

The full repository of code files for this project can be found here: https://github.com/s2105802-Nelson/COSC2673-A2-Group42

* Note: PyTorch models make use of custom data files, images_main.csv and images_extra.csv, which are included


Running Environment
--------------
Before running the PyTorch based Model files, please note the isGoogleColab variable near the top of the file, which by default is set to False
	If this file is being run locally, the scripts assume that the "Image_classification_data" folder exists in the same directory as the notebook
	If this file is to be run on Google Colab, set this variable to True. The script assumes there is a directory in the "My Drive" of your Google Drive account 
		called "COSC2673" and that both the notebook and the "Image_classification_data" folder are in this folder.
For testing the scripts, the variable useFullData in the PyTorch model files can be set to False. This will train the models on a subset of 1000 records only.


Baseline and Best Model Results:
--------------
For the Baseline Model with PyTorch, see the file: 05c.PyTorchBaseline.ipynb
For the Best performing models, CNN model with PyTorch, see the file 28.PyTorchCNN07.ipynb


Other Model Results:
--------------
For the Tensorflow NN with greyscaling and data augmentation, see the file: COSC2793_A2_Tensorflow_Cancerous.ipynb
For Cell Type Modelling with Semi-Supervised Learning, see the file: 31.PyTorchFullDataMulti01.ipynb


Other Files:
------------
For Exploratory Data Analysis (EDA) see the file: EDA.ipynb
For the Custom Main Images csv file that are pre-split by Train/Validation/Test, see the file: images_main.csv
	To regenerate the Custom Main images file, you can run the file: 02.ImageDataLoad.ipynb
For the Custom Extra Images csv file that are pre-split by Train/Validation/Test, see the file: images_extra.csv
	To regenerate the Custom Extra images file, you can run the file: 02.ImageExtraDataLoad.ipynb