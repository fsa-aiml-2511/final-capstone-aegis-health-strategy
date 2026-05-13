Project Details

General Information: 

*Start here*
Necessary Files:
    1) Four CSV files are needed for this project and need to be saved in ../data/raw. 
    The four files listed below can be found at: 

        1) clinical_codes_reference.csv
        2) patient_encounters_2023.csv
        3) patient_medication_feedback.csv
        4) retinal_labels.csv

    2) Retinal scan images are also needed for this project. The zip file below contains 3,662 image files which must be stored in ../data/raw/retinal_scan_images. The folder named retinal_scan_images must be created manually. 
    Zip file with retinal images can be found here: https://drive.google.com/file/d/1fLTFqaQ4I7-uqS7ipBdXzQy5Q9fl0-HG/view?usp=drive_link

Model 1: [placeholder]

Model 2: [placeholder]

Model 3: 
    1) Creating or downloading model (Choose one option)
        Creating model:
        Running train.py found in ../models/model3_cnn will generate a file called cnn_model.keras, which will be located in ../models/model3_cnn/saved_model.

        Downloading model:
        A copy of the model can be downloaded as an alternative to creating via train.py. 
        Download the model from the following location and save in ../models/model3_cnn/saved_model.
        Model location: https://huggingface.co/jfranklingoff/Capstone-Project-CNN-Model/blob/main/cnn_model.keras

    2) Predicting with model:
        Load images you with to classify in ../test_data/images.
        Execute predict.py found in ../models/model3_cnn.
        Once complete, results will be saved in ../test_data/model3_results.csv
        

Model 4: [placeholder]

Model 5: [placeholder]