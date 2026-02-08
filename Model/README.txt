This README describes how to run the developer python scripts locally and how to
run the deployment web app.

You can find this project on GitHub: https://github.com/bozhidar-bozhikov/machine-learning-W25
and on huggingface: https://huggingface.co/Veki21/ML_project_w2025

The GitHub repository has 2 branches: the main branch for the baseline model and a
"dropout_early_l2" branch for the enhanced model. The differences between the two models
are described in the final report. You may switch branches on git to quickly access either version.

To run the python scripts locally you need to download the Kaggle cats and dogs 
dataset from https://www.microsoft.com/en-us/download/details.aspx?id=54765
Afterwards extract the zip file so the file structure looks like this:

machine-learning-W25
└── Model
	├── PetImages
	│	├── Cat/
	│	└── Dog/
	└── binary_classifier.py (and other files)


To run the web app, you need to download both sets of weights from https://huggingface.co/Veki21/ML_project_w2025
Afterwards in the app.py file you need to update the WEIGHTS_PATH_V1 and WEIGHTS_PATH_V1 to point to the correct paths to the .pth weights. 
Then run the command «python app.py» to launch the web app.