## Dog Cat Classification

This problem contains 2 solutions.
1. Deep learning model from scratch using sequential model - **Refer dog_cat_classification.py**.
2. Deep leaning model using pre trained model (InceptionResNetV2) - Refer **dog_cat_classification_using_pre_trained_model.py**.

### Steps to run:

1. Fist download images from following kaggle website: https://www.kaggle.com/c/3362/download-all
2. Move train images to train folder and test images to test folder.
3. Run **dog_cat_classification.py** , it will generate model files model_keras.h5 and model_weights.h5 ( Remember as model files are already there with same names, better to move them to other folders)
4. Step 3 may take so much of time as we are running on simple computers (Usually in deep leaning, model training runs on GPUs). You can run prediction directly using **dog_cat_prediction.py** as I already trained and pushed model files.
5. You can see prediction images after completion on above step.

Simillarly you can run **dog_cat_classification_using_pre_trained_model.py** to generate model using pre trained model. Here you need to run training compulsary as I am unable to upload model files because of size limitation in github. Once training is completed, you can run **dog_cat_prediction_using_pretrained_model.py** to predict new images.  

