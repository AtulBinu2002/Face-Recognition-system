# Face-Recognition-system
To design a system that identifies and recongises face

Main.pyib contains main file code to preproess,augment data and also to desgin a model that helps in recongizing face.

Camera.py helps interface with camerma to help create datasets for the CNN model.

Face.py the main program that helps in accessing camera and recongnize the face present.
Steps
1. Creating the datasets(Taking 100 Images)
2. Cropping only the face
3. Spliting file into training,valadation and testing
4. Data Augmentation and Preprocressing
5. Designing a model using pretrained model
6. Appropriate Hyperparameters
7. Fine-Tunning
8. Evaluate the model
9. Repeat the previous 3 steps until the valadation accuracy about 90%
10. Test the model real-time.
