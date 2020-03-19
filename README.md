<h2> Key Points Identification in an Image to Create a Drag-And-Drop Function on Screen </h2>

The goal of this project is to identify the extreme points of two fingers from a human hand on a live camera and create executable functions based on the motion and directional movement of those two extreme points. The steps I am following in this process are as follows:

<ul>

<li> <b> Image Features Extraction </b> </li> </br>

  In order to create the model to recognize if a frame from a live camera contains a human hand or not, I will be dealing with spatial-bin features and HOG features and vectorize them as numpy arrays.

<li> <b> Classification and Modelling </b> </li> </br>

  To create the classifiers I will work on with Linear SVMs and Deep Neural Networks and proceed further with the one with less errors.

<li> <b> Key Points Identification and Tracking </b> </li> </br>

  After the classification, I will proceed with the recognition process by working with the concavexity defects.

<li> <b> Algorithm Design to Map the Recognition to Executable Functions </b> </li> </br>

  After the identification of the extreme points from a single frame, it will be tested if it is workable on a live camera. Based on the movement of those points, possible algorithms will be devised so that they could be mapped in order to make an object draggable on screen.

</ul>
