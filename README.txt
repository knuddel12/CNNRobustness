################################################
Annotations for the Python scripts

- baselineTraining.py
	script to train a baseline ResNet
- constBlurTraining.py
	script to train a ResNet with images of constant blur. Blur method can be selected.
- calculateMeanSD.py
	calculated the mean image for zero-centering of features, which was mentioned and shown in the appendix.
- FaceDataExchange.py, FaceDetection.py, and FacePreprocessing.py 
	contained code provided by Dermalog running the Align and Crop-preprocessing.
- EpochWiseBlurChangeTraining.py, StepWiseBlurChangeTraining.py
	script to train a ResNet with images of epoch-/step-wise blur change. Blur method can be selected.
- plotActivationFunctions.py
	plots the activations functions.
- plotTestResults.py
	for all plots which can be found in the results-chapter.
- test_nets.py
	file for testing the trained ResNets. Contains several lists, where similar ResNets of similar approaches were 
	collected. A list can be selected to test on a chosen blur method.
- transforms, util, training.py
	helper files to execute training


################################################
Annotations for the ResNets

The trained networks in the ResNets folder are subdivided into the training approaches. The naming of the files gives 
information about the training approach, the blur method, blur degree, learning rate, total amount of training epochs, and,
if saved in between training progress, the current epoch of training.

EXAMPLE: 
'const_box_blur3_lr0.005_epochs200_curr_60.pth'

This ResNet was trained with constantly blurred images. The box filter was used with a kernel size of 3x3. LR = 0.005. The total
training was set to 200 epochs, but this version of the network was trained for 60 epochs (current epochs = 60).

################################################
Annotations for the testresults

See above.
'continued' means that a pre-trained ResNet model (pre-training of 60 epochs, on a spec. blur method) was used.
'iterative' translate to epoch-wise blur change.