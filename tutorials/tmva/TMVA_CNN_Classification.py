#!/usr/bin/env python
# coding: utf-8




# / \file
# / \ingroup tutorial_tmva
# / \notebook
# /  TMVA Classification Example Using a Convolutional Neural Network
# /
# / This is an example of using a CNN in TMVA. We do classification using a toy image data set
# / that is generated when running the example macro
# /
# / \macro_image
# / \macro_output
# / \macro_code
# /
# / \author Neel Shah


 # TMVA Classification Example Using a Convolutional Neural Network


#  Helper function to create input images data
#  we create a signal and background 2D histograms from 2d gaussians
#  with a location (means in X and Y)  different for each event
#  The difference between signal and background is in the gaussian width.
#  The width for the background gaussian is slightly larger than the signal width by few % values
# 
# 





import ROOT
from ROOT import TMVA 
from array import array





opt=[1,1,1,1,1]
useTMVACNN = opt[0] if (len(opt) > 0) else False
useKerasCNN = opt[0] if (len(opt) > 1) else False
useTMVADNN = opt[0] if (len(opt) > 2) else False
useTMVABDT = opt[0] if (len(opt) > 3) else False
usePyTorchCNN = opt[0] if (len(opt) > 4) else False
useTMVACNN = False

writeOutputFile = True

num_threads = 0  # use default threads



## For PYMVA methods
TMVA.PyMethodBase.PyInitialize()

ROOT.TMVA.Tools.Instance()


# do enable MT running
if (num_threads >= 0):
  ROOT.EnableImplicitMT(num_threads)
  if (num_threads > 0):
     ROOT.gSystem.Setenv("OMP_NUM_THREADS", ROOT.TString.Format("%d",num_threads))

else:
  ROOT.gSystem.Setenv("OMP_NUM_THREADS", "1")

print("Running with nthreads  = " + str(ROOT.GetThreadPoolSize()) )


if __debug__:
    ROOT.gSystem.Setenv("KERAS_BACKEND", "tensorflow")
    # for using Keras
#     TMVA.PyMethodBase.PyInitialize()
else:
    useKerasCNN = False


if (writeOutputFile):
    outputFile = ROOT.TFile.Open("TMVA_CNN_ClassificationOutput.root", "RECREATE")

'''
      
    ## Create TMVA Factory
    Create the Factory class. Later you can choose the methods
    whose performance you'd like to investigate.

    The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass

    - The first argument is the base of the name of all the output
    weightfiles in the directory weight/ that will be created with the
    method parameters

    - The second argument is the output file for the training results

    - The third argument is a string option defining some general configuration for the TMVA session.
      For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the
   option string

    - note that we disable any pre-transformation of the input variables and we avoid computing correlations between
   input variables
'''

factory =  ROOT.TMVA.Factory (
  "TMVA_CNN_Classification", outputFile,
  "!V:ROC:!Silent:Color:AnalysisType=Classification:Transformations=None:!Correlations")

'''

  ## Declare DataLoader(s)

  The next step is to declare the DataLoader class that deals with input variables

  Define the input variables that shall be used for the MVA training
  note that you may also use variable expressions, which can be parsed by TTree::Draw( "expression" )]

  In this case the input data consists of an image of 16x16 pixels. Each single pixel is a branch in a ROOT TTree

'''
loader = ROOT.TMVA.DataLoader("dataset")


'''

   ## Setup Dataset(s)

   Define input data file and signal and background trees

'''

imgSize = 16 * 16
inputFileName = "images_data_16x16.root"

def MakeImagesTree(n, nh, nw):
    #  image size (nh x nw)
    ntot = nh * nw
    fileOutName = "images_data_"+str(nh)+"x"+str(nw)+".root"

    nRndmEvts = 10000 # number of events we use to fill each image
    delta_sigma = 0.1  # 5% difference in the sigma
    pixelNoise = 5

    sX1 = 3
    sY1 = 3
    sX2 = sX1 + delta_sigma
    sY2 = sY1 - delta_sigma

    h1 = ROOT.TH2D("h1", "h1", nh, 0, 10, nw, 0, 10)
    h2 = ROOT.TH2D("h2", "h2", nh, 0, 10, nw, 0, 10)

    f1 = ROOT.TF2("f1", "xygaus")
    f2 = ROOT.TF2("f2", "xygaus")
    sgn = ROOT.TTree("sig_tree", "signal_tree")
    bkg = ROOT.TTree("bkg_tree", "background_tree")

    f = ROOT.TFile(fileOutName, "RECREATE")

    x1 = [None]*ntot
    x2 = [None]*ntot

    # create signal and background trees with a single branch
    # an array of size nh x nw containing the image data

#     px1 = array(x1)
#     px2 = array(x2)

#     bkg.Branch("vars", "array", x1)
#     sgn.Branch("vars", "array", x2)

    # print("create tree")

    sgn.SetDirectory(f)
    bkg.SetDirectory(f)

    f1.SetParameters(1, 5, sX1, 5, sY1)
    f2.SetParameters(1, 5, sX2, 5, sY2)
    ROOT.gRandom.SetSeed(0)
    print("Filling ROOT tree \n")
    for i in range(n) :
      if (i % 1000 == 0):
         print("Generating image event ... " + str(i) )
      h1.Reset()
      h2.Reset()
      #generate random means in range [3,7] to be not too much on the border
      f1.SetParameter(1, ROOT.gRandom.Uniform(3, 7))
      f1.SetParameter(3, ROOT.gRandom.Uniform(3, 7))
      f2.SetParameter(1, ROOT.gRandom.Uniform(3, 7))
      f2.SetParameter(3, ROOT.gRandom.Uniform(3, 7))

      h1.FillRandom("f1", nRndmEvts)
      h2.FillRandom("f2", nRndmEvts)

      for k in range(nh):
         for l in range(nw):
            m = k * nw + l
            # add some noise in each bin
            x1[m] = h1.GetBinContent(k + 1, l + 1) + ROOT.gRandom.Gaus(0, pixelNoise)
            x2[m] = h2.GetBinContent(k + 1, l + 1) + ROOT.gRandom.Gaus(0, pixelNoise)


      sgn.Fill()
      bkg.Fill()

    sgn.Write()
    bkg.Write()

    ROOT.Info("MakeImagesTree", "Signal and background tree with images data written to the file" + str(f.GetName()))
    sgn.Print()
    bkg.Print()
    f.Close()

MakeImagesTree(5000, 16, 16)


inputFile = ROOT.TFile.Open(inputFileName)
if (inputFile == None):
    Error("TMVA_CNN_Classification", "Error opening input file %s - exit", inputFileName.Data())
signalTree = inputFile.Get("sig_tree")
backgroundTree = inputFile.Get("bkg_tree")

signalTree.Print()

nEventsSig = signalTree.GetEntries()
nEventsBkg = backgroundTree.GetEntries()
# global event weights per tree (see below for setting event-wise weights)
signalWeight = 1.0
backgroundWeight = 1.0

# You can add an arbitrary number of signal or background trees
loader.AddSignalTree(signalTree, signalWeight)
loader.AddBackgroundTree(backgroundTree, backgroundWeight)

## add event variables (image)
## use new method (from ROOT 6.20 to add a variable array for all image data)
 
for  i in range(0,imgSize):
    varName = "var"+str(i)
    loader.AddVariablesArray(varName,256,'F')

# Set individual event weights (the variables must exist in the original TTree)
#    for signal    : factory.SetSignalWeightExpression    ("weight1*weight2")
#    for background: factory.SetBackgroundWeightExpression("weight1*weight2")
# loader.SetBackgroundWeightExpression( "weight" )


# Apply additional cuts on the signal and background samples (can be different)
mycuts = ROOT.TCut("") # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
mycutb = ROOT.TCut("") # for example: TCut mycutb = "abs(var1)<0.5";

#  Tell the factory how to use the training and testing events
# 
#  If no numbers of events are given, half of the events in the tree are used
#  for training, and the other half for testing:
#     loader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
#  It is possible also to specify the number of training and testing events,
#  note we disable the computation of the correlation matrix of the input variables

nTrainSig = 0.8 * nEventsSig
nTrainBkg = 0.8 * nEventsBkg

#  build the string options for DataLoader::PrepareTrainingAndTestTree
prepareOptions = "nTrain_Signal="+str(nTrainSig)+":nTrain_Background="+str(nTrainBkg)+":SplitMode=Random:SplitSeed=100:NormMode=NumEvents:!V:!CalcCorrelations"
  

loader.PrepareTrainingAndTestTree(mycuts, mycutb, prepareOptions)

'''

   DataSetInfo              : [dataset] : Added class "Signal"
   : Add Tree sig_tree of type Signal with 10000 events
   DataSetInfo              : [dataset] : Added class "Background"
   : Add Tree bkg_tree of type Background with 10000 events



'''

# signalTree.Print();

'''
    # Booking Methods

    Here we book the TMVA methods. We book a Boosted Decision Tree method (BDT)

'''

# Boosted Decision Trees
# if (useTMVABDT):
#   factory.BookMethod(loader, ROOT.TMVA.Types.kBDT, "BDT","!V:NTrees=400:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:"+"UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20")

'''
#### Booking Deep Neural Network

Here we book the DNN of TMVA. See the example TMVA_Higgs_Classification.C for a detailed description of the
options
'''

if (useTMVADNN):
  layoutString = "Layout=DENSE|100|RELU,BNORM,DENSE|100|RELU,BNORM,DENSE|100|RELU,BNORM,DENSE|100|RELU,DENSE|1|LINEAR"

  #  Training strategies
  #  one can catenate several training strings with different parameters (e.g. learning rates or regularizations
  #  parameters) The training string must be concatenates with the `|` delimiter
  trainingString1 = "LearningRate=1e-3,Momentum=0.9,Repetitions=1,"+ "ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,"+"MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"+"Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0."
                          

  trainingStrategyString = "TrainingStrategy="
  trainingStrategyString += trainingString1 # + "|" + trainingString2 + ....

  # Build now the full DNN Option string

  dnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"+"WeightInitialization=XAVIER"
  dnnOptions+= ":"
  dnnOptions+= layoutString
  dnnOptions+= ":"
  dnnOptions+= trainingStrategyString

  dnnMethodName = "TMVA_DNN_CPU"

  dnnOptions += ":Architecture=CPU"


factory.BookMethod(loader, ROOT.TMVA.Types.kDL, dnnMethodName, dnnOptions)


# In[37]:


'''
### Book Convolutional Neural Network in TMVA

For building a CNN one needs to define

-  Input Layout :  number of channels (in this case = 1)  | image height | image width
-  Batch Layout :  batch size | number of channels | image size = (height*width)

Then one add Convolutional layers and MaxPool layers.

-  For Convolutional layer the option string has to be:
   - CONV | number of units | filter height | filter width | stride height | stride width | padding height | paddig
width | activation function

   - note in this case we are using a filer 3x3 and padding=1 and stride=1 so we get the output dimension of the
conv layer equal to the input

  - note we use after the first convolutional layer a batch normalization layer. This seems to help significantly the
convergence

 - For the MaxPool layer:
    - MAXPOOL  | pool height | pool width | stride height | stride width

The RESHAPE layer is needed to flatten the output before the Dense layer


Note that to run the CNN is required to have CPU  or GPU support

'''



inputLayoutString ="InputLayout=1|16|16"

#  Batch Layout
layoutString = "Layout=CONV|10|3|3|1|1|1|1|RELU,BNORM,CONV|10|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,"+"RESHAPE|FLAT,DENSE|100|RELU,DENSE|1|LINEAR"

#  Training strategies.
trainingString1 = "LearningRate=1e-3,Momentum=0.9,Repetitions=1,"+"ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,"+"MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"+"Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0.0"

trainingStrategyString = "TrainingStrategy="
trainingStrategyString += trainingString1 # + "|" + trainingString2 + "|" + trainingString3; for concatenating more training strings

# Build full CNN Options.


cnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:" +"WeightInitialization=XAVIER::Architecture=CPU"

cnnOptions +=  ":" + inputLayoutString
cnnOptions +=  ":" + layoutString
cnnOptions +=  ":" + trainingStrategyString
  ## New DL (CNN)
cnnMethodName = "TMVA_CNN_CPU"
# use GPU if available


cnnOptions += ":Architecture=CPU"
cnnMethodName = "TMVA_CNN_CPU"


factory.BookMethod(loader, ROOT.TMVA.Types.kDL, cnnMethodName, cnnOptions)


ROOT.Info("TMVA_CNN_Classification", "Building convolutional keras model")
#  create python script which can be executed
#  create 2 conv2d layer + maxpool + dense

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, BatchNormalization

model = Sequential()
model.add(Reshape((16, 16, 1), input_shape = (256, )))
model.add(Conv2D(10, kernel_size=(3,3), kernel_initializer='TruncatedNormal', activation='relu', padding='same' ) )
model.add(Conv2D(10, kernel_size=(3,3), kernel_initializer='glorot_normal', activation ='relu', padding = 'same') )
model.add(BatchNormalization())
model.add(Conv2D(10, kernel_size = (3,3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same') )
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1,1))) 
model.add(Flatten())
model.add(Dense(256, activation = 'relu')) 
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])
model.save('model_cnn.h5')
model.summary()



if (ROOT.gSystem.AccessPathName("model_cnn.h5")):
 Warning("TMVA_CNN_Classification", "Error creating Keras model file - skip using Keras")
else:
 #  book PyKeras method only if Keras model could be created
 ROOT.Info("TMVA_CNN_Classification", "Booking tf.Keras CNN model")


factory.BookMethod(loader, ROOT.TMVA.Types.kPyKeras, "PyKeras","H:!V:VarTransform=None:FilenameModel=model_cnn.h5:"+"FilenameTrainedModel=trained_model_cnn.h5:NumEpochs=20:BatchSize=128")

 if (usePyTorchCNN):

   ROOT.Info("TMVA_CNN_Classification", "Using Convolutional PyTorch Model")
   pyTorchFileName = str(ROOT.gROOT.GetTutorialDir()) + "/tmva/PyTorch_Generate_CNN_Model.py"
   print(pyTorchFileName)
   # check that pytorch can be imported and file defining the model and used later when booking the method is existing
   if (ROOT.gSystem.Exec("python -c 'import torch'")  | ROOT.gSystem.AccessPathName(pyTorchFileName) ):
      Warning("TMVA_CNN_Classification", "PyTorch is not installed or model building file is not existing - skip using PyTorch")
   else:
      # book PyTorch method only if PyTorch model could be created
      ROOT.Info("TMVA_CNN_Classification", "Booking PyTorch CNN model")
      methodOpt = "H:!V:VarTransform=None:FilenameModel=PyTorchModelCNN.pt:" + "FilenameTrainedModel=PyTorchTrainedModelCNN.pt:NumEpochs=20:BatchSize=100"
      methodOpt += ":UserCode=" + pyTorchFileName
      factory.BookMethod(loader, ROOT.TMVA.Types.kPyTorch, "PyTorch", methodOpt)


#  ## Train Methods

factory.TrainAllMethods()


## Test and Evaluate Methods

factory.TestAllMethods();

factory.EvaluateAllMethods();


## Plot ROC Curve

auto c1 = factory.GetROCCurve(loader);
c1->Draw();


# close outputfile to save output file
outputFile->Close();
