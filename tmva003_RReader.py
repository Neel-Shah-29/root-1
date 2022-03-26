# /// \file
# /// \ingroup tutorial_tmva
# /// \notebook -nodraw
# /// This tutorial shows how to apply with the modern interfaces models saved in
# /// TMVA XML files.
# ///
# /// \macro_code
# /// \macro_output
# ///
# /// \date March 2022
# /// \author Neel Shah

import ROOT
from ROOT import TMVA

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

filename = "http://root.cern.ch/files/Higgs_data.root"

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

inputFile = ROOT.TFile("Higgs_data.root")
output = ROOT.TFile.Open("TMVA.root", "RECREATE")

factory = ROOT.TMVA.Factory("tmva003",output, "!V:!DrawProgressBar:AnalysisType=Classification")

# Open trees with signal and background events
data = ROOT.TFile.Open("Higgs_data.root")
signal = data.Get("TreeS")
background = data.Get("TreeB")
# --- Register the training and test trees

# Add variables and register the trees with the dataloader
dataloader = ROOT.TMVA.DataLoader("tmva003_BDT")
variables = ["var1", "var2", "var3", "var4"]
for var in variables: 
  dataloader.AddVariable(var);
### We set now the input data trees in the TMVA DataLoader class

# global event weights per tree (see below for setting event-wise weights)
signalWeight = 1.0
backgroundWeight = 1.0

#  You can add an arbitrary number of signal or background trees
dataloader.AddSignalTree    ( signal,     signalWeight     )
dataloader.AddBackgroundTree( background, backgroundWeight )

dataloader.PrepareTrainingAndTestTree("", "")

# Train a TMVA method
factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT", "!V:!H:NTrees=300:MaxDepth=2")
factory.TrainAllMethods()

# Next, we load the model from the TMVA XML file.
model="tmva003_BDT/weights/tmva003_BDT.weights.xml"

# In case you need a reminder of the names and order of the variables during
# training, you can ask the model for it.
variables = model.GetVariableNames()

# The model can now be applied in different scenarios:
# 1) Event-by-event inference
# 2) Batch inference on data of multiple events
# 3) Inference as part of an RDataFrame graph

# 1) Event-by-event inference
# The event-by-event inference takes the values of the variables as a std::vector<float>.
# Note that the return value is as well a std::vector<float> since the reader
# is also capable to process models with multiple outputs.

prediction = model.Compute([0.5, 1.0, -0.2, 1.5])
print("Single-event inference: " + prediction[0] + "\n\n")

# 2) Batch inference on data of multiple events
# For batch inference, the data needs to be structured as a matrix. For this
# purpose, TMVA makes use of the RTensor class. For convenience, we use RDataFrame
# and the AsTensor utility to make the read-out from the ROOT file.
df = ROOT.RDataFrame("TreeS", filename)
df2 = df.Range(3) #Read only a small subset of the dataset
x = ROOT.AsTensor(df2, variables)
y = model.Compute(x)

print("RTensor input for inference on data of multiple events:\n" + x + "\n\n")
print("Prediction performed on multiple events: " + y + "\n\n")

# 3) Perform inference as part of an RDataFrame graph
# We write a small lambda function that performs for us the inference on
# a dataframe to omit code duplication.
def make_histo (treename) :
      df= ROOT.RDataFrame(treename, filename)
      df2 = df.Define("y", Compute<4, float>(model), variables)
      return df2.Histo1D({treename.c_str(), ";BDT score;N_{Events}", 30, -0.5, 0.5}, "y")


sig = make_histo("TreeS")
bkg = make_histo("TreeB")

# Make plot
ROOT.gStyle.SetOptStat(0)
c = ROOT.TCanvas("", "", 800, 800)

sig.SetLineColor(kRed)
bkg.SetLineColor(kBlue)
sig.SetLineWidth(2)
bkg.SetLineWidth(2)
bkg.Draw("HIST")
sig.Draw("HIST SAME")

legend = ROOT.TLegend(0.7, 0.7, 0.89, 0.89)
legend.SetBorderSize(0)
legend.AddEntry("TreeS", "Signal", "l")
legend.AddEntry("TreeB", "Background", "l")
legend.Draw()

c.DrawClone()
