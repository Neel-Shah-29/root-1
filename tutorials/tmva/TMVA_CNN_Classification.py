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
# / \author Lorenzo Moneta


 # TMVA Classification Example Using a Convolutional Neural Network


#  Helper function to create input images data
#  we create a signal and background 2D histograms from 2d gaussians
#  with a location (means in X and Y)  different for each event
#  The difference between signal and background is in the gaussian width.
#  The width for the background gaussian is slightly larger than the signal width by few % values
# 
# 
import ROOT
from array import array

def MakeImagesTree(n, nh, nw):
#  image size (nh x nw)
   ntot = nh * nw
   fileOutName = ROOT.TString.Format("images_data_%dx%d.root", nh, nw)

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

   x1 = array("ntot",[0])
   x2 = array("ntot",[0])

    # create signal and background trees with a single branch
    # an array of size nh x nw containing the image data

   px1 = array(x1)
   px2 = array(x2)

   bkg.Branch("vars", "array", px1)
   sgn.Branch("vars", "array", px2)

    # print("create tree")

   sgn.SetDirectory(f)
   bkg.SetDirectory(f)

   f1.SetParameters(1, 5, sX1, 5, sY1)
   f2.SetParameters(1, 5, sX2, 5, sY2)
   ROOT.gRandom.SetSeed(0)
   print("Filling ROOT tree \n")
   for i in range(n) :
      if (i % 1000 == 0)
         print("Generating image event ... " + i )
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

   ROOT.Info("MakeImagesTree", "Signal and background tree with images data written to the file %s", f.GetName());
   sgn.Print()
   bkg.Print()
   f.Close()
