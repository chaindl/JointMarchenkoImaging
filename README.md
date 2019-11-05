# Joint Marchenko Imaging Code Using Pylops

Each main function performs simultaneous Marchenko imaging of the baseline and the monitor dataset. 

Inputs:
  * depths [m] of the first line and the last line to be imaged respectively
  * for Marchenko-depthloop-JointRadon.py only 3rd input is the number of lines to be calulated in 
    parallel

Outputs:
Each point of the image is calculated by performing Marchenko redatuming at that point and once all 
points at one depth level have been calculated the line is saved additionaly to the full image at 
the end of the runtime.

Main functions:
  * Marchenko-depthloop-IndepLsqr.py
  * Marchenko-depthloop-JointLsqr.py
  * Marchenko-depthloop-JointRadon.py (sparse inversion using radon transform, uses parallel computing)
 
MarchenkoFunctions.py contains supplementary functions

Dependencies:
  * sys and numpy
  * Pylops
  * multiprocessing

