# Joint Marchenko Imaging Code Using Pylops

Each main function performs Marchenko imaging and creates angle gathers of the input datasets. The provided inputs are synthetics from a baseline and a monitor survey as well as files describing the acquisition geometry. Most of the options are hardcoded and can be changed by modifying the lines in the code. But, for the purpose of parallel computing, the functions require a start depth and an end depth as input.

Input Arguments:
  * Depths [m] of the first line and the last line to be imaged respectively
  
Input Files:
  * r.dat and s.dat: receiver and source positions
  * select_rfrac..:  receivers to be used (missing ones are gaps)
  * wav.dat:         wavelet
  * trav.dat:        traveltimes (used by LSQR functions)¹
  * vel_sm.dat:      smoothed velocity model used for angle gathers
  * Traveltimes/..:  cut up traveltime data (used by Radon functions)¹
  * R/dat1..:        baseline synthetics²
  * R/dat2..:        monitor synthetics²

1: Due to the file size the file is only available on request.
2: One example file is provided and the entire dataset is available upon request.

Outputs:

Each point of the image and angle gather is calculated by performing Marchenko redatuming at that point and once all 
points at one depth level have been calculated the line is saved. The image and the full angle gather are obtained by combining all the respective lines into an array.

Main functions:
  * Marchenko-depthloop-IndepLsqr.py:
      * standard Marchenko inversion
      * simultaneously processes both datasets
      * also calculates and saves single-scattering results
      * may be used to create reference solutions by replacing the restriction matrices with identity matrices
  * Marchenko-depthloop-JointLsqr.py
      * joint Marchenko inversion of the baseline and the monitor dataset without sparsity constraint
  * Marchenko-depthloop-IndepRadon.py
      * Marchenko inversion of a single dataset in a sparse domain (here a Radon transform in sliding windows)
      * by commenting and uncommenting a few lines one the inversion solvers LSQR, FISTA and SPGL1 may be chosen
      * due to the higher computational cost of this code only one dataset is processed at a time
      * for better performance when using parallel computing the traveltime input file has been divided up and the direct wave sections are loaded in the wrapper
  * Marchenko-depthloop-JointRadon.py
      * joint Marchenko inversion of the baseline and the monitor dataset in a sparse domain (here a Radon transform in sliding windows)
      * by commenting and uncommenting a few lines one the inversion solvers LSQR, FISTA and SPGL1 may be chosen
      * for better performance when using parallel computing the traveltime input file has been divided up and the direct wave sections are loaded in the wrapper

Dependencies: sys, numpy, pylops, scipy
