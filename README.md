# GDE
Codes for the follow papers:<br/>
Less is More: Reweighting Important Spectral Graph Features for Recommendation (SIGIR '22)<br/>
Less is More: Removing Redundancy of Graph Convolutional Networks for Recommendation (TOIS)

# Environment
The algorithm is implemented in Python 3.8.5, with the following libraries additionally needed to be installed:<br/>
* Pytorch+GPU==1.8.0<br/>
* Numpy==1.19.2<br/>
* Pandas==1.1.4<br/>

Due to the inefficiency of CPU, we only provide a GPU implementation. Feel free to modify the codes to adapt to your own environment.

# Get Started

Two steps to run the GDE algorithm:<br/>
1. Run preprocess_gde.py to generate the required spectral features for the dataset. You can change the number of smoothed spectral features by adjusting 'smooth_ratio'; similarly, by adjusting 'rough_ratio', you change the number of rough spectral features.
2. Run GDE.py to generate the accuracy on test sets. Explanation on hyperparameters is provided in the codes.

Similarly, two steps for SGDE algorithms.
