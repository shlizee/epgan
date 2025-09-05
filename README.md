Source code and Jupyter Notebook for testing pre-trained EP-GAN models (small and large HH-model) with respect to 9 experimental C. elegans neurons considered in the paper.

# Dependencies
NumPy, SciPy, Pandas, PyTorch (CPU version), Matplotlib, ipython, jupyter

# Steps for running the notebook

Step 1: Make sure your Python environment has all the required libraries listed above

Step 2: Download the repository in .zip file and unzip it at your desired location

Step 3: Manually download PyTorch EP-GAN models S1.pth (for small HH-model) and L1.pth (for large HH-model) from [Here](https://github.com/shlizee/epgan/tree/main/EPGAN/pretrained_models) and place them under epgan-main/EPGAN/pretrained_models

Step 4: Download experimental recording data exp.zip for 9 neurons from Mendeley Data and unzip it under epgan-main/EPGAN/data (There should be /sim and /exp folders after unzipping)

Step 5: Start Jupyter Notebook navigation interface via typing "jupyter notebook" prompt e.g., Anaconda Prompt (Windows) or Terminal (Mac/Linux)

Step 6: Load the Jupyter Notebook "EP-GAN testing on experimental neurons.ipynb" and execute each cell from top to bottom

# How to cite

If you are using this package please cite the following:
\
\
Kim, J., Peng, M., Chen, S., Liu, Q., & Shlizerman, E. (2025). ElectroPhysiomeGAN: Generation of Biophysical Neuron Model Parameters from Recorded Electrophysiological Responses. eLife, 13.
