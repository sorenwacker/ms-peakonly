Modified code from [Arseha](https://github.com/Arseha/peakonly)


This repository mainly contains the origial code with an added convenient high-level Python API for easier use. 
For example for interactive use in the JupyterLab. The original code contains code for a NeuralNetwork based approach
for peak detection in untargeted metabolomics. This high-level API makes it much easier to use the code in your
own Python script or notebook. In the following a small example on how to use it and how to install it. 

# Example usage
       
    from ms_peakonly import PeakOnly
    from glob import glob
    
    # Get a list of file names to process (I believe onl mzML files are supported)
    fns = glob('/my/metabolomics/directory/*.mzML')
    
    # Instantiate the engine if the neural network weights
    # are not already downloaded this will
    # also download the models. 
    po = PeakOnly(model_dir='/my/model/directory/')
    
    # Simply pass the list of filenames to the `process` method.
    table = po.process(fns)
    
# Installation
This API can be installed with `pip` or `python setup.py install` and is then available 
via a simple import in your Python environment as shown above.
   
    pip install git+https://github.com/sorenwacker/ms-peakonly
    
