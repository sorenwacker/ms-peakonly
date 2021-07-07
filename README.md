Modified code from [Arseha](https://github.com/Arseha/peakonly)


# High-level API for PeakOnly for simplified use
       
    from ms_peakonly import PeakOnly
    from glob import glob
    
    fns = glob('/my/metabolomics/directory/*.mzML')
    
    po = PeakOnly(model_dir='/my/model/directory/')
    
    table = po.process(fns)
    
# Installation

    git clone git@github.com:soerendip/ms-peakonly.git
    
    cd ms-peakonly
    
    pip install .
    
