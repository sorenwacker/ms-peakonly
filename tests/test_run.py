

from ms_peakonly import PeakOnly



def test__run_peak_only(tempdir):
    po = PeakOnly(model_dir=tempdir)
    fn = 'data/MH.mzML'
    table = po.process(fn)

    print(table)


