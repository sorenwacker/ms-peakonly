

from ms_peakonly import PeakOnly
from pathlib import Path as P
import pandas as pd



def test__run_peak_only_end_to_end(tmp_path):
    po = PeakOnly(model_dir=tmp_path)
    fn = 'tests/data/MH.mzML'

    assert P(fn).is_file(), fn

    table = po.process([fn])

    print(table)

    expected_columns = ['mz_mean', 'rt_min', 'rt_max', 'tests/data/MH.mzML']
    actual_columns = table.columns.to_list()
    assert isinstance(table, pd.DataFrame), f'Output is not a pandas.DataFrame but {type(table)}'
    assert len(table)>0, 'Output is empty'
    assert actual_columns == expected_columns, actual_columns


