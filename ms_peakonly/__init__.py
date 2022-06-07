import os
import io
import pandas as pd
import numpy as np
import torch
import urllib

from pathlib import Path as P

from .processing_utils.postprocess import ResultTable
from .processing_utils.roi import get_ROIs
from .processing_utils.matching import construct_mzregions, rt_grouping, align_component
from .processing_utils.run_utils import preprocess, get_borders
from .processing_utils.run_utils import border_correction, build_features, feature_collapsing
from .models.cnn_classifier import Classifier
from .models.cnn_segmentator import Segmentator


class PeakOnly():

    def __init__(self, mz_deviation=0.01, min_roi_length=15, max_zeros=3, min_peak_length=8, model_dir=None):
        self._mz_dev,  self._min_roi, self._max_zeros, self._min_peak_length, self._model_dir = \
            mz_deviation, min_roi_length, max_zeros, min_peak_length, model_dir
        self.results = None
        
    def process(self, fns):
        self.maybe_get_models()
        self.results = process_mzmls(fns, 
                    mz_dev=self._mz_dev,  
                    min_roi=self._min_roi, 
                    max_zeros=self._max_zeros, 
                    min_peak_length=self._min_peak_length,
                    model_dir=self._model_dir)
        return self.results

    def maybe_get_models(self):
        path = P(self._model_dir)

        if not path.is_dir():
            os.makedirs(path)

        fn_clf = path/'Classifier.pt'
        fn_seg = path/'Segmentator.pt'
        fn_rnn = path/'RecurrentCNN.pt'

        if not fn_clf.is_file():
        # Classifier
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/rAhl2u7WeIUGYA'
            urllib.request.urlretrieve(url, fn_clf)
        # Segmentator
        if not fn_seg.is_file():
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/9m5e3C0q0HKbuw'
            urllib.request.urlretrieve(url, fn_seg)
        # RecurrentCNN
        if not fn_rnn.is_file():
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/1IrXRWDWhANqKw'
            urllib.request.urlretrieve(url, fn_rnn)

    def as_mint_targets(self):
        return po_table_to_mint_peaklist(self.results)


def table_to_pandas(table):
    output = io.StringIO()
    table.to_csv(output)
    string = output.getvalue()
    return pd.read_csv( io.StringIO(string), index_col=0 ) 


def process_mzmls(fns, mz_dev, min_roi, max_zeros, min_peak_length, model_dir):
    print("Finding ROIs...")
    rois = {}
    for f in fns:
        rois[f] = get_ROIs(f, mz_dev, min_roi, max_zeros, None)

    mzregions = construct_mzregions(rois, mz_dev)
    components = rt_grouping(mzregions)

    print("Aligning ROIs...")
    aligned_components = []
    for i, component in enumerate(components):
        aligned_components.append(align_component(component))

    print("Finding peaks...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = Classifier().to(device)
    path2classifier_weights = os.path.join(model_dir, "Classifier.pt")
    classifier.load_state_dict(torch.load(path2classifier_weights, map_location=device))
    classifier.eval()
    segmentator = Segmentator().to(device)
    path2segmentator_weights = os.path.join(model_dir, "Segmentator.pt")
    segmentator.load_state_dict(torch.load(path2segmentator_weights, map_location=device))
    segmentator.eval()

    component_number = 0
    features = []
    for j, component in enumerate(aligned_components):  # run through components
        borders = {}
        to_delete = []
        for i, (sample, roi) in enumerate(zip(component.samples, component.rois)):
            signal = preprocess(roi.i, device, interpolate=True, length=256)
            classifier_output, _ = classifier(signal)
            classifier_output = classifier_output.data.cpu().numpy()
            label = np.argmax(classifier_output)
            if label == 1:
                _, segmentator_output = segmentator(signal)
                segmentator_output = segmentator_output.data.sigmoid().cpu().numpy()
                borders[sample] = get_borders(segmentator_output[0, 0, :], segmentator_output[0, 1, :],
                                              peak_minimum_points=min_peak_length,
                                              interpolation_factor=len(signal[0, 0]) / len(roi.i))
            else:
                to_delete.append(i)
        if len(borders) > len(fns) // 3:  # enough rois contain a peak
            component.pop(to_delete)  # delete ROIs which don't contain peaks
            border_correction(component, borders)
            features.extend(build_features(component, borders, component_number))
            component_number += 1

    features = feature_collapsing(features)
    to_delete = []
    for i, feature in enumerate(features):
        if len(feature) <= len(fns) // 3:  # to do: adjustable parameter
            to_delete.append(i)
    for j in to_delete[::-1]:
        features.pop(j)
    print('total number of features: {}'.format(len(features)))
    features.sort(key=lambda x: x.mz)
    table = ResultTable(fns, features)
    table.fill_zeros(mz_dev)
    df = table_to_pandas( table )
    df = df.rename(columns={'mz': 'mz_mean', 'rtmin': 'rt_min', 'rtmax': 'rt_max'})    
    return df


def po_table_to_mint_peaklist(df, unit='minutes'):
    df = df.copy()
    df = df.rename(columns={'mz': 'mz_mean', 'rtmin': 'rt_min', 'rtmax': 'rt_max'})
    df['rt'] = df[['rt_min', 'rt_max']].mean(axis=1)
    if unit == 'minutes':
        df['rt_min'] = df['rt_min']*60.
        df['rt_max'] = df['rt_max']*60.
        df['rt'] =  df['rt']*60.
    df['peak_label'] = df.mz_mean.apply(lambda x: str(np.round(x, 3))) + '@' + df.rt.apply(lambda x: str(np.round(x, 2)))
    df['mz_width'] = 10 
    df = df.set_index(['peak_label', 'mz_mean', 'mz_width', 'rt', 'rt_min', 'rt_max']).reset_index()
    return df
