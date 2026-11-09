import json
import re
from pathlib import Path
import unittest

from metan.config import ROOT,load_config
from metan.metrics import regression_metrics,mean_sd
from metan.runtime import sha256

class ReleaseContracts(unittest.TestCase):
    def test_runtime_model_names(self):
        retired_name=re.compile('whis'+'per',re.IGNORECASE)
        for folder in [ROOT/'metan',ROOT/'baselines']:
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix in ['.py','.R']:
                    self.assertIsNone(retired_name.search(path.read_text(encoding='utf8')),str(path.relative_to(ROOT)))

    def test_regression_metrics(self):
        result=regression_metrics([1,2,3],[1,2,3])
        self.assertEqual(result['Pearson'],1)
        self.assertEqual(result['MSE'],0)
        self.assertEqual(result['MAE'],0)
        self.assertEqual(result['RMSE'],0)
        self.assertEqual(result['R2'],1)
        self.assertEqual(result['prediction_SD_ratio'],1)

    def test_invalid_predictions_fail(self):
        with self.assertRaises(ValueError):regression_metrics([1,2],[1,float('nan')])

    def test_sample_sd(self):
        mean,sd=mean_sd([1,2,3])
        self.assertEqual(mean,2)
        self.assertEqual(sd,1)

    def test_frozen_sources(self):
        manifest=json.loads((ROOT/'metan/source_manifest.json').read_text())
        self.assertTrue(manifest)
        for name,digest in manifest.items():self.assertEqual(sha256(ROOT/name),digest)

    def test_configuration_contract(self):
        for path in (ROOT/'configs').glob('*/experiment.json'):
            spec=load_config(path)
            model=json.loads(Path(spec['model_path']).read_text())
            self.assertEqual(spec['traits'],model['output_layer']['phenotype_name'])
            self.assertEqual(spec['PCA'],model['pca_covariates']['n_components'])
            self.assertEqual(len(spec['seeds'])*len(spec['folds']),15)
            self.assertTrue(Path(spec['baseline_config_path']).is_file())

if __name__=='__main__':unittest.main()
