import sys
print(sys.executable)
import socket
print(socket.gethostname())
import os
os.chdir("/home/icb/alioguz.can/projects/campa_pt")
print(os.getcwd())
import sys
# Add the path to sys.path
sys.path.append('/home/icb/alioguz.can/projects/campa_pt')
from campa.tl import (
    # Cluster,
    # TorchEstimator,
    # TorchPredictor,
    TorchExperiment,
    # ModelComparator,
    run_torch_experiments
)
from campa.data import MPPData
from campa.utils import init_logging
from campa.constants import campa_config
from pathlib import Path


if __name__=="__main__":
    # init logging with level INFO=20, WARNING=30
    init_logging(level=30)
    # read correct campa_config -- created with setup.ipynb
    CAMPA_DIR = Path.cwd()
    campa_config.config_fname = CAMPA_DIR / "notebooks/params/campa.ini"
    print(campa_config)
    torch_exps = TorchExperiment.get_experiments_from_config("notebooks/params/example_experiment_params_torch.py")
    run_torch_experiments(torch_exps)