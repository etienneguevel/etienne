import os
from tvb.simulator.lab import *

data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            'data'
        )
)

berlin_subjects = ["DH_20120806","QL_20120814"]

def load_hcp_subject(subj):
    subj_root = os.path.join( data_root, "external", "hcp_connectomes")

    dataset = {}

    dataset['connectivity'] = connectivity.Connectivity.from_file( 
            os.path.abspath(
                os.path.join( subj_root, f"{subj}.zip")
            )
    )

    return dataset


def load_berlin_subject(subj):
    subj_root = os.path.join( data_root, "external", "berlin_subjects", subj)

    dataset = {}

    dataset['eeg'] = monitors.EEG.from_file(
            sensors_fname=os.path.join(
                subj_root, f"{subj}_EEGLocations.txt"
            ),
            projection_fname=os.path.join(
                subj_root, f"{subj}_ProjectionMatrix.mat"
            ),
            rm_f_name=os.path.join(
                subj_root, f"{subj}_RegionMapping.txt"
            ),
    )

    dataset['connectivity'] = connectivity.Connectivity.from_file( 
            os.path.abspath(
                os.path.join( subj_root, f"{subj}_Connectivity.zip")
            )
    )

    # TODO surfaces, sensor locations, BOLD if available, ...

    return dataset


def load_sensors(sensors_fname):
    eeg_sensors = sensors.SensorsEEG.from_file(sensors_fname)
    return eeg_sensors

