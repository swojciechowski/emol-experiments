import os

from emol.helpers.keel import parse_keel_dat

from sklearn.preprocessing import LabelEncoder

DEFAULT_KEEL_DIR = "keel"

def find_datasets(keel_path=DEFAULT_KEEL_DIR):
    for fname in os.listdir(keel_path):
        if ".dat" in fname:
            yield fname.split(".")[:-1]

def load_keel_dataset(ds_name, keel_path=DEFAULT_KEEL_DIR, encode_categorical=False):
    _, data, target = parse_keel_dat(os.path.join(keel_path, ds_name + ".dat"))

    if encode_categorical:
        data = encode_categorical(data)

    return data.to_numpy(), LabelEncoder().fit_transform(target.to_numpy())
