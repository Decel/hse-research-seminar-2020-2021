from typing import Callable

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def normalize_data_frames(dataframe: pd.DataFrame, target_field_name: str, predicate: Callable) -> (np.ndarray, np.ndarray):
    features = list(dataframe.columns.values)
    features_filtered = list(filter(predicate, features))

    # Separating out the features
    x_axis = dataframe.loc[:, features_filtered].values

    # Separating out the target
    y_axis = dataframe.loc[:, [target_field_name]].values

    # Standardizing the features
    x_axis = StandardScaler().fit_transform(x_axis)

    return x_axis, y_axis


def prepare_pca_dataframe(frame: pd.DataFrame, x_axis: np.ndarray, target_field_name: str) -> pd.DataFrame:
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x_axis)
    principal_dataframe = pd.DataFrame(data=principal_components,
                                       columns=['Principal Component 1', 'Principal Component 2'])

    return pd.concat([principal_dataframe, frame[target_field_name]], axis=1)


def prepare_umap_dataframe(frame: pd.DataFrame, x_axis: np.ndarray, target_field_name: str) -> pd.DataFrame:
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(x_axis)
    embedded_dataframe = pd.DataFrame(data=embedding,
                                      columns=['Principal Component 1', 'Principal Component 2'])
    return pd.concat([embedded_dataframe, frame[target_field_name]], axis=1)
