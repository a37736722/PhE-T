import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, TypedDict


class BooleanTraitInfo(TypedDict):
    name: int
    true_id: int
    false_id: int


class PhenotypeTokenizer:
    def __init__(self, n_bins: int = 10, binning: str = 'uniform'):
        self.n_bins = n_bins
        self.binnning = binning
        self.p_id_map: Dict[str, int] = {}
        self.p_name_map: Dict[int, str] = {}
        self.v_id_map: Dict[int, Dict[Union[str, int, float, bool], int]] = {}
        self.v_desc_map: Dict[int, Dict[int, Union[str, bool]]] = {}
        self.p_size = 0
        self.v_size = 0
        self.next_p_id = 0
        self.next_v_id = 0
        self.num_features: List[str] = []
        self.cat_features: List[str] = []
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.boolean_traits: Dict[str, BooleanTraitInfo] = {}
        self._add_special_tokens()
    
    def fit(self, df: pd.DataFrame, num_features: List[str], cat_features: List[str]):
        self.num_features = num_features
        self.cat_features = cat_features

        for feature in num_features:
            self._add_phenotype(feature)
            if self.binnning == 'uniform':
                bins, bin_edges = pd.cut(df[feature], bins=self.n_bins, duplicates='drop', retbins=True)
            elif self.binnning == 'quantile':
                bins, bin_edges = pd.qcut(df[feature], q=self.n_bins, duplicates='drop', retbins=True)
                
            self.bin_edges[feature] = bin_edges 
            for i in range(len(bin_edges) - 1):
                lower = round(bin_edges[i], 2)
                upper = round(bin_edges[i+1], 2)
                interval = f"({lower}, {upper}]"
                self._add_value(feature, interval)

        for feature in self.cat_features:
            self._add_phenotype(feature)
            unique_values = df[feature].unique()
            for value in unique_values:
                self._add_value(feature, self._convert_numpy_type(value))
            
            # Check if this is a bool trait
            if len(unique_values) == 2 and set(unique_values) == {True, False}:
                p_id = self.get_phenotype_id(feature)
                true_id = self.v_id_map[p_id][True]
                false_id = self.v_id_map[p_id][False]
                self.boolean_traits[p_id] = BooleanTraitInfo(
                    name=feature,
                    true_id=true_id,
                    false_id=false_id
                )
        
        self.p_size = self.next_p_id
        self.v_size = self.next_v_id

    def encode(self, row: Dict[str, Any]) -> Dict[str, int]:
        phenotype_ids = []
        value_ids = []
        for feature, value in row.items():
            p_id = self.get_phenotype_id(feature)
            if p_id == -1:
                raise ValueError(f"Unknown phenotype: {feature}")
                
            if feature in self.num_features:
                # For numerical features, find the appropriate bin
                bin_edges = self.bin_edges[feature]
                bin_index = np.digitize([value], bin_edges, right=True)[0]
                
                if bin_index == 0:
                    lower = round(bin_edges[bin_index], 2)
                    upper = round(bin_edges[bin_index + 1], 2)
                elif bin_index == len(bin_edges):
                    lower = round(bin_edges[-2], 2)
                    upper = round(bin_edges[-1], 2)
                else:
                    lower = round(bin_edges[bin_index - 1], 2)
                    upper = round(bin_edges[bin_index], 2)
                
                interval = f"({lower}, {upper}]"
                v_id = self.v_id_map[p_id].get(interval, -1)
            else:
                # For categorical features, directly map the value
                v_id = self.v_id_map[p_id].get(value, -1)
            
            if v_id == -1:
                raise ValueError(f"Unknown value for feature '{feature}': {value}. " 
                                 f"This value was not present in the training data. "
                                 f"For numerical features, ensure the value falls within "
                                 f"the range of the training data. For categorical features, "
                                 f"ensure the value was present in the training data.")
            phenotype_ids.append(p_id)
            value_ids.append(v_id)
        return {'phenotype_ids': phenotype_ids, 'value_ids': value_ids}

    def decode(self, row: Dict[str, List[int]]) -> Dict[str, Any]:
        decoded = {}
        phenotype_ids = row['phenotype_ids']
        value_ids = row['value_ids']
        
        for p_id, v_id in zip(phenotype_ids, value_ids):
            phenotype = self.get_phenotype_name(p_id)
            if phenotype == "Unknown":
                continue  # Skip unknown phenotypes
            value = self.get_value_description(p_id, v_id)
            if value == "Unknown":
                continue  # Skip unknown values
            
            if phenotype in self.num_features:
                # For numerical features, return the midpoint of the bin
                bin_range = value.strip('()[]').split(', ')
                decoded[phenotype] = (float(bin_range[0]) + float(bin_range[1])) / 2
            else:
                # For categorical features, return the value as is
                decoded[phenotype] = value
        
        return decoded

    def _add_phenotype(self, name: str) -> int:
        if name not in self.p_id_map:
            p_id = self.next_p_id
            self.p_id_map[name] = p_id
            self.p_name_map[p_id] = name
            self.v_id_map[p_id] = {}
            self.v_desc_map[p_id] = {}
            self.next_p_id += 1
        return self.p_id_map[name]

    def _add_value(self, phenotype: str, value: Union[str, int, float, bool]) -> int:
        p_id = self.p_id_map[phenotype]
        if value not in self.v_id_map[p_id]:
            v_id = self.next_v_id
            self.v_id_map[p_id][value] = v_id
            self.v_desc_map[p_id][v_id] = value
            self.next_v_id += 1
        return self.v_id_map[p_id][value]

    def get_phenotype_id(self, name: str) -> int:
        return self.p_id_map.get(name, -1)

    def get_phenotype_name(self, p_id: int) -> str:
        return self.p_name_map.get(p_id, "Unknown")

    def get_value_ids(self, p_id: int) -> List[int]:
        return list(self.v_desc_map[p_id].keys())

    def get_value_description(self, p_id: int, v_id: int) -> str:
        return self.v_desc_map[p_id].get(v_id, "Unknown")

    def get_boolean_trait_info(self, p_id: int) -> Union[BooleanTraitInfo, None]:
        return self.boolean_traits.get(p_id)
    
    def _add_special_tokens(self):
        self._add_phenotype('Special tokens')
        v_id = self._add_value('Special tokens', 'Mask token')
        self.mask_token_id = v_id

    @staticmethod
    def _convert_numpy_type(value: Any) -> Any:
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        elif isinstance(value, (np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        return value