
ROOT_URL_PATTERN="https://zenodo.org/api/records/"

NAME_PARTITION_TO_RECORID ={
    ("TEST", 0): "10998049",
    ("TEST", 1): "10998591",

    ("10786", 0): "8352275",
    ("10786", 1): "8352281",

    ("11120", 0): "8352296",
    ("11120", 1): "8352300",

    ("10166", 0): "8353634",
    ("10166", 1): "8353702",

    ("10280", 0): "8353722",
    ("10280", 1): "8353737",


    ("10374", 0): "8386774",
    ("10374", 1): "8386881",

    ("10409", 0): "8386582",
    ("10409", 1): "8386660",


    ("10399", 0): "11099225",
    ("10399", 1): "11093190",
    
    
    ("10648", 0): "8387534",
    ("10648", 1): "8390711",
    
    ("consensus_10648", 0): "11092025",
    ("consensus_10648", 1): "11092048",
    
    ("simulated_10648", 0): "11098206",
    ("simulated_10648", 1): "11098270",


}


RECORD_DEPENDENCIES = {"11092025":["8387534"], "11092048":["8390711"]}

_masks_base_url = "https://zenodo.org/record/11092291/files/"
NAME_TO_MASK_URL={
    "TEST":  f"{_masks_base_url}/11120_mask.mrc",
    "10166": f"{_masks_base_url}/10166_mask.mrc",

    "10648": f"{_masks_base_url}/10648_mask.mrc",
    "astex-5534": f"{_masks_base_url}/10648_mask.mrc",
    "consensus_10648":  f"{_masks_base_url}/10648_mask.mrc",
    "simulated_10648":  f"{_masks_base_url}/10648_mask.mrc",
    
    "10280": f"{_masks_base_url}/10280_mask.mrc",
    "10786": f"{_masks_base_url}/10786_mask.mrc",
    "10409": f"{_masks_base_url}/10409_mask.mrc",
    "11120": f"{_masks_base_url}/11120_mask.mrc",

    "10374": f"{_masks_base_url}/10374_mask.mrc",

    "10399": f"{_masks_base_url}/10399_mask.mrc",
}

