
ROOT_URL_PATTERN="https://zenodo.org/api/records/"

NAME_PARTITION_TO_RECORID ={
    ("TEST", 0): "8363635",
    ("TEST", 1): "8363677",

    ("10786", 0): "8352275",
    ("10786", 1): "8352281",

    ("11120", 0): "8352296",
    ("11120", 1): "8352300",

    ("10166", 0): "8353634",
    ("10166", 1): "8353702",

    ("10280", 0): "8353722",
    ("10280", 1): "8353737",

    ("10647", 0): "8387534",
    ("10647", 1): "8390711",

    ("10374", 0): "8386774",
    ("10374", 1): "8386881",

    ("10409", 0): "8386582",
    ("10409", 1): "8386660",

}

_masks_base_url = "https://zenodo.org/record/8392782/files/"
NAME_TO_MASK_URL={
    "TEST":  f"{_masks_base_url}/10166_mask.mrc",
    "10166": f"{_masks_base_url}/10166_mask.mrc",

    "10647": f"{_masks_base_url}/10647_mask.mrc",
    "astex-5534": f"{_masks_base_url}/10647_mask.mrc",

    "10280": f"{_masks_base_url}/10280_mask.mrc",
    "10786": f"{_masks_base_url}/10786_mask.mrc",
    "10409": f"{_masks_base_url}/10409_mask.mrc",
    "11120": f"{_masks_base_url}/11120_mask.mrc",

    "10374": f"{_masks_base_url}/10374_mask.mrc",

}