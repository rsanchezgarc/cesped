import os
import tempfile

import requests
import json
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
import argparse

from cesped.zenodo import tokens

#TODO: Ask for symmetry and store it.

SANDBOX = False
CHUNK_SIZE = 1024 ** 2 * 300  # 300MB
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dirname" , type=str, required=True)
    parser.add_argument("-e", "--empiarID" , type=int, required=True)
    parser.add_argument("-s", "--split" , type=int, required=True)
    parser.add_argument("-y", "--symmetry" , type=str, required=True)

    args = parser.parse_args()
    # dir_to_upload = "/homes/sanchezg/ScipionUserData/projects/EMPIAR-10166/Runs/000463_ProtRelionAutorefSplitData/extra/subset_1/"  # "/home/sanchezg/tmp/ConorData/bound/postprocess/"
    # dataset_name = "CESPED-10166_split1"
    dir_to_upload = args.dirname
    assert args.split in [0,1]
    dataset_name = f"CESPED-{args.empiarID}_split{args.split}"
    print(dir_to_upload)
    print(dataset_name)



    assert os.path.exists(dir_to_upload), f"Error, file {dir_to_upload} does not exists"

    if not SANDBOX:
        # zenodo
        ACCESS_TOKEN = tokens.ACCESS_TOKEN_ZENODO
        baseurl = 'https://zenodo.org/api/'
    else:
        # zenodo sandbox
        ACCESS_TOKEN = tokens.ACCESS_TOKEN_ZENODO_SANDBOX
        baseurl = 'https://sandbox.zenodo.org/api/'

    r = requests.get(baseurl + 'deposit/depositions', params={'access_token': ACCESS_TOKEN})
    assert r.status_code == 200, "Error, cannot connect"
    r.json()


    # Creates empty deposition
    headers = {"Content-Type": "application/json"}
    params = {'access_token': ACCESS_TOKEN}
    r = requests.post(baseurl + 'deposit/depositions', params=params, json={}, headers=headers)

    print("Empty deposition creation status", r.status_code)
    assert r.status_code == 201
    # 201
    print(r.json())

    print("URL TO CHECK IN THE BROWSER", r.json()["links"]["latest_draft_html"])

    bucket_url = r.json()["links"]["bucket"]
    deposition_id = r.json()["id"]

    '''
    The target URL is a combination of the bucket link with the desired filename
    seperated by a slash.
    '''

    ########## METADATA ##########
    data = {
        'metadata': {
            'title': dataset_name,
            'license': "cc-by-4.0",
            'upload_type': 'dataset',
            'version': '0.1',
            'description': f'{dataset_name} is part of the Cryo-EM Supervised Pose Estimation'
                           f' Dataset benchmark. The particles*.star file contains the metadata, and '
                           f'the .mrcs.chunk* contains the images',
            'creators': [{'name': 'Anonymous, Anonymous',
                          'affiliation': 'Anonymous'}]
        }
    }
    r = requests.put(baseurl + 'deposit/depositions/%s' % deposition_id,
                     params={'access_token': ACCESS_TOKEN}, data=json.dumps(data),
                     headers=headers)


    def upload_with_chunks(file_path, upload_url, web_filename, chunk_size=CHUNK_SIZE):
        file_size = os.stat(file_path).st_size
        if file_size <= chunk_size:
            upload(file_path, upload_url, web_filename)
            return
        chunk_number = 0
        with open(file_path, "rb") as fp:
            with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as t:
                f = CallbackIOWrapper(t.update, fp, "read")
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    chunk_filename = f"{web_filename}.chunk_{chunk_number}"
                    target_url = f"{upload_url}/{chunk_filename}"
                    #                print(chunk_filename)
                    r = requests.put(target_url, data=chunk, params=params)
                    assert r.status_code == 200
                    chunk_number += 1


    def upload(path, upload_url, web_filename):
        file_size = os.stat(path).st_size
        with open(path, "rb") as fp:
            with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as t:
                wrapped_file = CallbackIOWrapper(t.update, fp, "read")
                r = requests.put(
                    "%s/%s" % (upload_url, web_filename),
                    data=wrapped_file,
                    params=params,
                )
                assert r.status_code == 200, f"Error, {web_filename} was not uploaded"


    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, f"info_{args.split}.json")
        with open(temp_file_path, "w") as f:
            json.dump({"symmetry": args.symmetry.upper()}, f)
            f.seek(0)
        upload_with_chunks(f.name, bucket_url, os.path.basename(f.name))

    for filename in os.listdir(dir_to_upload):
        path = os.path.join(dir_to_upload, filename)
        print(f"Uploading {path}")
        upload_with_chunks(path, bucket_url, filename)




    print("URL TO CHECK IN THE BROWSER", r.json()["links"]["latest_draft_html"])
    ## WARNING, THE FOLLOWING LINE WILL PUBLISH THE RESULTS. ARE YOU SURE?
    # input(" THE FOLLOWING LINE WILL PUBLISH THE RESULTS. ARE YOU SURE?")
    # r = requests.post(baseurl+'deposit/depositions/%s/actions/publish' % deposition_id, params={'access_token': ACCESS_TOKEN} )