import os
import shutil
import tempfile
from hashlib import md5

import requests
from tqdm import tqdm

from cesped.zenodo.bechmarkUrls import ROOT_URL_PATTERN, NAME_TO_MASK_URL


def getDoneFname(destination_dir, record_id):
    return os.path.join(destination_dir, f"SUCCESSFUL_DOWNLOAD_{record_id}.txt")
def download_record(record_id, destination_dir, root_url=ROOT_URL_PATTERN):

    donefname = getDoneFname(destination_dir, record_id)
    if os.path.isfile(donefname):
        return
    root_url = f"{root_url}/{record_id}"
    r = requests.get(root_url)
    assert r.status_code == 200, f"Error, bad request response {r}"
    # print(r.json())
    os.makedirs(destination_dir, exist_ok=True)


    def check_checksum(content, expected_checksum):
        m = md5()
        m.update(content)
        calculated_checksum = m.hexdigest()
        return calculated_checksum == expected_checksum.replace("md5:", "")


    def download_file(url, expected_checksum, pbar, output_file, start_position):
        print(f"Downloading {url}")
        response = requests.get(url, stream=True)
        partial_content = b""
        if response.status_code != 200:
            raise RuntimeError(f"Error, it was not possible to download {url}. Try later. {response}")

        for chunk in response.iter_content(chunk_size=100*1024*2): #100MB chunks
            if chunk:
                partial_content += chunk
                pbar.update(len(chunk))

        if not check_checksum(partial_content, expected_checksum):
            print(f"Checksum mismatch for {url}")
            return False, start_position

        output_file.seek(start_position)
        output_file.write(partial_content)
        return True, start_position + len(partial_content)


    def download_and_concatenate_files(urls, sizes, checksums, output_file_path):
        total_size = sum(sizes)

        if os.path.exists(output_file_path):
            current_size = os.path.getsize(output_file_path)
        else:
            current_size = 0
        cum_size = 0
        with open(output_file_path, 'ab') as output_file:
            with tqdm(total=total_size, initial=current_size, unit='B', unit_scale=True,
                      desc="Downloading") as pbar:
                for url, size, checksum in zip(urls, sizes, checksums):
                    cum_size += size
                    if current_size >= cum_size:
                        print(f"Skipping already downloaded {url}")
                        continue
                    success, new_position = download_file(url, checksum, pbar, output_file, current_size)
                    if not success:
                        raise RuntimeError(f"Aborting process due to checksum mismatch for {url}")
                    current_size = new_position
        return current_size

    starFname = None
    mrcsFnames = []
    jsonFname = None
    files_info_dict = {}
    for fileRecord in r.json()["files"]:
        # print(fileRecord["key"], fileRecord["links"]["self"], fileRecord['checksum'], fileRecord['size'])
        try:
            fname = fileRecord.get("key",fileRecord.get("filename")) #It used to work with key, now it seems it is called filename
            link = f"https://zenodo.org/records/{record_id}/files/{fname}"
            size = fileRecord.get("size", fileRecord.get("filesize")) #It used to work with size, now it seems it is called filesize
            files_info_dict[fname] = (link, size, fileRecord.get('checksum'))
        except KeyError:
            print(fileRecord)
            raise
        if fname.endswith(".star"):
            starFname = fname
        elif ".mrcs" in fname:
            mrcsFnames.append(fname)
        elif fname.endswith(".json"):
            jsonFname = fname
        else:
            raise ValueError(f"Error, unexpected file {fname} found in the record")


    assert starFname, "Error, starfile not found in the record"
    url, size, checksum = files_info_dict[starFname]
    output_file_path = os.path.join(destination_dir, starFname)
    download_and_concatenate_files([url], [size], [checksum], output_file_path)
    print(f"Particle metadata downloaded at {output_file_path}")

    assert jsonFname, "Error, json file not found in the record"
    url, size, checksum = files_info_dict[jsonFname]
    output_file_path = os.path.join(destination_dir, jsonFname)
    download_and_concatenate_files([url], [size], [checksum], output_file_path)
    print(f"Particle json downloaded at {output_file_path}")

    assert mrcsFnames, "Error, mrcs files not found in the record"
    urls = []
    sizes = []
    checksums = []
    for fname in sorted(mrcsFnames, key=lambda x: int(x.split("mrcs.chunk_")[-1])):
        url, size, checksum = files_info_dict[fname]
        urls += [url]
        sizes += [size]
        checksums += [checksum]

    output_file_path = os.path.join(destination_dir, starFname.replace(".star", ".mrcs"))
    current_size = download_and_concatenate_files(urls, sizes, checksums, output_file_path)
    print(f"Particle stack downloaded at {output_file_path}")
    with open(donefname, "w") as f:
        f.write("%s\n"%record_id)
        f.write("%s\n"%current_size)
    print()

def download_mask(targetName, mask_fname):

    if os.path.exists(mask_fname):
        return
    mask_url = NAME_TO_MASK_URL[targetName]
    response = requests.get(mask_url, stream=True)
    assert response.status_code == 200, (f"Error downloading mask {mask_url}. {response} If you cannot download it "
                                         f"after retrying, place a "
                                         f"mask named {os.path.basename(mask_fname)} "
                                         f"on {os.path.split(mask_fname)[0]}")
    with tempfile.NamedTemporaryFile() as tmpf:
        for chunk in response.iter_content(chunk_size=100 * 1024 * 2):  # 100MB chunks
            tmpf.write(chunk)
        shutil.copyfile(tmpf.name, mask_fname)