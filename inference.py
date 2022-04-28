import torch
import torch.nn as nn
import torchvision.transforms as T
import webdataset as wds
import pandas as pd
import timm
import mmh3
import os
import json
import fsspec
import uuid
from braceexpand import braceexpand
from data import create_webdataset
from tqdm import tqdm

def inference(device, args):
    """
    Load the model, initialize DDP, and run inference.
    """
    num_workers = args.num_workers
    batch_size = args.batch_size
    output_num_samples = args.output_num_samples
    output_columns = args.output_columns


    no_of_available_devices = torch.cuda.device_count()
    model, transforms = load_model(device)
    if args.bucket_dir is None:
        output_folder = args.output_dir
        file_handler = open
        os.makedirs(output_folder, exist_ok=True)
    else:
        fs, output_folder = fsspec.core.url_to_fs(args.bucket_dir,
            client_kwargs={"endpoint_url":"https://bucket.vpce-06aadfc9fc5aabd58-bv32itci.s3.us-east-1.vpce.amazonaws.com/"})
        output_folder += "/"
        file_handler = fs.open
    


    all_urls = list(braceexpand(args.urls))
    # slice URLs to be sharded when multiple devices are being used. Ex: When there are two GPUs available [0,2,4...] urls will be processed on device 0 and [1,3,5..] urls on device 1 
    urls = all_urls[device::no_of_available_devices]
    dataset = create_webdataset(
        urls,
        transforms,
        enable_metadata=True,
    )
    dataloader = wds.WebLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate
    )

    # Run inference
    current_samples = []
    if device == 0:
        pbar = tqdm(total=args.num_samples)

    additional_columns = additional_columns_to_include(args)
    output_metadata_columns = ["pwatermark"] + additional_columns
    
    for batch in tqdm(dataloader, desc="processing.."):
        img = batch['image_tensor'].to(device)
        start = perf_counter()
        with torch.no_grad():
            out = model(img)
            out = torch.nn.functional.softmax(out, dim=1)
        current_samples.extend(statistics_to_array(out, batch, additional_columns))
        # Save current samples to parquet
        if len(current_samples) >= output_num_samples:
            df = pd.DataFrame(current_samples, columns=output_metadata_columns)
            df['pwatermark'] = df['pwatermark'].astype("float32")
            with file_handler(os.path.join(output_folder, str(uuid.uuid4())) + '.parquet', 'wb') as f:
                df.to_parquet(f)
            current_samples = []
    if device == 0:
        pbar.update(no_of_available_devices * args.batch_size)
        # aproximage pbar update? assuming that the other othe devices are processing batches at same speed as device 0
    df = pd.DataFrame(current_samples, columns=output_metadata_columns)
    df['pwatermark'] = df['pwatermark'].astype("float32")
    with file_handler(os.path.join(output_folder, str(uuid.uuid4())) + '.parquet', 'wb') as f:
        df.to_parquet(f)
    if device == 0:
        pbar.close()


def load_model(device):
    """
    Loads model.pt into a pretrained timm model.
    """
    transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = timm.create_model('efficientnet_b3a', pretrained=False, num_classes=2)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2)
    )

    # Load model weights
    state_dict = torch.load('./model.pt',
        map_location=torch.device(f"cuda:{device}")
    )['weights']
    
    model.load_state_dict(state_dict)
    model.eval().to(device)

    return model, transforms

def collate(arr):
    keys = arr[0].keys()
    ret_dict = {}
    for k in keys:
        ret_dict[k] = [x[k] for x in arr]
        if k == 'image_tensor':
            ret_dict[k] = torch.stack(ret_dict[k])
    
    return ret_dict

def statistics_to_array(out, batch, columns):
    output = []
    for i in range(out.shape[0]):
        row = [out[i][0].item(), batch['image_path'][i]]
        # for c in columns:
        #     if c in batch:
        #         row.append(batch[c][i])
        #     else:
        #         json_meta = json.loads(batch['metadata'][i])
        #         if c in json_meta:
        #             row.append(json_meta[c])
        #         else:
        #             row.append(None)
        output.append(row)
    return output

def compute_hash(url, text):
  if url is None:
    url = ''

  if text is None:
    text = ''
  
  total = (url + text).encode("utf-8")
  return mmh3.hash64(total)[0]

def additional_columns_to_include(args):
    columns = ["image_path"]
    return columns
    # c = set(args.output_columns)
    # columns += list(c)
    # return columns