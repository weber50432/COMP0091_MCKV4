import huggingface_hub
import os
from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

local_dir = os.path.join(os.path.expanduser("~"), "UCL/final_dissertation/prov-gigapath/data/aa3f0051-ea7f-47f2-a7aa-315c1bdc3f18")
# huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=True)
slide_path = os.path.join(local_dir, "TCGA-BH-A0C1-01Z-00-DX1.21FE357E-B182-4397-BFEF-7E96E994236A.svs")

print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides")
target_mpp = 0.5
level = find_level_for_target_mpp(slide_path, target_mpp)
if level is not None:
    print(f"Found level: {level}")
else:
    print("No suitable level found.")
