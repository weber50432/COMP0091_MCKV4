from gigapath.pipeline import tile_one_slide
import huggingface_hub
import os

# Set your Hugging Face token as environment variable:
# export HF_TOKEN="your_huggingface_token_here"
assert "HF_TOKEN" in os.environ, "Please set HF_TOKEN environment variable with your Hugging Face token"

local_dir = os.path.join("../data/tiles")
# huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=True)
slide_path = os.path.join("/home/weber50432/UCL/final_dissertation/TCGA-WSI/BLCA/TCGA-4Z-AA7O-01Z-00-DX1.A45C32E8-AA40-4182-8A6F-986DFE56A748.svs")

save_dir = os.path.join(local_dir, '')

print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. Please make sure to use the appropriate level for the 0.5 MPP")
tile_one_slide(slide_path, save_dir=save_dir, level=1)

print("NOTE: tiling dependency libraries can be tricky to set up. Please double check the generated tile images.")
