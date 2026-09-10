from diffusers import StableDiffusion3Pipeline
p = StableDiffusion3Pipeline.from_pretrained("/home/dell/models/sd3.5-large",
    torch_dtype="float16", local_files_only=True)
print("blocks", len(p.transformer.transformer_blocks))
print("patch", p.transformer.config.patch_size)
print("sample_size", p.transformer.config.sample_size)
print("in_channels", p.transformer.config.in_channels)
