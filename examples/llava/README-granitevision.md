# Granite Vision

Download the model and point your `GRANITE_MODEL` environment variable to the path.

```bash
$ git clone https://huggingface.co/ibm-granite/granite-vision-3.1-2b-preview
$ export GRANITE_MODEL=./granite-vision-3.1-2b-preview
```


### 1. Running llava surgery v2.
First, we need to run the llava surgery script as shown below:

`python llava_surgery_v2.py -C -m $GRANITE_MODEL`

You should see two new files (`llava.clip` and `llava.projector`) written into your model's directory, as shown below.

```bash
$ ls $GRANITE_MODEL | grep -i llava
llava.clip
llava.projector
```

We should see that the projector and visual encoder get split out into the llava files. Quick check to make sure they aren't empty:
```python
import os
import torch

MODEL_PATH = os.getenv("GRANITE_MODEL")
if not MODEL_PATH:
    raise ValueError("env var GRANITE_MODEL is unset!")

encoder_tensors = torch.load(os.path.join(MODEL_PATH, "llava.clip"))
projector_tensors = torch.load(os.path.join(MODEL_PATH, "llava.projector"))

assert len(encoder_tensors) > 0
assert len(projector_tensors) > 0
```

If you actually inspect the `.keys()` of the loaded tensors, you should see a lot of `vision_model` tensors in the `encoder_tensors`, and 5 tensors (`'multi_modal_projector.linear_1.bias'`, `'multi_modal_projector.linear_1.weight'`, `'multi_modal_projector.linear_2.bias'`, `'multi_modal_projector.linear_2.weight'`, `'image_newline'`) in the multimodal `projector_tensors`.


### 2. Creating the Visual Component GGUF
To create the GGUF for the visual components, we need to write a config for the visual encoder; make sure the config contains the correct `image_grid_pinpoints`


Note: we refer to this file as `$VISION_CONFIG` later on.
```json
{
    "_name_or_path": "siglip-model",
    "architectures": [
      "SiglipVisionModel"
    ],
    "image_grid_pinpoints": [
        [384,768],
        [384,1152],
        [384,1536],
        [384,1920],
        [384,2304],
        [384,2688],
        [384,3072],
        [384,3456],
        [384,3840],
        [768,384],
        [768,768],
        [768,1152],
        [768,1536],
        [768,1920],
        [1152,384],
        [1152,768],
        [1152,1152],
        [1536,384],
        [1536,768],
        [1920,384],
        [1920,768],
        [2304,384],
        [2688,384],
        [3072,384],
        [3456,384],
        [3840,384]
    ],
    "mm_patch_merge_type": "spatial_unpad",
    "hidden_size": 1152,
    "image_size": 384,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "patch_size": 14,
    "layer_norm_eps": 1e-6,
    "hidden_act": "gelu_pytorch_tanh",
    "projection_dim": 0,
    "vision_feature_layer": [-24, -20, -12, -1]
}
```

Create a new directory to hold the visual components, and copy the llava.clip/projector files, as well as the vision config into it.

```bash
$ ENCODER_PATH=$PWD/visual_encoder
$ mkdir $ENCODER_PATH

$ cp $GRANITE_MODEL/llava.clip $ENCODER_PATH/pytorch_model.bin
$ cp $GRANITE_MODEL/llava.projector $ENCODER_PATH/
$ cp $VISION_CONFIG $ENCODER_PATH/config.json
```

At which point you should have something like this:
```bash
$ ls $ENCODER_PATH
config.json             llava.projector         pytorch_model.bin
```

Now convert the components to GGUF; Note that we also override the image mean/std dev to `[.5,.5,.5]` since we use the siglip visual encoder - in the transformers model, you can find these numbers in the [preprocessor_config.json](https://huggingface.co/ibm-granite/granite-vision-3.1-2b-preview/blob/main/preprocessor_config.json).
```bash
$ python convert_image_encoder_to_gguf.py \
    -m $ENCODER_PATH \
    --llava-projector $ENCODER_PATH/llava.projector \
    --output-dir $ENCODER_PATH \
    --clip-model-is-vision \
    --clip-model-is-siglip \
    --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5
```

this will create the first GGUF file at `$ENCODER_PATH/mmproj-model-f16.gguf`; we will refer to the abs path of this file as the `$VISUAL_GGUF_PATH.`


### 3. Creating the LLM GGUF.
The granite vision model contains a granite LLM as its language model. For now, the easiest way to get the GGUF for LLM is by loading the composite model in `transformers` and exporting the LLM so that it can be directly converted with the normal conversion path.

First, set the `LLM_EXPORT_PATH` to the path to export the `transformers` LLM to.
```
$ export LLM_EXPORT_PATH=$PWD/granite_vision_llm
```

```python
import os
import transformers

MODEL_PATH = os.getenv("GRANITE_MODEL")
if not MODEL_PATH:
    raise ValueError("env var GRANITE_MODEL is unset!")

LLM_EXPORT_PATH = os.getenv("LLM_EXPORT_PATH")
if not MODEL_PATH:
    raise ValueError("env var LLM_EXPORT_PATH is unset!")

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

# NOTE: granite vision support was added to transformers very recently (4.49);
# if you get size mismatches, your version is too old.
# If you are running with an older version, set `ignore_mismatched_sizes=True`
# as shown below; it won't be loaded correctly, but the LLM part of the model that
# we are exporting will be loaded correctly.
model = transformers.AutoModelForImageTextToText.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True)

tokenizer.save_pretrained(LLM_EXPORT_PATH)
model.language_model.save_pretrained(LLM_EXPORT_PATH)
```

Now you can convert the exported LLM to GGUF with the normal converter in the root of the llama cpp project.
```bash
$ LLM_GGUF_PATH=$LLM_EXPORT_PATH/granite_llm.gguf
...
$ python convert_hf_to_gguf.py --outfile $LLM_GGUF_PATH $LLM_EXPORT_PATH
```


### 4. Running the Model in Llama cpp
Build llama cpp normally; you should have a target binary named `llama-llava-cli`, which you can pass two binaries to. Sample usage:

Note - the test image shown below can be found [here](https://github-production-user-asset-6210df.s3.amazonaws.com/10740300/415512792-d90d5562-8844-4f34-a0a5-77f62d5a58b5.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250221%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250221T054145Z&X-Amz-Expires=300&X-Amz-Signature=86c60be490aa49ef7d53f25d6c973580a8273904fed11ed2453d0a38240ee40a&X-Amz-SignedHeaders=host).

```bash
$ ./build/bin/llama-llava-cli -m $LLM_GGUF_PATH \
    --mmproj $VISUAL_GGUF_PATH \
    --image cherry_blossom.jpg \
    -c 16384 \
    -p "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|user|>\n\<image>\nWhat type of flowers are in this picture?\n<|assistant|>\n" \
    --temp 0
```

Sample response: `The flowers in the picture are cherry blossoms, which are known for their delicate pink petals and are often associated with the beauty of spring.`
