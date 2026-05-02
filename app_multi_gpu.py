import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['SDPA_BACKEND'] = 'cudnn'

import cv2
import numpy as np
import base64
import io
import shutil
from datetime import datetime
from typing import Tuple
import torch
from PIL import Image
import gradio as gr
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8
DEFAULT_MODE = 3
DEFAULT_STEP = 3


css = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')).read()
css_start = css.find('css = """')
css_end = css.find('"""', css_start + 9)
css_content = css[css_start + 9:css_end]

head_start = css.find('head = """')
head_end = css.find('"""', head_start + 10)
head_content = css[head_start + 10:head_end]

empty_html_start = css.find('empty_html = f"""')
empty_html_end = css.find('"""', empty_html_start + 17)
empty_html_content = css[empty_html_start + 17:empty_html_end]


num_gpus = torch.cuda.device_count()
devices = [f'cuda:{i}' for i in range(num_gpus)]
primary_device = devices[0]

print(f"Available GPUs: {num_gpus}")
print(f"Loading pipeline and distributing models across {devices}...")

pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
pipeline.to_multi_gpu(devices)

print("Model distribution:")
for name, device in pipeline._model_devices.items():
    print(f"  {name}: {device}")

envmap = {
    'forest': EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device=primary_device
    )),
    'sunset': EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/sunset.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device=primary_device
    )),
    'courtyard': EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/courtyard.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device=primary_device
    )),
}

for i in range(len(MODES)):
    icon = Image.open(MODES[i]['icon'])
    buffered = io.BytesIO()
    icon.convert("RGB").save(buffered, format="jpeg", quality=85)
    MODES[i]['icon_base64'] = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def image_to_base64(image):
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)


def preprocess_image(image: Image.Image) -> Image.Image:
    return pipeline.preprocess_image(image)


def pack_state(latents):
    shape_slat, tex_slat, res = latents
    return {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }


def unpack_state(state):
    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).to(primary_device),
        coords=torch.from_numpy(state['coords']).to(primary_device),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).to(primary_device))
    return shape_slat, tex_slat, state['res']


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(image, seed, resolution, ss_guidance_strength, ss_guidance_rescale,
                ss_sampling_steps, ss_rescale_t, shape_slat_guidance_strength,
                shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
                tex_slat_guidance_strength, tex_slat_guidance_rescale,
                tex_slat_sampling_steps, tex_slat_rescale_t, req: gr.Request,
                progress=gr.Progress(track_tqdm=True)):
    max_retries = 3
    for attempt in range(max_retries):
        current_seed = seed + attempt if attempt > 0 else seed
        outputs, latents = pipeline.run(
            image, seed=current_seed, preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
            },
            pipeline_type={"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[resolution],
            return_latent=True,
        )
        if len(outputs) > 0:
            break
    if len(outputs) == 0:
        raise gr.Error("No mesh was generated after multiple attempts. Please try a different seed or image.")
    mesh = outputs[0]
    mesh.simplify(16777216)
    mesh_device = mesh.vertices.device if hasattr(mesh.vertices, 'device') else primary_device
    local_envmap = {k: EnvMap(v.image.to(mesh_device)) for k, v in envmap.items()}
    images = render_utils.render_snapshot(mesh, resolution=1024, r=2, fov=36, nviews=STEPS, envmap=local_envmap, device=mesh_device)
    state = pack_state(latents)
    torch.cuda.empty_cache()

    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            unique_id = f"view-m{m_idx}-s{s_idx}"
            is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_visible else ""
            img_base64 = image_to_base64(Image.fromarray(images[mode['render_key']][s_idx]))
            images_html += f'<img id="{unique_id}" class="previewer-main-image {vis_class}" src="{img_base64}" loading="eager">'

    btns_html = ""
    for idx, mode in enumerate(MODES):
        active_class = "active" if idx == DEFAULT_MODE else ""
        btns_html += f'<img src="{mode["icon_base64"]}" class="mode-btn {active_class}" onclick="selectMode({idx})" title="{mode["name"]}">'

    full_html = f"""
    <div class="previewer-container">
        <div class="tips-wrapper">
            <div class="tips-icon">💡Tips</div>
            <div class="tips-text">
                <p>● <b>Render Mode</b> - Click on the circular buttons to switch between different render modes.</p>
                <p>● <b>View Angle</b> - Drag the slider to change the view angle.</p>
            </div>
        </div>
        <div class="display-row">{images_html}</div>
        <div class="mode-row" id="btn-group">{btns_html}</div>
        <div class="slider-row">
            <input type="range" id="custom-slider" min="0" max="{STEPS - 1}" value="{DEFAULT_STEP}" step="1" oninput="onSliderChange(this.value)">
        </div>
    </div>
    """
    return state, full_html


def extract_glb(state, decimation_target, texture_size, req: gr.Request, progress=gr.Progress(track_tqdm=True)):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shape_slat, tex_slat, res = unpack_state(state)
    meshes = pipeline.decode_latent(shape_slat, tex_slat, res)
    if len(meshes) == 0:
        raise gr.Error("No mesh was generated. The model produced an empty result.")
    mesh = meshes[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout, grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=decimation_target,
        texture_size=texture_size, remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    glb.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    return glb_path, glb_path


gpu_info = ", ".join([f"GPU {i}" for i in range(num_gpus)])

with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown(f"""
    ## Image to 3D Asset with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2) (Multi-GPU: {gpu_info})
    * Upload an image (preferably with an alpha-masked foreground object) and click Generate to create a 3D asset.
    * Click Extract GLB to export and download the generated GLB file if you're satisfied with the result. Otherwise, try another time.
    """)
    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)
            resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            decimation_target = gr.Slider(100000, 1000000, label="Decimation Target", value=500000, step=10000)
            texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)
            generate_btn = gr.Button("Generate")
            with gr.Accordion(label="Advanced Settings", open=False):
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                gr.Markdown("Stage 2: Shape Generation")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                gr.Markdown("Stage 3: Material Generation")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
        with gr.Column(scale=10):
            with gr.Walkthrough(selected=0) as walkthrough:
                with gr.Step("Preview", id=0):
                    preview_output = gr.HTML(empty_html_content, label="3D Asset Preview", show_label=True, container=True)
                    extract_btn = gr.Button("Extract GLB")
                with gr.Step("Extract", id=1):
                    glb_output = gr.Model3D(label="Extracted GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                    download_btn = gr.DownloadButton(label="Download GLB")
        with gr.Column(scale=1, min_width=172):
            examples = gr.Examples(
                examples=[f'assets/example_image/{img}' for img in os.listdir("assets/example_image")],
                inputs=[image_prompt], fn=preprocess_image, outputs=[image_prompt],
                run_on_click=True, examples_per_page=18,
            )
    output_buf = gr.State()

    demo.load(start_session)
    demo.unload(end_session)
    image_prompt.upload(preprocess_image, inputs=[image_prompt], outputs=[image_prompt])
    generate_btn.click(get_seed, inputs=[randomize_seed, seed], outputs=[seed]).then(
        lambda: gr.Walkthrough(selected=0), outputs=walkthrough
    ).then(
        image_to_3d,
        inputs=[image_prompt, seed, resolution,
                ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
                shape_slat_guidance_strength, shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
                tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t],
        outputs=[output_buf, preview_output],
    )
    extract_btn.click(
        lambda: gr.Walkthrough(selected=1), outputs=walkthrough
    ).then(
        extract_glb, inputs=[output_buf, decimation_target, texture_size],
        outputs=[glb_output, download_btn],
    )

if __name__ == '__main__':
    os.makedirs(TMP_DIR, exist_ok=True)
    demo.launch(css=css_content, head=head_content)
