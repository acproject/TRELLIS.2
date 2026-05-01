import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
os.environ['SDPA_BACKEND'] = 'cudnn'
import cv2
import imageio
from PIL import Image
import torch
import torch.multiprocessing as mp
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel


def process_on_gpu(gpu_id, image_paths, output_dir):
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.low_vram = False
    pipeline.to(f'cuda:{gpu_id}')

    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device=f'cuda:{gpu_id}'
    ))

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path)
        mesh = pipeline.run(image)[0]
        mesh.simplify(16777216)

        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        video_path = os.path.join(output_dir, f"{image_name}.mp4")
        imageio.mimsave(video_path, video, fps=15)

        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   1000000,
            texture_size        =   4096,
            remesh              =   True,
            remesh_band         =   1,
            remesh_project      =   0,
            verbose             =   True
        )
        glb_path = os.path.join(output_dir, f"{image_name}.glb")
        glb.export(glb_path, extension_webp=True)
        print(f"[GPU {gpu_id}] {image_name} done -> {glb_path}")


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    image_dir = "assets/example_image"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

    if num_gpus <= 1:
        process_on_gpu(0, image_paths, output_dir)
    else:
        chunks = [[] for _ in range(num_gpus)]
        for i, path in enumerate(image_paths):
            chunks[i % num_gpus].append(path)

        processes = []
        for gpu_id, chunk in enumerate(chunks):
            if not chunk:
                continue
            p = mp.Process(target=process_on_gpu, args=(gpu_id, chunk, output_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("All done!")