from io import BytesIO
import requests
import gradio as gr
import requests
import torch
from tqdm import tqdm
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
from torchvision.transforms import ToPILImage
from utils import preprocess, prepare_mask_and_masked_image, recover_image, resize_and_crop

gr.close_all()
topil = ToPILImage()

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe_inpaint = pipe_inpaint.to("cuda")

## Good params for editing that we used all over the paper --> decent quality and speed   
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 100

def pgd(X, targets, model, criterion, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  
        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean - targets).norm()
        pbar.set_description(f"Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])
        
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    
        
        if mask is not None:
            X_adv.data *= mask
            
    return X_adv

def get_target():
    target_url = 'https://www.rtings.com/images/test-materials/2015/204_Gray_Uniformity.png'
    response = requests.get(target_url)
    target_image = Image.open(BytesIO(response.content)).convert("RGB")
    target_image = target_image.resize((512, 512))
    return target_image

def immunize_fn(init_image, mask_image):
    with torch.autocast('cuda'):
        mask, X = prepare_mask_and_masked_image(init_image, mask_image)
        X = X.half().cuda()
        mask = mask.half().cuda()

        targets = pipe_inpaint.vae.encode(preprocess(get_target()).half().cuda()).latent_dist.mean

        adv_X = pgd(X, 
                    targets = targets,
                    model=pipe_inpaint.vae.encode, 
                    criterion=torch.nn.MSELoss(), 
                    clamp_min=-1, 
                    clamp_max=1,
                    eps=0.1, 
                    step_size=0.01, 
                    iters=200,
                    mask=1-mask
                   )

        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
        
        adv_image = topil(adv_X[0]).convert("RGB")
        adv_image = recover_image(adv_image, init_image, mask_image, background=True)
        return adv_image        

def run(image, prompt, seed, immunize=False):
    seed = int(seed)
    torch.manual_seed(seed)

    init_image = Image.fromarray(image['image'])
    init_image = resize_and_crop(init_image, (512,512))
    mask_image = ImageOps.invert(Image.fromarray(image['mask']).convert('RGB')).resize(init_image.size)
    mask_image = resize_and_crop(mask_image, init_image.size)
    
    if immunize:
        immunized_image = immunize_fn(init_image, mask_image)
        
    image_edited = pipe_inpaint(prompt=prompt, 
                         image=init_image if not immunize else immunized_image, 
                         mask_image=mask_image, 
                         height = init_image.size[0],
                         width = init_image.size[1],
                         eta=1,
                         guidance_scale=GUIDANCE_SCALE,
                         num_inference_steps=NUM_INFERENCE_STEPS,
                        ).images[0]
        
    image_edited = recover_image(image_edited, init_image, mask_image)
    
    if immunize:
        return [(immunized_image, 'Immunized Image'), (image_edited, 'Edited After Immunization')]
    else:
        return [(image_edited, 'Edited Image')]


demo = gr.Interface(fn=run, 
                    inputs=[
                        gr.ImageMask(label='Input Image'),
                        gr.Textbox(label='Prompt', placeholder='A photo of a man in a wedding'),
                        gr.Textbox(label='Seed', placeholder='1234'),
                        gr.Checkbox(label='Immunize', value=False),
                    ], 
                    outputs=[gr.Gallery(
                            label="Generated images", 
                            show_label=False, 
                            elem_id="gallery").style(grid=[1,2], height="auto")],
                    examples=[
                    ['./images/hadi_and_trevor.jpg', 'man attending a wedding', '329357'],
                    ['./images/trevor_2.jpg', 'two men in prison', '329357'],
                    ['./images/trevor_3.jpg', 'man in a private jet', '329357'],
                    ['./images/elon_2.jpg', 'man in a metro station', '214213'],
                    ],
                    examples_per_page=20,
                    allow_flagging='never',
                    title="Immunize your photos against manipulation by Stable Diffusion",
                    description='''<u>Official</u> demo of our paper: <br>
                    **Raising the Cost of Malicious AI-Powered Image Editing** <br>
                    *Hadi Salman\*, Alaa Khaddaj\*, Guillaume Leclerc\*, Andrew Ilyas, Aleksander Madry* <br>
                    [Paper](https://arxiv.org/abs/2302.06588) 
                    &nbsp;&nbsp;[Blog post](https://gradientscience.org/photoguard/) 
                    &nbsp;&nbsp;[![](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/MadryLab/photoguard)
                    <br />
                    We present an approach to mitigating the risks of malicious image editing posed by large diffusion models. The key idea is to immunize images so as to make them resistant to manipulation by these models. This immunization relies on injection of imperceptible adversarial perturbations designed to disrupt the operation of the targeted diffusion models, forcing them to generate unrealistic images.
                    <br />
                    <br />
                    **Demo steps:**
 + Upload an image (or select from the below examples!)
 + Mask the parts of the image you want to maintain unedited (e.g., faces of people)
 + Add a prompt to edit the image accordingly (see examples below)
 + Play with the seed and click submit until you get a realistic edit that you are happy with (we have good seeds for you below)
 
 Now let's immunize your image and try again! 
 + Click on the "immunize" button, then submit.
 + You will get the immunized image (which looks identical to the original one) and the edited image, which is now hopefully unrealistic!                   
                    <br />
                    **This is a research project and is not production-ready.**
                    ''',
                   )

demo.launch()
# demo.launch(server_name='0.0.0.0', share=False, server_port=7860, inline=False, )