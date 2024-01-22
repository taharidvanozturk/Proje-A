import os
import PIL
import numpy as np
import torch
import torch.nn.functional as f
import torchvision.models as models
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp



img_dir = 'images'
img_files = os.listdir(img_dir)

for img_name in img_files:
    img_path = os.path.join(img_dir, img_name)

    pil_img = PIL.Image.open(img_path)
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
    torch_img = f.interpolate(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)

    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    densenet = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
    squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)


    cam_dict = dict()

    alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_11', input_size=(224, 224))
    alexnet_gradcam = GradCAM(alexnet_model_dict, True)
    alexnet_gradcampp = GradCAMpp(alexnet_model_dict, True)
    cam_dict['AlexNet'] = [alexnet_gradcam, alexnet_gradcampp]

    vgg_model_dict = dict(type='vgg', arch=vgg, layer_name='features_29', input_size=(224, 224))
    vgg_gradcam = GradCAM(vgg_model_dict, True)
    vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
    cam_dict['VGG-16'] = [vgg_gradcam, vgg_gradcampp]

    resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
    resnet_gradcam = GradCAM(resnet_model_dict, True)
    resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    cam_dict['ResNet-101'] = [resnet_gradcam, resnet_gradcampp]

    densenet_model_dict = dict(type='densenet', arch=densenet, layer_name='features_norm5', input_size=(224, 224))
    densenet_gradcam = GradCAM(densenet_model_dict, True)
    densenet_gradcampp = GradCAMpp(densenet_model_dict, True)
    cam_dict['DenseNet-161'] = [densenet_gradcam, densenet_gradcampp]

    squeezenet_model_dict = dict(type='squeezenet', arch=squeezenet,
                                 layer_name='features_12_expand3x3_activation', input_size=(224, 224))
    squeezenet_gradcam = GradCAM(squeezenet_model_dict, True)
    squeezenet_gradcampp = GradCAMpp(squeezenet_model_dict, True)
    cam_dict['SqueezeNet'] = [squeezenet_gradcam, squeezenet_gradcampp]

    images = []
    for model_name, (gradcam, gradcam_pp) in cam_dict.items():
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))

    images = make_grid(torch.cat(images, 0), nrow=5)


    numpy_img = images.permute(1, 2, 0).cpu().numpy()


    plt.imshow(numpy_img)
    for i, (model_name, _) in enumerate(cam_dict.items()):
        plt.text(-250, i * 224+180, f"{model_name}", color="black")


    labels = ["Girdi", "GradCAM\nIsı\nHaritası", "GradCAM++\nIsı\nHaritası", "GradCam\nSonucu", "GradCam++\nSonucu"]
    for i, label in enumerate(labels):
        plt.text(i * 224, -40, f"{label}", color="black")

    plt.axis('off')


    output_path = os.path.join("outputs", img_name)
    plt.savefig(output_path)

    plt.clf()
