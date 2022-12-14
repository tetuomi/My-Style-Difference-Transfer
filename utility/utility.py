import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn

from torchvision import transforms

import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage
import cv2

import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.log = open(logfile_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

# post processing for images
postpa = transforms.Compose([
                            # transforms.Lambda(lambda x: x.mul_(1./255)),
                            # transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                            #                     std=[1,1,1]),
                            # # transforms.Normalize(mean=[0,0,0], #subtract imagenet mean
                            # #                         std=[1/0.5,1/0.5,1/0.5]),
                            # # transforms.Normalize(mean=[-0.5,-0.5,-0.5], #subtract imagenet mean
                            # #                         std=[1,1,1]),
                            # transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                            ])
postpb = transforms.Compose([
                            transforms.ToPILImage(),
                            ])

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        Fe = input.view(b, c, h*w)
        G = torch.bmm(Fe, Fe.transpose(1,2))
        return G.div(b*c*h*w)

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

def postp(tensor, image_size, invert): # to clip results in the range [0,1]
    # t = postpa(tensor)
    t = tensor.clamp(0, 1)

    img = postpb(t)
    if invert:
        img = PIL.ImageOps.invert(img)
    # img = transforms.functional.resize(img,[image_size, image_size])
    return img

def custom_postp(tensor, image_size, output_path):
    # Post processing
    t = transforms.Lambda(lambda x: x.mul_(1./255))(tensor)
    ## Save histogram
    plt.plot(torch.histc(t[0],bins=255,min=t[0].min(),max=t.max()).numpy(), color='blue')
    plt.plot(torch.histc(t[1],bins=255,min=t[1].min(),max=t.max()).numpy(), color='green')
    plt.plot(torch.histc(t[2],bins=255,min=t[2].min(),max=t.max()).numpy(), color='red')
    plt.savefig(output_path + "_before.jpg")
    ## Save before image
    bef_img = transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])(t)
    bef_img = transforms.ToPILImage()(bef_img)
    bef_img = transforms.Resize([image_size,image_size])(bef_img)
    bef_img.save(output_path + "_before.bmp")
    ## Unnormalize
    t = transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1])(t)
    ## Map to [0,1]
    # a = 0
    # b = 1
    # t = (t - t.min())*(b-a)/(t.max()-t.min()) + a
    ## Return to RGB
    t = transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])(t)
    ## Cut invalid values
    t[t>1] = 1
    t[t<0] = 0
    ## Save histogram
    plt.plot(torch.histc(t[2],bins=255,min=0,max=1).numpy(), color='blue')
    plt.plot(torch.histc(t[1],bins=255,min=0,max=1).numpy(), color='green')
    plt.plot(torch.histc(t[0],bins=255,min=0,max=1).numpy(), color='red')
    plt.savefig(output_path + "_after.jpg")
    ## Transform to PIL image
    img = transforms.ToPILImage()(t)
    ## Resize
    img = transforms.Resize([image_size,image_size])(img)
    ## Return
    return img

def preprocessing(img,img_size=64,margin=0,threshold_w=0,threshold_h=0):
    img_white = cv2.bitwise_not(img)#np.where(img_org>0,0,255) #文字領域白
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0
    # 行の処理
    for i in range(img_white.shape[0]):
        if np.sum(img_white[i,:]) > 0:
            y_min = i
            break
    for i in reversed(range(img_white.shape[0])):
        if np.sum(img_white[i,:]) > 0:
            y_max = i+1 #rangeが0~n-1なので，arrayのインデックス調整で+1する　例：img[0:N] これも0~N-1になっているので+1しておかないとバグる
            break
    # 列の処理
    for i in range(img_white.shape[1]):
        if np.sum(img_white[:,i]) > 0:
            x_min = i
            break
    for i in reversed(range(img_white.shape[1])):
        if np.sum(img_white[:,i]) > 0:
            x_max = i+1
            break
    img = img_white[y_min:y_max,x_min:x_max]
    h = img.shape[0]
    w = img.shape[1]
    if (h<threshold_h) or (w<threshold_w):
        # print('error')
        return 0
    if margin>0:
        img = np.pad(img,[(margin,margin),(margin,margin)],'constant')
    size = max(w,h)
    ratio = img_size/size #何倍すれば良いか
    img_resize = cv2.resize(img, (int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_CUBIC)
    # img_resize = cv2.bitwise_not(img_resize) #文字領域黒
    #0埋めの幅を決める
    if w > h:
        pad = int((img_size - h*ratio)/2)
        #np.pad()の第二引数[(上，下),(左，右)]にpaddingする行・列数
        img_resize = np.pad(img_resize,[(pad,pad),(0,0)],'constant')
    elif h > w:
        pad = int((img_size - w*ratio)/2)
        img_resize = np.pad(img_resize,[(0,0),(pad,pad)],'constant')
    #最終的にきれいに100x100にresize
    img_resize = cv2.resize(img_resize,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
    img_resize = cv2.bitwise_not(img_resize)#np.where(img_resize!=0,0,255)

    return img_resize

# Function to load images
def load_images(img_dir, device, crop=True):
    # prep = transforms.Compose([transforms.Resize((img_size,img_size)),
    #                         # transforms.RandomRotation(angle),
    #                         transforms.ToTensor(),
    #                         transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
    #                         transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
    #                                                 std=[1,1,1]),
    #                     #    transforms.Normalize(mean=[0.5, 0.5, 0.5], #add imagenet mean
    #                     #                         std=[0.5,0.5,0.5]),
    #                         transforms.Lambda(lambda x: x.mul_(255)),
    #                         ])
    # # Load & invert image
    # image = Image.open(img_dir)
    # image = image.convert('RGB')
    # if invert:
    #     image = PIL.ImageOps.invert(image)
    # # Make torch variable
    # img_torch = prep(image)
    # img_torch = Variable(img_torch.unsqueeze(0).to(device))

    img = cv2.imread(img_dir, 0)

    if crop == True:
        margin = 10
        img = preprocessing(img, margin=margin)
    else:
        img = cv2.resize(img,(64, 64),interpolation=cv2.INTER_CUBIC)
    img = img / 255.0
    img_torch = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)

    return img_torch

def load_mono_images(img_dir, img_size, device, invert):
    prep = transforms.Compose([transforms.Resize((img_size,img_size)),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.5, ), (0.5, )),
                                # transforms.Lambda(lambda x: x.div(255)),
                                ])
    # Load & invert image
    image = PIL.Image.open(img_dir)
    image = image.convert('L')
    if invert:
        image = PIL.ImageOps.invert(image)
    # img_np = np.array(image) / 255
    # Make torch variable
    img_torch = prep(image).to(device)
    # img_torch = img_torch.unsqueeze(0).to(device)

    return img_torch

# Function to save images
def save_images(content_image, opt_img, style_image1, style_image2, image_size, output_path, n_iter, content_invert, style_invert, result_invert):

    fnt = ImageFont.truetype('/usr/share/fonts/ubuntu/UbuntuMono-R.ttf', 13)

    # Save style image 1
    style_image1 = postp(style_image1, image_size, style_invert)
    d = ImageDraw.Draw(style_image1)
    # d.text((0,0), "Style1", font=fnt, fill=(0,0,0))
    # d.text((0,0), "Style1", font=fnt)
    style_image1.save(output_path + 'style1.jpg')

    # Save style image 2
    style_image2 = postp(style_image2, image_size, style_invert)
    d = ImageDraw.Draw(style_image2)
    # d.text((0,0), "Style2", font=fnt, fill=(0,0,0))
    # d.text((0,0), "Style2", font=fnt)
    style_image2.save(output_path + 'style2.jpg')

    # Save content image
    content_image = postp(content_image, image_size, content_invert)
    d = ImageDraw.Draw(content_image)
    # d.text((0,0), "Content", font=fnt, fill=(0,0,0))
    # d.text((0,0), "Content", font=fnt)
    content_image.save(output_path + 'content.jpg')

    # Save optimized images
    out_img = postp(opt_img, image_size, result_invert)
    d = ImageDraw.Draw(out_img)
    # d.text((0,0), "Generated", font=fnt, fill=(0,0,0))
    # d.text((0,0), "Generated", font=fnt)
    out_img.save(output_path + '/{}.jpg'.format(n_iter))

    images = [style_image1, style_image2, content_image, out_img]
    # widths, heights = zip(*(i.size for i in images))
    widths = [i.size[0] for i in images]
    heights = [i.size[1] for i in images]
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(output_path + '/all.bmp')


def make_folders(output_path):
    try:
        os.mkdir(output_path)
    except:
        pass
    try:
        os.mkdir(output_path+'/outputs')
    except:
        pass

"""
Input tensor
Outputs distance transform
"""
def dist_cv2(input_tensor, device, image_size, content_invert):
    out_img = postp(input_tensor.data[0].cpu().squeeze(), image_size, content_invert)
    # out_img = PIL.ImageOps.invert(out_img)
    # out_img = PIL.ImageOps.grayscale(out_img)
    out_img = out_img.convert('L')

    img = np.asarray(out_img)

    img = ndimage.grey_erosion(img, size=(3,3))

    img_dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    # plt.imshow(img_dist, cmap="Blues")
    # plt.colorbar()
    # plt.savefig("dist_img.png")
    # cv2.imwrite("dist_img.bmp", img_dist)
    cont_dist = torch.from_numpy(img_dist).float().to(device)
    f = cont_dist.unsqueeze(0)
    a = torch.cat((f,f,f),0)
    a = a.unsqueeze(0)
    return a