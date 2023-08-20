import torch
import torch.nn as nn
import torch.nn.functional as F

class L1_Charbonnier(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self, eps=1e-3):
        super(L1_Charbonnier, self).__init__()
        '''MSRNet uses 1e-3 as default'''
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.mean(error)
        return loss




class Edge_loss(nn.Module):
    def __init__(self):
        super(Edge_loss, self).__init__()
        self.loss = L1_Charbonnier()

    def rgb2gray(self, img):
        out_img = img.clone().detach()
        R, G, B = out_img[:, 0, :, :], out_img[:, 1, :, :], out_img[:, 2, :, :]
        out_img[:, 0, :, :] = 0.299 * R + 0.587 * G + 0.114 * B
        gray = out_img[:, 0:1, :, :]
        return gray

    def laplacian_op(self, img):
        '''Input a gray-scale image'''
        if img.shape[1] == 3:
            img = self.rgb2gray(img)
            # print("Color image")
        sobel_kernel = torch.tensor(
            [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]])
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        weight = torch.autograd.Variable(sobel_kernel)
        edge_detect = F.conv2d(img, weight)
        edge_detect = edge_detect.squeeze().detach()
        return edge_detect

    def forward(self, X, Y):
        edge_X, edge_Y = self.laplacian_op(X), self.laplacian_op(Y)
        return self.loss(edge_X, edge_Y)

# image1 = torch.rand(16, 3, 768, 768)
# image2 = torch.rand(16, 3, 768, 768)
#
# loss = Edge_loss()
# print(loss(image1, image2))


