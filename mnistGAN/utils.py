import matplotlib.pyplot as plt

def imshow(img):
    """
        >>> rand_noise = torch.randn(1, 784).cuda()
        >>> rand_img = g(rand_noise)

        >>> imshow(rand_img)
    """
    
    figure = plt.figure(figsize=(1.5, 1.5))
    plt.imshow(img.reshape(28, 28).detach().cpu().numpy(), cmap="gray", figure=figure)
    plt.show()
    
    
def grid_imshow(imgs):
    """
        >>> rand_noise_grid = torch.randn(10, 784).cuda()
        >>> rand_imgs = g(rand_noise_grid)

        >>> grid_imshow(rand_imgs)
    """
    figure = plt.figure(figsize=(1.5 * imgs.shape[0], 1.5))
    
    reshaped_imgs = imgs.reshape(imgs.shape[0], 28, 28)
    
    for n in range(1, imgs.shape[0] + 1):
        figure.add_subplot(1, imgs.shape[0], n)
        plt.axis("off")
        plt.imshow(reshaped_imgs[n - 1].cpu().detach().numpy(), cmap="gray")
