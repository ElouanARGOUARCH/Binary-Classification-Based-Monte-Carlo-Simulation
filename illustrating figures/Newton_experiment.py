import torch
from matplotlib import image
import matplotlib
import numpy
import matplotlib.pyplot as plt
from models_bcbmcs import *

import torch

class logit():
    def __init__(self, alpha = 1e-2):
        self.alpha = alpha

    def transform(self,x, alpha = None):
        assert torch.all(x<=1) and torch.all(x>=0), 'can only transform value between 0 and 1'
        if alpha is None:
            alpha = self.alpha
        return torch.logit(alpha*torch.ones_like(x) + x*(1-2*alpha))

    def inverse_transform(self, x, alpha = None):
        if alpha is None:
            alpha = self.alpha
        return (torch.sigmoid(x)-alpha*torch.ones_like(x))/(1-2*alpha)

    def log_det(self,x, alpha = None ):
        if alpha is None:
            alpha = self.alpha
        return torch.sum(torch.log((1-2*alpha)*(torch.reciprocal(alpha*torch.ones_like(x) + x*(1-2*alpha)) + torch.reciprocal((1-alpha)*torch.ones_like(x) - x*(1-2*alpha)))), dim = -1)

def plot_2d_function(f,range = [[-10,10],[-10,10]], bins = [50,50], alpha = 0.7,show = True):
    with torch.no_grad():
        tt_x = torch.linspace(range[0][0], range[0][1], bins[0])
        tt_y = torch.linspace(range[1][0],range[1][1], bins[1])
        mesh = torch.cartesian_prod(tt_x, tt_y)
        with torch.no_grad():
            plt.pcolormesh(tt_x,tt_y,f(mesh).numpy().reshape(bins[0],bins[1]).T, cmap = matplotlib.cm.get_cmap('viridis'), alpha = alpha, lw = 0)
    if show:
        plt.show()

def plot_image_2d_points(samples, bins=(200, 200), range=None, alpha = 1.,show = True):
    assert samples.shape[-1] == 2, 'Requires 2-dimensional points'
    hist, x_edges, y_edges = numpy.histogram2d(samples[:, 0].numpy(), samples[:, 1].numpy(), bins,range)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.pcolormesh(x_edges, y_edges, hist.T, cmap=matplotlib.cm.get_cmap('viridis'),alpha=alpha, lw=0)
    if show:
        plt.show()

rgb = image.imread("newton.jpg")
lines, columns = rgb.shape[:-1]

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
grey = torch.tensor(rgb2gray(rgb))


fig = plt.figure(figsize =(8,12))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.imshow(grey, aspect = 'auto')

#Sample data according to image
vector_density = grey.flatten()
vector_density = vector_density/torch.sum(vector_density)
lines, columns = grey.shape
num_samples = 500000
cat = torch.distributions.Categorical(probs = vector_density)
categorical_samples = cat.sample([num_samples])
target_samples = torch.cat([((categorical_samples%columns + torch.rand(num_samples))/columns).unsqueeze(-1),((1-(categorical_samples//columns + torch.rand(num_samples))/lines)).unsqueeze(-1)], dim = -1)

#Apply logit transform to data
logit_transform = logit(alpha = 1e-2)
transformed_samples = logit_transform.transform(target_samples)

#Estimate instrumental distribution and sample to constitute dataset
mean = torch.mean(transformed_samples, dim = 0)
cov = torch.cov(transformed_samples.T)
instrumental = torch.distributions.MultivariateNormal(mean, (cov + cov.T)/2)
proposed_samples = instrumental.sample([num_samples])

#train binary classifier to distinguish samples
binary_classif = BinaryClassifier(proposed_samples, transformed_samples,[512,512,512])
binary_classif.train(200,10000,lr = 1e-3, weight_decay = 0, verbose = True)
binary_classif.train(200,10000,lr = 5e-4, weight_decay = 5e-6, verbose = True)
binary_classif.train(200,10000,lr = 1e-4, weight_decay = 5e-6, verbose = True)
torch.save(binary_classif, 'model_newton.sav')

#sample the energy based model with Sampling Importance Resampling
proposed_samples = instrumental.sample([num_samples])
unormalized_weights = torch.exp(binary_classif.logit_r(proposed_samples)).squeeze(-1)
normalized_weights = unormalized_weights/torch.sum(unormalized_weights)
cat = torch.distributions.Categorical(normalized_weights).sample([num_samples])
samples = proposed_samples[cat]

#display original image
fig = plt.figure(figsize =(8,12))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.imshow(grey, aspect = 'auto')
plt.show()

#display unormalized energy
fig = plt.figure(figsize =(8,12))
plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
plot_2d_function(lambda x: torch.exp(binary_classif.logit_r(logit_transform.transform(x)).squeeze(-1) + instrumental.log_prob(logit_transform.transform(x)) + logit_transform.log_det(x)), bins = (lines, columns), range =([[0.,1.],[0.,1.]]))
plt.show()

#display obtained samples
inverse_samples = logit_transform.inverse_transform(samples)
fig = plt.figure(figsize =(8,12))
plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
plot_image_2d_points(inverse_samples)
plt.show()