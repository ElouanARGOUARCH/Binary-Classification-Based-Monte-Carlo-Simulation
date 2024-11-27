import torch
import matplotlib.pyplot as plt
import math
import sklearn
import matplotlib
from sklearn import datasets
from models import *

def plot_2d_function(f,range = [[-10,10],[-10,10]], bins = [50,50], alpha = 0.7,show = True):
    with torch.no_grad():
        tt_x = torch.linspace(range[0][0], range[0][1], bins[0])
        tt_y = torch.linspace(range[1][0],range[1][1], bins[1])
        mesh = torch.cartesian_prod(tt_x, tt_y)
        with torch.no_grad():
            plt.pcolormesh(tt_x,tt_y,f(mesh).numpy().reshape(bins[0],bins[1]).T, cmap = matplotlib.cm.get_cmap('viridis'), alpha = alpha, lw = 0)
    if show:
        plt.show()

class Orbits():
    def __init__(self, number_planets=7, means_target=None, covs_target=None, weights_target=None):
        self.number_planets = number_planets
        if means_target is None:
            self.means_target = 2.5 * torch.view_as_real(
                torch.pow(torch.exp(torch.tensor([2j * 3.1415 / number_planets])), torch.arange(0, number_planets)))
        else:
            assert means_target.shape != [self.number_planets, 2], "wrong size of means"
            self.means_target = means_target

        if covs_target is None:
            self.covs_target = 0.04 * torch.eye(2).unsqueeze(0).repeat(number_planets, 1, 1)
        else:
            assert covs_target.shape != [self.number_planets, 2, 2], 'wrong size of covs'
            self.covs_target = covs_target

        if weights_target is None:
            self.weights_target = torch.ones(self.number_planets)
        else:
            assert weights_target.shape != [self.number_planets], 'wrong size of weights'
            self.weights_target = weights_target

    def sample(self, num_samples, joint=False):
        mvn_target = torch.distributions.MultivariateNormal(self.means_target, self.covs_target)
        all_x = mvn_target.sample(num_samples)
        cat = torch.distributions.Categorical(self.weights_target.log().softmax(dim=0))
        pick = cat.sample(num_samples)
        if joint:
            return all_x[range(all_x.shape[0]), pick, :], pick
        else:
            return all_x[range(all_x.shape[0]), pick, :]

    def log_prob(self, x):
        mvn_target = torch.distributions.MultivariateNormal(self.means_target.to(x.device),
                                                            self.covs_target.to(x.device))
        cat = torch.distributions.Categorical(self.weights_target.softmax(dim=0).to(x.device))
        return torch.distributions.MixtureSameFamily(cat, mvn_target).log_prob(x)

class TwoCircles():
    def __init__(self):
        super().__init__()
        self.means = torch.tensor([1., 2.])
        self.weights = torch.tensor([.5, .5])
        self.noise = torch.tensor([0.125])

    def sample(self, num_samples, joint=False):
        angle = torch.rand(num_samples) * 2 * math.pi
        cat = torch.distributions.Categorical(self.weights).sample(num_samples)
        x, y = self.means[cat] * torch.cos(angle) + torch.randn_like(angle) * self.noise, self.means[
            cat] * torch.sin(angle) + torch.randn_like(angle) * self.noise
        if not joint:
            return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        else:
            return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1), cat

    def log_prob(self, x):
        r = torch.norm(x, dim=-1).unsqueeze(-1)
        cat = torch.distributions.Categorical(self.weights.to(x.device))
        mvn = torch.distributions.MultivariateNormal(self.means.to(x.device).unsqueeze(-1),
                                                     torch.eye(1).to(x.device).unsqueeze(0).repeat(2, 1,
                                                                                                   1) * self.noise.to(
                                                         x.device))
        mixt = torch.distributions.MixtureSameFamily(cat, mvn)
        return mixt.log_prob(r)

class SCurve():
    def __init__(self):
        super().__init__()

    def sample(self, num_samples):
        X, t = datasets.make_s_curve(num_samples[0], noise=0.05)
        X = torch.tensor(sklearn.preprocessing.StandardScaler().fit_transform(X)).float()
        return torch.cat([X[:,0].unsqueeze(-1), X[:,-1].unsqueeze(-1)], dim = -1)

#load target
target = Orbits()
num_samples = 10000
target_samples = target.sample([num_samples])

#compute instrumental distribution and sample
instrumental_distribution = torch.distributions.MultivariateNormal(torch.mean(target_samples, dim = 0), torch.cov(target_samples.T))
instrumental_samples = instrumental_distribution.sample([num_samples])

#train binary classifier
bc = BinaryClassifier(instrumental_samples, target_samples,[128,128,128])
bc.train(200,1000,lr = 5e-4, weight_decay= 5e-5,verbose = True)

#sample with AR
sampler = BinaryClassifierSampler(bc, instrumental_distribution)
samples = sampler.AR_sampling(num_samples)

#compute range
all_samples = torch.cat([instrumental_samples, target_samples, samples], dim = 0)
range_X = [torch.min(all_samples[:,0]),torch.max(all_samples[:,0])]
range_Y = [torch.min(all_samples[:,1]),torch.max(all_samples[:,1])]

#display target samples
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(target_samples[:,0],target_samples[:,1], color = 'red')
ax.set_xlim(range_X)
ax.set_ylim(range_Y)
plt.show()

#display instrumental distribution
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(instrumental_samples[:,0], instrumental_samples[:,1], color = 'green')
ax.set_xlim(range_X)
ax.set_ylim(range_Y)
plt.show()

#display samples
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(samples[:,0], samples[:,1], color = 'purple')
ax.set_xlim(range_X)
ax.set_ylim(range_Y)
plt.show()

#display ratio
fig = plt.figure(figsize = (10,10))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plot_2d_function(lambda x: torch.exp(bc.logit_r(x)), range = [range_X, range_Y], bins = [500,500])

#load dataset
target = TwoCircles()
target_samples = target.sample([num_samples])

#compute and sample instrumental distribution
instrumental_distribution = torch.distributions.MultivariateNormal(torch.mean(target_samples, dim = 0), torch.cov(target_samples.T))
instrumental_samples = instrumental_distribution.sample([num_samples])

#train binary classifier
bc = BinaryClassifier(instrumental_samples, target_samples,[128,128,128])
bc.train(200,1000,lr = 5e-4, weight_decay= 5e-5,verbose = True)

#sample with IMH
sampler = BinaryClassifierSampler(bc, instrumental_distribution)
samples = sampler.IMH_sampling(num_samples,50)

#compute range
all_samples = torch.cat([instrumental_samples, target_samples, samples], dim = 0)
range_X = [torch.min(all_samples[:,0]),torch.max(all_samples[:,0])]
range_Y = [torch.min(all_samples[:,1]),torch.max(all_samples[:,1])]

#display target samples
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(target_samples[:,0],target_samples[:,1], color = 'red')
ax.set_xlim(range_X)
ax.set_ylim(range_Y)
plt.show()

#display instrumental distribution
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(instrumental_samples[:,0], instrumental_samples[:,1], color = 'green')
ax.set_xlim(range_X)
ax.set_ylim(range_Y)
plt.show()

#display samples
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(samples[:,0], samples[:,1], color = 'purple')
ax.set_xlim(range_X)
ax.set_ylim(range_Y)
plt.show()

#display ratio
fig = plt.figure(figsize = (10,10))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plot_2d_function(lambda x: torch.exp(bc.logit_r(x)), range = [range_X, range_Y], bins = [500,500])

#load dataset
target = SCurve()
target_samples = target.sample([num_samples])
instrumental_distribution = torch.distributions.MultivariateNormal(torch.mean(target_samples, dim = 0), torch.cov(target_samples.T))
instrumental_samples = instrumental_distribution.sample([num_samples])

bc = BinaryClassifier(instrumental_samples, target_samples,[128,128,128])
bc.train(200,1000,lr = 5e-3, weight_decay= 5e-5,verbose = True)

sampler = BinaryClassifierSampler(bc, instrumental_distribution)
samples = sampler.SIS_sampling(num_samples)

#compute range
all_samples = torch.cat([instrumental_samples, target_samples, samples], dim = 0)
range_X = [torch.min(all_samples[:,0]),torch.max(all_samples[:,0])]
range_Y = [torch.min(all_samples[:,1]),torch.max(all_samples[:,1])]

#display target samples
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(target_samples[:,0],target_samples[:,1], color = 'red')
ax.set_xlim(range_X)
ax.set_ylim(range_Y)
plt.show()

#display instrumental distribution
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(instrumental_samples[:,0], instrumental_samples[:,1], color = 'green')
ax.set_xlim(range_X)
ax.set_ylim(range_Y)
plt.show()

#display samples
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.scatter(samples[:,0], samples[:,1], color = 'purple')
ax.set_xlim(range_X)
ax.set_ylim(range_Y)
plt.show()

#display ratio
fig = plt.figure(figsize = (10,10))
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plot_2d_function(lambda x: torch.exp(bc.logit_r(x)), range = [range_X, range_Y], bins = [500,500])



