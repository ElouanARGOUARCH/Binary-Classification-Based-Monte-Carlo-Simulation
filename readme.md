This Github Project is associated to the paper "Binary Classification Monte Carlo Sampling" https://arxiv.org/abs/2307.16035; and provides with reusable code which implements the proposed methods. 

**Summary of the paper:**
---
Sampling has became paramount in many statistical field and Monte Carlo estimation methods usually takes interest in sampling from a distribution which we know via its probability density function. Such methods include in particular: Accept-Reject sampling, Sampling-Importance-Resampling and Independent-Metropolis-Hastings; and these three methods share in common that the sampling mechanism involves the ratio of the target pdf and the pdf of a chosen instrumental distribution.
The score point of our contribution is to realize that when the target distribution is known only via a set of recorded samples and the pdf is otherwise unknown, the three sampling algorithms can still be applied even though the ratio of pdf is also unknown. 
Indeed, we propose to use a classifier which is trained to distinguish samples from the target distribution from that from the instrumental distribution and wich can indeed be turned into an approximation of the pdf ratio.

**How to use the code in practice:**
---
First install the package with using the shell command: 
```shell 
pip install git+https://github.com/ElouanARGOUARCH/Binary-Classification-Based-Monte-Carlo-Simulation
```
and import the corresponding package in python 
```python
import models_bcbmcs
```
The proposed methods takes as input a set of samples (target_samples in our example) of shape [n,d] from the target distribution and involves three objects: (i) an easy-to-samples instrumental distribution, ideally close to the target distribution,(ii) a binary classifier trained via Binary Cross entropy criterion to distinguish between target samples and instrumental samples, and (iii) a sampling algorithm.
For small dimensional example, using a Gaussian instrumental distribution with estimated mean an covariance matrix usually provides with satisfactory results, at least for illustration purposes. This can be achieved with:
```python 
mean = torch.mean(target_samples, dim = 0)
cov = torch.cov(target_samples.T)
instrumental = torch.distributions.MultivariateNormal(mean, (cov + cov.T)/2)
proposed_samples = instrumental.sample([num_samples])
```

In this repository, we propose reusable code for the methods and provide an implementation of binary classifier trained via Binary cross entropy in models/binary_classifier.py. 
This class in instanciated by specifying samples from the instrumental disitribution (instrumental_samples in this example), samples from the target distribution (target_samples in this example), and the structure of the underlying fully connected neural network in the classifier ([128,128,128] is three hidden-layers with each 128 units in this example). This binary classifier can be trained using Adam gradient-based optimization with:
```python
bc = BinaryClassifier(instrumental_samples, target_samples,[128,128,128])
bc.train(epochs, batch_size)
````


Once the binary classifier is train, one can proceed to approximate Monte Carlo sampling from the target distribution using the instrumental distribution and the binary classifier with the following class located in models/samplers.py.
This class in instanciated by specifying a trained binary classifier (bc in this example) and the corresponding instrumental distribution (object of type torch.distribution). This class has three methods for the three sampling mechanisms:
- in the AR case, the method takes num_proposed_samples as input which is the number of samples proposed from the instrumental distribution;
- in the IMH case, the method T is the length of the MCMC chain and num_chains is the number of parallel chains;
- in the SIS case, num_samples is the number of proposed importance samples and the number of resampled samples (assumed identical in this implementation).

```python
sampler = BinaryClassifierSampler(bc, instrumental_distribution)
ar_samples = sampler.AR_sampling(num_proposed_samples)
imh_samples = sampler.IMH_sampling(T, num_chains)
sis_samples = sampler.SIS_sampling(num_samples)
log_prob = sampler.log_prob(samples)
```
Finally, if the instrumental distribution indeed has tractable pdf, we can access the approximately normalized unormalized log_pdf (which is the energy function, refer to the last section of the paper) of a given set of samples with using the log_prob method. If the instrumental distribution has untractable pdf or has no such method implement, this log_prob methods raises an exception.

  
**Going further**
--

To clone the project and access more elaborate usage examples: 
```shell
git clone https://github.com/ElouanARGOUARCH/Binary-Classification-Based-Monte-Carlo-Simulation
cd Binary-Classification-Based-Monte-Carlo-Simulation
pip install -r requirements.txt
```


