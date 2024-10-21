import torch
import matplotlib.pyplot as plt
class Uniform:
    def __init__(self, lower, upper):
        self.p = upper.shape[0]
        self.lower = lower
        self.upper = upper
        assert torch.sum(upper>lower) == self.p, 'upper bound should be greater or equal to lower bound'
        self.log_scale = torch.log(self.upper - self.lower)
        self.location = (self.upper + self.lower)/2

    def log_prob(self, samples):
        condition = ((samples > self.lower).sum(-1) == self.p) * ((samples < self.upper).sum(-1) == self.p)*1
        inverse_condition = torch.logical_not(condition) * 1
        true = -torch.logsumexp(self.log_scale, dim = -1) * condition
        false = torch.nan_to_num(-torch.inf*inverse_condition, nan = 0)
        return (true + false)

    def sample(self, num_samples):
        desired_size = num_samples.copy()
        desired_size.append(self.p)
        return self.lower.expand(desired_size) + torch.rand(desired_size)*torch.exp(self.log_scale.expand(desired_size))
q = torch.distributions.Normal(torch.zeros(1), 1.5*torch.ones(1))
p = Uniform(torch.tensor([-1.5]), torch.tensor([1.5]))
diff = p.upper - p.lower
c = 1/(torch.exp(q.log_prob(p.upper))*diff)
print(c)
fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot(1, 2, 1)   #top and bottom left
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
tt = torch.linspace(-5,5,200)
plt.plot(tt, c*torch.exp(q.log_prob(tt)),color = 'black', label=r'$c q(x)$')
plt.plot(tt, torch.exp(p.log_prob(tt.unsqueeze(-1))),color = 'black', linestyle ='--',label = r'$p(x)$')
plt.fill_between(tt,torch.exp(p.log_prob(tt.unsqueeze(-1))),c*torch.exp(q.log_prob(tt)), color = 'C3', alpha=.5)
plt.fill_between(tt,torch.exp(p.log_prob(tt.unsqueeze(-1))), color = 'C2', alpha =.5)
plt.legend()
ax = fig.add_subplot(2, 2, 2)   #top right
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
tt = torch.linspace(-5,5,500)
plt.fill_between(tt,torch.exp(q.log_prob(tt))  - torch.exp(p.log_prob(tt.unsqueeze(-1)))/c, color = 'C3', alpha=.5,label = r'reminder')
ax.legend()
ax = fig.add_subplot(2, 2, 4)   #bottom right
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.fill_between(tt,torch.exp(p.log_prob(tt.unsqueeze(-1)))/c, color = 'C2', alpha =.5, label = r'target $p(x)$')
ax.legend()
plt.show()