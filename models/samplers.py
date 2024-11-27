import torch

class BinaryClassifierSampler(torch.nn.Module):
    def __init__(self,binary_classifier, proposal_distribution):
        super().__init__()
        self.binary_classifier = binary_classifier
        self.proposal_distribution = proposal_distribution


    def AR_sampling(self, num_proposed_samples):
        proposed_samples = self.proposal_distribution.sample([num_proposed_samples])
        constant = torch.max(torch.exp(self.binary_classifier.logit_r(proposed_samples).squeeze(-1)))
        accepted_mask = torch.rand(proposed_samples.shape[0]) < (
                torch.exp(self.binary_classifier.logit_r(proposed_samples).squeeze(-1)) / constant)
        return proposed_samples[accepted_mask]

    def IMH_sampling(self,T, num_chains):
        samples = self.proposal_distribution.sample([num_chains])
        for t in range(T):
            proposed_samples = self.proposal_distribution.sample([num_chains])
            mask = torch.rand(num_chains) < torch.exp(self.binary_classifier.logit_r(proposed_samples) - self.binary_classifier.logit_r(samples)).squeeze(-1)
            samples = torch.cat([proposed_samples[mask], samples[~mask]], dim=0)
        return samples

    def SIS_sampling(self, num_samples):
        proposed_samples = self.proposal_distribution.sample([num_samples])
        unormalized_weights = torch.exp(self.binary_classifier.logit_r(proposed_samples)).squeeze(-1)
        normalized_weights = unormalized_weights / torch.sum(unormalized_weights)
        cat = torch.distributions.Categorical(normalized_weights).sample([num_samples])
        return proposed_samples[cat]

    def log_prob(self,samples):
        try:
            return self.binary_classif.logit_r(samples.squeeze(-1)) + self.instrumental.log_prob(samples)
        except AttributeError:
            print("Instrumental distribution has no method log_prob")
