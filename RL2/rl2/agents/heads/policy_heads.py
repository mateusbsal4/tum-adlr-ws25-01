"""
Policy heads for RL^2 agents.
"""

import torch as tc
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)

class LinearPolicyHead(tc.nn.Module):
    """
    Policy head for a reinforcement learning agent.
    """
    def __init__(self, num_features, num_actions):
        super().__init__()
        self._num_features = num_features
        self._num_actions = num_actions
        
        # Add a small network instead of a single linear layer
        self._net = tc.nn.Sequential(
            tc.nn.Linear(num_features, num_features // 2),
            tc.nn.LayerNorm(num_features // 2),  # Add normalization
            tc.nn.ReLU(),
            tc.nn.Linear(num_features // 2, num_actions),
        )
        
        # Initialize with smaller weights
        for layer in self._net:
            if isinstance(layer, tc.nn.Linear):
                tc.nn.init.orthogonal_(layer.weight, gain=0.01)
                tc.nn.init.zeros_(layer.bias)

    def forward(self, features: tc.FloatTensor) -> tc.distributions.Categorical:
        """
        Computes a policy distribution from features and returns it.

        Args:
            features: a tc.FloatTensor of shape [B, ..., F].

        Returns:
            tc.distributions.Categorical over actions, with batch shape [B, ...]
        """
        # print("\nDEBUG - Policy Head:")
        # print(f"Input features range: {features.min().item():.3f} to {features.max().item():.3f}")
        
        logits = self._net(features)
        # print(f"Raw logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
        
        # Add numerical stability
        # logits = tc.clamp(logits, -20.0, 20.0)
        # print(f"Clamped logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
        
        return tc.distributions.Categorical(logits=logits)


    # def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    #     """
    #     Forward pass in all the networks (actor and critic)

    #     :param obs: Observation
    #     :param deterministic: Whether to sample or use deterministic actions
    #     :return: action, value and log probability of the action
    #     """
    #     # Preprocess the observation if needed
    #     features = self.extract_features(obs)
    #     if self.share_features_extractor:
    #         latent_pi, latent_vf = self.mlp_extractor(features)
    #     else:
    #         pi_features, vf_features = features
    #         latent_pi = self.mlp_extractor.forward_actor(pi_features)
    #         latent_vf = self.mlp_extractor.forward_critic(vf_features)
    #     distribution = self._get_action_dist_from_latent(latent_pi)
    #     
    #     actions = distribution.get_actions(deterministic=deterministic)
    #     log_prob = distribution.log_prob(actions)
    #     actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
    #     return actions, values, log_prob

    # def make_features_extractor(self) -> BaseFeaturesExtractor:
    #     """Helper method to create a features extractor."""
    #     return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
    
    # def extract_features(  # type: ignore[override]
    #     self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    # ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
    #     """
    #     Preprocess the observation if needed and extract features.

    #     :param obs: Observation
    #     :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
    #     :return: The extracted features. If features extractor is not shared, returns a tuple with the
    #         features for the actor and the features for the critic.
    #     """
    #     if self.share_features_extractor:
    #         return super().extract_features(obs, self.features_extractor if features_extractor is None else features_extractor)
    #     else:
    #         if features_extractor is not None:
    #             warnings.warn(
    #                 "Provided features_extractor will be ignored because the features extractor is not shared.",
    #                 UserWarning,
    #             )

    #         pi_features = super().extract_features(obs, self.pi_features_extractor)
    #         vf_features = super().extract_features(obs, self.vf_features_extractor)
    #         return pi_features, vf_features

    # def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
    #     """
    #     Retrieve action distribution given the latent codes.

    #     :param latent_pi: Latent code for the actor
    #     :return: Action distribution
    #     """
    #     mean_actions = self.action_net(latent_pi)

    #     if isinstance(self.action_dist, DiagGaussianDistribution):
    #         return self.action_dist.proba_distribution(mean_actions, self.log_std)
    #     elif isinstance(self.action_dist, CategoricalDistribution):
    #         # Here mean_actions are the logits before the softmax
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, MultiCategoricalDistribution):
    #         # Here mean_actions are the flattened logits
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, BernoulliDistribution):
    #         # Here mean_actions are the logits (before rounding to get the binary actions)
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, StateDependentNoiseDistribution):
    #         return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
    #     else:
    #         raise ValueError("Invalid action distribution")

    # def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
    #     """
    #     Get the action according to the policy for a given observation.

    #     :param observation:
    #     :param deterministic: Whether to use stochastic or deterministic actions
    #     :return: Taken action according to the policy
    #     """
    #     return self.get_distribution(observation).get_actions(deterministic=deterministic)

    # def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
    #     """
    #     Evaluate actions according to the current policy,
    #     given the observations.

    #     :param obs: Observation
    #     :param actions: Actions
    #     :return: estimated value, log likelihood of taking those actions
    #         and entropy of the action distribution.
    #     """
    #     # Preprocess the observation if needed
    #     features = self.extract_features(obs)
    #     if self.share_features_extractor:
    #         latent_pi, latent_vf = self.mlp_extractor(features)
    #     else:
    #         pi_features, vf_features = features
    #         latent_pi = self.mlp_extractor.forward_actor(pi_features)
    #         latent_vf = self.mlp_extractor.forward_critic(vf_features)
    #     distribution = self._get_action_dist_from_latent(latent_pi)
    #     log_prob = distribution.log_prob(actions)
    #     values = self.value_net(latent_vf)
    #     entropy = distribution.entropy()
    #     return values, log_prob, entropy

    # def get_distribution(self, obs: PyTorchObs) -> Distribution:
    #     """
    #     Get the current policy distribution given the observations.

    #     :param obs:
    #     :return: the action distribution.
    #     """
    #     features = super().extract_features(obs, self.pi_features_extractor)
    #     latent_pi = self.mlp_extractor.forward_actor(features)
    #     return self._get_action_dist_from_latent(latent_pi)

    # def predict_values(self, obs: PyTorchObs) -> th.Tensor:
    #     """
    #     Get the estimated values according to the current policy given the observations.

    #     :param obs: Observation
    #     :return: the estimated values.
    #     """
    #     features = super().extract_features(obs, self.vf_features_extractor)
    #     latent_vf = self.mlp_extractor.forward_critic(features)
    #     return self.value_net(latent_vf)