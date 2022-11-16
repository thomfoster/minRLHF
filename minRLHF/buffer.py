from __future__ import annotations
import torch
import torch.nn.functional as func
from typing import Callable, Optional

from torch_discounted_cumsum import discounted_cumsum_right

def default_reward_augmenter(buf: Buffer) -> None:
    buf.reward_augmentation_buffer[:, :] = 0


class Buffer:
    def __init__(self, vocab_size:int, device: Optional[str]=None, max_episodes: int=32, 
                 max_ep_length: int=50, reward_augmenter: Callable[[Buffer], None]=default_reward_augmenter):
        """
        Store batches of generated strings + rewards + value estimates. 
        Compute augmented rewards and advantages.
        """
        self.vocab_size = vocab_size
        self.device = torch.device(device) if device is not None \
            else torch.device('cuda:0') if torch.cuda.is_available() \
                else torch.device('cpu')
        self.max_episodes = max_episodes
        self.max_ep_length = max_ep_length
        self.reward_augmenter = reward_augmenter
        
        self.ptr = 0

        # Empty buffers are filled up in batches using the store method
        # TODO: This really should be sharded
        self.state_buffer               = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.long).to(self.device)
        self.prompt_mask_buffer         = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.long).to(self.device)
        self.completion_mask_buffer     = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.long).to(self.device)
        self.reward_buffer              = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.float32).to(self.device)
        self.value_estimates_buffer     = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.float32).to(self.device)
        self.pi_0_logprobs_buffer       = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.float32).to(self.device)
        self.pi_t_logprobs_buffer       = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.float32).to(self.device)
        # If you want to do exact KL div reward augmentation, you also need to store the entire next token probability dist for each token.
        # self.pi_0_buffer                = torch.empty(size=(self.max_episodes, self.max_ep_length, self.vocab_size), dtype=torch.float32).to(self.device)
        # self.pi_t_buffer                = torch.empty(size=(self.max_episodes, self.max_ep_length, self.vocab_size), dtype=torch.float32).to(self.device)
    
        # The following buffers are filled with computations performed during the `.get` method
        self.critic_targets_buffer      = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.float32).to(self.device)
        self.advantages_buffer          = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.float32).to(self.device)
        
        # `.get` computes augmented_reward_buffer = reward_buffer + beta * reward_augmentation_buffer
        # zeros by default if no reward_augmenter function given during init
        self.reward_augmentation_buffer = torch.zeros(size=(self.max_episodes, self.max_ep_length), dtype=torch.float32).to(self.device)
        self.augmented_reward_buffer    = torch.empty(size=(self.max_episodes, self.max_ep_length), dtype=torch.float32).to(self.device)
        
        
    def store(self, state: torch.Tensor, prompt_mask: torch.Tensor, completion_mask: torch.Tensor, reward: torch.Tensor, 
              value_estimates: torch.Tensor, pi_0_logprobs: torch.Tensor, pi_t_logprobs: torch.Tensor, pi_0: torch.Tensor, pi_t: torch.Tensor):
        """
        Add batches of episodes to the buffer.
        """
        # TODO: Allow both "batch" and "state by state" storage.
        
        # Check we won't overflow the buffers
        batch_size = state.shape[0]
        assert prompt_mask.shape[0] == batch_size
        assert completion_mask.shape[0] == batch_size
        assert reward.shape[0] == batch_size
        assert value_estimates.shape[0] == batch_size
        assert pi_0_logprobs.shape[0] == batch_size
        assert pi_t_logprobs.shape[0] == batch_size
        # assert pi_0.shape[0] == batch_size
        # assert pi_t.shape[0] == batch_size
        assert self.ptr + batch_size <= self.max_episodes
        
        assert state.shape[1] == self.max_ep_length
        assert prompt_mask.shape[1] == self.max_ep_length
        assert completion_mask.shape[1] == self.max_ep_length
        assert value_estimates.shape[1] == self.max_ep_length
        assert pi_0_logprobs.shape[1] == self.max_ep_length
        assert pi_t_logprobs.shape[1] == self.max_ep_length
        assert pi_0.shape[1] == self.max_ep_length
        assert pi_t.shape[1] == self.max_ep_length
        
        # assert pi_0.shape[2] == self.vocab_size
        # assert pi_t.shape[2] == self.vocab_size
        
        # ? Do we need to deepcopy here?
        self.state_buffer[self.ptr:self.ptr+batch_size] = state
        self.prompt_mask_buffer[self.ptr:self.ptr+batch_size] = prompt_mask
        self.completion_mask_buffer[self.ptr:self.ptr+batch_size] = completion_mask
        self.reward_buffer[self.ptr:self.ptr+batch_size] = reward
        self.value_estimates_buffer[self.ptr:self.ptr+batch_size] = value_estimates
        self.pi_0_logprobs_buffer[self.ptr:self.ptr+batch_size] = pi_0_logprobs
        self.pi_t_logprobs_buffer[self.ptr:self.ptr+batch_size] = pi_t_logprobs
        # self.pi_0_buffer[self.ptr:self.ptr+batch_size] = pi_0
        # self.pi_t_buffer[self.ptr:self.ptr+batch_size] = pi_t
        
        # ! Critical (duh)
        self.ptr += batch_size
        

    def get(self, gamma: float, lam: float, beta: float):
        """
        Compute advantages etc, then return tensors.

        Inputs: 
        Outputs: data: dict[str:Tensor] - all tensors needed for training
                 info: dict[str: float] - extra info, like perplexity, kld from 
        """
        # TODO: Extend with "state by state" style return for use with standard libraries.
        # TODO: Add batch_size parameter for iterated return of data, to allow gradient accumulation.
        # TODO: Extend for arbitrary augmentation function.
        # TODO: Add returnd device map
        # TODO: Add `last_val` option for controlling end of sequence rewards.
        
        assert self.ptr == self.max_episodes    # Asserting full makes tensor logic much easier
        self.ptr = 0
        
        self.reward_augmenter(self)     # Side effect: Fills augmentation buffer
        self.augmented_reward_buffer = self.reward_buffer + beta * self.reward_augmentation_buffer     # actually augment rewards
        
        self._compute_critic_targets(gamma)      # Side effect: Fills critic targets buffer
        self._compute_advantages(gamma, lam)      # Side effect: Fills advantages buffer
        
        # Normalise advantages to zero mean and variance
        mu, sigma = self.advantages_buffer.mean(dim=-1).unsqueeze(-1), self.advantages_buffer.std(dim=-1).unsqueeze(-1)
        self.advantages_buffer = (self.advantages_buffer - mu) / sigma
        
        # ? Do we need to return copies here?
        return {
            'ids': self.state_buffer,
            'prompt_mask': self.prompt_mask_buffer,
            'completion_mask': self.completion_mask_buffer,
            'reward': self.reward_buffer,
            'value_estimates': self.value_estimates_buffer,
            'pi_0_logprobs': self.pi_0_logprobs_buffer,
            'pi_t_logprobs': self.pi_t_logprobs_buffer,
            # 'pi_0': self.pi_0_buffer,
            # 'pi_t': self.pi_t_buffer,
            'critic_targets': self.critic_targets_buffer,
            'advantages': self.advantages_buffer,
            'augmented_reward': self.augmented_reward_buffer,
        }
    
    
    def _compute_critic_targets(self, gamma):
        """
        For each generated sequence of length n in batch:
            advantages[i] = sum_j=i_n+1 [ gamma^(j-1) reward[j]]
            where reward[n+1] = 0 if episode ended and v[n] otherwise.
            Ideally would be v[n+1] but most enironments won't give you access to that state.
        """
        zerod_rewards = self.completion_mask_buffer * self.augmented_reward_buffer
        for idx in range(self.max_episodes):
            self.critic_targets_buffer[idx, :] = \
                discounted_cumsum_right(zerod_rewards[idx].unsqueeze(0), gamma)
    
    
    def _compute_advantages(self, gamma, lam):
        """
        For each generated sequence of length n in batch:
            deltas[i] = rewards[i] + gamma*value_estimates[i] - value-estimates[i-1] for i>0
            deltas[i] = rewards[i] + gamma*value_estimates[i] for i == 0
        """
        zeord_rewards = self.completion_mask_buffer * self.augmented_reward_buffer
        padded_value_estimates = func.pad(self.value_estimates_buffer, (1, 0))
        deltas = zeord_rewards \
            + gamma*padded_value_estimates[:, 1:] \
                - padded_value_estimates[:, :-1]
        zerod_deltas = self.completion_mask_buffer * deltas
        for idx in range(self.max_episodes):
            self.advantages_buffer[idx, :] = \
                discounted_cumsum_right(zerod_deltas[idx].unsqueeze(0), gamma*lam)