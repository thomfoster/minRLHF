

# %%
import minRLHF
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForTokenClassification
)

# %%
from minRLHF.environment import Environment
import random
from transformers.pipelines import pipeline

reward_model = pipeline(
    "text-classification",
    model='bhadresh-savani/distilbert-base-uncased-emotion', 
    return_all_scores=True
)

class MyEnv(Environment):
    def get_input_prompt(self) -> str:
        return random.choice([
            'I went for a walk one day and',
            'A long time ago, in a galaxy far far away',
            'Oops! I'
        ])
        
    def score_generation(self, text: str) -> float:
        sentiment_scores = reward_model(text)[0]
        sentiment_scores = {d['label']: d['score'] for d in sentiment_scores}
        return sentiment_scores['joy']

# %%
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda')
reference = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda')
critic = AutoModelForTokenClassification.from_pretrained('gpt2', num_labels=1).to('cuda')

# Instantiate envrionment
env = MyEnv(tokenizer, batch_size=32)

# %%
from minRLHF.ppo_trainer import PPOTrainer

# Create PPO trainer
ppo_trainer = PPOTrainer(
    actor_model=model,
    critic_model=critic,
    reference_model=reference,
    env=env,
)

# %%
ppo_trainer.train()


