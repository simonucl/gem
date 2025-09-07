import random
import time
import logging
from typing import Dict, Any
import gem
from gem.wrappers.wrapper_factory import get_wrapper_fns
from RL2.utils.agent import AgentBase


def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def apply_no_template(observation: str) -> str:
    return observation

def apply_qwen3_general_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def apply_code_template(observation: str) -> str:
    return (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
        f"\nQuestion: {observation}"
        "\nPlease reason step by step, and write your code in markdown format, e.g., ```python\n# YOUR CODE HERE\n```."
    )

TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "code": apply_code_template,
}


class GEMAgent(AgentBase):
    """
    GEM agent that inherits from AgentBase.
    
    Implements the required __init__, step, and reset methods for GEM environment integration.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.tokenizer = kwargs.get('tokenizer')
        
        self.config = config
        
        wrappers = get_wrapper_fns(self.config.get("wrappers", []), tokenizer=self.tokenizer)
        
        self.env = gem.make_vec(
            [self.config["env_id"]],
            vec_kwargs=[{"seed": self.config.get("seed", 233) + kwargs.get("agent_id", 0)}],
            wrappers=wrappers,
            async_mode=False,
        )
        
        logging.info(f"Initialized GEM environment: {self.config['env_id']}")

    def format_observation(self, observation: str) -> str:
        prompt_template = self.config.get("prompt_template", "no")
        template_fn = TEMPLATE_FACTORY.get(prompt_template, apply_no_template)
        formatted_obs = template_fn(observation)
        
        return formatted_obs

    async def reset(self, **kwargs):
        seed = kwargs.get('seed', int(time.time() * 1000) % (2**31))
        random.seed(seed)
        
        observation, _ = self.env.reset(seed=seed)
        
        if isinstance(observation, list):
            observation = observation[0]
        
        return {"obs": observation}

    async def step(self, action: str, train: bool = True, **kwargs) -> Dict[str, Any]:
        next_obs, reward, terminated, truncated, info = self.env.step([action])
        
        if isinstance(next_obs, list):
            next_obs = next_obs[0]
        if isinstance(reward, list):
            reward = reward[0]
        if isinstance(terminated, list):
            terminated = terminated[0]
        if isinstance(truncated, list):
            truncated = truncated[0]
            
        done = terminated or truncated
        
        return {
            "obs": next_obs,
            "reward": float(reward),
            "done": bool(done),
            "info": info if info else {}
        }