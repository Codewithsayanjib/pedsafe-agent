import time
from typing import Dict, Tuple, Any, Optional

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class MLXModelManager:
    """
    Singleton manager for loading and caching local MLX models in unified memory.
    Ensures that multiple agents sharing the same model path do not duplicate memory.
    """
    _instance = None
    _loaded_models: Dict[str, Tuple[Any, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLXModelManager, cls).__new__(cls)
        return cls._instance

    def get_model(self, model_path: str):
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX / mlx_lm is not installed in the active environment.")
        
        if model_path not in self._loaded_models:
            print(f"[*] Initializing model into unified memory: {model_path} ...")
            start = time.time()
            model, tokenizer = load(model_path)
            self._loaded_models[model_path] = (model, tokenizer)
            elapsed = time.time() - start
            print(f"[✓] Loaded {model_path} in {elapsed:.2f}s")
        
        return self._loaded_models[model_path]

    def run_inference(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes generation and returns generated text along with latency and timing stats.
        """
        model, tokenizer = self.get_model(model_path)
        
        start_time = time.time()
        sampler = make_sampler(temp=temperature)
        # mlx_lm generate call
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )
        latency = time.time() - start_time
        
        return {
            "text": response.strip(),
            "latency_seconds": round(latency, 3),
            "model_path": model_path
        }
