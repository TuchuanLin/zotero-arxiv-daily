from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np
@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging
    
            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

            # 从配置中提取 encode_kwargs，避免重复参数
            if self.config.reranker.local.encode_kwargs:
                encode_kwargs = dict(self.config.reranker.local.encode_kwargs)
            else:
                encode_kwargs = {}
            
            # 从 encode_kwargs 中移除 trust_remote_code，避免重复
            encode_kwargs.pop('trust_remote_code', None)
            
            # 创建编码器时显式设置 trust_remote_code
            encoder = SentenceTransformer(
                self.config.reranker.local.model, 
                trust_remote_code=True
            )
            
            s1_feature = encoder.encode(s1, **encode_kwargs, show_progress_bar=True)
            s2_feature = encoder.encode(s2, **encode_kwargs, show_progress_bar=True)
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
