"""GenEval HTTP reward client wrapper."""

from io import BytesIO
import pickle
from typing import List
import importlib

import numpy as np
from PIL import Image


class GenEvalHTTPRewardModel:
    """Call an external GenEval reward server over HTTP."""

    def __init__(
        self,
        url: str = "",
        batch_size: int = 64,
        timeout: int = 120,
        only_strict: bool = True,
        retries: int = 6,
    ):
        self.url = url
        self.batch_size = batch_size
        self.timeout = timeout
        self.only_strict = only_strict

        self.sess = None
        try:
            requests = importlib.import_module("requests")
            adapters = importlib.import_module("requests.adapters")
            HTTPAdapter = getattr(adapters, "HTTPAdapter")
            Retry = getattr(adapters, "Retry")

            self.sess = requests.Session()
            retry_cfg = Retry(
                total=retries,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=False,
            )
            self.sess.mount("http://", HTTPAdapter(max_retries=retry_cfg))
            self.sess.mount("https://", HTTPAdapter(max_retries=retry_cfg))
        except Exception as e:
            print(f"[WARNING] requests init failed for GenEval HTTP: {e}")

    def _encode_jpeg_batch(self, images: List[Image.Image]) -> List[bytes]:
        encoded = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            buf = BytesIO()
            image.save(buf, format="JPEG")
            encoded.append(buf.getvalue())
        return encoded

    def __call__(self, images: List, prompts: List[str]) -> List[float]:
        if self.sess is None:
            return [0.0 for _ in images]

        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")

        all_scores = []
        for start in range(0, len(images), self.batch_size):
            image_batch = images[start : start + self.batch_size]
            prompt_batch = prompts[start : start + self.batch_size]

            payload = {
                "images": self._encode_jpeg_batch(image_batch),
                # For compatibility with Flow-style GenEval servers.
                "meta_datas": list(prompt_batch),
                "only_strict": self.only_strict,
            }
            data_bytes = pickle.dumps(payload)

            try:
                response = self.sess.post(self.url, data=data_bytes, timeout=self.timeout)
                response.raise_for_status()
                response_data = pickle.loads(response.content)

                if "scores" in response_data:
                    scores = response_data["scores"]
                elif "rewards" in response_data:
                    scores = response_data["rewards"]
                elif "outputs" in response_data:
                    scores = response_data["outputs"]
                else:
                    raise KeyError("No scores/rewards/outputs field in GenEval response")

                all_scores.extend([float(x) for x in scores])
            except Exception as e:
                print(f"[WARNING] GenEval HTTP request failed: {e}")
                all_scores.extend([0.0 for _ in image_batch])

        return all_scores
