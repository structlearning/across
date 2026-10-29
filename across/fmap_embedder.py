from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy import integrate, interpolate
from torch import Tensor


def _pdf(w: np.ndarray, T: float) -> np.ndarray:
    R_G = 2 * np.sin(w * T / 2) ** 2 / w**2 + T * np.sin(w * T) / w
    I_G = np.sin(w * T) / w**2 - T * np.cos(w * T) / w
    return np.abs(R_G) + np.abs(I_G)


def _sample_frequencies(
    num_samples: int, T: float, a: float, b: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Importance-sample num_samples frequencies from the dominance kernel spectral
    density p(ω) ∝ |R_G(ω)| + |I_G(ω)| over [a, b] via inverse-CDF.
    Returns (samples, normalised_pdfs), both float32, shape (num_samples,).
    """
    x = np.linspace(a, b, 100_000)
    y = _pdf(x, T)
    cdf = np.cumsum(y)
    cdf /= cdf.max()
    inv_cdf = interpolate.interp1d(cdf, x)
    N = integrate.quad(_pdf, a, b, points=0, limit=10_000, args=T)[0]
    uniform = np.random.uniform(1e-5, 1 - 1e-5, num_samples)
    samples = inv_cdf(uniform).astype(np.float32)
    pdfs = np.array([_pdf(float(w), T) / N for w in samples], dtype=np.float32)
    return samples, pdfs


class FMapEmbedder:
    """
    Asymmetric Fourier Map embedder (Choice A).

    For each output dimension i, the scalar (W^{(t)} h)_i is mapped to 4m Fourier
    features that approximate the dominance kernel's characteristic function.
    Following FourierHashNet, each embedding dimension gets its own m independently
    sampled frequencies, so ws has shape (d, m) with d*m total samples.
    """

    def __init__(self, config):
        d = config.gmn.filters_3
        m = config.fmap_embedder.m_use
        T = config.fmap_embedder.T
        a = config.fmap_embedder.freq_a
        b = config.fmap_embedder.freq_b
        num_samples = d * m

        cache_path = (
            Path(config.base_dir)
            / "allPklDumps"
            / config.dataset.name
            / f"fmap_freqs_d{d}_m{m}_T{T}_a{a}_b{b}.pt"
        )

        if cache_path.exists():
            logger.info(f"Loading FMap frequencies from {cache_path}")
            saved = torch.load(cache_path, weights_only=True)
            ws = saved["ws"].numpy()      # (d*m,)
            pdfs = saved["pdfs"].numpy()  # (d*m,)
        else:
            logger.info(f"Sampling {num_samples} FMap frequencies (d={d}, m={m}, T={T})")
            ws, pdfs = _sample_frequencies(num_samples, T, a, b)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"ws": torch.tensor(ws), "pdfs": torch.tensor(pdfs)}, cache_path)
            logger.info(f"Saved FMap frequencies to {cache_path}")

        # Reshape to (d, m) — each embedding dimension has its own m frequencies
        ws   = ws.reshape(d, m)
        pdfs = pdfs.reshape(d, m)

        R_G = 2 * np.sin(ws * T / 2) ** 2 / ws**2 + T * np.sin(ws * T) / ws  # (d, m)
        I_G = np.sin(ws * T) / ws**2 - T * np.cos(ws * T) / ws                # (d, m)
        sqrt_pdf = np.sqrt(pdfs)                                                # (d, m)

        # Query amplitude factors — shape (d, m)
        # fmap1/fmap2: sign(R_G) * sqrt(|R_G|) / sqrt(pdf)
        # fmap3:      -sign(I_G) * sqrt(|I_G|) / sqrt(pdf)
        # fmap4:       sign(I_G) * sqrt(|I_G|) / sqrt(pdf)
        self.amp_q_RG     = torch.tensor(np.sign(R_G) * np.sqrt(np.abs(R_G)) / sqrt_pdf)
        self.amp_q_IG_neg = torch.tensor(-np.sign(I_G) * np.sqrt(np.abs(I_G)) / sqrt_pdf)
        self.amp_q_IG_pos = torch.tensor(np.sign(I_G) * np.sqrt(np.abs(I_G)) / sqrt_pdf)

        # Corpus amplitude factors — shape (d, m)
        # fmap1/fmap2: sqrt(|R_G|) / sqrt(pdf)
        # fmap3/fmap4: sqrt(|I_G|) / sqrt(pdf)
        self.amp_c_RG = torch.tensor(np.sqrt(np.abs(R_G)) / sqrt_pdf)
        self.amp_c_IG = torch.tensor(np.sqrt(np.abs(I_G)) / sqrt_pdf)

        self.ws = torch.tensor(ws)  # (d, m)

    def to(self, device: str) -> "FMapEmbedder":
        self.ws           = self.ws.to(device)
        self.amp_q_RG     = self.amp_q_RG.to(device)
        self.amp_q_IG_neg = self.amp_q_IG_neg.to(device)
        self.amp_q_IG_pos = self.amp_q_IG_pos.to(device)
        self.amp_c_RG     = self.amp_c_RG.to(device)
        self.amp_c_IG     = self.amp_c_IG.to(device)
        return self

    def get_embs(
        self,
        query_embs: Tensor,
        corpus_embs: Tensor,
        curr_params: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        query_embs, corpus_embs : (B, d1)  pre-last-layer embeddings
        curr_params             : (d, d1)  W^{(t)}
        returns                 : (B, d, 4m), (B, d, 4m)
        """
        q_final = (curr_params @ query_embs.T).T   # (B, d)
        c_final = (curr_params @ corpus_embs.T).T  # (B, d)
        return self._fmap(q_final, is_query=True), self._fmap(c_final, is_query=False)

    def _fmap(self, embs: Tensor, *, is_query: bool) -> Tensor:
        # embs: (B, d),  self.ws: (d, m)  →  thetas: (B, d, m)
        thetas = embs.unsqueeze(-1) * self.ws
        cos_t  = torch.cos(thetas)
        sin_t  = torch.sin(thetas)

        if is_query:
            f1 = self.amp_q_RG     * cos_t   # sign(R_G)*√|R_G|/√pdf · cos
            f2 = self.amp_q_RG     * sin_t   # sign(R_G)*√|R_G|/√pdf · sin
            f3 = self.amp_q_IG_neg * sin_t   # -sign(I_G)*√|I_G|/√pdf · sin
            f4 = self.amp_q_IG_pos * cos_t   # sign(I_G)*√|I_G|/√pdf · cos
        else:
            f1 = self.amp_c_RG * cos_t       # √|R_G|/√pdf · cos
            f2 = self.amp_c_RG * sin_t       # √|R_G|/√pdf · sin
            f3 = self.amp_c_IG * cos_t       # √|I_G|/√pdf · cos
            f4 = self.amp_c_IG * sin_t       # √|I_G|/√pdf · sin

        return torch.cat([f1, f2, f3, f4], dim=-1)  # (B, d, 4m)
