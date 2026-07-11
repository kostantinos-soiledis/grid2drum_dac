from __future__ import annotations

from contextlib import contextmanager
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from runtime_compat import apply_runtime_compat

apply_runtime_compat()

PACKAGE_ROOT = Path(__file__).resolve().parents[3]

import torch
import torch.nn.functional as F


DEFAULT_CODEC_FAMILY = "encodec"
DEFAULT_CODEC_MODEL_ID = "facebook/encodec_32khz"
DEFAULT_DAC_CODEC_MODEL_ID = "descript/dac_44khz"
DEFAULT_SHAME_CODEC_MODEL_ID = "stabilityai/SAME-L"
DEFAULT_ENCODEC_BANDWIDTH = 2.2
LEGACY_TARGET_LAYOUT = "framewise_sum_t128"
DEFAULT_TARGET_LAYOUT = "framewise_sum"
PCA_TARGET_LAYOUT = "framewise_pca"


@dataclass(frozen=True)
class AudioCodecMetadata:
    codec_family: str
    codec_model_id: str
    codec_sample_rate: int
    codec_audio_channels: int
    codec_frame_rate: float
    codec_codebook_size: int
    codec_num_codebooks: int
    codec_target_dim: int
    encodec_bandwidth: float | None = None
    dac_num_quantizers: int | None = None
    codec_hop_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.encodec_bandwidth is None:
            payload.pop("encodec_bandwidth", None)
        if self.dac_num_quantizers is None:
            payload.pop("dac_num_quantizers", None)
        if self.codec_hop_length is None:
            payload.pop("codec_hop_length", None)
        return payload


def _legacy_dac_hop_length(mapping: Mapping[str, Any]) -> int | None:
    codec_family = infer_codec_family(
        codec_family=None if mapping.get("codec_family") is None else str(mapping.get("codec_family")),
        codec_model_id=None if mapping.get("codec_model_id") is None else str(mapping.get("codec_model_id")),
    )
    codec_model_id = str(mapping.get("codec_model_id", "")).strip().lower()
    if codec_family == "dac" and codec_model_id == DEFAULT_DAC_CODEC_MODEL_ID:
        sample_rate = int(mapping.get("codec_sample_rate", mapping.get("sample_rate", 0)) or 0)
        if int(sample_rate) == 44100:
            return 512
    return None


def _resolve_codec_hop_length_from_mapping(mapping: Mapping[str, Any]) -> int | None:
    value = mapping.get("codec_hop_length", mapping.get("hop_length"))
    try:
        hop = int(value)
    except (TypeError, ValueError):
        hop = int(_legacy_dac_hop_length(mapping) or 0)
    if int(hop) <= 0:
        return None
    return int(hop)


def resolve_device(device: str = "auto") -> str:
    device = str(device).strip().lower()
    if device in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


@contextmanager
def _temporary_full_context_encodec(codec_model: Any):
    config = getattr(codec_model, "config", None)
    if config is None:
        yield
        return
    old_chunk_length_s = getattr(config, "chunk_length_s", None)
    old_overlap = getattr(config, "overlap", None)
    try:
        if hasattr(config, "chunk_length_s"):
            config.chunk_length_s = None
        if hasattr(config, "overlap"):
            config.overlap = None
        yield
    finally:
        if hasattr(config, "chunk_length_s"):
            config.chunk_length_s = old_chunk_length_s
        if hasattr(config, "overlap"):
            config.overlap = old_overlap


def infer_codec_family(*, codec_family: str | None = None, codec_model_id: str | None = None) -> str:
    family_eff = str(codec_family or "").strip().lower()
    if family_eff in {"encodec", "dac", "shame"}:
        return family_eff
    model_id = str(codec_model_id or "").strip().lower()
    if model_id.startswith("stabilityai/same") or model_id == "same-l":
        return "shame"
    if "dac" in model_id:
        return "dac"
    return "encodec"


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _maybe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _audio_codec_meta_from_mapping(mapping: Mapping[str, Any] | None) -> AudioCodecMetadata | None:
    if mapping is None:
        return None
    if "codec_family" not in mapping and "encodec_model_id" in mapping:
        return AudioCodecMetadata(
            codec_family="encodec",
            codec_model_id=str(mapping["encodec_model_id"]).strip(),
            codec_sample_rate=int(mapping.get("codec_sample_rate", mapping.get("sample_rate", 32000))),
            codec_audio_channels=int(mapping.get("codec_audio_channels", mapping.get("audio_channels", 1))),
            codec_frame_rate=float(mapping.get("codec_frame_rate", mapping.get("frame_rate", 50.0))),
            codec_codebook_size=int(mapping.get("codec_codebook_size", mapping.get("codebook_size", 2048))),
            codec_num_codebooks=int(mapping.get("codec_num_codebooks", mapping.get("num_codebooks", 4))),
            codec_target_dim=int(mapping.get("codec_target_dim", mapping.get("target_dim", 128))),
            encodec_bandwidth=float(mapping.get("encodec_bandwidth", mapping.get("bandwidth", DEFAULT_ENCODEC_BANDWIDTH))),
            codec_hop_length=_resolve_codec_hop_length_from_mapping(mapping),
        )
    if "codec_family" not in mapping or "codec_model_id" not in mapping:
        return None
    codec_hop_length = _resolve_codec_hop_length_from_mapping(mapping)
    return AudioCodecMetadata(
        codec_family=str(mapping["codec_family"]).strip().lower(),
        codec_model_id=str(mapping["codec_model_id"]).strip(),
        codec_sample_rate=int(mapping["codec_sample_rate"]),
        codec_audio_channels=int(mapping["codec_audio_channels"]),
        codec_frame_rate=float(mapping["codec_frame_rate"]),
        codec_codebook_size=int(mapping["codec_codebook_size"]),
        codec_num_codebooks=int(mapping["codec_num_codebooks"]),
        codec_target_dim=int(mapping["codec_target_dim"]),
        encodec_bandwidth=(
            None
            if mapping.get("encodec_bandwidth") is None
            else float(mapping["encodec_bandwidth"])
        ),
        dac_num_quantizers=(
            None
            if mapping.get("dac_num_quantizers") is None
            else int(mapping["dac_num_quantizers"])
        ),
        codec_hop_length=codec_hop_length,
    )


def resolve_codec_num_codebooks_for_encodec(
    *,
    frame_rate: float,
    codebook_size: int,
    bandwidth: float,
    max_num_codebooks: int,
) -> int:
    bits_per_codebook = math.log2(float(codebook_size))
    raw = int(math.floor((float(bandwidth) * 1000.0) / (float(bits_per_codebook) * float(frame_rate))))
    return int(max(1, min(int(max_num_codebooks), int(raw))))


def _resolve_encodec_codebook_size(model: Any) -> int:
    layers = list(getattr(getattr(model, "quantizer", None), "layers", []) or [])
    if not layers:
        raise RuntimeError("encodec model has no quantizer layers")
    embed = torch.as_tensor(layers[0].codebook.embed)
    return int(embed.shape[0])


def _resolve_encodec_target_dim(model: Any) -> int:
    layers = list(getattr(getattr(model, "quantizer", None), "layers", []) or [])
    if not layers:
        raise RuntimeError("encodec model has no quantizer layers")
    embed = torch.as_tensor(layers[0].codebook.embed)
    return int(embed.shape[-1])


def _build_encodec_metadata(
    model: Any,
    *,
    codec_model_id: str,
    bandwidth: float | None = None,
) -> AudioCodecMetadata:
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("encodec model is missing config")
    sample_rate = int(getattr(config, "sampling_rate"))
    audio_channels = int(getattr(config, "audio_channels", 1))
    if int(audio_channels) != 1:
        raise ValueError(
            f"phase-1 codec support is mono-only; {codec_model_id} reports audio_channels={audio_channels}"
        )
    frame_rate = float(getattr(config, "frame_rate"))
    codebook_size = int(getattr(config, "codebook_size", _resolve_encodec_codebook_size(model)))
    max_num_codebooks = int(len(list(getattr(getattr(model, "quantizer", None), "layers", []) or [])))
    if int(max_num_codebooks) <= 0:
        raise RuntimeError("encodec model has no quantizer layers")
    target_bandwidths = list(getattr(config, "target_bandwidths", []) or [])
    bandwidth_eff = (
        float(bandwidth)
        if bandwidth is not None
        else float(target_bandwidths[-1] if target_bandwidths else DEFAULT_ENCODEC_BANDWIDTH)
    )
    num_codebooks = resolve_codec_num_codebooks_for_encodec(
        frame_rate=float(frame_rate),
        codebook_size=int(codebook_size),
        bandwidth=float(bandwidth_eff),
        max_num_codebooks=int(max_num_codebooks),
    )
    return AudioCodecMetadata(
        codec_family="encodec",
        codec_model_id=str(codec_model_id),
        codec_sample_rate=int(sample_rate),
        codec_audio_channels=int(audio_channels),
        codec_frame_rate=float(frame_rate),
        codec_codebook_size=int(codebook_size),
        codec_num_codebooks=int(num_codebooks),
        codec_target_dim=int(_resolve_encodec_target_dim(model)),
        encodec_bandwidth=float(bandwidth_eff),
    )


def _build_dac_metadata(
    model: Any,
    *,
    codec_model_id: str,
    num_quantizers: int | None = None,
) -> AudioCodecMetadata:
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("dac model is missing config")
    sample_rate = int(getattr(config, "sampling_rate"))
    audio_channels = int(getattr(config, "audio_channels", 1) or 1)
    if int(audio_channels) != 1:
        raise ValueError(
            f"phase-1 codec support is mono-only; {codec_model_id} reports audio_channels={audio_channels}"
    )
    frame_rate = float(getattr(config, "frame_rate"))
    hop_length = int(getattr(config, "hop_length", 0) or 0)
    if int(hop_length) <= 0 and str(codec_model_id).strip().lower() == DEFAULT_DAC_CODEC_MODEL_ID:
        if int(sample_rate) == 44100:
            hop_length = 512
    if int(hop_length) > 0:
        frame_rate = float(sample_rate) / float(hop_length)
    max_codebooks = int(getattr(config, "n_codebooks"))
    num_quantizers_eff = int(max(1, min(int(max_codebooks), int(num_quantizers or max_codebooks))))
    return AudioCodecMetadata(
        codec_family="dac",
        codec_model_id=str(codec_model_id),
        codec_sample_rate=int(sample_rate),
        codec_audio_channels=int(audio_channels),
        codec_frame_rate=float(frame_rate),
        codec_codebook_size=int(getattr(config, "codebook_size")),
        codec_num_codebooks=int(num_quantizers_eff),
        codec_target_dim=int(getattr(config, "hidden_size")),
        dac_num_quantizers=int(num_quantizers_eff),
        codec_hop_length=int(hop_length) if int(hop_length) > 0 else None,
    )


def attach_audio_codec_metadata(model: Any, metadata: AudioCodecMetadata) -> Any:
    setattr(model, "_pca_diffusion_codec_metadata", metadata.to_dict())
    return model


def get_audio_codec_metadata(model: Any) -> AudioCodecMetadata | None:
    attached = getattr(model, "_pca_diffusion_codec_metadata", None)
    meta = _audio_codec_meta_from_mapping(attached if isinstance(attached, Mapping) else None)
    if meta is not None:
        return meta

    config = getattr(model, "config", None)
    if config is None:
        return None
    model_type = str(getattr(config, "model_type", "")).strip().lower()
    if model_type == "dac":
        try:
            return _build_dac_metadata(model, codec_model_id="descript/dac_44khz")
        except Exception:
            return None
    try:
        return _build_encodec_metadata(model, codec_model_id=DEFAULT_CODEC_MODEL_ID)
    except Exception:
        return None


def load_audio_codec_model(
    *,
    codec_family: str | None = None,
    codec_model_id: str | None = None,
    device: str = "auto",
    encodec_bandwidth: float | None = None,
    dac_num_quantizers: int | None = None,
    metadata: AudioCodecMetadata | Mapping[str, Any] | None = None,
) -> tuple[Any, str, AudioCodecMetadata]:
    metadata_mapping = dict(metadata) if isinstance(metadata, Mapping) else None
    if metadata is not None:
        meta_obj = metadata if isinstance(metadata, AudioCodecMetadata) else _audio_codec_meta_from_mapping(metadata)
        if meta_obj is None:
            raise ValueError("metadata must be AudioCodecMetadata or a mapping with codec_* keys")
        codec_family = str(meta_obj.codec_family)
        codec_model_id = str(meta_obj.codec_model_id)
        encodec_bandwidth = meta_obj.encodec_bandwidth
        dac_num_quantizers = meta_obj.dac_num_quantizers

    family = infer_codec_family(codec_family=codec_family, codec_model_id=codec_model_id)
    model_id = str(
        codec_model_id
        or (
            DEFAULT_DAC_CODEC_MODEL_ID
            if family == "dac"
            else DEFAULT_SHAME_CODEC_MODEL_ID
            if family == "shame"
            else DEFAULT_CODEC_MODEL_ID
        )
    ).strip()
    model_path = Path(model_id).expanduser()
    if model_id and not model_path.is_absolute():
        packaged_model_path = PACKAGE_ROOT / model_path
        if packaged_model_path.exists():
            model_id = str(packaged_model_path)
    resolved_device = resolve_device(device)
    torch_device = torch.device(resolved_device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device)

    def _load_pretrained_model(model_cls: Any) -> Any:
        try:
            return model_cls.from_pretrained(model_id, local_files_only=True)
        except Exception as local_exc:
            raise RuntimeError(
                f"Could not load codec model {model_id!r} from local files. "
                "The anonymous demo package is expected to include all runtime model files."
            ) from local_exc

    if family == "encodec":
        try:
            from transformers import EncodecModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not import transformers.EncodecModel. Check the local transformers installation."
            ) from exc
        model = _load_pretrained_model(EncodecModel).to(resolved_device).eval()
        meta_obj = _build_encodec_metadata(model, codec_model_id=model_id, bandwidth=encodec_bandwidth)
    elif family == "dac":
        try:
            from transformers import DacModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not import transformers.DacModel. Check the local transformers installation."
            ) from exc
        model = _load_pretrained_model(DacModel).to(resolved_device).eval()
        meta_obj = _build_dac_metadata(model, codec_model_id=model_id, num_quantizers=dac_num_quantizers)
    elif family == "shame":
        try:
            from SHAME.shame_same import (
                build_shame_codec_metadata,
                load_same_model,
                same_latent_dim,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "SHAME/SAME codec metadata was requested, but stable_audio_tools is not available in this Python. "
                "Install stable_audio_tools in the active Python environment."
            ) from exc
        same_model_id = str((metadata_mapping or {}).get("same_model_id") or model_id or DEFAULT_SHAME_CODEC_MODEL_ID)
        try:
            model, same_model_config, resolved_device, same_use_half = load_same_model(
                model_id=same_model_id,
                device=resolved_device,
                half="auto",
            )
        except ModuleNotFoundError as exc:  # pragma: no cover
            if str(getattr(exc, "name", "")) != "stable_audio_tools":
                raise
            raise RuntimeError(
                "SHAME/SAME codec metadata was requested, but stable_audio_tools is not available in this Python. "
                "Install stable_audio_tools in the active Python environment."
            ) from exc
        if metadata is not None:
            meta_obj = metadata if isinstance(metadata, AudioCodecMetadata) else _audio_codec_meta_from_mapping(metadata)
            if meta_obj is None:
                raise ValueError("metadata must be AudioCodecMetadata or a mapping with codec_* keys")
            loaded_dim = int(same_latent_dim(model, same_model_config, fallback=int(meta_obj.codec_target_dim)))
            if int(meta_obj.codec_target_dim) != int(loaded_dim):
                raise ValueError(
                    f"SAME latent dim mismatch: metadata={int(meta_obj.codec_target_dim)} loaded_model={int(loaded_dim)}"
                )
        else:
            nested_model_cfg = same_model_config.get("model") if isinstance(same_model_config, Mapping) else None
            nested_latent_dim = (
                int(nested_model_cfg["latent_dim"])
                if isinstance(nested_model_cfg, Mapping) and nested_model_cfg.get("latent_dim") is not None
                else None
            )
            latent_dim = int(same_latent_dim(model, same_model_config, fallback=nested_latent_dim))
            meta_obj = _audio_codec_meta_from_mapping(
                build_shame_codec_metadata(
                    model=model,
                    model_config=same_model_config,
                    model_id=same_model_id,
                    latent_dim=int(latent_dim),
                )
            )
            if meta_obj is None:
                raise RuntimeError("could not build SHAME codec metadata")
        setattr(model, "_pca_diffusion_same_model_config", dict(same_model_config or {}))
        setattr(model, "_pca_diffusion_same_device", str(resolved_device))
        setattr(model, "_pca_diffusion_same_use_half", bool(same_use_half))
    else:  # pragma: no cover
        raise ValueError(f"unsupported codec_family={family!r}")

    for param in model.parameters():
        param.requires_grad_(False)
    attach_audio_codec_metadata(model, meta_obj)
    return model, resolved_device, meta_obj


def load_encodec_model(model_id: str = DEFAULT_CODEC_MODEL_ID, device: str = "auto") -> tuple[Any, str]:
    model, resolved_device, _metadata = load_audio_codec_model(
        codec_family="encodec",
        codec_model_id=str(model_id),
        device=device,
    )
    return model, resolved_device


def _resolve_codec_family_from_model(model: Any, metadata: AudioCodecMetadata | None = None) -> str:
    if metadata is not None:
        return str(metadata.codec_family)
    config = getattr(model, "config", None)
    if str(getattr(config, "model_type", "")).strip().lower() == "dac":
        return "dac"
    if callable(getattr(model, "decode_audio", None)) and callable(getattr(model, "encode_audio", None)):
        return "shame"
    return "encodec"


def resolve_audio_codec_sample_rate(model: Any, default: int = 32000) -> int:
    meta_obj = get_audio_codec_metadata(model)
    if meta_obj is not None and int(meta_obj.codec_sample_rate) > 0:
        return int(meta_obj.codec_sample_rate)
    config = getattr(model, "config", None)
    value = getattr(config, "sampling_rate", None)
    if value is None:
        return int(default)
    return int(value)


def extract_codebook_embeddings(
    codec_model: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    metadata: AudioCodecMetadata | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    meta_obj = (
        metadata
        if isinstance(metadata, AudioCodecMetadata)
        else _audio_codec_meta_from_mapping(metadata if isinstance(metadata, Mapping) else None)
    ) or get_audio_codec_metadata(codec_model)
    family = _resolve_codec_family_from_model(codec_model, meta_obj)

    if family == "shame":
        raise ValueError("SHAME/SAME is a continuous latent codec and does not expose RVQ codebook embeddings")
    if family == "dac":
        quantizers = list(getattr(getattr(codec_model, "quantizer", None), "quantizers", []) or [])
        if not quantizers:
            raise RuntimeError("dac model has no quantizer modules")
        num_codebooks = int(meta_obj.codec_num_codebooks if meta_obj is not None else len(quantizers))
        embeds: list[torch.Tensor] = []
        for quantizer in list(quantizers[: int(num_codebooks)]):
            base_embed = torch.as_tensor(
                quantizer.codebook.weight.detach(),
                dtype=dtype,
                device=device,
            ).transpose(0, 1).unsqueeze(0)
            projected = quantizer.out_proj(base_embed).squeeze(0).transpose(0, 1).contiguous()
            embeds.append(projected)
        return torch.stack(embeds, dim=0).contiguous()

    layers = list(getattr(getattr(codec_model, "quantizer", None), "layers", []) or [])
    if not layers:
        raise RuntimeError("encodec model has no quantizer layers")
    num_codebooks = int(meta_obj.codec_num_codebooks if meta_obj is not None else len(layers))
    embeds = [
        layer.codebook.embed.detach().to(device=device, dtype=dtype).contiguous()
        for layer in list(layers[: int(num_codebooks)])
    ]
    return torch.stack(embeds, dim=0).contiguous()


def extract_dac_latent_subspace_basis(
    codec_model: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    metadata: AudioCodecMetadata | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    diagnostics = compute_dac_latent_subspace_diagnostics(
        codec_model,
        device=device,
        dtype=dtype,
        metadata=metadata,
    )
    basis_dq = torch.as_tensor(diagnostics["basis_dq"], dtype=dtype, device=device)
    rank = int(diagnostics["rank"])
    if int(rank) <= 0:
        raise RuntimeError("DAC latent subspace basis rank is zero")
    return basis_dq[:, : int(rank)].contiguous()


def compute_dac_latent_subspace_diagnostics(
    codec_model: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    metadata: AudioCodecMetadata | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    meta_obj = (
        metadata
        if isinstance(metadata, AudioCodecMetadata)
        else _audio_codec_meta_from_mapping(metadata if isinstance(metadata, Mapping) else None)
    ) or get_audio_codec_metadata(codec_model)
    family = _resolve_codec_family_from_model(codec_model, meta_obj)
    if family != "dac":
        raise ValueError(f"DAC latent subspace basis requires codec_family='dac', got {family!r}")
    quantizers = list(getattr(getattr(codec_model, "quantizer", None), "quantizers", []) or [])
    if not quantizers:
        raise RuntimeError("dac model has no quantizer modules")
    num_codebooks = int(meta_obj.codec_num_codebooks if meta_obj is not None else len(quantizers))
    mats = []
    for quantizer in list(quantizers[: int(num_codebooks)]):
        weight = torch.as_tensor(
            quantizer.out_proj.weight.detach(),
            dtype=dtype,
            device=device,
        ).squeeze(-1)
        if int(weight.dim()) != 2:
            raise RuntimeError(f"unexpected DAC out_proj weight shape: {tuple(weight.shape)}")
        mats.append(weight.contiguous())
    stacked = torch.cat(mats, dim=1).contiguous()
    stacked_f32 = stacked.to(dtype=torch.float32)
    singular_values = torch.linalg.svdvals(stacked_f32)
    max_sv = float(singular_values.max().item()) if int(singular_values.numel()) > 0 else 0.0
    rank_tol = float(max_sv * float(max(stacked_f32.shape)) * float(torch.finfo(torch.float32).eps))
    rank = int(torch.count_nonzero(singular_values > float(rank_tol)).item())
    if int(rank) <= 0:
        raise RuntimeError("DAC latent subspace basis rank is zero")
    basis_dq, _r = torch.linalg.qr(stacked, mode="reduced")
    first_discarded = (
        float(singular_values[int(rank)].item())
        if int(rank) < int(singular_values.numel())
        else 0.0
    )
    return {
        "basis_dq": basis_dq[:, : int(rank)].contiguous(),
        "rank": int(rank),
        "rank_tolerance": float(rank_tol),
        "matrix_shape": [int(dim) for dim in stacked_f32.shape],
        "singular_value_max": float(max_sv),
        "singular_value_min_retained": float(singular_values[int(rank) - 1].item()),
        "singular_value_first_discarded": float(first_discarded),
    }


def _flatten_encodec_audio_codes(audio_codes_fbqt: torch.Tensor) -> torch.Tensor:
    if int(audio_codes_fbqt.dim()) != 4:
        raise RuntimeError(f"unexpected EnCodec code tensor shape: {tuple(audio_codes_fbqt.shape)}")
    return (
        audio_codes_fbqt.permute(1, 2, 0, 3)
        .contiguous()
        .view(int(audio_codes_fbqt.shape[1]), int(audio_codes_fbqt.shape[2]), -1)
    )


def encode_audio_batch_to_codes(
    codec_model: Any,
    audio_bct: torch.Tensor,
    *,
    device: torch.device | str,
    metadata: AudioCodecMetadata | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    audio = torch.as_tensor(audio_bct, dtype=torch.float32, device=device)
    if int(audio.dim()) == 2:
        audio = audio.unsqueeze(0)
    if int(audio.dim()) != 3:
        raise ValueError(f"audio_bct must be [B,C,T] or [C,T], got {tuple(audio.shape)}")
    meta_obj = (
        metadata
        if isinstance(metadata, AudioCodecMetadata)
        else _audio_codec_meta_from_mapping(metadata if isinstance(metadata, Mapping) else None)
    ) or get_audio_codec_metadata(codec_model)
    family = _resolve_codec_family_from_model(codec_model, meta_obj)

    if family == "shame":
        raise ValueError("SHAME/SAME is a continuous latent codec and does not expose discrete audio codes")
    if family == "dac":
        encoded = codec_model.encode(
            audio,
            n_quantizers=None if meta_obj is None else meta_obj.dac_num_quantizers,
        )
        audio_codes = getattr(encoded, "audio_codes", None)
        if audio_codes is None and isinstance(encoded, tuple) and len(encoded) >= 3:
            audio_codes = encoded[2]
        if audio_codes is None:
            raise RuntimeError("DAC encode did not return audio_codes")
        audio_codes = torch.as_tensor(audio_codes, dtype=torch.long, device=device)
        if int(audio_codes.dim()) != 3:
            raise RuntimeError(f"unexpected DAC code tensor shape: {tuple(audio_codes.shape)}")
        return audio_codes.contiguous()

    padding_mask = torch.ones_like(audio, dtype=torch.bool, device=device)
    with _temporary_full_context_encodec(codec_model):
        encoded = codec_model.encode(
            audio,
            padding_mask=padding_mask,
            bandwidth=None if meta_obj is None else meta_obj.encodec_bandwidth,
        )
    audio_codes = getattr(encoded, "audio_codes", None)
    if audio_codes is None and isinstance(encoded, tuple) and len(encoded) >= 1:
        audio_codes = encoded[0]
    if audio_codes is None:
        raise RuntimeError("Encodec encode did not return audio_codes")
    audio_codes = torch.as_tensor(audio_codes, dtype=torch.long, device=device)
    return _flatten_encodec_audio_codes(audio_codes).contiguous()


def encode_audio_chunks_to_codes(
    codec_model: Any,
    chunks_ct: list[torch.Tensor],
    *,
    device: torch.device | str,
    metadata: AudioCodecMetadata | Mapping[str, Any] | None = None,
) -> list[torch.Tensor]:
    if not chunks_ct:
        return []
    lengths = [int(torch.as_tensor(chunk).shape[-1]) for chunk in chunks_ct]
    expected_len = int(lengths[0])
    if any(int(length) != int(expected_len) for length in lengths):
        raise ValueError("all chunks_ct must share the same sample length")
    batch = torch.stack(
        [torch.as_tensor(chunk, dtype=torch.float32) for chunk in chunks_ct],
        dim=0,
    ).contiguous()
    codes_bct = encode_audio_batch_to_codes(
        codec_model,
        batch,
        device=device,
        metadata=metadata,
    )
    return [
        codes_bct[int(idx)].detach().to(device="cpu", dtype=torch.long).contiguous()
        for idx in range(int(codes_bct.shape[0]))
    ]


def token_ids_to_codebook_embeddings(
    token_ids_bct: torch.Tensor,
    codebook_embed_ckd: torch.Tensor,
    *,
    valid_bt: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(token_ids_bct.dim()) != 3:
        raise ValueError(f"token_ids_bct must be [B,C,T], got {tuple(token_ids_bct.shape)}")
    if int(codebook_embed_ckd.dim()) != 3:
        raise ValueError(f"codebook_embed_ckd must be [C,K,D], got {tuple(codebook_embed_ckd.shape)}")
    batch_size, num_codebooks, _num_frames = [int(x) for x in list(token_ids_bct.shape)]
    if int(codebook_embed_ckd.shape[0]) != int(num_codebooks):
        raise ValueError(
            f"codebook embeddings must match codebook count, got {tuple(codebook_embed_ckd.shape)} "
            f"for token shape {tuple(token_ids_bct.shape)}"
        )
    embeds = []
    for codebook_idx in range(int(num_codebooks)):
        embed_kd = codebook_embed_ckd[int(codebook_idx)].to(device=token_ids_bct.device, dtype=torch.float32)
        token_idx_bt = token_ids_bct[:, int(codebook_idx), :].to(device=token_ids_bct.device, dtype=torch.long)
        token_idx_bt = token_idx_bt.clamp(min=0, max=int(embed_kd.shape[0]) - 1)
        embeds.append(F.embedding(token_idx_bt, embed_kd).unsqueeze(1))
    out = torch.cat(embeds, dim=1).contiguous()
    if valid_bt is not None:
        valid_eff = valid_bt.to(device=out.device, dtype=torch.bool)
        out = out.masked_fill(~valid_eff[:, None, :, None], 0.0)
    return out.contiguous()


def rvq_sum_latents(
    codebook_latents_bctd: torch.Tensor,
    *,
    valid_bt: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(codebook_latents_bctd.dim()) != 4:
        raise ValueError(
            f"codebook_latents_bctd must be [B,C,T,D], got {tuple(codebook_latents_bctd.shape)}"
        )
    summed = codebook_latents_bctd.to(dtype=torch.float32).sum(dim=1)
    if valid_bt is not None:
        summed = summed.masked_fill(~valid_bt.to(device=summed.device, dtype=torch.bool)[:, :, None], 0.0)
    return summed.contiguous()


def summed_frame_latents_from_code_ids(
    token_ids_ct: torch.Tensor,
    codebook_embed_ckd: torch.Tensor,
) -> torch.Tensor:
    if int(token_ids_ct.dim()) != 2:
        raise ValueError(f"token_ids_ct must be [C,T], got {tuple(token_ids_ct.shape)}")
    token_ids_bct = token_ids_ct.unsqueeze(0).to(dtype=torch.long)
    valid_bt = torch.ones((1, int(token_ids_ct.shape[1])), device=token_ids_ct.device, dtype=torch.bool)
    latents_bctd = token_ids_to_codebook_embeddings(
        token_ids_bct,
        codebook_embed_ckd,
        valid_bt=valid_bt,
    )
    return rvq_sum_latents(latents_bctd, valid_bt=valid_bt)[0].to(dtype=torch.float32).contiguous()


def pooled_segment_latent_from_code_ids(
    token_ids_ct: torch.Tensor,
    codebook_embed_ckd: torch.Tensor,
) -> torch.Tensor:
    return summed_frame_latents_from_code_ids(token_ids_ct, codebook_embed_ckd).sum(dim=0).to(dtype=torch.float32)


def _is_cudnn_engine_failure(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "unable to find an engine" in message
        or "cudnn" in message and "engine" in message
    )


def _decode_dac_with_cudnn_fallback(codec_model: Any, **kwargs: Any) -> torch.Tensor:
    try:
        return codec_model.decode(**kwargs).audio_values
    except RuntimeError as exc:
        if not _is_cudnn_engine_failure(exc):
            raise
        with torch.backends.cudnn.flags(enabled=False):
            return codec_model.decode(**kwargs).audio_values


def decode_codes_to_audio_b1t(
    codec_model: Any,
    codes_bct: torch.Tensor,
    *,
    device: torch.device | str,
    metadata: AudioCodecMetadata | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    codes = torch.as_tensor(codes_bct, dtype=torch.long, device=device)
    if int(codes.dim()) == 2:
        codes = codes.unsqueeze(0)
    meta_obj = (
        metadata
        if isinstance(metadata, AudioCodecMetadata)
        else _audio_codec_meta_from_mapping(metadata if isinstance(metadata, Mapping) else None)
    ) or get_audio_codec_metadata(codec_model)
    family = _resolve_codec_family_from_model(codec_model, meta_obj)
    if family == "dac":
        codebook_embed_ckd = extract_codebook_embeddings(
            codec_model,
            device=device,
            metadata=meta_obj,
        )
        valid_bt = codes.ge(0).all(dim=1).to(dtype=torch.bool)
        latents_bctd = token_ids_to_codebook_embeddings(
            codes.clamp_min(0),
            codebook_embed_ckd,
            valid_bt=valid_bt,
        )
        quantized_latent = rvq_sum_latents(latents_bctd, valid_bt=valid_bt)
        return decode_quantized_latent_to_audio(codec_model, quantized_latent)
    if family == "shame":
        raise ValueError("SHAME/SAME is a continuous latent codec and does not expose discrete audio codes")
    audio_codes = codes.unsqueeze(0).contiguous()
    decoded = codec_model.decode(audio_codes, [None], padding_mask=None).audio_values
    audio = decoded.to(device=device, dtype=torch.float32)
    if int(audio.dim()) == 2:
        audio = audio.unsqueeze(1)
    if int(audio.dim()) != 3:
        raise RuntimeError(f"unexpected decoded audio shape: {tuple(audio.shape)}")
    return audio.contiguous()


def decode_quantized_latent_to_audio(
    codec_model: Any,
    latent_btd: torch.Tensor,
) -> torch.Tensor:
    latent = torch.as_tensor(latent_btd, dtype=torch.float32)
    if int(latent.dim()) != 3:
        raise ValueError(f"latent_btd must be [B,T,D], got {tuple(latent.shape)}")
    meta_obj = get_audio_codec_metadata(codec_model)
    family = _resolve_codec_family_from_model(codec_model, meta_obj)
    z_q = latent.transpose(1, 2).contiguous()
    if family == "dac":
        audio = _decode_dac_with_cudnn_fallback(codec_model, quantized_representation=z_q)
        audio = torch.as_tensor(audio, dtype=torch.float32, device=latent.device)
        if int(audio.dim()) == 2:
            audio = audio.unsqueeze(1)
        return audio.contiguous()
    if family == "shame":
        try:
            from SHAME.shame_same import decode_same_latent_to_audio
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "SHAME/SAME decode requires stable_audio_tools. "
                "Install stable_audio_tools in the active Python environment."
            ) from exc
        same_model_config = dict(getattr(codec_model, "_pca_diffusion_same_model_config", {}) or {})
        same_device = str(getattr(codec_model, "_pca_diffusion_same_device", "") or "")
        if not same_device:
            try:
                same_device = str(next(codec_model.parameters()).device)
            except Exception:
                same_device = str(latent.device)
        same_use_half = bool(getattr(codec_model, "_pca_diffusion_same_use_half", False))
        audio = decode_same_latent_to_audio(
            codec_model,
            latent,
            model_config=same_model_config,
            device=same_device,
            use_half=bool(same_use_half),
        )
        return audio.to(device=latent.device, dtype=torch.float32).contiguous()
    return codec_model.decoder(z_q)


def load_target_pca_basis(
    source: str | Path | Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        payload = dict(
            torch.load(Path(source).expanduser().resolve(), map_location="cpu", weights_only=False)
        )
    elif isinstance(source, Mapping):
        payload = dict(source)
    else:
        raise TypeError(f"unsupported PCA basis source: {type(source).__name__}")
    mean = torch.as_tensor(payload["mean"], dtype=dtype, device=device).view(-1).contiguous()
    components = torch.as_tensor(payload["components"], dtype=dtype, device=device).contiguous()
    if int(components.dim()) != 2:
        raise ValueError(f"PCA basis components must be [K,D], got {tuple(components.shape)}")
    if int(components.shape[-1]) != int(mean.shape[0]):
        raise ValueError(
            f"PCA basis mean/components mismatch: mean={tuple(mean.shape)} components={tuple(components.shape)}"
        )
    out: dict[str, Any] = {**payload}
    out["mean"] = mean
    out["components"] = components
    out["k"] = int(payload.get("k", int(components.shape[0])))
    out["target_dim"] = int(payload.get("target_dim", int(components.shape[0])))
    out["full_target_dim"] = int(payload.get("full_target_dim", int(components.shape[-1])))
    out["target_layout"] = str(payload.get("target_layout", PCA_TARGET_LAYOUT)).strip().lower()
    return out


def project_latent_to_pca(
    latent_btd: torch.Tensor,
    pca_basis: Mapping[str, Any],
) -> torch.Tensor:
    latent = torch.as_tensor(latent_btd, dtype=torch.float32)
    basis = load_target_pca_basis(
        pca_basis,
        device=latent.device,
        dtype=latent.dtype,
    )
    mean = torch.as_tensor(basis["mean"], dtype=latent.dtype, device=latent.device)
    components = torch.as_tensor(basis["components"], dtype=latent.dtype, device=latent.device)
    if int(latent.shape[-1]) != int(components.shape[-1]):
        raise ValueError(
            f"latent/PCA basis mismatch: latent={tuple(latent.shape)} components={tuple(components.shape)}"
        )
    return torch.matmul(latent - mean.view(*([1] * (int(latent.dim()) - 1)), -1), components.transpose(0, 1))


def reconstruct_latent_from_pca(
    latent_btd: torch.Tensor,
    pca_basis: Mapping[str, Any] | None,
) -> torch.Tensor:
    latent = torch.as_tensor(latent_btd, dtype=torch.float32)
    if pca_basis is None:
        return latent.contiguous()
    basis = load_target_pca_basis(
        pca_basis,
        device=latent.device,
        dtype=latent.dtype,
    )
    mean = torch.as_tensor(basis["mean"], dtype=latent.dtype, device=latent.device)
    components = torch.as_tensor(basis["components"], dtype=latent.dtype, device=latent.device)
    if int(latent.shape[-1]) != int(components.shape[0]):
        raise ValueError(
            f"latent/PCA basis mismatch: latent={tuple(latent.shape)} components={tuple(components.shape)}"
        )
    return (
        torch.matmul(latent, components)
        + mean.view(*([1] * (int(latent.dim()) - 1)), -1)
    ).contiguous()


@contextmanager
def _temporary_full_context_encodec(codec_model: Any):
    config = getattr(codec_model, "config", None)
    if config is None:
        yield
        return
    old_chunk_length_s = getattr(config, "chunk_length_s", None)
    old_overlap = getattr(config, "overlap", None)
    if hasattr(config, "chunk_length_s"):
        config.chunk_length_s = None
    if hasattr(config, "overlap"):
        config.overlap = None
    try:
        yield
    finally:
        if hasattr(config, "chunk_length_s"):
            config.chunk_length_s = old_chunk_length_s
        if hasattr(config, "overlap"):
            config.overlap = old_overlap

@torch.inference_mode()
def encode_audio_to_codes_bct(
    codec_model: Any,
    audio_bct: torch.Tensor,
    *,
    device: torch.device | str,
    metadata: AudioCodecMetadata | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    return encode_audio_batch_to_codes(
        codec_model,
        audio_bct,
        device=device,
        metadata=metadata,
    )


def requantize_latent_to_codes_bct(
    codec_model: Any,
    latent_btd: torch.Tensor,
    *,
    device: torch.device | str,
    metadata: AudioCodecMetadata | Mapping[str, Any] | None = None,
    target_pca_basis: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    latent = reconstruct_latent_from_pca(
        torch.as_tensor(latent_btd, dtype=torch.float32, device=device),
        target_pca_basis,
    )
    if int(latent.dim()) != 3:
        raise ValueError(f"latent_btd must be [B,T,D], got {tuple(latent.shape)}")
    meta_obj = (
        metadata
        if isinstance(metadata, AudioCodecMetadata)
        else _audio_codec_meta_from_mapping(metadata if isinstance(metadata, Mapping) else None)
    ) or get_audio_codec_metadata(codec_model)
    family = _resolve_codec_family_from_model(codec_model, meta_obj)
    z_q = latent.transpose(1, 2).contiguous()
    if family == "dac":
        _zq, audio_codes, _projected, _commitment, _codebook = codec_model.quantizer(
            z_q,
            None if meta_obj is None else meta_obj.dac_num_quantizers,
        )
        if int(audio_codes.dim()) != 3:
            raise RuntimeError(f"unexpected DAC quantizer output shape: {tuple(audio_codes.shape)}")
        return audio_codes.contiguous().to(dtype=torch.long)
    bandwidth = None if meta_obj is None else meta_obj.encodec_bandwidth
    codes_cbt = codec_model.quantizer.encode(z_q, bandwidth=bandwidth)
    if int(codes_cbt.dim()) != 3:
        raise RuntimeError(f"unexpected quantizer.encode output shape: {tuple(codes_cbt.shape)}")
    if int(codes_cbt.shape[1]) != int(latent.shape[0]):
        raise RuntimeError(
            f"quantizer.encode batch mismatch: latent batch={int(latent.shape[0])}, codes={tuple(codes_cbt.shape)}"
        )
    return codes_cbt.permute(1, 0, 2).contiguous().to(dtype=torch.long)


def legacy_source_cache_codec_metadata() -> AudioCodecMetadata:
    return AudioCodecMetadata(
        codec_family="encodec",
        codec_model_id=DEFAULT_CODEC_MODEL_ID,
        codec_sample_rate=32000,
        codec_audio_channels=1,
        codec_frame_rate=50.0,
        codec_codebook_size=2048,
        codec_num_codebooks=4,
        codec_target_dim=128,
        encodec_bandwidth=DEFAULT_ENCODEC_BANDWIDTH,
    )


def resolve_codec_metadata_from_cache_config(cache_root: str | Path) -> AudioCodecMetadata:
    cache_root_path = Path(cache_root).expanduser().resolve()
    config_payload = _maybe_read_json(cache_root_path / "config.json")
    meta_obj = _audio_codec_meta_from_mapping(config_payload)
    if meta_obj is not None:
        return meta_obj
    return legacy_source_cache_codec_metadata()


def resolve_codec_metadata_from_payload(
    payload: Mapping[str, Any],
    *,
    fallback: AudioCodecMetadata | None = None,
) -> AudioCodecMetadata:
    meta_obj = _audio_codec_meta_from_mapping(payload)
    if meta_obj is not None:
        return meta_obj
    if "codec_metadata" in payload and isinstance(payload["codec_metadata"], Mapping):
        meta_obj = _audio_codec_meta_from_mapping(payload["codec_metadata"])
        if meta_obj is not None:
            return meta_obj
    if fallback is not None:
        return fallback
    return legacy_source_cache_codec_metadata()


def normalize_target_payload(
    payload: Mapping[str, Any],
    *,
    fallback_target_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    target_layout = str(
        payload.get(
            "target_layout",
            PCA_TARGET_LAYOUT if payload.get("target_pc_tk") is not None else DEFAULT_TARGET_LAYOUT,
        )
        or DEFAULT_TARGET_LAYOUT
    ).strip().lower()
    if target_layout == PCA_TARGET_LAYOUT:
        target_pc_tk = payload.get("target_pc_tk")
        target_pc_pool_k = payload.get("target_pc_pool_k")
        if target_pc_tk is None:
            raise KeyError("cache payload is missing target_pc_tk for framewise_pca targets")
        target_pc_tk_t = torch.as_tensor(target_pc_tk, dtype=torch.float32).contiguous()
        if target_pc_pool_k is None:
            target_pc_pool_k_t = target_pc_tk_t.sum(dim=0).to(dtype=torch.float32).contiguous()
        else:
            target_pc_pool_k_t = torch.as_tensor(target_pc_pool_k, dtype=torch.float32).contiguous()
        if int(target_pc_tk_t.dim()) != 2:
            raise RuntimeError(f"expected target_pc_tk [T,K], got {tuple(target_pc_tk_t.shape)}")
        if int(target_pc_pool_k_t.dim()) != 1:
            raise RuntimeError(f"expected target_pc_pool_k [K], got {tuple(target_pc_pool_k_t.shape)}")
        target_dim = int(payload.get("target_dim", fallback_target_dim or int(target_pc_tk_t.shape[-1])))
        if int(target_pc_tk_t.shape[-1]) != int(target_dim):
            raise RuntimeError(
                f"target_dim mismatch: target_pc_tk has {target_pc_tk_t.shape[-1]} dims vs target_dim={target_dim}"
            )
        if int(target_pc_pool_k_t.shape[0]) != int(target_dim):
            raise RuntimeError(
                f"target_pc_pool_k mismatch: target_pc_pool_k has {target_pc_pool_k_t.shape[0]} dims vs target_dim={target_dim}"
            )
        return target_pc_tk_t, target_pc_pool_k_t, int(target_dim)

    target_sum_td = payload.get("target_sum_td", payload.get("target_sum_t128"))
    target_sum_pool_d = payload.get("target_sum_pool_d", payload.get("target_sum_pool_128"))
    if target_sum_td is None or target_sum_pool_d is None:
        missing = []
        if target_sum_td is None:
            missing.append("target_sum_td")
        if target_sum_pool_d is None:
            missing.append("target_sum_pool_d")
        raise KeyError(f"cache payload is missing target tensors {missing}")
    target_sum_td_t = torch.as_tensor(target_sum_td, dtype=torch.float32).contiguous()
    target_sum_pool_d_t = torch.as_tensor(target_sum_pool_d, dtype=torch.float32).contiguous()
    if int(target_sum_td_t.dim()) != 2:
        raise RuntimeError(f"expected target_sum_td [T,D], got {tuple(target_sum_td_t.shape)}")
    if int(target_sum_pool_d_t.dim()) != 1:
        raise RuntimeError(f"expected target_sum_pool_d [D], got {tuple(target_sum_pool_d_t.shape)}")
    target_dim = int(payload.get("target_dim", fallback_target_dim or int(target_sum_td_t.shape[-1])))
    if int(target_sum_td_t.shape[-1]) != int(target_dim):
        raise RuntimeError(
            f"target_dim mismatch: target_sum_td has {target_sum_td_t.shape[-1]} dims vs target_dim={target_dim}"
        )
    if int(target_sum_pool_d_t.shape[0]) != int(target_dim):
        raise RuntimeError(
            f"target_sum_pool_d mismatch: target_sum_pool_d has {target_sum_pool_d_t.shape[0]} dims vs target_dim={target_dim}"
        )
    return target_sum_td_t, target_sum_pool_d_t, int(target_dim)


def resolve_target_dim_from_cache_config(cache_root: str | Path, default: int = 128) -> int:
    cache_root_path = Path(cache_root).expanduser().resolve()
    config_payload = _maybe_read_json(cache_root_path / "config.json") or {}
    if "target_dim" in config_payload:
        return int(config_payload["target_dim"])
    return int(default)


def resolve_target_layout_from_cache_config(
    cache_root: str | Path,
    default: str = DEFAULT_TARGET_LAYOUT,
) -> str:
    cache_root_path = Path(cache_root).expanduser().resolve()
    config_payload = _maybe_read_json(cache_root_path / "config.json") or {}
    layout = str(config_payload.get("target_layout") or "").strip().lower()
    return layout or str(default).strip().lower()


def resolve_target_pca_basis_path_from_cache_config(cache_root: str | Path) -> Path | None:
    cache_root_path = Path(cache_root).expanduser().resolve()
    config_payload = _maybe_read_json(cache_root_path / "config.json") or {}
    rel_path = str(config_payload.get("pca_basis_path") or "").strip()
    if rel_path:
        candidate = (cache_root_path / rel_path).resolve()
        if candidate.is_file():
            return candidate
    fallback = (cache_root_path / "pca_basis.pt").resolve()
    if fallback.is_file():
        return fallback
    return None
