"""Core TTS generation service wrapping VibeVoice model."""

import torch
import numpy as np
from typing import Iterator, List, Optional, Union
from transformers import set_seed
import logging

logger = logging.getLogger(__name__)

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

from api.config import Settings
from api.services.chunking import split_script_into_chunks


class TTSService:
    """Service for TTS generation using VibeVoice model."""
    
    def __init__(self, settings: Settings):
        """
        Initialize TTS service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self._model_loaded = False
    
    def load_model(self):
        """Load VibeVoice model and processor."""
        if self._model_loaded:
            print("Model already loaded")
            return
        
        print(f"Loading VibeVoice model from {self.settings.vibevoice_model_path}")
        
        # Get device and dtype
        self.device = self.settings.get_device()
        self.dtype = self.settings.get_dtype()
        attn_implementation = self.settings.get_attn_implementation()
        
        print(f"Using device: {self.device}, dtype: {self.dtype}, attention: {attn_implementation}")

        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.settings.vibevoice_model_path)

        # Determine if we should load to CPU first for quantization
        # This avoids loading full precision model to GPU then quantizing (wastes VRAM)
        load_to_cpu_first = (
            self.settings.vibevoice_quantization
            and self.device == "cuda"
        )

        if load_to_cpu_first:
            print("Loading model to CPU first for quantization (saves GPU memory)...")
            # Use sdpa for CPU loading since flash_attention_2 requires CUDA
            cpu_attn = "sdpa" if attn_implementation == "flash_attention_2" else attn_implementation
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.settings.vibevoice_model_path,
                torch_dtype=self.dtype,
                device_map="cpu",
                attn_implementation=cpu_attn,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            # Apply quantization on CPU
            self._apply_quantization()

            # Now move to CUDA
            print("Moving quantized model to CUDA...")
            self.model = self.model.to("cuda")

            # Log final VRAM usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_final = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"Final VRAM usage after moving to GPU: {vram_final:.2f} GB")
        else:
            # Standard loading path (no quantization or non-CUDA device)
            try:
                if self.device == "mps":
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.settings.vibevoice_model_path,
                        torch_dtype=self.dtype,
                        attn_implementation=attn_implementation,
                        device_map=None,
                    )
                    self.model.to("mps")
                elif self.device == "cuda":
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.settings.vibevoice_model_path,
                        torch_dtype=self.dtype,
                        device_map="cuda",
                        attn_implementation=attn_implementation,
                    )
                else:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.settings.vibevoice_model_path,
                        torch_dtype=self.dtype,
                        device_map="cpu",
                        attn_implementation=attn_implementation,
                    )
            except Exception as e:
                if attn_implementation == 'flash_attention_2':
                    print(f"Flash attention failed: {e}")
                    print("Falling back to SDPA attention")
                    attn_implementation = "sdpa"

                    if self.device == "mps":
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            attn_implementation=attn_implementation,
                            device_map=None,
                        )
                        self.model.to("mps")
                    elif self.device == "cuda":
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            device_map="cuda",
                            attn_implementation=attn_implementation,
                        )
                    else:
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            device_map="cpu",
                            attn_implementation=attn_implementation,
                        )
                else:
                    raise e

            self.model.eval()

        # Apply torch.compile for optimized inference
        if self.settings.torch_compile:
            try:
                compile_mode = self.settings.torch_compile_mode
                self.model = torch.compile(self.model, mode=compile_mode, dynamic=True)
                print(f"Model compiled with torch.compile(mode='{compile_mode}', dynamic=True)")
            except Exception as e:
                print(f"torch.compile() failed: {e}, continuing without compilation")

        # Use the model's default noise scheduler (DPMSolverMultistepScheduler
        # with the trained beta_schedule and the deterministic dpmsolver++
        # algorithm). Earlier versions overrode this to sde-dpmsolver++ +
        # squaredcos_cap_v2 — that injected stochastic noise at every reverse
        # step (audible as "bleed" at the start of segments) and replaced the
        # trained beta schedule (audible as poor pronunciation/accent). The
        # ComfyUI front-end and Microsoft's reference inference demo both use
        # the model defaults.

        # Set inference steps
        self.model.set_ddpm_inference_steps(num_steps=self.settings.vibevoice_inference_steps)
        
        self._model_loaded = True
        print("Model loaded successfully")

    def _apply_quantization(self):
        """Apply quantization to the model based on settings."""
        quant_method = self.settings.vibevoice_quantization

        if quant_method == "int8_torchao":
            self._apply_torchao_quant(bits=8)
        elif quant_method == "int4_torchao":
            self._apply_torchao_quant(bits=4)
        else:
            logger.warning(f"Unknown quantization method: {quant_method}, skipping quantization")

    def _apply_torchao_quant(self, bits: int = 8):
        """
        Apply torchao weight-only quantization to the language model.

        This selectively quantizes only the LLM (Qwen2) decoder and lm_head,
        keeping audio components (tokenizers, diffusion head, connectors) at full precision.

        Args:
            bits: 8 for INT8 (~40% VRAM reduction) or 4 for INT4 (~60% VRAM reduction, faster)
        """
        try:
            from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
        except ImportError:
            logger.error(
                "torchao not installed. Install with: pip install torchao\n"
                "Falling back to full precision."
            )
            return

        # Select quantization function based on bits
        if bits == 4:
            quant_fn = int4_weight_only()
            quant_name = "INT4"
        else:
            quant_fn = int8_weight_only()
            quant_name = "INT8"

        # Check if model is on CUDA (for memory logging)
        model_on_cuda = next(self.model.parameters()).is_cuda

        logger.info(f"Applying torchao {quant_name} weight-only quantization...")
        if model_on_cuda:
            logger.info("Model is on CUDA - quantizing in place")
        else:
            logger.info("Model is on CPU - quantizing before moving to GPU (saves VRAM)")

        # Quantize only the language model (Qwen2 decoder) - this is the largest component
        # The audio components (acoustic_tokenizer, semantic_tokenizer, prediction_head, connectors)
        # are kept at full precision to maintain audio quality
        try:
            logger.info("Quantizing language_model (Qwen2 decoder)...")
            quantize_(self.model.model.language_model, quant_fn)

            logger.info("Quantizing lm_head...")
            quantize_(self.model.lm_head, quant_fn)

        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            logger.info("Continuing with full precision model")
            return

        logger.info(f"{quant_name} quantization applied successfully")

        # Force garbage collection
        import gc
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def generate_speech(
        self,
        text: str,
        voice_samples: List[np.ndarray],
        cfg_scale: float = 1.3,
        inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        stream: bool = False,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_words_per_chunk: int = 0,
        chunk_silence_ms: int = 0,
        voice_sources: Optional[List[str]] = None,
    ) -> Union[np.ndarray, Iterator[np.ndarray]]:
        """
        Generate speech from text.

        Args:
            text: Input text (formatted with Speaker labels)
            voice_samples: List of voice sample arrays
            cfg_scale: Classifier-free guidance scale
            inference_steps: Number of diffusion steps (None = use default)
            seed: Random seed for reproducibility
            stream: Whether to return streaming iterator
            do_sample: Use sampling (True) vs greedy (False). Auto-True if
                temperature/top_p provided and this is None.
            temperature: Sampling temperature (only used when do_sample=True)
            top_p: Nucleus sampling top_p (only used when do_sample=True)
            max_words_per_chunk: If > 0, split long scripts into chunks of at
                most this many words and synthesize sequentially. 0 disables.
            chunk_silence_ms: Silence (ms) inserted between concatenated chunks
                in non-streaming mode. 0 disables.
            voice_sources: Optional list of human-readable source identifiers
                (preset key, file path, or "base64:N bytes") for each voice
                sample, in the same order as `voice_samples`. Used only for
                debug logging.

        Returns:
            Generated audio array or iterator of audio chunks
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if seed is not None:
            set_seed(seed)

        if inference_steps is not None:
            self.model.set_ddpm_inference_steps(num_steps=inference_steps)

        if do_sample is None:
            do_sample = temperature is not None or top_p is not None

        chunks = split_script_into_chunks(text, max_words_per_chunk)
        if len(chunks) > 1:
            logger.info(f"Split script into {len(chunks)} chunks (max {max_words_per_chunk} words/chunk)")

        self._log_generation_request(
            text=text,
            voice_samples=voice_samples,
            voice_sources=voice_sources,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            seed=seed,
            stream=stream,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_words_per_chunk=max_words_per_chunk,
            chunk_silence_ms=chunk_silence_ms,
            num_chunks=len(chunks),
        )

        if stream:
            return self._generate_streaming_chunks(
                chunks, voice_samples, cfg_scale, do_sample, temperature, top_p
            )

        audio_pieces: List[np.ndarray] = []
        silence = (
            np.zeros(int(24000 * chunk_silence_ms / 1000.0), dtype=np.float32)
            if chunk_silence_ms > 0 else None
        )
        for i, chunk_text in enumerate(chunks):
            audio_pieces.append(
                self._generate_full(chunk_text, voice_samples, cfg_scale, do_sample, temperature, top_p)
            )
            if silence is not None and i < len(chunks) - 1:
                audio_pieces.append(silence)

        if len(audio_pieces) == 1:
            return audio_pieces[0]
        # Ensure consistent shape (1D) for concatenation
        flat = [a.reshape(-1) if a.ndim > 1 else a for a in audio_pieces]
        return np.concatenate(flat)

    def _log_generation_request(
        self,
        text: str,
        voice_samples: List[np.ndarray],
        voice_sources: Optional[List[str]],
        cfg_scale: float,
        inference_steps: Optional[int],
        seed: Optional[int],
        stream: bool,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        max_words_per_chunk: int,
        chunk_silence_ms: int,
        num_chunks: int,
    ) -> None:
        """Emit a DEBUG line summarising the exact params used for generation.

        Mirrors the call we will make to model.generate() and dumps voice
        sample stats so quality issues (bleed, accent, level) can be triaged
        from logs alone.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        effective_steps = (
            inference_steps if inference_steps is not None
            else self.settings.vibevoice_inference_steps
        )
        sample_rate = 24000

        sample_lines = []
        for i, sample in enumerate(voice_samples):
            src = voice_sources[i] if voice_sources and i < len(voice_sources) else "?"
            try:
                arr = np.asarray(sample)
                n = int(arr.size)
                duration_s = n / sample_rate
                peak = float(np.max(np.abs(arr))) if n else 0.0
                rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))) if n else 0.0
                dtype = arr.dtype
                shape = arr.shape
            except Exception as e:
                sample_lines.append(f"  speaker[{i}] src={src} <stat-error: {e}>")
                continue
            sample_lines.append(
                f"  speaker[{i}] src={src} samples={n} duration={duration_s:.3f}s "
                f"sr={sample_rate} peak={peak:.4f} rms={rms:.4f} dtype={dtype} shape={shape}"
            )

        text_preview = text if len(text) <= 200 else (text[:200] + "...")

        logger.debug(
            "TTS generate() params:\n"
            f"  device={self.device} dtype={self.dtype} model={self.settings.vibevoice_model_path}\n"
            f"  cfg_scale={cfg_scale} inference_steps={effective_steps} (request={inference_steps}) "
            f"seed={seed} stream={stream}\n"
            f"  do_sample={bool(do_sample)} temperature={temperature} top_p={top_p}\n"
            f"  max_words_per_chunk={max_words_per_chunk} chunk_silence_ms={chunk_silence_ms} "
            f"num_chunks={num_chunks}\n"
            f"  refresh_negative=True return_speech=True verbose=False show_progress_bar=False\n"
            f"  voice_samples ({len(voice_samples)}):\n" + "\n".join(sample_lines) + "\n"
            f"  text_preview={text_preview!r}"
        )

    def _build_inputs(self, text: str, voice_samples: List[np.ndarray]) -> dict:
        inputs = self.processor(
            text=[text],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        target_device = self.device if self.device in ("cuda", "mps") else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)
        return inputs

    def _build_generation_config(
        self,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> dict:
        cfg: dict = {"do_sample": bool(do_sample)}
        if do_sample:
            if temperature is not None:
                cfg["temperature"] = float(temperature)
            if top_p is not None:
                cfg["top_p"] = float(top_p)
        return cfg

    def _generate_full(
        self,
        text: str,
        voice_samples: List[np.ndarray],
        cfg_scale: float,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> np.ndarray:
        inputs = self._build_inputs(text, voice_samples)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config=self._build_generation_config(do_sample, temperature, top_p),
                return_speech=True,
                verbose=False,
                refresh_negative=True,
                show_progress_bar=False,
            )
        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            raise RuntimeError("No audio generated")
        audio = outputs.speech_outputs[0]
        if torch.is_tensor(audio):
            if audio.dtype == torch.bfloat16:
                audio = audio.float()
            audio = audio.cpu().numpy()
        return audio

    def _generate_streaming_chunks(
        self,
        chunks: List[str],
        voice_samples: List[np.ndarray],
        cfg_scale: float,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> Iterator[np.ndarray]:
        for chunk_text in chunks:
            inputs = self._build_inputs(chunk_text, voice_samples)
            yield from self._generate_streaming(
                inputs, cfg_scale, do_sample, temperature, top_p
            )

    def _generate_streaming(
        self,
        inputs: dict,
        cfg_scale: float,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[np.ndarray]:
        """
        Generate speech with streaming.
        
        Args:
            inputs: Processed model inputs
            cfg_scale: CFG scale
            
        Yields:
            Audio chunks as numpy arrays
        """
        # Create audio streamer
        audio_streamer = AudioStreamer(
            batch_size=1,
            stop_signal=None,
            timeout=None
        )
        
        # Start generation in background
        import threading
        
        gen_config = self._build_generation_config(do_sample, temperature, top_p)

        def generate():
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config=gen_config,
                    audio_streamer=audio_streamer,
                    return_speech=True,
                    verbose=False,
                    refresh_negative=True,
                    show_progress_bar=False
                )
        
        generation_thread = threading.Thread(target=generate)
        generation_thread.start()
        
        # Yield chunks as they arrive
        audio_stream = audio_streamer.get_stream(0)
        for chunk in audio_stream:
            if torch.is_tensor(chunk):
                # Convert bfloat16 to float32 before converting to numpy
                if chunk.dtype == torch.bfloat16:
                    chunk = chunk.float()
                chunk = chunk.cpu().numpy()
            yield chunk
        
        # Wait for generation to complete
        generation_thread.join(timeout=10.0)
    
    def format_script_for_single_speaker(self, text: str, speaker_id: int = 0) -> str:
        """
        Format plain text as single-speaker script.
        
        Args:
            text: Plain text input
            speaker_id: Speaker ID to use
            
        Returns:
            Formatted script
        """
        # Split into sentences/paragraphs
        lines = text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(f"Speaker {speaker_id}: {line}")
        
        return '\n'.join(formatted_lines)

