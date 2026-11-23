from enum import Enum
import os
import uuid
import wave
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import numpy as np
logger = logging.getLogger(__name__)


class ASR(ABC):
    @staticmethod
    def _save_audio_to_file(audio_data, file_path):
        """将音频数据保存为WAV文件"""
        try:
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b''.join(audio_data))
            #logger.info(f"ASR识别文件录音保存到：{file_path}")
        except Exception as e:
            logger.error(f"An error occurred when saving the audio file: {e}")
            raise

    @abstractmethod
    def recognizer(self, stream_in_audio):
        """处理输入音频流并返回识别的文本，子类必须实现"""
        pass
    @staticmethod
    def volume_augment(audio, dB=80):
        """对音频进行音量增强。"""
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32)+dB
        audio = audio.astype(np.int16).tobytes()
        return audio

class FunASR(ASR):
    def __init__(self, config):
        from funasr import AutoModel

        self.model_dir = config.get("model_dir")
        self.vad_dir = config.get("vad_dir", "fsmn-vad")
        self.output_dir = config.get("output_file")
        #device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel(
            model=self.model_dir,
            vad_kwargs={"max_single_segment_time": 3000},
            disable_update=True,
            vad_model=self.vad_dir,
            #device="cuda:0",  # 如果有GPU，可以解开这行并指定设备
        )
        
    def recognizer(self, stream_in_audio):
        from .utils.utils import detect_language
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        try:
            tmpfile = os.path.join(self.output_dir, f"asr-{datetime.now().date()}@{uuid.uuid4().hex}.wav")
            # Enhance audio volume
            stream_in_audio = list(map(lambda audio: self.volume_augment(audio), stream_in_audio))
            self._save_audio_to_file(stream_in_audio, tmpfile)
            res = self.model.generate(
                input=tmpfile,
                cache={},
                language="auto",  # 语言选项: "zn", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False,
                batch_size_s=60,
                merge_vad=False,  #
                merge_length_s=15,
            )
            text = rich_transcription_postprocess(res[0]["text"])   
            lang=detect_language(text)
            logger.info(f"recognize text: {text}，language:{lang}")
            if lang=="en" or lang=="zh":
                 return text, tmpfile,lang
            else:
                return "", tmpfile,""

        except Exception as e:
            logger.error(f"An error occurred during the ASR recognition process: {e}")
            return None, None

class WhisperASR(ASR):
    LANGUAGE_CODES = {"english": "en", "finnish": "fi", "chinese": "zh"}

    def __init__(self, config):
        from transformers import pipeline
        import torch
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_file")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"


        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_dir,
            chunk_length_s=30,
            device=device,
        )
    def recognizer(self, stream_in_audio, *, keep_wav=True, raise_on_error=False):
        self.last_detected_language = None
        try:
            pcm = b"".join(stream_in_audio)
            if not pcm:
                return "", None, ""

            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            tmpfile = None
            if keep_wav:
                os.makedirs(self.output_dir, exist_ok=True)
                tmpfile = os.path.join(self.output_dir, f"asr-{datetime.now().date()}@{uuid.uuid4().hex}.wav")
                self._save_audio_to_file([pcm], tmpfile)

            result = self.pipe(
                audio,
                return_language=True,
                generate_kwargs={
                    "task": "transcribe",
                    "forced_decoder_ids": None,
                    # Transformers 4.51.1 otherwise strips the detected language token.
                    # The pipeline already limits each generation call to a 30-second chunk.
                    "force_unique_generate_call": True,
                },
            )
            text = result["text"]
            detected_language = next(
                (chunk["language"] for chunk in result.get("chunks", []) if chunk.get("language")),
                None,
            )
            self.last_detected_language = detected_language
            language = self.LANGUAGE_CODES.get(detected_language, "")
            logger.info(f"recognize text: {text}, language: {language}")
            return text, tmpfile, language

        except Exception as e:
            if raise_on_error:
                raise
            logger.error(f"An error occurred during the ASR recognition process: {e}")
            return "", None, ""

def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")
    
