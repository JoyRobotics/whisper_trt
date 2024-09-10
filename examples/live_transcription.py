# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import datetime
import json
import os
import numpy as np
import time
import pyaudio
from multiprocessing import Process, Queue, Event
from collections import deque
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional

import requests
from whisper_trt.vad import load_vad
from whisper_trt.model import load_trt_model
from base_redis import BaseRedis
from tuning import Tuning
import usb.core
import soundfile as sf

from nano_llm.utils import audio_db,audio_rms


def find_respeaker_audio_device_index():

    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount")

    for i in range(num_devices):

        device_info = p.get_device_info_by_host_api_device_index(0, i)
        
        if "respeaker" in device_info.get("name").lower():

            device_index = i

    return device_index


@contextmanager
def get_respeaker_audio_stream(
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 6,
        bitwidth: int = 2
    ):

    if device_index is None:
        device_index = find_respeaker_audio_device_index()

    if device_index is None:
        raise RuntimeError("Could not find Respeaker device.")
    
    p = pyaudio.PyAudio()

    stream = p.open(
        rate=sample_rate,
        format=p.get_format_from_width(bitwidth),
        channels=channels,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=1024 * 2
    )

    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


def audio_numpy_from_bytes(audio_bytes: bytes):
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio


def audio_numpy_slice_channel(audio_numpy: np.ndarray, channel_index: int, 
                      num_channels: int = 6):
    return audio_numpy[channel_index::num_channels]

def audio_numpy_normalize(audio_numpy: np.ndarray):
    return audio_numpy.astype(np.float32) / 32768


@dataclass
class AudioChunk:
    audio_raw: bytes
    audio_numpy: np.ndarray
    audio_numpy_normalized: np.ndarray
    voice_prob: float | None = None
    energy: float = 0.0  # Add energy attribute
    time_ts: float = 0.0,
    doa: int = 0,
    is_speech: bool = False,
    is_mic_voice: bool = False,
    is_aec_silence: bool = True


@dataclass
class AudioSegment:
    chunks: AudioChunk


class Microphone(Process):

    def __init__(self, 
                 output_queue: Queue, 
                 chunk_size: int = 1536, 
                 device_index: int | None = None,
                 use_channel: int = 0, 
                 num_channels: int = 6,
                 sample_rate: int = 16000):
        super().__init__()
        self.output_queue = output_queue
        self.chunk_size = chunk_size
        self.use_channel = use_channel
        self.num_channels = num_channels
        self.device_index = device_index
        self.sample_rate = sample_rate

        self.mic_tuning = Tuning(self.get_usb_device())
        self.mic_tuning.set_vad_threshold(2)
        self.mic_tuning.write("AECNORM",16)
        self.mic_tuning.write("AGCONOFF",0)
        self.mic_tuning.write("AGCGAIN",15.918223351240158)
        # self.mic_tuning.write("AGCDESIREDLEVEL",1e-5)

    def calculate_audio_energy(self, audio_numpy: np.ndarray) -> float:
        # Calculate the energy of the audio signal
        return np.sum(audio_numpy ** 2)

    def get_usb_device(self):
        """
        只有ReSpeaker 需要配置权限
        在非 root 用户下，需要确认权限。如果执行报错：usb.core.USBError: [Errno 13] Access denied (insufficient permissions)
        则需要修改设备的权限

        ```
        lsusb # 查找要操作的设备匹配的厂商 ID 和产品 ID

        ```

        方案一：临时解决，重启后失效
        ```
        sudo chmod 666 /dev/bus/usb/XXX/YYY
        ```

        方案二：重启后继续生效，在root 用户下
        > echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", GROUP="users", MODE="0666"' > /etc/udev/rules.d/99-usb-permissions.rules

        sudo udevadm control --reload-rules && sudo udevadm trigger
        """
        try:
            return usb.core.find(idVendor=0x2886, idProduct=0x0018)
        except Exception as err:
            self.logger.error(f"get_usb_device error: {err}")

    def run(self):
        with get_respeaker_audio_stream(sample_rate=self.sample_rate, 
                                        device_index=self.device_index, 
                                        channels=self.num_channels) as stream:
            while True:
                audio_raw = stream.read(self.chunk_size)
                audio_numpy = audio_numpy_from_bytes(audio_raw)
                audio_numpy = np.stack([audio_numpy_slice_channel(audio_numpy, i, self.num_channels) for i in range(self.num_channels)])
                audio_numpy_normalized = audio_numpy_normalize(audio_numpy)
                audio_energy = self.calculate_audio_energy(audio_numpy_normalized[self.use_channel])
                doa = int(self.mic_tuning.direction)
                is_speech = bool(self.mic_tuning.is_speech)
                is_mic_voice = bool(self.mic_tuning.is_voice)
                is_aec_silence = bool(self.mic_tuning.aecsilence_mode)
                audio_energy = audio_db(audio_raw)
                audio = AudioChunk(
                    audio_raw=audio_raw,
                    audio_numpy=audio_numpy,
                    audio_numpy_normalized=audio_numpy_normalized,
                    energy=audio_energy,
                    time_ts=time.time(),
                    doa=doa,
                    is_speech=is_speech,
                    is_mic_voice=is_mic_voice,
                    is_aec_silence=is_aec_silence
                    
                )
                self.output_queue.put(audio)


class VAD(Process):

    def __init__(self,
            input_queue: Queue, 
            output_queue: Queue,
            sample_rate: int = 16000,
            use_channel: int = 0,
            speech_threshold: float = 0.5,
            max_filter_window: int = 1,
            ready_flag = None,
            speech_start_flag = None,
            speech_end_flag = None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.use_channel = use_channel
        self.speech_threshold = speech_threshold
        self.max_filter_window = max_filter_window
        self.ready_flag = ready_flag
        self.speech_start_flag = speech_start_flag
        self.speech_end_flag = speech_end_flag
        self._redis = BaseRedis().get_instance()

    def run(self):

        vad = load_vad()
        
        # warmup run
        vad(np.zeros(1536, dtype=np.float32), sr=self.sample_rate)
        

        max_filter_window = deque(maxlen=self.max_filter_window)

        speech_chunks = []

        prev_is_voice = False

        start = False

        listen_ack_flag = False

        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:
            
            audio_chunk = self.input_queue.get()
            voice_prob = float(vad(audio_chunk.audio_numpy_normalized[self.use_channel], sr=self.sample_rate).flatten()[0])
            chunk = AudioChunk(
                audio_raw=audio_chunk.audio_raw,
                audio_numpy=audio_chunk.audio_numpy,
                audio_numpy_normalized=audio_chunk.audio_numpy_normalized,
                voice_prob=voice_prob,
                doa=audio_chunk.doa,
                energy=audio_chunk.energy,
                time_ts=audio_chunk.time_ts,
                is_speech=audio_chunk.is_speech,
                is_mic_voice=audio_chunk.is_mic_voice,
                is_aec_silence=audio_chunk.is_aec_silence
            ) 
            
            max_filter_window.append(chunk)
            is_voice = any(c.voice_prob > self.speech_threshold for c in max_filter_window)
            is_speech_count = sum(1 for c in max_filter_window if c.is_speech > 0)
            is_mic_voice_count = sum(1 for c in max_filter_window if c.is_mic_voice > 0)


            send_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            if chunk.is_aec_silence:
                max_filter = 2
            else:
                max_filter = 4

            if is_mic_voice_count >= max_filter and is_speech_count >= max_filter and not start and chunk.energy > -25:
                start = True

            if is_voice > prev_is_voice and start:
                speech_chunks = [chunk for chunk in max_filter_window]
                # start voice
                speech_chunks.append(chunk)
                if self.speech_start_flag is not None:
                    self.speech_start_flag.set()
                json_str = json.dumps({"type":1}, indent=4)
                self._redis.publish(f"SENSOR:VISION:10001", json_str)

                print(f"[{send_time}]开始说话: is_voice({is_voice}) , doa({chunk.doa}) , mic({chunk.is_speech},{chunk.is_mic_voice}) , energy({chunk.energy}) , voice_prob({voice_prob}) , time_ts({chunk.time_ts}) ,is_aec_silence({chunk.is_aec_silence})")
                start_time = time.time()
            elif is_voice < prev_is_voice and start:
                # end voice
                print(f"[{send_time}]说话结束: is_voice({is_voice}) , doa({chunk.doa}) , mic({chunk.is_speech},{chunk.is_mic_voice})  , energy({chunk.energy}) , voice_prob({voice_prob}) , time_ts({chunk.time_ts}) ,is_aec_silence({chunk.is_aec_silence})")
                start = False
                listen_ack_flag = False
                segment = AudioSegment(chunks=speech_chunks)
                self.output_queue.put(segment)
                if self.speech_end_flag is not None:
                    self.speech_end_flag.set()
            elif is_voice and start:
                # in progress
                speech_chunks.append(chunk)
                in_progress_time = time.time() - start_time
                if in_progress_time > 0.5:
                    self._redis.publish(f"ROBOT:10001:INTERRUPT", 1)
                if in_progress_time > 2 and not listen_ack_flag:
                    listen_ack_flag = True
                    self._redis.publish(f"ROBOT:10001:LISTEN_ACK", 1)
                print(f"[{send_time}]说话中: is_voice({is_voice}) , doa({chunk.doa}) , mic({chunk.is_speech},{chunk.is_mic_voice})  , energy({chunk.energy}) , voice_prob({voice_prob}) , time_ts({chunk.time_ts}) ,is_aec_silence({chunk.is_aec_silence})")

            else:
                # no voice
                start = False
                is_voice = False
                print(f"[{send_time}]无人说话: is_voice({is_voice}) , doa({chunk.doa}) , mic({chunk.is_speech},{chunk.is_mic_voice})  , energy({chunk.energy}) , voice_prob({voice_prob}) , time_ts({chunk.time_ts}) ,is_aec_silence({chunk.is_aec_silence})")

            prev_is_voice = is_voice



class ASR(Process):

    def __init__(self, model: str, backend: str, input_queue, use_channel: int = 0, ready_flag = None):
        super().__init__()
        self.model = model
        self.input_queue = input_queue
        self.use_channel = use_channel
        self.ready_flag = ready_flag
        self.backend = backend
        self._redis = BaseRedis().get_instance()

    def check_and_return_string(self,s: str) -> str:
        if (s.startswith("(") and s.endswith(")")) or \
        (s.startswith("[") and s.endswith("]")) or \
        (s.startswith("{") and s.endswith("}")):
            return ""
        return s

    def run(self):
        
        if self.backend == "whisper_trt":
            from whisper_trt import load_trt_model
            model = load_trt_model(self.model)
        elif self.backend == "whisper":
            from whisper import load_model
            model = load_model(self.model)
        elif self.backend == "faster_whisper":
            from faster_whisper import WhisperModel
            class FasterWhisperWrapper:
                def __init__(self, model):
                    self.model = model
                def transcribe(self, audio):
                    segs, info = self.model.transcribe(audio)
                    text = "".join([seg.text for seg in segs])
                    return {"text": text}
                
            model = FasterWhisperWrapper(WhisperModel(self.model))

        # warmup
        model.transcribe(np.zeros(1536, dtype=np.float32))

        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:

            speech_segment = self.input_queue.get()

            t0 = time.perf_counter_ns()
            audio = np.concatenate([chunk.audio_numpy_normalized[self.use_channel] for chunk in speech_segment.chunks])
            text = model.transcribe(audio)['text']

            t1 = time.perf_counter_ns()

            text = self.check_and_return_string(text)

                # 使用时间戳命名规则
            # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 生成时间戳，例如 20230905_120530
            # filename = f"output_{timestamp}.wav"  # 根据时间戳生成文件名
            # full_filename = os.path.join("/opt/deploy/joy_robotic/libs/mic", filename)  # 生成完整路径

            # # 保存音频为文件
            # sf.write(full_filename, audio, 16000)  # 保存到指定路径下的带时间戳的文件

            if text != "":
                result_dict = {
                        # "agesex": f"twenties:{gender}",
                        "person": "37",
                        "content": text,
                        "index": 0,
                        "start_speak_time": 0,
                        "end_speak_time": 0,
                        "doa": 0,
                        # "duration": self.calculate_duration(audio_data, self.sample_rate),
                        # "energy": self.calculate_energy(audio_data),
                        # "speaker": "1",
                        # "speaker_count": 1,

                    }

                json_str = json.dumps(result_dict, indent=4)
                _ = self._redis.publish(
                    f"SENSOR:AUDIO:10001", json_str)
            send_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            print(f"Text: {text}")
            print(f"[asr][{send_time}]Transcription Time: {(t1 - t0) / 1e9}")


class StartEndMonitor(Process):

    def __init__(self, start_flag: Event, end_flag):
        super().__init__()
        self.start_flag = start_flag
        self.end_flag = end_flag

    def run(self):
        while True:
            self.start_flag.wait()
            self.start_flag.clear()
            # print(f"==========开始说话==========")
            self.end_flag.wait()
            self.end_flag.clear()
            # print(f"==========说话结束==========")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--backend", type=str, default="whisper_trt")
    args = parser.parse_args()

    audio_chunks = Queue()
    speech_segments = Queue()
    vad_ready = Event()
    asr_ready = Event()
    speech_start = Event()
    speech_end = Event()


    asr = ASR(args.model, args.backend, speech_segments, ready_flag=asr_ready)

    vad = VAD(audio_chunks, speech_segments, max_filter_window=5, ready_flag=vad_ready, speech_start_flag=speech_start, speech_end_flag=speech_end)

    mic = Microphone(audio_chunks)
    mon = StartEndMonitor(speech_start, speech_end)

    vad.start()
    asr.start()
    mon.start()

    vad_ready.wait()
    asr_ready.wait()

    mic.start()

    mic.join()
    vad.join()
    asr.join()
    mon.join()