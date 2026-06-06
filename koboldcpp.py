#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# KoboldCpp is an easy-to-use AI text-generation software for GGML models.
# It's a single self contained distributable from Concedo, that builds off llama.cpp,
# and adds a versatile Kobold API endpoint, additional format support,
# backward compatibility, as well as a fancy UI with persistent stories,
# editing tools, save formats, memory, world info, author's note, characters,
# scenarios and everything Kobold and KoboldAI Lite have to offer.

import os
try:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # try set GPU to PCI order first thing
except Exception:
    pass
import copy
import ctypes
import multiprocessing
import math
import re
import argparse
import platform
import base64
import struct
import json
import sys
import http.server
import time
import asyncio
import socket
import threading
import html
import random
import hashlib
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Tuple
import shutil
import subprocess
import gzip
import queue

# constants
sampler_order_max = 7
tensor_split_max = 16
images_max = 16
audio_max = 16
bias_min_value = -100.0
bias_max_value = 100.0
logprobs_max = 10
default_draft_amount = 8
default_ttsmaxlen = 4096
default_embeddingsmaxctx = 4096
default_visionmaxres = 1024
net_save_slots = 12
savestate_limit_default = 5
savestate_limit = 0 #savestate slots start at 0, only set when load model
default_vae_tile_threshold = 640
default_sdvaedevice = 'main'
default_sdclipdevice = 'CPU'
default_native_ctx = 16384
default_genlen = 1024
overridekv_max = 16
default_autofit_padding = 1024
lora_filenames_max = 4
multiuser_concurrent_limit = 10
swa_padding_default = 0
default_reqtimeout = 600 # 10 min default

# abuse prevention
stop_token_max = 256
ban_token_max = 768
logit_bias_max = 512
dry_seq_break_max = 128
extra_images_max = 4 # for kontext/qwen img

# global vars
KcppVersion = "1.115"
showdebug = True
kcpp_instance = None #global running instance
global_memory = {"tunnel_url": "", "restart_target":"", "input_to_exit":False, "load_complete":False, "restart_override_base_config":"", "last_active_timestamp":datetime.now(), "triggered_sleeping":False, "current_model":"initial_model", "base_config":"", "swapReqType": None, "autoswapmode": False}
using_gui_launcher = False

handle = None
friendlymodelname = "inactive"
friendlysdmodelname = "inactive"
friendlyembeddingsmodelname = "inactive"
autoswapmode = False
textName = None
sttName = None
ttsName = None
embedName = None
musicName = None
imageName = None
mmprojName = None
lastgeneratedcomfyimg = b''
lastuploadedcomfyimg = b''
fullsdmodelpath = ""  #if empty, it's not initialized
password = "" #if empty, no auth key required
fullwhispermodelpath = "" #if empty, it's not initialized
ttsmodelpath = "" #if empty, not initialized
embeddingsmodelpath = "" #if empty, not initialized
musicllmmodelpath = "" #if empty, not initialized
musicdiffusionmodelpath = "" #if empty, not initialized
imglora_preload = []   # all preloaded LoRAs
imglora_bypath = {}    # len(imglora_bypath) == 0 <==> static loras
imglora_name2path = {}
imglora_cached = True
imglora_initial_fixed = True
maxctx = 8192
maxhordectx = 0 #set to whatever maxctx is if 0
maxhordelen = 1024
modelbusy = threading.Lock()
batched_lock = threading.Lock()
batched_cond = threading.Condition(batched_lock)
batched_request_runner_count = 0 #incremented when a batched request is running, prevents all non-batched requests
requestsinqueue = 0
ratelimitlookup = {}
defaultport = 5001
showsamplerwarning = True
showmaxctxwarning = True
showusedmemwarning = True
showmultigpuwarning = True
session_kudos_earned = 0
session_jobs = 0
session_starttime = None
exitcounter = -1
punishcounter = 0 #causes a timeout if too many errors
rewardcounter = 0 #reduces error counts for successful jobs
totalgens = 0
currentusergenkey = "" #store a special key so polled streaming works even in multiuser
pendingabortkey = "" #if an abort is received for the non-active request, remember it (at least 1) to cancel later
args = None #global args
runmode_untouched = True
modelfile_extracted_meta = None
calulated_gpu_overhead = 0 # may be populated at runtime, can also be missing if undetected
importvars_in_progress = False
has_multiplayer = False
has_audio_support = False
has_vision_support = False
has_whisper = False
cached_chat_template = None
cached_sd_info = {}
cached_jinja_kwargs = None
savedata_obj = None
mcp_connections = [] #every element is linked to one mcp source, contains obj {"client":obj, "tools":[]}
mcp_lock = threading.Lock()
multiplayer_story_data_compressed = None #stores the full compressed story of the current multiplayer session
multiplayer_turn_major = 1 # to keep track of when a client needs to sync their stories
multiplayer_turn_minor = 1
multiplayer_dataformat = "" # used to tell what is the data payload in saved story. set by client
multiplayer_lastactive = {} # timestamp of last activity for each unique player
websearch_lastquery = ""
websearch_lastresponse = []
preloaded_story = None
chatcompl_adapter = None
chatcompl_adapter_list = None #if using autoguess, will populate this will potential adapters
embedded_kailite = None
embedded_kailite_gz = None
embedded_kcpp_docs = None
embedded_kcpp_docs_gz = None
embedded_kcpp_sdui = None
embedded_kcpp_sdui_gz = None
embedded_lcpp_ui_gz = None
embedded_musicui = None
embedded_musicui_gz = None
preloaded_custom_jinja = ""
voicebank = {}
voicelist = ["kobo","cheery","sleepy","shouty","chatty"]
sslvalid = False
nocertify = False
start_time = time.time()
last_req_time = time.time()
last_non_horde_req_time = time.time()
currfinishreason = None
zenity_recent_dir = os.getcwd()
zenity_permitted = True
default_rpc_port = 5551
thinkformats = [{"start":"<|channel|>analysis<|message|>","end":"<|start|>assistant<|channel|>final<|message|>"},
                {"start":"<think>","end":"</think>"},
                {"start":"<seed:think>","end":"</seed:think>"},
                {"start":"<|channel>thought","end":"<channel|>"}]
tool_call_pairs = [ #third element is whether its stream-handleable
    ("<tool_call>", "</tool_call>", True),
    ("<seed:tool_call>", "</seed:tool_call>", True),
    ("<|tool_call_begin|>", "<|tool_call_end|>", True),
    ("<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>", True),
    ("<minimax:tool_call>", "</minimax:tool_call>", True),
    ("<|tool_call>", "<tool_call|>", True),
    ("<|end|><|start|>assistant<|channel|>commentary to=", "", False),
]
deprecated_keys = {
    "hordeconfig",
    "sdconfig",
    "noblas",
    "nommap",
    "pipelineparallel",
    "sdnotile",
    "forceversion",
    "sdgendefaults",
    "flashattention",
    "useswa",
}

saved_stdout = None
saved_stderr = None
saved_stdout_py = None
saved_stderr_py = None
stdout_nullfile = None
stdout_nullfile_py = None

CUDevices = ["0","1","2","3","All"]
CUDevicesNames = ["","","","",""]
VKDevicesNames = ["","","",""]
VKIsDGPU = [0,0,0,0]
MaxMemory = [0]
MaxFreeMemory = [0]

class logit_bias(ctypes.Structure):
    _fields_ = [("token_id", ctypes.c_int32),
                ("bias", ctypes.c_float)]

class token_count_outputs(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int),
                ("ids", ctypes.POINTER(ctypes.c_int))]

class detokenize_inputs(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int),
                ("ids", ctypes.POINTER(ctypes.c_int)),
                ("special", ctypes.c_bool)]

# returns top 5 logprobs per token
class logprob_item(ctypes.Structure):
     _fields_ = [("option_count", ctypes.c_int),
                ("selected_token", ctypes.c_char_p),
                ("selected_logprob", ctypes.c_float),
                ("selected_token_id", ctypes.c_int32),
                ("tokens", ctypes.c_char_p * logprobs_max),
                ("token_ids", ctypes.c_int32 * logprobs_max),
                ("logprobs", ctypes.POINTER(ctypes.c_float))]
class last_logprobs_outputs(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int),
                ("logprob_items", ctypes.POINTER(logprob_item))]

class load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("blasthreads", ctypes.c_int),
                ("max_context_length", ctypes.c_int),
                ("low_vram", ctypes.c_bool),
                ("use_mmq", ctypes.c_bool),
                ("splitmode", ctypes.c_int),
                ("executable_path", ctypes.c_char_p),
                ("model_filename", ctypes.c_char_p),
                ("lora_filename", ctypes.c_char_p),
                ("draftmodel_filename", ctypes.c_char_p),
                ("draft_amount", ctypes.c_int),
                ("draft_gpulayers", ctypes.c_int),
                ("draft_gpusplit", ctypes.c_float * tensor_split_max),
                ("mmproj_filename", ctypes.c_char_p),
                ("mmproj_cpu", ctypes.c_bool),
                ("visionmaxres", ctypes.c_int),
                ("visionmintokens", ctypes.c_int),
                ("visionmaxtokens", ctypes.c_int),
                ("use_mmap", ctypes.c_bool),
                ("use_mlock", ctypes.c_bool),
                ("use_smartcontext", ctypes.c_bool),
                ("use_contextshift", ctypes.c_bool),
                ("use_fastforward", ctypes.c_bool),
                ("kcpp_main_gpu", ctypes.c_int),
                ("batchsize", ctypes.c_int),
                ("autofit", ctypes.c_bool),
                ("autofit_tax_mb", ctypes.c_int),
                ("gpulayers", ctypes.c_int),
                ("rope_freq_scale", ctypes.c_float),
                ("rope_freq_base", ctypes.c_float),
                ("overridenativecontext", ctypes.c_int),
                ("moe_experts", ctypes.c_int),
                ("moecpu", ctypes.c_int),
                ("no_bos_token", ctypes.c_bool),
                ("load_guidance", ctypes.c_bool),
                ("override_kv", ctypes.c_char_p * overridekv_max),
                ("override_tensors", ctypes.c_char_p),
                ("flash_attention", ctypes.c_bool),
                ("tensor_split", ctypes.c_float * tensor_split_max),
                ("quant_k", ctypes.c_int),
                ("quant_v", ctypes.c_int),
                ("check_slowness", ctypes.c_bool),
                ("jinja_template", ctypes.c_char_p),
                ("highpriority", ctypes.c_bool),
                ("prevent_swa", ctypes.c_bool),
                ("swa_padding", ctypes.c_int),
                ("smartcache", ctypes.c_bool),
                ("smartcacheslots", ctypes.c_int),
                ("pipelineparallel", ctypes.c_bool),
                ("lora_multiplier", ctypes.c_float),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int),
                ("continuous_batching_slots", ctypes.c_int),
                ("rpc_mode", ctypes.c_int),
                ("rpc_targets", ctypes.c_char_p)]

class generation_inputs(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int),
                ("prompt", ctypes.c_char_p),
                ("memory", ctypes.c_char_p),
                ("negative_prompt", ctypes.c_char_p),
                ("guidance_scale", ctypes.c_float),
                ("images_len", ctypes.c_int),
                ("images", ctypes.POINTER(ctypes.c_char_p)),
                ("audio_len", ctypes.c_int),
                ("audio", ctypes.POINTER(ctypes.c_char_p)),
                ("max_context_length", ctypes.c_int),
                ("max_length", ctypes.c_int),
                ("temperature", ctypes.c_float),
                ("top_k", ctypes.c_int),
                ("top_a", ctypes.c_float),
                ("top_p", ctypes.c_float),
                ("min_p", ctypes.c_float),
                ("typical_p", ctypes.c_float),
                ("tfs", ctypes.c_float),
                ("nsigma", ctypes.c_float),
                ("rep_pen", ctypes.c_float),
                ("rep_pen_range", ctypes.c_int),
                ("rep_pen_slope", ctypes.c_float),
                ("presence_penalty", ctypes.c_float),
                ("mirostat", ctypes.c_int),
                ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float),
                ("xtc_threshold", ctypes.c_float),
                ("xtc_probability", ctypes.c_float),
                ("sampler_order", ctypes.c_int * sampler_order_max),
                ("sampler_len", ctypes.c_int),
                ("allow_eos_token", ctypes.c_bool),
                ("bypass_eos_token", ctypes.c_bool),
                ("tool_call_fix", ctypes.c_bool),
                ("render_special", ctypes.c_bool),
                ("stream_sse", ctypes.c_bool),
                ("grammar", ctypes.c_char_p),
                ("grammar_retain_state", ctypes.c_bool),
                ("dynatemp_range", ctypes.c_float),
                ("dynatemp_exponent", ctypes.c_float),
                ("smoothing_factor", ctypes.c_float),
                ("smoothing_curve", ctypes.c_float),
                ("adaptive_target", ctypes.c_float),
                ("adaptive_decay", ctypes.c_float),
                ("dry_multiplier", ctypes.c_float),
                ("dry_base", ctypes.c_float),
                ("dry_allowed_length", ctypes.c_int),
                ("dry_penalty_last_n", ctypes.c_int),
                ("dry_sequence_breakers_len", ctypes.c_int),
                ("dry_sequence_breakers", ctypes.POINTER(ctypes.c_char_p)),
                ("stop_sequence_len", ctypes.c_int),
                ("stop_sequence", ctypes.POINTER(ctypes.c_char_p)),
                ("logit_biases_len", ctypes.c_int),
                ("logit_biases", ctypes.POINTER(logit_bias)),
                ("banned_tokens_len", ctypes.c_int),
                ("banned_tokens", ctypes.POINTER(ctypes.c_char_p)),
                ("reasoning_budget", ctypes.c_int)]

class generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("stopreason", ctypes.c_int),
                ("prompt_tokens", ctypes.c_int),
                ("completion_tokens", ctypes.c_int),
                ("text", ctypes.c_char_p)]

class sd_load_model_inputs(ctypes.Structure):
    _fields_ = [("model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_device", ctypes.c_int),
                ("threads", ctypes.c_int),
                ("quant", ctypes.c_int),
                ("flash_attention", ctypes.c_bool),
                ("offload_cpu", ctypes.c_bool),
                ("use_mmap", ctypes.c_bool),
                ("kcpp_vae_device", ctypes.c_int),
                ("kcpp_clip_device", ctypes.c_int),
                ("diffusion_conv_direct", ctypes.c_bool),
                ("vae_conv_direct", ctypes.c_bool),
                ("taesd", ctypes.c_bool),
                ("tiled_vae_threshold", ctypes.c_int),
                ("t5xxl_filename", ctypes.c_char_p),
                ("clip1_filename", ctypes.c_char_p),
                ("clip2_filename", ctypes.c_char_p),
                ("vae_filename", ctypes.c_char_p),
                ("audio_vae_filename", ctypes.c_char_p),
                ("lora_len", ctypes.c_int),
                ("lora_filenames", ctypes.POINTER(ctypes.c_char_p)),
                ("lora_multipliers", ctypes.POINTER(ctypes.c_float)),
                ("lora_apply_mode", ctypes.c_int),
                ("photomaker_filename", ctypes.c_char_p),
                ("upscaler_filename", ctypes.c_char_p),
                ("img_hard_limit", ctypes.c_int),
                ("img_soft_limit", ctypes.c_int),
                ("max_vram", ctypes.c_float),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class sd_generation_inputs(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("negative_prompt", ctypes.c_char_p),
                ("init_images", ctypes.c_char_p),
                ("mask", ctypes.c_char_p),
                ("extra_images_len", ctypes.c_int),
                ("extra_images", ctypes.POINTER(ctypes.c_char_p)),
                ("reverse_refimg", ctypes.c_bool),
                ("flip_mask", ctypes.c_bool),
                ("denoising_strength", ctypes.c_float),
                ("cfg_scale", ctypes.c_float),
                ("distilled_guidance", ctypes.c_float),
                ("shifted_timestep", ctypes.c_int),
                ("flow_shift", ctypes.c_float),
                ("sample_steps", ctypes.c_int),
                ("width", ctypes.c_int),
                ("height", ctypes.c_int),
                ("seed", ctypes.c_int),
                ("sample_method", ctypes.c_char_p),
                ("scheduler", ctypes.c_char_p),
                ("eta", ctypes.c_float),
                ("clip_skip", ctypes.c_int),
                ("vid_req_frames", ctypes.c_int),
                ("vid_fps", ctypes.c_int),
                ("video_output_type", ctypes.c_int),
                ("remove_limits", ctypes.c_bool),
                ("circular_x", ctypes.c_bool),
                ("circular_y", ctypes.c_bool),
                ("cache_mode", ctypes.c_char_p),
                ("cache_options", ctypes.c_char_p),
                ("upscale", ctypes.c_bool),
                ("lora_len", ctypes.c_int),
                ("lora_filenames", ctypes.POINTER(ctypes.c_char_p)),
                ("lora_multipliers", ctypes.POINTER(ctypes.c_float))]

class sd_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("animated", ctypes.c_int),
                ("data", ctypes.c_char_p),
                ("data_extra", ctypes.c_char_p),
                ("final_frame", ctypes.c_char_p),
                ("info", ctypes.c_char_p)]

class sd_upscale_inputs(ctypes.Structure):
    _fields_ = [("init_images", ctypes.c_char_p),
                ("upscaling_resize", ctypes.c_int)]

class sd_info_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class whisper_load_model_inputs(ctypes.Structure):
    _fields_ = [("model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class whisper_generation_inputs(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("audio_data", ctypes.c_char_p),
                ("suppress_non_speech", ctypes.c_bool),
                ("langcode", ctypes.c_char_p)]

class whisper_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class tts_load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("ttc_model_filename", ctypes.c_char_p),
                ("cts_model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("gpulayers", ctypes.c_int),
                ("flash_attention", ctypes.c_bool),
                ("ttsmaxlen", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class tts_generation_inputs(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("speaker_seed", ctypes.c_int),
                ("audio_seed", ctypes.c_int),
                ("custom_speaker_voice", ctypes.c_char_p),
                ("custom_speaker_text", ctypes.c_char_p),
                ("custom_speaker_data", ctypes.c_char_p),
                ("reference_audio", ctypes.c_char_p),
                ("speaker_instruction", ctypes.c_char_p)]

class tts_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class embeddings_load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("model_filename", ctypes.c_char_p),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("gpulayers", ctypes.c_int),
                ("flash_attention", ctypes.c_bool),
                ("use_mmap", ctypes.c_bool),
                ("embeddingsmaxctx", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class embeddings_generation_inputs(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("truncate", ctypes.c_bool)]

class embeddings_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("count", ctypes.c_int),
                ("data", ctypes.c_char_p)]

class music_load_model_inputs(ctypes.Structure):
    _fields_ = [("musicllm_filename", ctypes.c_char_p),
                ("musicembedding_filename", ctypes.c_char_p),
                ("musicdiffusion_filename", ctypes.c_char_p),
                ("musicvae_filename", ctypes.c_char_p),
                ("lowvram", ctypes.c_bool),
                ("executable_path", ctypes.c_char_p),
                ("kcpp_main_gpu", ctypes.c_int),
                ("devices_override", ctypes.c_char_p),
                ("quiet", ctypes.c_bool),
                ("debugmode", ctypes.c_int)]

class music_generation_inputs(ctypes.Structure):
    _fields_ = [("is_planner_mode", ctypes.c_bool),
                ("stereo", ctypes.c_bool),
                ("use_mp3", ctypes.c_bool),
                ("gen_codes", ctypes.c_bool),
                ("rewrite_caption", ctypes.c_bool),
                ("input_json", ctypes.c_char_p),
                ("music_reference_audio_data", ctypes.c_char_p)]

class music_generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("music_output_json", ctypes.c_char_p),
                ("data", ctypes.c_char_p)]

class StdoutRedirector:
    def __init__(self, writer):
        self.writer = writer
        self.terminal = sys.__stdout__
    def write(self, message):
        try:
            # Always write to terminal, then duplicate to pipe writer
            self.terminal.write(message)
            self.terminal.flush()
            if self.writer:
                try:
                    self.writer.write(message)
                    self.writer.flush()
                except Exception:
                    self.writer = None
        except Exception:
            pass
    def flush(self):
        self.terminal.flush()

class MCPStdioClient:
    def resolve_command(self, command):
        resolved = shutil.which(command)
        if resolved:
            return resolved
        return command # fallback

    def __init__(self,command,largs,env=None,cwd=None):
        if isinstance(command, str):
            command = self.resolve_command(command)
            cmd = [command]
        else:
            cmd = list(command)
        if largs:
            cmd.extend(largs)
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=full_env,
            cwd=cwd
        )
        self.lock = threading.Lock()
        self.stderr_buffer = []
        self.stderr_limit = 20
        self.alive = True
        self.stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True
        )
        self.stderr_thread.start()

        self._pending = {}
        self._pending_lock = threading.Lock()
        self.stdout_thread = threading.Thread(
            target=self._read_stdout,
            daemon=True
        )
        self.stdout_thread.start()

    def _read_stderr(self):
        try:
            for line in self.process.stderr:
                if not line:
                    break
                line = line.rstrip()
                self.stderr_buffer.append(line)
                if len(self.stderr_buffer) > self.stderr_limit:
                    self.stderr_buffer.pop(0)
        finally:
            self.alive = False
    def _read_stdout(self):  # notifications (no id) are silently dropped
        try:
            for line in self.process.stdout:
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                msg_id = msg.get("id")
                if msg_id is not None:
                    with self._pending_lock:
                        q = self._pending.get(msg_id)
                    if q:
                        q.put(msg)
                    else:
                        print(f"[MCP] Unexpected response id: {msg_id}")
        finally:
            self.alive = False
            with self._pending_lock:
                for q in self._pending.values():
                    q.put(None)

    def send(self, message: dict, await_response=True) -> dict: # Send JSON-RPC request and wait for response.
        line = json.dumps(message)
        msg_id = message.get("id")

        if await_response and msg_id is None:
            raise ValueError("Cannot await response for a message without an 'id' field")

        response_q = queue.Queue()

        try:
            with self._pending_lock:
                if await_response and msg_id is not None:
                    self._pending[msg_id] = response_q
                with self.lock:
                    if self.process.stdin.closed:
                        raise RuntimeError("MCP server stdin is closed")
                    self.process.stdin.write(line + "\n")
                    self.process.stdin.flush()
        except Exception:
            if await_response and msg_id is not None:
                with self._pending_lock:
                    self._pending.pop(msg_id, None)
            raise

        if not await_response:
            return None
        try:
            response = response_q.get(timeout=180)
        except queue.Empty:
            raise RuntimeError("MCP server timed out (no response in 180s)")
        finally:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
        if response is None:
            errmsg = "\n".join(self.stderr_buffer[-10:])
            print(f"[MCP Server Error!]\n{errmsg}")
            raise RuntimeError("MCP server closed stdout")
        return response
    def notify(self, message: dict) -> None: # Send JSON-RPC notification (no response expected).
        line = json.dumps(message)
        with self.lock:
            if self.process.stdin.closed:
                raise RuntimeError("MCP server stdin is closed")
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
    def terminate(self):
        self.process.terminate()

class MCPHTTPClient:
    def __init__(self, url, headers=None, timeout=180.0):
        global nocertify
        self.url = url
        self.headers = {"Content-Type": "application/json","Accept": "application/json, text/event-stream"}
        if headers:
            self.headers.update(headers)
        self.timeout = timeout
        ssl_cert_dir = os.environ.get('SSL_CERT_DIR')
        if not ssl_cert_dir and not nocertify and os.name != 'nt':
            os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'

    def _read_sse(self, response) -> bytes:
        json_events = []
        buf = []
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line: # end of SSE event
                if buf:
                    payload = "\n".join(buf)
                    if payload and payload[0] in "{[":
                        json_events.append(payload)
                    buf = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                buf.append(line[5:].lstrip())
        if buf: # flush last event
            payload = "\n".join(buf)
            if payload and payload[0] in "{[":
                json_events.append(payload)
        if not json_events:
            raise RuntimeError("MCP HTTP server returned no JSON SSE response")
        return json_events[-1].encode("utf-8")


    def send(self, message: dict, await_response=True) -> dict: # Send JSON-RPC request and return response.
        data = json.dumps(message).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                sid = response.headers.get("MCP-Session-Id","92604d65-d82c-468a-96e9-cf4463ba68fc")
                if sid:
                    self.headers["MCP-Session-Id"] = sid
                ctype = response.headers.get("Content-Type","")
                body = self._read_sse(response) if "text/event-stream" in ctype else response.read()
        except urllib.error.HTTPError as e: # HTTP error with possible body
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MCP HTTP error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"MCP HTTP connection failed: {e.reason}") from e
        if not await_response:
            return None
        if not body:
            raise RuntimeError("MCP HTTP server returned empty response")
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"MCP HTTP server returned invalid JSON: {body!r}") from e

    def notify(self, message: dict) -> None: # Send JSON-RPC notification (no response expected).
        data = json.dumps(message).encode("utf-8")
        req = urllib.request.Request(self.url,data=data,headers=self.headers,method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except urllib.error.HTTPError as e: # Notifications may still return 204/empty; HTTPError means failure
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MCP HTTP notification failed ({e.code}): {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"MCP HTTP notification connection failed: {e.reason}") from e


def getdirpath():
    return os.path.dirname(os.path.realpath(__file__))
def getabspath():
    return os.path.dirname(os.path.abspath(__file__))
def file_exists(filename):
    return os.path.exists(os.path.join(getdirpath(), filename))

def suppress_stdout():
    global saved_stdout, saved_stderr, saved_stdout_py, saved_stderr_py, stdout_nullfile, stdout_nullfile_py
    if not saved_stdout and not saved_stderr and not saved_stdout_py and not saved_stderr_py and not stdout_nullfile and not stdout_nullfile_py:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(sys.stdout.fileno())
        saved_stderr = os.dup(sys.stderr.fileno())
        saved_stderr_py = sys.stderr
        saved_stdout_py = sys.stdout
        stdout_nullfile = os.open(os.devnull, os.O_WRONLY)
        stdout_nullfile_py = open(os.devnull, 'w')
        os.dup2(stdout_nullfile, sys.stdout.fileno())
        os.dup2(stdout_nullfile, sys.stderr.fileno())
        sys.stderr = sys.stdout = stdout_nullfile_py

def restore_stdout():
    global saved_stdout, saved_stderr, saved_stdout_py, saved_stderr_py, stdout_nullfile, stdout_nullfile_py
    if saved_stdout and saved_stderr and saved_stdout_py and saved_stderr_py and stdout_nullfile and stdout_nullfile_py:
        sys.stdout = saved_stdout_py
        sys.stderr = saved_stderr_py
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.dup2(saved_stderr, sys.stderr.fileno())
        os.close(stdout_nullfile)
        stdout_nullfile_py.close()
        os.close(saved_stdout)
        os.close(saved_stderr)
        saved_stdout = saved_stderr = saved_stdout_py = saved_stderr_py = stdout_nullfile = stdout_nullfile_py = None

def get_default_threads():
    physical_core_limit = 1
    if os.cpu_count() is not None and os.cpu_count()>1:
        physical_core_limit = os.cpu_count() // 2
    default_threads = (physical_core_limit if physical_core_limit<=3 else max(3,physical_core_limit-1))
    processor = platform.processor()
    if 'Intel' in processor:
        default_threads = (8 if default_threads > 8 else default_threads) #this helps avoid e-cores.
    if default_threads > 64:
        print(f"Auto CPU Threads capped at 64 (instead of {default_threads}). You can override this by passing an explicit number of --threads.")
        default_threads = 64
    return default_threads

def pick_existant_file(ntoption,nonntoption):
    precompiled_prefix = "precompiled_"
    ntexist = file_exists(ntoption)
    nonntexist = file_exists(nonntoption)
    precompiled_ntexist = file_exists(precompiled_prefix+ntoption)
    precompiled_nonntexist = file_exists(precompiled_prefix+nonntoption)
    if os.name == 'nt':
        if not ntexist and precompiled_ntexist:
            return (precompiled_prefix+ntoption)
        if nonntexist and not ntexist:
            return nonntoption
        return ntoption
    else:
        if not nonntexist and precompiled_nonntexist:
            return (precompiled_prefix+nonntoption)
        if ntexist and not nonntexist:
            return ntoption
        return nonntoption

lib_default = pick_existant_file("koboldcpp_default.dll","koboldcpp_default.so")
lib_failsafe = pick_existant_file("koboldcpp_failsafe.dll","koboldcpp_failsafe.so")
lib_noavx2 = pick_existant_file("koboldcpp_noavx2.dll","koboldcpp_noavx2.so")
lib_vulkan_failsafe = pick_existant_file("koboldcpp_vulkan_failsafe.dll","koboldcpp_vulkan_failsafe.so")
lib_cublas = pick_existant_file("koboldcpp_cublas.dll","koboldcpp_cublas.so")
lib_hipblas = pick_existant_file("koboldcpp_hipblas.dll","koboldcpp_hipblas.so")
lib_vulkan = pick_existant_file("koboldcpp_vulkan.dll","koboldcpp_vulkan.so")
lib_vulkan_noavx2 = pick_existant_file("koboldcpp_vulkan_noavx2.dll","koboldcpp_vulkan_noavx2.so")
libname = ""
lib_option_pairs = [
    (lib_default, "Use CPU"),
    (lib_cublas, "Use CUDA"),
    (lib_hipblas, "Use hipBLAS (ROCm)"),
    (lib_vulkan, "Use Vulkan"),
    (lib_noavx2, "Use CPU (Old CPU)"),
    (lib_vulkan_noavx2, "Use Vulkan (Old CPU)"),
    (lib_vulkan_failsafe, "Use Vulkan (Older CPU)"),
    (lib_failsafe, "Failsafe Mode (Older CPU)")]
default_option, cublas_option, hipblas_option, vulkan_option, noavx2_option, vulkan_noavx2_option, vulkan_failsafe_option, failsafe_option = (opt if file_exists(lib) or (os.name == 'nt' and file_exists(opt + ".dll")) else None for lib, opt in lib_option_pairs)
runopts = [opt for lib, opt in lib_option_pairs if file_exists(lib)]

def init_library():
    global handle, args, libname
    global lib_default,lib_failsafe,lib_noavx2,lib_vulkan_failsafe,lib_cublas,lib_hipblas,lib_vulkan,lib_vulkan_noavx2

    libname = lib_default

    if not args: # debug helper: koboldcpp.py loaded by external script
        pass
    elif args.noavx2: #failsafe implies noavx2 always
        if args.failsafe and (args.usevulkan is not None) and file_exists(lib_vulkan_failsafe):
            libname = lib_vulkan_failsafe
        elif (args.usevulkan is not None) and file_exists(lib_vulkan_noavx2):
            libname = lib_vulkan_noavx2
        elif (args.failsafe) and file_exists(lib_failsafe):
            print("!!! Attempting to use FAILSAFE MODE !!!")
            libname = lib_failsafe
        elif file_exists(lib_noavx2):
            libname = lib_noavx2
    elif (args.usecuda is not None):
        if file_exists(lib_cublas):
            libname = lib_cublas
        elif file_exists(lib_hipblas):
            libname = lib_hipblas
    elif (args.usevulkan is not None):
        if file_exists(lib_vulkan):
            libname = lib_vulkan
        elif file_exists(lib_vulkan_noavx2):
            libname = lib_vulkan_noavx2
    elif libname == lib_default and not file_exists(lib_default) and file_exists(lib_noavx2):
        libname = lib_noavx2

    print("Initializing dynamic library: " + libname)
    dir_path = getdirpath()
    abs_path = getabspath()

    #add all potential paths
    if os.name=='nt':
        os.add_dll_directory(dir_path)
        os.add_dll_directory(abs_path)
        os.add_dll_directory(os.getcwd())
        if libname == lib_cublas and "CUDA_PATH" in os.environ:
            newpath = os.path.join(os.environ["CUDA_PATH"], "bin")
            if os.path.exists(newpath):
                os.add_dll_directory(newpath)
        if libname == lib_hipblas and "HIP_PATH" in os.environ:
            newpath = os.path.join(os.environ["HIP_PATH"], "bin")
            if os.path.exists(newpath):
                os.add_dll_directory(newpath)

    handle = ctypes.CDLL(os.path.join(dir_path, libname))

    handle.load_model.argtypes = [load_model_inputs]
    handle.load_model.restype = ctypes.c_bool
    handle.generate.argtypes = [generation_inputs]
    handle.generate.restype = generation_outputs
    handle.new_token.restype = ctypes.c_char_p
    handle.new_token.argtypes = [ctypes.c_int]
    handle.get_stream_count.restype = ctypes.c_int
    handle.has_finished.restype = ctypes.c_bool
    handle.batch_generate_enabled.restype = ctypes.c_bool
    handle.batch_generate_submit.argtypes = [generation_inputs]
    handle.batch_generate_submit.restype = ctypes.c_int
    handle.batch_generate_has_finished.argtypes = [ctypes.c_int]
    handle.batch_generate_has_finished.restype = ctypes.c_bool
    handle.batch_generate_stream_count.argtypes = [ctypes.c_int]
    handle.batch_generate_stream_count.restype = ctypes.c_int
    handle.batch_generate_new_token.argtypes = [ctypes.c_int, ctypes.c_int]
    handle.batch_generate_new_token.restype = ctypes.c_char_p
    handle.batch_generate_pending_output.argtypes = [ctypes.c_int]
    handle.batch_generate_pending_output.restype = ctypes.c_char_p
    handle.batch_generate_result.argtypes = [ctypes.c_int]
    handle.batch_generate_result.restype = generation_outputs
    handle.batch_generate_abort.argtypes = [ctypes.c_int]
    handle.batch_generate_abort.restype = ctypes.c_bool
    handle.batch_generate_release.argtypes = [ctypes.c_int]
    handle.batch_generate_release.restype = None
    handle.has_audio_support.restype = ctypes.c_bool
    handle.has_vision_support.restype = ctypes.c_bool
    handle.get_last_eval_time.restype = ctypes.c_float
    handle.get_last_process_time.restype = ctypes.c_float
    handle.get_last_token_count.restype = ctypes.c_int
    handle.get_last_input_count.restype = ctypes.c_int
    handle.get_last_seed.restype = ctypes.c_int
    handle.get_last_draft_success.restype = ctypes.c_int
    handle.get_last_draft_failed.restype = ctypes.c_int
    handle.get_total_img_gens.restype = ctypes.c_int
    handle.get_total_tts_gens.restype = ctypes.c_int
    handle.get_total_transcribe_gens.restype = ctypes.c_int
    handle.get_total_gens.restype = ctypes.c_int
    handle.get_last_stop_reason.restype = ctypes.c_int
    handle.abort_generate.restype = ctypes.c_bool
    handle.token_count.restype = token_count_outputs
    handle.get_pending_output.restype = ctypes.c_char_p
    handle.get_chat_template.restype = ctypes.c_char_p
    handle.calc_new_state_kv.restype = ctypes.c_size_t
    handle.calc_new_state_tokencount.restype = ctypes.c_size_t
    handle.calc_old_state_kv.argtypes = [ctypes.c_int]
    handle.calc_old_state_kv.restype = ctypes.c_size_t
    handle.calc_old_state_tokencount.argtypes = [ctypes.c_int]
    handle.calc_old_state_tokencount.restype = ctypes.c_size_t
    handle.save_state_kv.argtypes = [ctypes.c_int]
    handle.save_state_kv.restype = ctypes.c_size_t
    handle.load_state_kv.argtypes = [ctypes.c_int]
    handle.load_state_kv.restype = ctypes.c_bool
    handle.clear_state_kv.restype = ctypes.c_bool
    handle.sd_load_model.argtypes = [sd_load_model_inputs]
    handle.sd_load_model.restype = ctypes.c_bool
    handle.sd_generate.argtypes = [sd_generation_inputs]
    handle.sd_generate.restype = sd_generation_outputs
    handle.sd_upscale.argtypes = [sd_upscale_inputs]
    handle.sd_upscale.restype = sd_generation_outputs
    handle.sd_get_info.argtypes = []
    handle.sd_get_info.restype = sd_info_outputs
    handle.whisper_load_model.argtypes = [whisper_load_model_inputs]
    handle.whisper_load_model.restype = ctypes.c_bool
    handle.whisper_generate.argtypes = [whisper_generation_inputs]
    handle.whisper_generate.restype = whisper_generation_outputs
    handle.tts_load_model.argtypes = [tts_load_model_inputs]
    handle.tts_load_model.restype = ctypes.c_bool
    handle.tts_generate.argtypes = [tts_generation_inputs]
    handle.tts_generate.restype = tts_generation_outputs
    handle.embeddings_load_model.argtypes = [embeddings_load_model_inputs]
    handle.embeddings_load_model.restype = ctypes.c_bool
    handle.embeddings_generate.argtypes = [embeddings_generation_inputs]
    handle.embeddings_generate.restype = embeddings_generation_outputs
    handle.music_load_model.argtypes = [music_load_model_inputs]
    handle.music_load_model.restype = ctypes.c_bool
    handle.music_generate.argtypes = [music_generation_inputs]
    handle.music_generate.restype = music_generation_outputs
    handle.last_logprobs.restype = last_logprobs_outputs
    handle.detokenize.argtypes = [detokenize_inputs]
    handle.detokenize.restype = ctypes.c_char_p
    handle.launch_rpc_server.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    handle.launch_rpc_server.restype = ctypes.c_bool
    handle.set_environment_variable.restype = ctypes.c_int
    handle.set_environment_variable.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

def set_backend_props(inputs):
    # we must force an explicit tensor split
    # otherwise the default will divide equally and multigpu crap will slow it down badly
    inputs.kcpp_main_gpu = -1
    if(args.maingpu is not None and args.maingpu>=0):
        inputs.kcpp_main_gpu = args.maingpu

    if args.usecuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not args.tensor_split:
        if (args.usecuda and "0" in args.usecuda):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["HIP_VISIBLE_DEVICES"] = "0"
            inputs.kcpp_main_gpu = 0
        elif (args.usecuda and "1" in args.usecuda):
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            os.environ["HIP_VISIBLE_DEVICES"] = "1"
            inputs.kcpp_main_gpu = 0
        elif (args.usecuda and "2" in args.usecuda):
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            os.environ["HIP_VISIBLE_DEVICES"] = "2"
            inputs.kcpp_main_gpu = 0
        elif (args.usecuda and "3" in args.usecuda):
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            os.environ["HIP_VISIBLE_DEVICES"] = "3"
            inputs.kcpp_main_gpu = 0
    else:
        if(args.maingpu is None or args.maingpu<0):
            if (args.usecuda and "0" in args.usecuda):
                inputs.kcpp_main_gpu = 0
            elif (args.usecuda and "1" in args.usecuda):
                inputs.kcpp_main_gpu = 1
            elif (args.usecuda and "2" in args.usecuda):
                inputs.kcpp_main_gpu = 2
            elif (args.usecuda and "3" in args.usecuda):
                inputs.kcpp_main_gpu = 3

    if "GGML_VK_VISIBLE_DEVICES" not in os.environ:
        if args.usevulkan: # is an empty array if using vulkan without defined gpu
            vulkangpus = ','.join([str(g) for g in args.usevulkan])
            handle.set_environment_variable("GGML_VK_VISIBLE_DEVICES".encode("UTF-8"),vulkangpus.encode("UTF-8"))

    # set universal flags
    inputs.devices_override = (args.device if args.device else "").encode("UTF-8")
    inputs.quiet = args.quiet
    inputs.debugmode = args.debugmode
    inputs.executable_path = (getdirpath()+"/").encode("UTF-8")

    return inputs

def end_trim_to_sentence(input_text):
    enders = ['.', '!', '?', '*', '"', ')', '}', '`', ']', ';', '…']
    last = -1
    for ender in enders:
        last = max(last, input_text.rfind(ender))
    nl = input_text.rfind("\n")
    last = max(last, nl)
    if last > 0:
        return input_text[:last + 1].strip()
    return input_text.strip()

def tryparseint(value,fallback):
    if value is None:
        return fallback
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value == "true":
            return 1
        if lower_value == "false":
            return 0
    try:
        return int(value)
    except ValueError:
        return fallback
def tryparsefloat(value,fallback):
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback

def replace_last_in_string(text: str, match: str, replacement: str) -> str:
    if match == "":
        return text
    head, sep, tail = text.rpartition(match)
    if sep == "":
        return text  # old not found
    return head + replacement + tail

def is_incomplete_utf8_sequence(byte_seq): #note, this will only flag INCOMPLETE sequences, corrupted ones will be ignored.
    try:
        byte_seq.decode('utf-8')
        return False  # Valid UTF-8
    except UnicodeDecodeError as e:
        if e.reason == 'unexpected end of data':
            return True #incomplete sequence
        return False #invalid sequence, but not incomplete

def strip_base64_prefix(encoded_data):
    if not encoded_data:
        return ""
    if encoded_data.startswith("data:image"):
        encoded_data = encoded_data.split(',', 1)[-1]
    return encoded_data

def fix_unquoted_keys(s: str) -> str:
    """
    Fix JSON with unquoted keys by only quoting identifiers that appear
    in key position (after '{' or ',' at object level, before ':').
    Uses a state machine to track position in the JSON structure.
    """
    result = []
    i = 0
    n = len(s)
    def skip_whitespace():
        nonlocal i
        while i < n and s[i].isspace():
            result.append(s[i])
            i += 1
    def read_string():
        """Read a quoted string, handling escape sequences correctly."""
        nonlocal i
        assert s[i] == '"'
        result.append(s[i])
        i += 1
        while i < n:
            ch = s[i]
            result.append(ch)
            i += 1
            if ch == '\\':
                if i < n:
                    result.append(s[i])
                    i += 1
            elif ch == '"':
                break
    def read_value():
        """Read any JSON value."""
        nonlocal i
        skip_whitespace()
        if i >= n:
            return
        ch = s[i]
        if ch == '{':
            read_object()
        elif ch == '[':
            read_array()
        elif ch == '"':
            read_string()
        else:
            while i < n and s[i] not in ',}]':
                result.append(s[i])
                i += 1
    def read_object():
        nonlocal i
        result.append(s[i])
        i += 1
        skip_whitespace()
        if i < n and s[i] == '}':
            result.append(s[i])
            i += 1
            return
        while i < n:
            skip_whitespace()
            if i < n and s[i] == '"':
                read_string()
            elif i < n and re.match(r'[a-zA-Z_]', s[i]):
                key = []
                while i < n and re.match(r'[a-zA-Z0-9_]', s[i]):
                    key.append(s[i])
                    i += 1
                result.append('"' + ''.join(key) + '"')
            skip_whitespace()
            if i < n and s[i] == ':':
                result.append(s[i])
                i += 1
            read_value()
            skip_whitespace()
            if i >= n or s[i] == '}':
                break
            if s[i] == ',':
                result.append(s[i])
                i += 1
        if i < n and s[i] == '}':
            result.append(s[i])
            i += 1
    def read_array():
        nonlocal i
        result.append(s[i])
        i += 1
        skip_whitespace()
        if i < n and s[i] == ']':
            result.append(s[i])
            i += 1
            return
        while i < n:
            read_value()
            skip_whitespace()
            if i >= n or s[i] == ']':
                break
            if s[i] == ',':
                result.append(s[i])
                i += 1
        if i < n and s[i] == ']':
            result.append(s[i])
            i += 1
    read_value()
    return ''.join(result)

def old_cpu_check(): #return -1 for pass, 0 if has avx2, 1 if has avx, 2 if has nothing
    shouldcheck = ((sys.platform == "linux" and platform.machine().lower() in ("x86_64", "amd64")) or
                  (os.name == 'nt' and platform.machine().lower() in ("amd64", "x86_64")))
    if not shouldcheck:
        return -1 #doesnt deal with avx at all.
    try:
        retflags = 0
        if sys.platform == "linux":
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                cpuinfo = cpuinfo.lower()
                if 'avx' not in cpuinfo and 'avx2' not in cpuinfo:
                    retflags = 2
                elif 'avx2' not in cpuinfo:
                    retflags = 1
        elif os.name == 'nt':
            basepath = os.path.abspath(os.path.dirname(__file__))
            output = ""
            data = None
            output = subprocess.run([os.path.join(basepath, "simplecpuinfo.exe")], capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS, encoding='utf-8', timeout=6).stdout
            data = json.loads(output)
            if data["avx2"]==0 and data["avx"]==0:
                retflags = 2
            elif data["avx2"]==0:
                retflags = 1
        return retflags
    except Exception:
        return -1 #cannot determine

def has_valid_model():
    return args.model_param or args.sdmodel or args.whispermodel or args.ttsmodel or args.embeddingsmodel or args.musicdiffusion or args.musicllm or args.mcpfile or args.nomodel or args.rpcmode=="host"

def unpack_to_dir(destpath = ""):
    srcpath = os.path.abspath(os.path.dirname(__file__))
    cliunpack = False if destpath == "" else True
    print("Attempt to unpack KoboldCpp into directory...")

    if not cliunpack:
        from tkinter import messagebox
        destpath = zentk_askdirectory(title='Select an empty folder to unpack KoboldCpp')
        if not destpath:
            return

    if not os.path.isdir(destpath):
        os.makedirs(destpath)

    if os.path.isdir(srcpath) and os.path.isdir(destpath) and not os.listdir(destpath):
        try:
            if cliunpack:
                print(f"KoboldCpp will be extracted to {destpath}\nThis process may take several seconds to complete.")
            else:
                messagebox.showinfo("Unpack Starting", f"KoboldCpp will be extracted to {destpath}\nThis process may take several seconds to complete.")
            pyds_dir = os.path.join(destpath, 'pyds')
            using_pyinstaller_6 = False
            try:
                import pkg_resources
                piver = pkg_resources.get_distribution("pyinstaller").version
                print(f"PyInstaller Version: {piver}")
                if piver.startswith("6."):
                    using_pyinstaller_6 = True
                    os.makedirs(os.path.join(destpath, "_internal"), exist_ok=True)
                    pyds_dir = os.path.join(os.path.join(destpath, "_internal"), 'pyds')
            except Exception:
                pass
            os.makedirs(pyds_dir, exist_ok=True)
            for item in os.listdir(srcpath):
                s = os.path.join(srcpath, item)
                d = os.path.join(destpath, item)
                d2 = d  #this will be modified for pyinstaller 6 and unmodified for pyinstaller 5
                if using_pyinstaller_6:
                    d2 = os.path.join(os.path.join(destpath, "_internal"), item)
                if using_pyinstaller_6 and item.startswith('koboldcpp-launcher'):  # Move koboldcpp-launcher to its intended location
                    shutil.copy2(s, d)
                    continue
                if item.endswith('.pyd'):  # relocate pyds files to subdirectory
                    pyd = os.path.join(pyds_dir, item)
                    shutil.copy2(s, pyd)
                    continue
                if os.path.isdir(s):
                    shutil.copytree(s, d2, False, None)
                else:
                    shutil.copy2(s, d2)
            if cliunpack:
                print(f"KoboldCpp successfully extracted to {destpath}")
            else:
                messagebox.showinfo("KoboldCpp Unpack Success", f"KoboldCpp successfully extracted to {destpath}")
        except Exception as e:
            if cliunpack:
                print(f"An error occurred while unpacking: {e}")
            else:
                messagebox.showerror("Error", f"An error occurred while unpacking: {e}")
    else:
        if cliunpack:
            print("The target folder is not empty or invalid. Please select an empty folder.")
        else:
            messagebox.showwarning("Invalid Selection", "The target folder is not empty or invalid. Please select an empty folder.")

def exit_with_error(code, message, title="Error"):
    global using_gui_launcher
    print("")
    time.sleep(1)
    if using_gui_launcher:
        show_gui_msgbox(title, message)
    else:
        print(message, flush=True)
    time.sleep(2)
    sys.exit(code)

def utfprint(str, importance = 2): #0 = only debugmode, 1 = except quiet, 2 = always print
    if args.quiet and importance<2: #quiet overrides debugmode
        return
    if args.debugmode < 1:
        if importance==1 and (args.debugmode == -1 or args.quiet):
            return
        if importance==0:
            return
    maxlen = 40000
    if args.debugmode >= 1:
        maxlen = 240000
    try:
        strlength = len(str)
        if strlength > maxlen: #limit max output len
            str = str[:maxlen] + f"... (+{strlength-maxlen} chars)"
    except Exception:
        pass

    try:
        print(str)
    except UnicodeEncodeError:
        # Replace or omit the problematic character
        utf_string = str.encode('ascii', 'ignore').decode('ascii',"ignore")
        utf_string = utf_string.replace('\a', '') #remove bell characters
        print(utf_string)

def bring_terminal_to_foreground():
    if os.name=='nt':
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 9)
        ctypes.windll.user32.SetForegroundWindow(ctypes.windll.kernel32.GetConsoleWindow())

def simple_lcg_hash(input_string): #turns any string into a number between 10000 and 99999
    a = 1664525
    c = 1013904223
    m = 89999  # Modulo
    hash_value = 25343
    for char in input_string:
        hash_value = (a * hash_value + ord(char) + c) % m
    hash_value += 10000
    return hash_value

def string_has_overlap(str_a, str_b, maxcheck):
    max_overlap = min(maxcheck, len(str_a), len(str_b))
    for i in range(1, max_overlap + 1):
        if str_a[-i:] == str_b[:i]:
            return True
    return False

def string_contains_or_overlaps_sequence_substring(inputstr, sequences):
    if inputstr=="":
        return False
    for s in sequences:
        if s.strip()=="":
            continue
        if s.strip() in inputstr.strip() or inputstr.strip() in s.strip():
            return True
        if string_has_overlap(inputstr, s, 10):
            return True
    return False

def truncate_long_json(data, max_length):
    def truncate_middle(s, max_length):
        if len(s) <= max_length or max_length < 5:
            return s
        half = (max_length - 3) // 2
        return s[:half] + "..." + s[-half:]

    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                new_data[key] = truncate_middle(value, max_length)
            else:
                new_data[key] = truncate_long_json(value, max_length)
        return new_data
    elif isinstance(data, list):
        return [truncate_long_json(item, max_length) for item in data]
    elif isinstance(data, str):
        return truncate_middle(data, max_length)
    else:
        return data

def convert_json_to_gbnf(json_obj):
    try:
        from json_to_gbnf import SchemaConverter
        prop_order = []
        converter = SchemaConverter(
        prop_order={name: idx for idx, name in enumerate(prop_order)},
        allow_fetch=False,
        dotall=False,
        raw_pattern=False)
        schema = json.loads(json.dumps(json_obj))
        schema = converter.resolve_refs(schema, '')
        converter.visit(schema, '')
        outstr = converter.format_grammar()
        return outstr
    except Exception as e:
        print(f"JSON to GBNF failed: {e}")
        return ""

def get_capabilities():
    global savedata_obj, has_multiplayer, KcppVersion, friendlymodelname, friendlysdmodelname, fullsdmodelpath, password, fullwhispermodelpath, ttsmodelpath, embeddingsmodelpath, musicdiffusionmodelpath, musicllmmodelpath, has_audio_support, has_vision_support, mcp_connections
    global autoswapmode, textName, sttName, ttsName, embedName, musicName, imageName, mmprojName
    has_llm = not (friendlymodelname=="inactive") or (autoswapmode and textName is not None)
    has_txt2img = not (friendlysdmodelname=="inactive" or fullsdmodelpath=="") or (autoswapmode and imageName is not None)
    has_password = (password!="")
    has_whisper = (fullwhispermodelpath!="") or (autoswapmode and sttName is not None)
    has_search = True if args.websearch else False
    has_tts = (ttsmodelpath!="") or (autoswapmode and ttsName is not None)
    has_embeddings = (embeddingsmodelpath!="") or (autoswapmode and embedName is not None)
    has_music = (musicdiffusionmodelpath!="" or musicllmmodelpath!="") or (autoswapmode and musicName is not None)
    visionSupport = (has_vision_support) or (autoswapmode and mmprojName is not None) #todo: not always correct
    audioSupport = (has_audio_support) #todo: not always correct
    has_guidance = True if args.enableguidance else False
    has_jinja = True if args.jinja else False
    has_mcp = True if (args.mcpfile and mcp_connections and len(mcp_connections) > 0) else False
    admin_type = (2 if args.admin and args.admindir and args.adminpassword else (1 if args.admin and args.admindir else 0))
    has_router = True if args.routermode else False
    return {"result":"KoboldCpp", "version":KcppVersion, "protected":has_password, "llm":has_llm, "txt2img":has_txt2img,"vision":visionSupport,"audio":audioSupport,"transcribe":has_whisper,"multiplayer":has_multiplayer,"websearch":has_search,"tts":has_tts, "embeddings":has_embeddings, "music":has_music, "savedata":(savedata_obj is not None), "admin": admin_type, "router":has_router, "guidance": has_guidance, "jinja": has_jinja, "mcp":has_mcp}


def scan_directory(dirpath, valid_exts, depth):
    files = []
    for entry in sorted(os.listdir(dirpath)): # Scan top-level directory
        full_path = os.path.join(dirpath, entry)
        if os.path.isfile(full_path) and entry.lower().endswith(valid_exts): # If toplevel file
            files.append(entry)
        elif depth > 0 and os.path.isdir(full_path): #if dir, scan up to 1 level deep
            for subentry in sorted(os.listdir(full_path)):
                sub_full_path = os.path.join(full_path, subentry)
                if os.path.isfile(sub_full_path) and subentry.lower().endswith(valid_exts):
                    rel_path = os.path.join(entry, subentry)
                    files.append(rel_path)
    return files


def get_current_admindir_list():
    opts = []
    if args.admin and args.admindir:
        dirpath = os.path.abspath(args.admindir)
        valid_exts = (".kcpps", ".kcppt", ".gguf")
        opts = scan_directory(dirpath, valid_exts, 1)
        opts.append("initial_model")
        opts.append("unload_model")
    return opts


def dump_gguf_metadata(file_path): #if you're gonna copy this into your own project at least credit concedo
    chunk_size = 1024*1024*20  # read first 20mb of file
    try:
        data = None
        fptr = 0
        dt_table = ["u8","i8","u16","i16","u32","i32","f32","bool","str","arr","u64","i64","f64"] #13 types, else error
        tt_table = ["f32","f16","q4_0","q4_1","q4_2","q4_3","q5_0","q5_1","q8_0","q8_1","q2_k","q3_k","q4_k","q5_k","q6_k","q8_k","iq2_xxs","iq2_xs","iq3_xxs","iq1_s","iq4_nl","iq3_s","iq2_s","iq4_xs","i8","i16","i32","i64","f64","iq1_m","bf16","q4_0_4_4","q4_0_4_8","q4_0_8_8","tq1_0","tq2_0","iq4_nl_4_4","iq4_nl_4_8","iq4_nl_8_8","mxfp4","nvfp4","q1_0","unknown","unknown","unknown","unknown"]
        def read_data(datatype):
            nonlocal fptr, data, dt_table
            if datatype=="u32":
                val_bytes = data[fptr:fptr + 4]
                val = struct.unpack('<I', val_bytes)[0]
                fptr += 4
                return val
            if datatype=="u64":
                val_bytes = data[fptr:fptr + 8]
                val = struct.unpack('<Q', val_bytes)[0]
                fptr += 8
                return val
            if datatype=="i32":
                val_bytes = data[fptr:fptr + 4]
                val = struct.unpack('<i', val_bytes)[0]
                fptr += 4
                return val
            if datatype=="bool":
                val_bytes = data[fptr:fptr + 1]
                val = struct.unpack('<B', val_bytes)[0]
                fptr += 1
                return val
            if datatype=="f32":
                val_bytes = data[fptr:fptr + 4]
                val = struct.unpack('<f', val_bytes)[0]
                fptr += 4
                return val
            if datatype=="str":
                val_bytes = data[fptr:fptr + 8]
                str_len = struct.unpack('<Q', val_bytes)[0]
                fptr += 8
                val_bytes = data[fptr:fptr + str_len]
                str_val = val_bytes.split(b'\0', 1)[0].decode('utf-8')
                fptr += str_len
                return str_val
            if datatype == "u16":
                val_bytes = data[fptr:fptr + 2]
                val = struct.unpack('<H', val_bytes)[0]
                fptr += 2
                return val
            if datatype == "i16":
                val_bytes = data[fptr:fptr + 2]
                val = struct.unpack('<h', val_bytes)[0]
                fptr += 2
                return val
            if datatype == "u8":
                val_bytes = data[fptr:fptr + 1]
                val = struct.unpack('<B', val_bytes)[0]
                fptr += 1
                return val
            if datatype == "i8":
                val_bytes = data[fptr:fptr + 1]
                val = struct.unpack('<b', val_bytes)[0]
                fptr += 1
                return val
            if datatype=="arr":
                val_bytes = data[fptr:fptr + 4]
                arr_type = struct.unpack('<I', val_bytes)[0]
                fptr += 4
                val_bytes = data[fptr:fptr + 8]
                arr_elems = struct.unpack('<Q', val_bytes)[0]
                fptr += 8
                arr_vals = []
                for i in range(arr_elems):
                    dt_translated = dt_table[arr_type]
                    arr_val = read_data(dt_translated)
                    arr_vals.append(arr_val)
                return arr_vals
            print(f"Unknown Datatype: {datatype}")
            return

        fsize = os.path.getsize(file_path)
        if fsize < 512: #ignore files under file size limit
            print("This GGUF file is too small to analyze. Please ensure it is valid.")
            return
        with open(file_path, 'rb') as f:
            file_header = f.read(4)
            if file_header != b'GGUF': #file is not GGUF
                print(f"File does not seem to be a GGUF: {file_header}")
                return
            data = f.read(chunk_size)
            read_ver = read_data("u32")
            if read_ver < 2:
                print(f"This GGUF file is too old. Version detected: {read_ver}")
                return
            read_tensorcount = read_data("u64")
            read_kvcount = read_data("u64")
            print(f"*** GGUF FILE METADATA ***\nGGUF.version = {read_ver}\nGGUF.tensor_count = {read_tensorcount}\nGGUF.kv_count = {read_kvcount}")
            for kn in range(read_kvcount):
                curr_key = read_data("str")
                curr_datatype = read_data("u32")
                dt_translated = dt_table[curr_datatype]
                curr_val = read_data(dt_translated)
                if dt_translated=="arr":
                    print(f"{dt_translated}: {curr_key} = [{len(curr_val)}]")
                elif dt_translated=="str":
                    print(f"{dt_translated}: {curr_key} = {curr_val[:256]}")
                else:
                    print(f"{dt_translated}: {curr_key} = {curr_val}")
            print("\n*** GGUF TENSOR INFO ***")
            for kn in range(read_tensorcount):
                tensor_name = read_data("str")
                dims = read_data("u32")
                dim_val_str = "["
                for d in range(dims):
                    dim_val = read_data("u64")
                    dim_val_str += f"{'' if d==0 else ', '}{dim_val}"
                dim_val_str += "]"
                tensor_type = read_data("u32")
                read_data("u64") # tensor_offset not used
                tensor_type_str = tt_table[tensor_type]
                print(f"{kn:<3}: {tensor_type_str:<8} | {tensor_name:<30} | {dim_val_str}")
            print(f"Metadata and TensorInfo Bytes: {fptr}")
    except Exception as e:
        print(f"Error Analyzing File: {e}")
        return

def read_gguf_metadata(file_path):
    chunk_size = 16384  # read only first 16kb of file
    try:
        def read_gguf_key(keyname,data,maxval):
            keylen = len(keyname)
            index = data.find(keyname)  # Search for the magic number, Read 2 chunks of 4 byte numbers
            if index != -1 and index + keylen + 8 <= chunk_size:
                start_index = index + keylen
                first_value_bytes = data[start_index:start_index + 4]
                second_value_bytes = data[start_index + 4:start_index + 8]
                # Unpack each 4 bytes as an unsigned int32 in little-endian format
                value1 = struct.unpack('<I', first_value_bytes)[0] #4 means its a uint32
                value2 = struct.unpack('<I', second_value_bytes)[0]
                if value1 == 4 and value2 > 0 and value2 <= maxval:
                    return value2 #contains the desired value
                return 0
            else:
                return 0 #not found

        fsize = os.path.getsize(file_path)
        if fsize < (chunk_size+256): #ignore files under 16kb
            return None
        with open(file_path, 'rb') as f:
            file_header = f.read(4)
            if file_header != b'GGUF': #file is not GGUF
                return None
            data = f.read(chunk_size)
            layercount = read_gguf_key(b'.block_count',data,512)
            head_count_kv = read_gguf_key(b'.attention.head_count_kv',data,8192)
            key_length = read_gguf_key(b'.attention.key_length',data,8192)
            val_length = read_gguf_key(b'.attention.value_length',data,8192)
            return [layercount,head_count_kv, max(key_length,val_length)]
    except Exception:
        return None

def extract_modelfile_params(filepath,sdfilepath,whisperfilepath,mmprojfilepath,draftmodelpath,ttsmodelpath,embdmodelpath,musicllmpath,musicditpath):
    global modelfile_extracted_meta
    modelfile_extracted_meta = None
    sdfsize = 0
    whisperfsize = 0
    mmprojsize = 0
    draftmodelsize = 0
    ttsmodelsize = 0
    embdmodelsize = 0
    musicllmsize = 0
    musicditsize = 0
    if sdfilepath and os.path.exists(sdfilepath):
        sdfsize = os.path.getsize(sdfilepath)
    if whisperfilepath and os.path.exists(whisperfilepath):
        whisperfsize = os.path.getsize(whisperfilepath)
    if mmprojfilepath and os.path.exists(mmprojfilepath):
        mmprojsize = os.path.getsize(mmprojfilepath)
    if draftmodelpath and os.path.exists(draftmodelpath):
        draftmodelsize = os.path.getsize(draftmodelpath)
    if ttsmodelpath and os.path.exists(ttsmodelpath):
        ttsmodelsize = os.path.getsize(ttsmodelpath)
    if embdmodelpath and os.path.exists(embdmodelpath):
        embdmodelsize = os.path.getsize(embdmodelpath)
    if musicllmpath and os.path.exists(musicllmpath):
        musicllmsize = os.path.getsize(musicllmpath)
    if musicditpath and os.path.exists(musicditpath):
        musicditsize = os.path.getsize(musicditpath)
    if filepath and os.path.exists(filepath):
        try:
            fsize = os.path.getsize(filepath)
            if fsize>10000000: #dont bother with models < 10mb as they are probably bad
                ggufmeta = read_gguf_metadata(filepath)
                modelfile_extracted_meta = [filepath,ggufmeta,fsize,sdfsize,whisperfsize,mmprojsize,draftmodelsize,ttsmodelsize,embdmodelsize,musicllmsize,musicditsize] #extract done. note that meta may be null
        except Exception:
            modelfile_extracted_meta = None

def autoset_gpu_layers(ctxsize, sdquanted, bbs, musiclowvram): #shitty algo to determine how many layers to use
    global showusedmemwarning, showmultigpuwarning, modelfile_extracted_meta, calulated_gpu_overhead # reference cached values instead
    gpumem = MaxMemory[0]
    usedmem = 0
    if MaxFreeMemory[0]>0:
        usedmem = MaxMemory[0]-MaxFreeMemory[0]
        if showusedmemwarning and usedmem > (2.5*1024*1024*1024):
            showusedmemwarning = False
            print(f"Note: KoboldCpp has detected that a significant amount of GPU VRAM ({usedmem/1024/1024} MB) is currently used by another application.\nFor best results, you may wish to close that application and then restart KoboldCpp.\n***")
    reservedmem = max(1.25*1024*1024*1024,(0.5*1024*1024*1024 + usedmem)) # determine vram overhead
    try:
        if not modelfile_extracted_meta:
            return 0
        layerlimit = 0
        fsize = modelfile_extracted_meta[2]
        fname = modelfile_extracted_meta[0]
        if fsize > (10*1024*1024): #dont bother with models < 10mb
            cs = ctxsize
            mem = gpumem
            if "-00001-of-00" in fname:
                match = re.search(r'-(\d{5})-of-(\d{5})\.', fname)
                if match:
                    total_parts = int(match.group(2))
                    if total_parts > 1 and total_parts <= 999:
                        if showmultigpuwarning:
                            showmultigpuwarning = False
                            print("Multi-Part GGUF detected. Layer estimates may not be very accurate - recommend setting layers manually.")
                        fsize *= total_parts

            calulated_gpu_overhead = 0
            unsubmitted_overhead = 0 #this overhead is used to calculate for local estimate but not sent to backend
            musicoh1 = 0
            musicoh2 = 0
            if modelfile_extracted_meta[3] > 1024*1024*1024*5: #sdxl tax
                calulated_gpu_overhead += 1024*1024*1024*(9 - sdquanted * 1.5) # 9, 7.5, 6
            elif modelfile_extracted_meta[3] > 1024*1024*512: #normal sd tax
                calulated_gpu_overhead += 1024*1024*1024*(4.25 - sdquanted * 0.5) # 4.25, 3.75, 3.25
            if modelfile_extracted_meta[4] > 1024*1024*10: #whisper tax
                calulated_gpu_overhead += max(350*1024*1024,modelfile_extracted_meta[4]*1.5)
            if modelfile_extracted_meta[5] > 1024*1024*10: #mmproj tax (now internal to kcpp)
                unsubmitted_overhead += max(350*1024*1024,modelfile_extracted_meta[5]*1.5)
            if modelfile_extracted_meta[6] > 1024*1024*10: #draft model tax
                calulated_gpu_overhead += (modelfile_extracted_meta[6] * 1.5)
            if modelfile_extracted_meta[7] > 1024*1024*10: #tts model tax
                if modelfile_extracted_meta[7] < 1024*1024*1024: #less than 1gb probably means outetts, which needs more vram
                    calulated_gpu_overhead += max(600*1024*1024, modelfile_extracted_meta[7] * 3)
                else:
                    calulated_gpu_overhead += max(600*1024*1024, (150*1024*1024 + modelfile_extracted_meta[7] * 1.3))
            if modelfile_extracted_meta[8] > 1024*1024*10: #embeddings model tax
                calulated_gpu_overhead += max(350*1024*1024, modelfile_extracted_meta[8] * 1.5)
            if modelfile_extracted_meta[9] > 1024*1024*10: #music llm tax
                musicoh1 = modelfile_extracted_meta[9] * 1.05
            if modelfile_extracted_meta[10] > 1024*1024*10: #music dit tax
                musicoh2 = modelfile_extracted_meta[10] * 1.05 + (600*1024*1024)
            if musiclowvram:
                calulated_gpu_overhead += max(musicoh1,musicoh2)
            else:
                calulated_gpu_overhead += musicoh1 + musicoh2

            mem -= calulated_gpu_overhead
            mem -= unsubmitted_overhead
            mem = 0 if mem < 0 else mem

            csmul = (cs/4096) if cs >= 8192 else 1.8 if cs > 4096 else 1.2 if cs > 2048 else 1.0
            ggufmeta = modelfile_extracted_meta[1]
            if not ggufmeta or ggufmeta[0]==0: #fail to read or no layers
                sizeperlayer = fsize*csmul*0.052
                layerlimit = int(min(200,(mem-usedmem)/sizeperlayer))
            else:
                layers = ggufmeta[0]
                headcount = ggufmeta[1]
                headkvlen = (ggufmeta[2] if ggufmeta[2] > 0 else 128)
                ratio = (mem-usedmem)/(fsize*csmul*1.6*(1.0 if bbs <= 512 else 1.2))
                if headcount > 0:
                    # rubbish random formula. apply batchsize calculations if over 512
                    fattn_discount = 1.0
                    mem1 = layers*(4 if bbs <= 512 else (bbs/128))*headkvlen*cs*fattn_discount*4*1.45
                    mem2 = layers*headcount*headkvlen*cs*fattn_discount*4*1.15
                    ratio = max(ratio,(mem - reservedmem - mem1) / (fsize + mem2))
                layerlimit = min(int(ratio*layers), (layers + 1))
        layerlimit = (0 if layerlimit<=2 else layerlimit)
        return layerlimit
    except Exception:
        return 0

def detect_memory_cu(gpumem_ignore_limit_min, gpumem_ignore_limit_max):
        FetchedCUdevices = []
        FetchedCUdeviceMem = []
        FetchedCUfreeMem = []

        AMDgpu = None
        try: # Get NVIDIA GPU names
            output = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10).stdout
            FetchedCUdevices = [line.split(",")[0].strip() for line in output.splitlines()]
            FetchedCUdeviceMem = [line.split(",")[1].strip().split(" ")[0].strip() for line in output.splitlines()]
            FetchedCUfreeMem = [line.split(",")[2].strip().split(" ")[0].strip() for line in output.splitlines()]
        except Exception:
            FetchedCUdeviceMem = []
            FetchedCUfreeMem = []
            pass
        if len(FetchedCUdevices)==0:
            try: # Get AMD ROCm GPU names and VRAM from rocminfo
                output = subprocess.run(['rocminfo'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10).stdout
                device_name = None
                current_agent_is_gpu = False
                in_pool_section = False

                for line in output.splitlines(): # read through the output line by line
                    line = line.strip()
                    if line.startswith("Agent ") and "Agent" in line:
                        # Reset state for new agent
                        device_name = None
                        current_agent_is_gpu = False
                        in_pool_section = False
                    elif line.startswith("Marketing Name:"):
                        device_name = line.split(":", 1)[1].strip() # if we find a named device, temporarily save the name
                    elif line.startswith("Device Type:") and "GPU" in line and device_name is not None:
                        # if the following Device Type is a GPU (not a CPU) then add it to devices list
                        FetchedCUdevices.append(device_name)
                        current_agent_is_gpu = True
                        AMDgpu = True
                    elif line.startswith("Device Type:") and "GPU" not in line:
                        device_name = None
                        current_agent_is_gpu = False
                    elif line.startswith("Pool Info:") and current_agent_is_gpu:
                        in_pool_section = True
                    elif in_pool_section and current_agent_is_gpu and line.startswith("Segment:") and "GLOBAL" in line and "COARSE GRAINED" in line:
                        # This is the main VRAM pool for this GPU
                        continue
                    elif in_pool_section and current_agent_is_gpu and line.startswith("Size:"):
                        # Extract VRAM size in KB and convert to MB
                        size_match = re.search(r'(\d+)\(0x[0-9a-fA-F]+\)\s*KB', line)
                        if size_match:
                            vram_kb = int(size_match.group(1))
                            vram_mb = vram_kb // 1024
                            FetchedCUdeviceMem.append(str(vram_mb))
                            in_pool_section = False

                if FetchedCUdevices and FetchedCUdeviceMem:
                    print(f"Detected AMD GPU VRAM from rocminfo: {list(zip(FetchedCUdevices, FetchedCUdeviceMem))} MB")
            except Exception:
                FetchedCUdeviceMem = []
                FetchedCUfreeMem = []
                pass
        lowestcumem = 0
        lowestfreecumem = 0
        try:
            for idx in range(0,4):
                if(len(FetchedCUdevices)>idx):
                    CUDevicesNames[idx] = FetchedCUdevices[idx]
            for idx in range(0,4):
                if(len(FetchedCUdevices)>idx):
                    if len(FetchedCUdeviceMem)>idx:
                        dmem = (int(FetchedCUdeviceMem[idx])*1024*1024) if AMDgpu else (int(FetchedCUdeviceMem[idx])*1024*1024)
                        lowestcumem = dmem if lowestcumem==0 else (dmem if dmem<lowestcumem else lowestcumem)
                    if len(FetchedCUfreeMem)>idx:
                        dmem = (int(FetchedCUfreeMem[idx])*1024*1024)
                        lowestfreecumem = dmem if lowestfreecumem==0 else (dmem if dmem<lowestfreecumem else lowestfreecumem)
        except Exception:
            lowestcumem = 0
            lowestfreecumem = 0

        return lowestcumem, lowestfreecumem


def detect_memory_vk(gpumem_ignore_limit_min, gpumem_ignore_limit_max):

        try: # Get Vulkan names
            foundVkGPU = False
            lowestvkmem = 0
            output = subprocess.run(['vulkaninfo','--summary'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10).stdout
            devicelist = [line.split("=")[1].strip() for line in output.splitlines() if "deviceName" in line]
            devicetypes = [line.split("=")[1].strip() for line in output.splitlines() if "deviceType" in line]
            idx = 0
            for dname in devicelist:
                if idx<len(VKDevicesNames):
                    VKDevicesNames[idx] = dname
                    idx += 1
            if len(devicetypes) == len(devicelist):
                idx = 0
                for dvtype in devicetypes:
                    if idx<len(VKIsDGPU):
                        typeflag = (1 if dvtype=="PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" else 0)
                        VKIsDGPU[idx] = typeflag
                        if typeflag:
                            foundVkGPU = True
                        idx += 1

            if foundVkGPU:
                try: # Try get vulkan memory (experimental)
                    output = subprocess.run(['vulkaninfo'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10).stdout
                    devicechunks = output.split("VkPhysicalDeviceMemoryProperties")[1:]
                    gpuidx = 0
                    for chunk in devicechunks:
                        heaps = chunk.split("memoryTypes:")[0].split("memoryHeaps[")[1:]
                        for heap in heaps:  # Check all heaps, not just the first one
                            if "MEMORY_HEAP_DEVICE_LOCAL_BIT" in heap and "size" in heap:
                                match = re.search(r"size\s*=\s*(\d+)", heap)
                                if match:
                                    dmem = int(match.group(1))
                                    if dmem > gpumem_ignore_limit_min and dmem < gpumem_ignore_limit_max:
                                        lowestvkmem = dmem if lowestvkmem==0 else (dmem if dmem<lowestvkmem else lowestvkmem)
                        gpuidx += 1
                except Exception: # failed to get vulkan vram
                    pass
            return lowestvkmem
        except Exception:
            pass

        return 0


def fetch_gpu_properties(testCU,testVK,testmemory=False):
    gpumem_ignore_limit_min = 1024*1024*600 #600 mb min
    gpumem_ignore_limit_max = 1024*1024*1024*300 #300 gb max

    if testCU:
        cumem, freecumem = detect_memory_cu(gpumem_ignore_limit_min, gpumem_ignore_limit_max)
        MaxMemory[0] = max(cumem,MaxMemory[0])
        MaxFreeMemory[0] = max(freecumem,MaxFreeMemory[0])
        if testmemory:
            print(f'detected CUDA memory: {cumem/(1024*1024)} MB, {freecumem/(1024*102)} MB free')

    if testVK:
        vkmem = detect_memory_vk(gpumem_ignore_limit_min, gpumem_ignore_limit_max)
        MaxMemory[0] = max(vkmem,MaxMemory[0])
        if testmemory:
            print(f'detected Vulkan memory: {vkmem/(1024*1024)} MB')

    # Check VRAM detection after all backends have been tested
    if MaxMemory[0] < (1024*1024*256):
        print("Unable to detect VRAM.")

    return

def auto_set_backend_cli():
    fetch_gpu_properties(True,True)
    found_new_backend = False

    # check for avx2 and avx support
    is_oldpc_ver = "Use CPU" not in runopts #on oldcpu ver, default lib does not exist
    cpusupport = old_cpu_check() # 0 if has avx2, 1 if has avx, 2 if has nothing
    eligible_cuda = (cpusupport<1 and not is_oldpc_ver) or (cpusupport<2 and is_oldpc_ver)
    if not eligible_cuda:
        if cpusupport==1:
            args.noavx2 = True
        elif cpusupport==2:
            args.noavx2 = True
            args.failsafe = True

    if eligible_cuda and exitcounter < 100 and MaxMemory[0]>3500000000 and (("Use CUDA" in runopts and CUDevicesNames[0]!="") or "Use hipBLAS (ROCm)" in runopts) and any(CUDevicesNames):
        if "Use CUDA" in runopts or "Use hipBLAS (ROCm)" in runopts:
            args.usecuda = ["normal","mmq"]
            print(f"Auto Selected CUDA Backend (flag={cpusupport})\n")
            found_new_backend = True
    elif exitcounter < 100 and (1 in VKIsDGPU) and ("Use Vulkan" in runopts or "Use Vulkan (Old CPU)" in runopts):
        for i in range(0,len(VKIsDGPU)):
            if VKIsDGPU[i]==1:
                args.usevulkan = []
                print(f"Auto Selected Vulkan Backend (flag={cpusupport})\n")
                found_new_backend = True
                break
    if not found_new_backend:
        print(f"Auto Selected Default Backend (flag={cpusupport})\n")

def load_model(model_filename):
    global args, calulated_gpu_overhead, savestate_limit
    inputs = load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs.max_context_length = maxctx #initial value to use for ctx, can be overwritten
    inputs.threads = args.threads
    inputs.low_vram = True if args.lowvram else False
    inputs.use_mmq = False if args.nommq else True
    inputs.splitmode = splitmode_choices_to_int(args.splitmode) #layer=1, row=2, tensor=3
    inputs.blasthreads = args.blasthreads
    inputs.use_mmap = args.usemmap
    inputs.use_mlock = args.usemlock
    inputs.lora_filename = "".encode("UTF-8")
    inputs.lora_multiplier = args.loramult
    if args.lora:
        inputs.lora_filename = args.lora[0].encode("UTF-8")

    inputs.draftmodel_filename = args.draftmodel.encode("UTF-8") if args.draftmodel else "".encode("UTF-8")
    inputs.draft_amount = args.draftamount
    inputs.draft_gpulayers = args.draftgpulayers
    for n in range(tensor_split_max):
        if args.draftgpusplit and n < len(args.draftgpusplit):
            inputs.draft_gpusplit[n] = float(args.draftgpusplit[n])
        else:
            inputs.draft_gpusplit[n] = 0
    inputs.mmproj_filename = args.mmproj.encode("UTF-8") if args.mmproj else "".encode("UTF-8")
    inputs.mmproj_cpu = (True if args.mmprojcpu else False)
    inputs.visionmaxres = (512 if args.visionmaxres < 512 else (2048 if args.visionmaxres > 2048 else args.visionmaxres))
    vmintk = args.visionmintokens
    vmaxtk = args.visionmaxtokens
    vmintk = -1 if vmintk<-1 else vmintk
    vmaxtk = -1 if vmaxtk<-1 else vmaxtk
    if(vmintk!=-1 or vmaxtk!=-1) and (vmintk==-1 or vmaxtk==-1): #if exactly one of the args is -1
        vmintk = max(vmintk,vmaxtk)
        vmaxtk = max(vmintk,vmaxtk)
    inputs.visionmintokens = vmintk
    inputs.visionmaxtokens = vmaxtk
    inputs.use_smartcontext = args.smartcontext
    if args.parallelrequests > 1 and not args.noshift:
        print("\nWarning: Continuous batching is enabled, so context shifting has been disabled automatically.\n")
        args.noshift = True
    inputs.use_contextshift = (0 if args.noshift else 1)
    inputs.use_fastforward = (0 if args.nofastforward else 1)
    inputs.flash_attention =  (False if args.noflashattention else True)
    if args.quantkv:
        qkvstr = str(args.quantkv).lower()
        qkvval = 0
        if qkvstr=="bf16" or qkvstr=="3": #migration for old index based values
            qkvval = 1
        elif qkvstr=="q8_0" or qkvstr=="1":
            qkvval = 2
        elif qkvstr=="q5_1":
            qkvval = 3
        elif qkvstr=="q4_0" or qkvstr=="2":
            qkvval = 4
        if args.noflashattention:
            inputs.quant_k = qkvval
            inputs.quant_v = 0 if qkvval!=1 else qkvval
            if qkvval>1:
                print("\nWarning: Quantized KV was used without flash attention! This is NOT RECOMMENDED!\nOnly K cache can be quantized, and performance can suffer.\nIn some cases, it might even use more VRAM when doing a full offload.\nYou are strongly encouraged to use flash attention if you want to use quantkv.")
        else:
            inputs.quant_k = inputs.quant_v = qkvval
    else:
        inputs.quant_k = inputs.quant_v = 0
    inputs.batchsize = args.batchsize
    inputs.autofit = args.autofit
    inputs.autofit_tax_mb = int(args.autofitpadding) + int(calulated_gpu_overhead/(1024*1024))
    inputs.gpulayers = args.gpulayers
    if args.overridenativecontext and args.overridenativecontext>0:
        inputs.overridenativecontext = args.overridenativecontext
        inputs.rope_freq_scale = 0
        inputs.rope_freq_base = 10000
    else:
        inputs.overridenativecontext = 0
        inputs.rope_freq_scale = args.ropeconfig[0]
        if len(args.ropeconfig)>1:
            inputs.rope_freq_base = args.ropeconfig[1]
        else:
            inputs.rope_freq_base = 10000

    for n in range(tensor_split_max):
        if args.tensor_split and n < len(args.tensor_split):
            inputs.tensor_split[n] = float(args.tensor_split[n])
        else:
            inputs.tensor_split[n] = 0

    inputs.moe_experts = args.moeexperts
    inputs.no_bos_token = args.nobostoken
    inputs.load_guidance = args.enableguidance
    okv = []
    if args.overridekv and str(args.overridekv).count(",")>0 and str(args.overridekv).count("=")>1 and str(args.overridekv).count(":")==str(args.overridekv).count("="):
        okv = [x.strip() for x in str(args.overridekv).split(",")]
        okv = [item for item in okv if item and item.strip()]
    elif args.overridekv:
        okv = [args.overridekv]
    for n in range(overridekv_max):
        if not okv or n >= len(okv):
            inputs.override_kv[n] = "".encode("UTF-8")
        else:
            inputs.override_kv[n] = okv[n].encode("UTF-8")
    inputs.override_tensors = args.overridetensors.encode("UTF-8") if args.overridetensors else "".encode("UTF-8")
    inputs.moecpu = (200 if args.moecpu > 200 else args.moecpu)
    inputs.check_slowness = (not args.highpriority and os.name == 'nt' and 'Intel' in platform.processor())
    inputs.jinja_template = preloaded_custom_jinja.encode("UTF-8")
    inputs.highpriority = args.highpriority
    inputs.prevent_swa = args.noswa
    inputs.swa_padding = 0 if args.noswa else args.swapadding
    scint = int(args.smartcache)
    inputs.smartcache = False if scint<=0 else True
    sclimit = (savestate_limit_default if scint<=1 else scint)
    savestate_limit = sclimit
    inputs.smartcacheslots = sclimit
    inputs.pipelineparallel = (not args.nopipelineparallel)
    inputs.continuous_batching_slots = args.parallelrequests if (args.parallelrequests>1) else 0
    inputs.rpc_mode = (2 if args.rpcmode=="host" else (1 if args.rpcmode=="connect" else 0))
    inputs.rpc_targets = (args.rpctargets if args.rpcmode=="connect" else "").encode("UTF-8")

    inputs = set_backend_props(inputs)
    ret = handle.load_model(inputs)
    return ret

def generate(genparams, stream_flag=False):
    global maxctx, args, currentusergenkey, totalgens, pendingabortkey
    default_adapter = {} if chatcompl_adapter is None else chatcompl_adapter
    adapter_obj = genparams.get('adapter', default_adapter)

    prompt = genparams.get('prompt', "")
    memory = genparams.get('memory', "")
    negative_prompt = genparams.get('negative_prompt', "")
    guidance_scale = tryparsefloat(genparams.get('guidance_scale', 1.0),1.0)
    images = genparams.get('images', [])
    audio = genparams.get('audio', [])
    max_context_length = tryparseint(genparams.get('max_context_length', maxctx),maxctx)
    max_length = tryparseint(genparams.get('max_length', args.defaultgenamt),args.defaultgenamt)
    temperature = tryparsefloat(genparams.get('temperature', adapter_obj.get("temperature", 0.75)),0.75)
    top_k = tryparseint(genparams.get('top_k', adapter_obj.get("top_k", 100)),100)
    top_a = tryparsefloat(genparams.get('top_a', 0.0),0.0)
    top_p = tryparsefloat(genparams.get('top_p', adapter_obj.get("top_p", 0.92)),0.92)
    min_p = tryparsefloat(genparams.get('min_p', adapter_obj.get("min_p", 0.0)),0.0)
    typical_p = tryparsefloat(genparams.get('typical', 1.0),1.0)
    tfs = tryparsefloat(genparams.get('tfs', 1.0),1.0)
    nsigma = tryparsefloat(genparams.get('nsigma', 0.0),0.0)
    rep_pen = tryparsefloat(genparams.get('rep_pen', adapter_obj.get("rep_pen", 1.0)),1.0)
    rep_pen_range = tryparseint(genparams.get('rep_pen_range', 320),320)
    rep_pen_slope = tryparsefloat(genparams.get('rep_pen_slope', 1.0),1.0)
    presence_penalty = tryparsefloat(genparams.get('presence_penalty', 0.0),0.0)
    mirostat = tryparseint(genparams.get('mirostat', 0),0)
    mirostat_tau = tryparsefloat(genparams.get('mirostat_tau', 5.0),5.0)
    mirostat_eta = tryparsefloat(genparams.get('mirostat_eta', 0.1),0.1)
    dry_multiplier = tryparsefloat(genparams.get('dry_multiplier', 0.0),0.0)
    dry_base = tryparsefloat(genparams.get('dry_base', 1.75),1.75)
    dry_allowed_length = tryparseint(genparams.get('dry_allowed_length', 2),2)
    dry_penalty_last_n = tryparseint(genparams.get('dry_penalty_last_n', 320),320)
    dry_sequence_breakers = genparams.get('dry_sequence_breakers', [])
    xtc_threshold = tryparsefloat(genparams.get('xtc_threshold', 0.2),0.2)
    xtc_probability = tryparsefloat(genparams.get('xtc_probability', 0),0)
    sampler_order = genparams.get('sampler_order', [6, 0, 1, 3, 4, 2, 5])
    seed = tryparseint(genparams.get('sampler_seed', -1),-1)
    stop_sequence = genparams.get('stop_sequence', [])
    ban_eos_token = genparams.get('ban_eos_token', False)
    stream_sse = stream_flag
    grammar = genparams.get('grammar', '')

    #translate grammar if its json
    try:
        grammarjson = json.loads(grammar)
        decoded = convert_json_to_gbnf(grammarjson)
        if decoded:
            grammar = decoded
    except Exception:
        pass
    grammar_retain_state = genparams.get('grammar_retain_state', False)
    genkey = genparams.get('genkey', '')
    trimstop = genparams.get('trim_stop', True)
    dynatemp_range = tryparsefloat(genparams.get('dynatemp_range', 0.0),0.0)
    dynatemp_exponent = tryparsefloat(genparams.get('dynatemp_exponent', 1.0),1.0)
    smoothing_factor = tryparsefloat(genparams.get('smoothing_factor', 0.0),0.0)
    smoothing_curve = tryparsefloat(genparams.get('smoothing_curve', 1.0),1.0)
    adaptive_target = tryparsefloat(genparams.get('adaptive_target', -1.0),-1.0)
    adaptive_decay = tryparsefloat(genparams.get('adaptive_decay', 0.9),0.9)
    adaptive_decay = 0.01 if adaptive_decay < 0.01 else (0.99 if adaptive_decay > 0.99 else adaptive_decay)
    if adaptive_target>0 and min_p<=0 and top_p>=1.0: #adaptive p sampler requires a truncation sampler first, force a tiny min-p
        min_p = 0.002
    logit_biases = genparams.get('logit_bias', {})
    render_special = genparams.get('render_special', False)
    banned_strings = genparams.get('banned_strings', []) # SillyTavern uses that name
    banned_tokens = genparams.get('banned_tokens', banned_strings)
    bypass_eos_token = genparams.get('bypass_eos', False)
    tool_call_fix = genparams.get('using_openai_tools', False)
    custom_token_bans = genparams.get('custom_token_bans', '')

    for tok in custom_token_bans.split(','):
        tok = tok.strip()  # Remove leading/trailing whitespace
        if tok.isdigit():
            logit_biases[tok] = bias_min_value

    inputs = generation_inputs()
    inputs.prompt = prompt.encode("UTF-8")
    inputs.memory = memory.encode("UTF-8")
    inputs.negative_prompt = negative_prompt.encode("UTF-8")
    inputs.guidance_scale = guidance_scale

    images = images[-images_max:]
    inputs.images_len = len(images)
    inputs.images = (ctypes.c_char_p * inputs.images_len)()
    for n, item in enumerate(images):
        inputs.images[n] = item.encode("UTF-8")
    audio = audio[-audio_max:]
    inputs.audio_len = len(audio)
    inputs.audio = (ctypes.c_char_p * inputs.audio_len)()
    for n, item in enumerate(audio):
        inputs.audio[n] = item.encode("UTF-8")

    global showmaxctxwarning
    if max_context_length > maxctx:
        if showmaxctxwarning:
            print(f"\n!!! ====== !!!\n(Warning! Request max_context_length={max_context_length} exceeds allocated context size of {maxctx}. It will be reduced to fit. Consider launching with increased --contextsize to avoid issues. This message will only show once per session.)\n!!! ====== !!!")
            showmaxctxwarning = False
        max_context_length = maxctx
    min_remain_hardlimit = max(min(max_context_length-4, 16),int(max_context_length*0.2))
    min_remain_softlimit = max(min(max_context_length-4, 16),int(max_context_length*0.4))
    if args.genlimit > 0 and max_length > args.genlimit:
        max_length = args.genlimit
    if max_length >= (max_context_length-min_remain_softlimit):
        print(f"\n!!! ====== !!!\nWarning: You are trying to generate text with max_length ({max_length}) near or exceeding max_context_length limit ({max_context_length}).\nMost of the context will be removed, and your outputs will not be very coherent.\nConsider launching with increased --contextsize to avoid issues.\n!!! ====== !!!")
        if max_length >= (max_context_length-min_remain_hardlimit):
            max_length = max_context_length-min_remain_hardlimit

    reasoning_effort = genparams.get('reasoning_effort', '')
    reasoning_effort = reasoning_effort.strip().lower() if reasoning_effort else ''
    reasoning_budget = -1
    if reasoning_effort == "none":
        reasoning_budget = 0
    elif reasoning_effort == "minimal":
        reasoning_budget = tryparseint(0.1 * max_length,-1)  # 10% of gen amount
    elif reasoning_effort == "low":
        reasoning_budget = tryparseint(0.25 * max_length,-1)  # 25% of gen amount
    elif reasoning_effort == "medium":
        reasoning_budget = tryparseint(0.5 * max_length,-1)  # 50% of gen amount
    else:
        pass #unrestricted

    inputs.max_context_length = max_context_length   # this will resize the context buffer if changed
    inputs.max_length = max_length
    inputs.temperature = temperature
    inputs.top_k = top_k
    inputs.top_a = top_a
    inputs.top_p = top_p
    inputs.min_p = min_p
    inputs.typical_p = typical_p
    inputs.tfs = tfs
    inputs.nsigma = nsigma
    inputs.rep_pen = rep_pen
    inputs.rep_pen_range = rep_pen_range
    inputs.rep_pen_slope = rep_pen_slope
    inputs.presence_penalty = presence_penalty
    inputs.stream_sse = stream_sse
    inputs.dynatemp_range = dynatemp_range
    inputs.dynatemp_exponent = dynatemp_exponent
    inputs.smoothing_factor = smoothing_factor
    inputs.smoothing_curve = smoothing_curve
    inputs.adaptive_target = adaptive_target
    inputs.adaptive_decay = adaptive_decay
    inputs.grammar = grammar.encode("UTF-8")
    inputs.grammar_retain_state = grammar_retain_state
    inputs.allow_eos_token = not ban_eos_token
    inputs.bypass_eos_token = bypass_eos_token
    inputs.tool_call_fix = tool_call_fix
    inputs.render_special = render_special
    if mirostat in (1, 2):
        inputs.mirostat = mirostat
        inputs.mirostat_tau = mirostat_tau
        inputs.mirostat_eta = mirostat_eta
    else:
        inputs.mirostat = inputs.mirostat_tau = inputs.mirostat_eta = 0
    inputs.dry_multiplier = dry_multiplier
    inputs.dry_base = dry_base
    inputs.xtc_threshold = xtc_threshold
    inputs.xtc_probability = xtc_probability
    inputs.dry_allowed_length = dry_allowed_length
    inputs.dry_penalty_last_n = dry_penalty_last_n
    # Handle dry_sequence_breakers being passed as a json-encoded array of
    # strings, rather than as an array of strings itself. This is to support
    # SillyTavern, which passes sequence breakers to Oobabooga that way.
    if dry_multiplier > 0 and isinstance(dry_sequence_breakers, str):
        try:
            dry_sequence_breakers = json.loads(dry_sequence_breakers)
        except ValueError as e:
            print(f"ERROR: dry_sequence_breakers must be an array of strings or a json encoded array of strings. Could not parse '{dry_sequence_breakers}': " + str(e))
            dry_sequence_breakers = []

    if dry_multiplier <= 0 or dry_sequence_breakers is None: # prevent explicitly set to None, retain old behavior
        dry_sequence_breakers = []

    dry_sequence_breakers = dry_sequence_breakers[:dry_seq_break_max]
    inputs.dry_sequence_breakers_len = len(dry_sequence_breakers)
    inputs.dry_sequence_breakers = (ctypes.c_char_p * inputs.dry_sequence_breakers_len)()

    for n, breaker in enumerate(dry_sequence_breakers):
        inputs.dry_sequence_breakers[n] = breaker.encode("UTF-8")

    if sampler_order and 0 < len(sampler_order) <= sampler_order_max:
        try:
            for i, sampler in enumerate(sampler_order):
                inputs.sampler_order[i] = sampler
            inputs.sampler_len = len(sampler_order)
            global showsamplerwarning
            if showsamplerwarning and inputs.mirostat==0 and inputs.sampler_len>0 and (inputs.sampler_order[0]!=6 or inputs.sampler_order[inputs.sampler_len-1]!=5):
                print("\n(Note: Non-default sampler_order detected. Recommended sampler values are [6,0,1,3,4,2,5]. This message will only show once per session.)")
                showsamplerwarning = False
        except TypeError as e:
            print("ERROR: sampler_order must be a list of integers: " + str(e))
    inputs.seed = seed

    inputs.stop_sequence_len = len(stop_sequence)
    inputs.stop_sequence = (ctypes.c_char_p * inputs.stop_sequence_len)()

    for n, sequence in enumerate(stop_sequence):
        if sequence:
            inputs.stop_sequence[n] = sequence.encode("UTF-8")
        else:
            inputs.stop_sequence[n] = "".encode("UTF-8")

    bias_list = []
    try:
        if logit_biases and len(logit_biases) > 0:
            bias_list = [{"key": key, "value": value} for key, value in logit_biases.items()]
    except Exception as ex:
        print(f"Logit bias dictionary is invalid: {ex}")

    bias_list = bias_list[:logit_bias_max]
    inputs.logit_biases_len = len(bias_list)
    inputs.logit_biases = (logit_bias * inputs.logit_biases_len)()
    for n, lb in enumerate(bias_list):
        try:
            t_id = int(lb['key'])
            bias = float(lb['value'])
            t_id = -1 if t_id < 0 else t_id
            bias = (bias_max_value if bias > bias_max_value else (bias_min_value if bias < bias_min_value else bias))
            inputs.logit_biases[n] = logit_bias(t_id, bias)
        except Exception as ex:
            inputs.logit_biases[n] = logit_bias(-1, 0.0)
            print(f"Skipped unparsable logit bias:{ex}")

    if banned_tokens is None:
        banned_tokens = []
    banned_tokens = banned_tokens[:ban_token_max]
    inputs.banned_tokens_len = len(banned_tokens)
    inputs.banned_tokens = (ctypes.c_char_p * inputs.banned_tokens_len)()
    for n, tok in enumerate(banned_tokens):
        inputs.banned_tokens[n] = tok.encode("UTF-8")

    inputs.reasoning_budget = reasoning_budget

    currentusergenkey = genkey
    totalgens += 1
    #early exit if aborted

    if pendingabortkey!="" and pendingabortkey==genkey:
        print(f"\nDeferred Abort for GenKey: {pendingabortkey}")
        pendingabortkey = ""
        return {"text":"","status":-1,"stopreason":-1, "prompt_tokens":0, "completion_tokens": 0, "total_tokens": 0}
    else:
        batch_request_id = -1
        if args.parallelrequests > 1:
            try:
                batch_request_id = handle.batch_generate_submit(inputs)
            except Exception:
                batch_request_id = -1
        if batch_request_id >= 0:
            genparams['_batch_request_id'] = batch_request_id
            ret = handle.batch_generate_result(batch_request_id)
        else:
            genparams['_batch_fallback'] = True
            ret = handle.generate(inputs)
        outstr = ""
        if ret.status==1:
            outstr = ret.text.decode("UTF-8","ignore")
        if batch_request_id >= 0 and not stream_flag:
            handle.batch_generate_release(batch_request_id)
            genparams.pop('_batch_request_id', None)
            genparams.pop('_batch_expected', None)
            genparams.pop('_batch_fallback', None)
        if trimstop:
            for trim_str in stop_sequence:
                sindex = outstr.find(trim_str)
                if sindex != -1 and trim_str!="":
                    outstr = outstr[:sindex]
        return {"text":outstr,"status":ret.status,"stopreason":ret.stopreason,"prompt_tokens":ret.prompt_tokens, "completion_tokens": ret.completion_tokens}

def continuous_batching_python_eligible(genparams, api_format):
    if not args.parallelrequests or args.parallelrequests <= 1 or api_format <= 0:
        return False
    model_path = str(getattr(args, "model_param", "") or "").lower()
    if model_path and not model_path.endswith(".gguf"):
        utfprint("Batching disabled due to file format",2)
        return False
    if not getattr(args, "noshift", False) or getattr(args, "smartcontext", False) or getattr(args, "draftmodel", "") or getattr(args, "enableguidance", False):
        utfprint("Batching disabled due to loaded settings",2)
        return False
    if genparams.get("negative_prompt") or genparams.get("images") or genparams.get("audio"):
        utfprint("Batching disabled due to media",2)
        return False
    if genparams.get("grammar") or genparams.get("grammar_retain_state") or genparams.get("banned_tokens") or genparams.get("banned_strings"):
        utfprint("Batching disabled due to grammar or bans",2)
        return False
    if tryparsefloat(genparams.get("dry_multiplier", 0), 0) or tryparseint(genparams.get("mirostat", 0), 0) or tryparsefloat(genparams.get("xtc_probability", 0), 0) or tryparsefloat(genparams.get("nsigma", 0), 0):
        utfprint("Batching disabled due to samplers set 1",2)
        return False
    if tryparsefloat(genparams.get("smoothing_factor", 0), 0) or tryparsefloat(genparams.get("adaptive_target", -1), -1) > 0 or genparams.get("using_openai_tools", False):
        utfprint("Batching disabled due to samplers set 2",2)
        return False
    if tryparsefloat(genparams.get("top_a", 0), 0) or tryparsefloat(genparams.get("tfs", 1), 1) != 1 or tryparsefloat(genparams.get("dynatemp_range", 0), 0):
        utfprint("Batching disabled due to samplers set 3",2)
        return False
    if genparams.get("sampler_order") and genparams.get("sampler_order") != [6, 0, 1, 3, 4, 2, 5]:
        utfprint("Batching disabled due to sampler order",2)
        return False
    if genparams.get("reasoning_effort"):
        utfprint("Batching disabled due to reasoning",2)
        return False
    return True

def sd_get_info():
    info = handle.sd_get_info()
    if info.status == 0:
        try:
            return json.loads(info.data)
        except Exception:
            print("An error occurred while decoding sd metadata info")
    else:
        print("An error occurred while getting sd metadata info")
    return {}

sampler_aliases = [
    # sd.cpp name, UI name, aliases
    ['euler',         'Euler',    'k_euler'],
    ['euler_a',       'Euler A',  'k_euler_a', 'euler a', 'euler_ancestral'],
    ['heun',          'Heun',     'k_heun'],
    ['dpm2',          'DPM2',     'k_dpm_2', 'dpm_2'],
    ['lcm',           'LCM',      'k_lcm'],
    ['dpm++2m',       'DPM++ 2M', 'k_dpmpp_2m', 'dpm++ 2m karras', 'dpm++ 2m', 'dpmpp_2m'],
    ['ddim_trailing', 'DDIM',     'ddim'],
    ['res_multistep', 'Res Multistep', 'k_res_multistep', 'res multistep'],
    ['res_2s',        'Res 2s',        'k_res_2s', 'res 2s'],
]

def sd_sampler_canonical_name(name):
    global cached_sd_info
    available = cached_sd_info.get('available_samplers', [])
    alias_map = {}
    for aliases in sampler_aliases:
        for alias in aliases:
            alias_map[alias] = aliases[0]
            alias_map[alias.lower()] = aliases[0]
    cname = alias_map.get(name.lower(), name)
    if cname in available:
        return cname
    return 'default'

def sd_sdapi_samplers():
    global cached_sd_info
    result = []
    available = set(cached_sd_info.get('available_samplers', []))
    # ensure we only advertise supported samplers
    smap = {}
    for aliases in sampler_aliases:
        if aliases[0] in available:
            smap[aliases[1]] = aliases[0:1] + aliases[2:]
            available.remove(aliases[0])
    for sampler in available:
        if sampler not in smap:
            smap[sampler] = []
    result = [{'name': k, 'aliases': v, 'options':{}}
                  for k, v in smap.items()]
    return result


sd_convdirect_choices = ['off', 'vaeonly', 'full']

def sd_convdirect_option(value):
    if not value:
        value = ''
    value = value.lower()
    if value in ['disabled', 'disable', 'none', 'off', '0', '']:
        return 'off'
    elif value in ['vae', 'vaeonly']:
        return 'vaeonly'
    elif value in ['enabled', 'enable', 'on', 'full']:
        return 'full'
    raise argparse.ArgumentTypeError(f"Invalid sdconvdirect option \"{value}\". Must be one of {sd_convdirect_choices}.")

sd_quant_choices = ['off','q8','q4']

def sd_quant_option(value):
    try:
        lvl = sd_quant_choices.index(value)
        return lvl
    except Exception:
        return 0

sd_device_choices = ['CPU', 'main', '0', '1', '2', '3']

def sd_get_device_number(name):
    '''maps device name to device number'''
    if name is None: # default handling should be done by sd_resolve_device
        return None
    if not name:
        return -1
    name = name.lower()
    aliases = {"cpu": -2, "gpu": -1, "": -1, "main": -1, "default": -1}
    if name in aliases:
        return aliases[name]
    return tryparseint(name, -1)

def sd_get_device_name(value):
    '''maps device number to device name'''
    if value <= -2:
        return "CPU"
    if value == -1:
        return "main"
    return str(value)

def sd_resolve_device(name, default_=-1):
    '''maps device name to device number, handling the default device'''
    if name is None:
        name = default_
    if isinstance(name, int):
        name = str(max(name, -2))
    return sd_get_device_number(name)

def sd_load_model(model_filename,vae_filename,t5xxl_filename,clip1_filename,clip2_filename,photomaker_filename,upscaler_filename,audio_vae_filename):
    global args
    inputs = sd_load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    thds = args.threads

    if args.sdthreads and args.sdthreads > 0:
        sdt = int(args.sdthreads)
        if sdt > 0:
            thds = sdt

    inputs.threads = thds
    inputs.quant = args.sdquant
    inputs.flash_attention = args.sdflashattention
    inputs.offload_cpu = args.sdoffloadcpu
    inputs.use_mmap = args.usemmap
    inputs.kcpp_vae_device = sd_resolve_device(args.sdvaedevice, default_sdvaedevice)
    inputs.kcpp_clip_device = sd_resolve_device(args.sdclipdevice, default_sdclipdevice)
    sdconvdirect = sd_convdirect_option(args.sdconvdirect)
    inputs.diffusion_conv_direct = sdconvdirect == 'full'
    inputs.vae_conv_direct = sdconvdirect in ['vaeonly', 'full']
    inputs.taesd = True if args.sdvaeauto else False
    inputs.tiled_vae_threshold = args.sdtiledvae
    inputs.vae_filename = vae_filename.encode("UTF-8")
    inputs.audio_vae_filename = audio_vae_filename.encode("UTF-8")
    inputs.t5xxl_filename = t5xxl_filename.encode("UTF-8")
    inputs.clip1_filename = clip1_filename.encode("UTF-8")
    inputs.clip2_filename = clip2_filename.encode("UTF-8")
    inputs.photomaker_filename = photomaker_filename.encode("UTF-8")
    inputs.upscaler_filename = upscaler_filename.encode("UTF-8")
    inputs.max_vram = (args.sdvramlimit/1024.0) if args.sdvramlimit > 0 else 0

    lora_filenames, lora_multipliers = prepare_initial_lora_multipliers()
    inputs.lora_len = len(lora_filenames)
    inputs.lora_filenames = (ctypes.c_char_p * inputs.lora_len)(*lora_filenames)
    inputs.lora_multipliers = (ctypes.c_float * inputs.lora_len)(*lora_multipliers)
    # auto if no zero-weight lora, dynamic otherwise
    lora_apply_mode = 0 # auto
    if imglora_bypath:
        lora_dynamic = 1 << 3 # accept changes at runtime
        lora_cache   = 1 << 4 if imglora_cached else 0 # cache the preloaded LoRAs
        lora_apply_mode = lora_dynamic | lora_cache
    inputs.lora_apply_mode = lora_apply_mode

    inputs.img_hard_limit = args.sdclamped
    inputs.img_soft_limit = args.sdclampedsoft
    inputs = set_backend_props(inputs)
    inputs.kcpp_main_device = sd_resolve_device(args.sdmaingpu, 'main')
    ret = handle.sd_load_model(inputs)
    return ret

def sd_oai_transform_params(genparams):
    size = genparams.get('size') or ''
    pattern = r'^\D*(\d+)x(\d+)$'
    match = re.fullmatch(pattern, size)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        genparams["width"] = width
        genparams["height"] = height
    return genparams

def sd_comfyui_tranform_params(genparams):
    promptobj = genparams.get('prompt', None)
    if promptobj and isinstance(promptobj, dict):
        for node_id, node_data in promptobj.items():
            class_type = node_data.get("class_type","")
            if class_type == "KSampler" or class_type == "KSamplerAdvanced":
                inp = node_data.get("inputs",{})

                # sampler settings from this node
                genparams["seed"] = inp.get("seed", -1)
                genparams["steps"] = inp.get("steps", 20)
                genparams["cfg_scale"] = inp.get("cfg", 5)
                genparams["sampler_name"] = inp.get("sampler_name", "euler")
                genparams["scheduler"] = inp.get("scheduler", "default")

                pos = inp.get("positive",[]) #positive prompt node
                neg = inp.get("negative",[]) #negative prompt node
                latentimg = inp.get("latent_image",[]) #image size node

                if latentimg and isinstance(latentimg, list) and len(latentimg) > 0:
                    temp = promptobj.get(str(latentimg[0]), {}) #now, this may be a VAEEncode or EmptyLatentImage
                    nodetype = temp.get("class_type", "") #if its a VAEEncode, it will have pixels
                    temp = temp.get('inputs', {})
                    if nodetype=="VAEEncode" and lastuploadedcomfyimg!="": #img2img
                        genparams["init_images"] = [lastuploadedcomfyimg]
                    genparams["width"] = temp.get("width", 512)
                    genparams["height"] = temp.get("height", 512)
                if neg and isinstance(neg, list) and len(neg) > 0:
                    temp = promptobj.get(str(neg[0]), {})
                    temp = temp.get('inputs', {})
                    genparams["negative_prompt"] = temp.get("text", "")
                if pos and isinstance(pos, list) and len(pos) > 0:
                    temp = promptobj.get(str(pos[0]), {})
                    temp = temp.get('inputs', {})
                    genparams["prompt"] = temp.get("text", "")
                    break
        if genparams.get("prompt","")=="": #give up, set generic prompt
            genparams["prompt"] = "high quality"
    else:
        print("Warning: ComfyUI Payload Missing!")
    return genparams

# json with top-level dict
def parse_json_object(value, field, silent=False):
    if not value:
        return None
    broken = False
    if isinstance(value, str):
        retry = False
        try: # Try parsing as-is
            value = json.loads(value)
            retry = False
        except json.JSONDecodeError:
            retry = True

        if retry and ":" in value: # Try wrapping in braces for loose key/value strings
            try:
                value = json.loads(f"{{{value}}}")
                retry = False
            except json.JSONDecodeError:
                retry = True

        if retry and '\\"' in value:  #try handle double escape
            try:
                tmp = json.loads(f"\"{value}\"")
                value = json.loads(tmp)
                retry = False
            except json.JSONDecodeError:
                retry = True

        if retry:
            broken = True
    if isinstance(value, dict):
        return value
    elif broken:
        if value:
            try:
                import ast
                # Convert JS literals to Python literals
                fixed = re.sub(r'\btrue\b', 'True', value)
                fixed = re.sub(r'\bfalse\b', 'False', fixed)
                fixed = re.sub(r'\bnull\b', 'None', fixed)
                value = ast.literal_eval(fixed)
                if value and isinstance(value, dict):
                    return value
            except Exception:
                pass
        if not silent:
            print(f"Warning: couldn't parse {field} field.")
    else:
        if not silent:
            print(f"Warning: {field} field - not a JSON object.")
    return None

def gendefaults_parse_meta_field(value):
    alias_map = {
        'cfg-scale': 'cfg_scale',
        'guidance': 'distilled_guidance',
        'sampler': 'sampler_name',
        'sampling-method': 'sampler_name',
        'timestep-shift': 'shifted_timestep',
        'flow-shift': 'flow_shift',
        'cache-mode': 'cache_mode',
        'cache-options': 'cache_options',
        # match sd.cpp flag
        'cache-option': 'cache_options',
        'cache_option': 'cache_options',
    }
    parsed = parse_json_object(value, 'gendefaults') or {}
    result = {}
    # First pass: apply aliases only if canonical key is not explicitly present
    for key, value in parsed.items():
        canonical = alias_map.get(key, key)
        if canonical not in parsed:
            result[canonical] = value
    result.update(parsed)  # Second pass: explicit keys override aliases
    return result

def sd_upscale(genparams):
    init_images = genparams.get("image", "")
    inputs = sd_upscale_inputs()
    inputs.init_images = init_images.encode("UTF-8")
    inputs.upscaling_resize = tryparseint(genparams.get("upscaling_resize", 2),2) # how many times to upscale
    ret = handle.sd_upscale(inputs)
    data_main = ""
    if ret.status==1:
        data_main = ret.data.decode("UTF-8","ignore")
    return data_main

def sanitize_lora_list(sdlora):
    if not sdlora:
        sdlora = []
    elif isinstance(sdlora, str):
        sdlora = [sdlora]
    elif not isinstance(sdlora, list):
        sdlora = []
    return sdlora

def sanitize_lora_multipliers(sdloramult):
    if sdloramult is None:
        sdloramult = [1.0]
    elif not isinstance(sdloramult, list):
        sdloramult = [sdloramult]
    sdloramult = [tryparsefloat(m, 0.) for m in sdloramult]
    return sdloramult

def prepare_initial_lora_multipliers():
    res_paths = []
    res_multipliers = []
    num_loras = len(imglora_preload)
    if num_loras > lora_filenames_max:
        print(f'Warning: more than {lora_filenames_max} preloaded LoRAs, extra ones will be ignored')
        num_loras = lora_filenames_max
    for info in imglora_preload[:num_loras]:
        res_paths.append(info['fullpath'].encode("UTF-8"))
        res_multipliers.append(info['multiplier'])
    return res_paths, res_multipliers

def prepare_lora_multipliers_backend(request_list, imglora_bypath):
    req_dedup = {}
    for r in request_list:
        if not isinstance(r, dict):
            continue
        path = r.get('path')
        multiplier = tryparsefloat(r.get('multiplier'), 0.)
        if not path or not isinstance(path, str) or not multiplier:
            continue
        info = imglora_bypath.get(path)
        if info:
            fullpath = info["fullpath"]
            req_dedup[fullpath] = req_dedup.get(fullpath, 0.) + multiplier
    res_paths = []
    res_multipliers = []
    for fullpath, multiplier in req_dedup.items():
        if multiplier != 0.0:
            res_paths.append(fullpath.encode("UTF-8"))
            res_multipliers.append(multiplier)
    # enforce lora_filenames_max
    max_requests = lora_filenames_max - len(imglora_preload)
    if len(res_paths) > max_requests:
        msg_preloaded = ""
        if len(imglora_preload) > 0:
            msg_preloaded = f" (including {len(imglora_preload)} preloaded)"
        print(f'Warning: more than {lora_filenames_max} requested LoRAs{msg_preloaded}, extra ones will be ignored')
        res_paths = res_paths[:max_requests]
        res_multipliers = res_multipliers[:max_requests]
    return res_paths, res_multipliers

def prepare_lora_multipliers(request_list):
    return prepare_lora_multipliers_backend(request_list, imglora_bypath)

def mk_sdapi_lora_list(imglora_bypath):
    return [
        {'name': info['name'], 'path': info['path']}
            for info in imglora_bypath.values()
                if not info.get('fixed')
    ]

def extract_loras_from_prompt(prompt):
    pattern = r'<lora:([^:>]+):([^>]+)>'
    lora_data = []
    matches = list(re.finditer(pattern, prompt))
    for match in matches:
        raw_path = match.group(1)
        raw_mul = match.group(2)
        try:
            mul = float(raw_mul)
        except ValueError:
            continue
        is_high_noise = False
        prefix = "|high_noise|"
        if raw_path.startswith(prefix):
            raw_path = raw_path[len(prefix):]
            is_high_noise = True
        item = {'name': raw_path, 'multiplier': mul}
        if is_high_noise:
            item["is_high_noise"] = is_high_noise
        lora_data.append(item)
        prompt = prompt.replace(match.group(0), "", 1)
    return prompt, lora_data

def lora_map_name_to_path(request_list):
    result = []
    for req in request_list:
        out = dict(req)
        name = out.pop('name')
        path = imglora_name2path.get(name)
        if not path:
            print(f'LoRA {name} not found')
            continue
        info = imglora_bypath.get(path)
        if info:
            out['path'] = info['path']
            result.append(out)
    return result

def sd_generate(genparams):
    global maxctx, args, currentusergenkey, totalgens, pendingabortkey, chatcompl_adapter

    job_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    default_adapter = {} if chatcompl_adapter is None else chatcompl_adapter
    adapter_obj = genparams.get('adapter', default_adapter)
    forced_negprompt = adapter_obj.get("add_sd_negative_prompt", "")
    forced_posprompt = adapter_obj.get("add_sd_prompt", "")
    forced_steplimit = tryparseint(adapter_obj.get("add_sd_step_limit", genparams.get("add_sd_step_limit",80)),80)
    forced_maxcfg = tryparsefloat(adapter_obj.get("add_sd_cfg_limit", genparams.get("add_sd_cfg_limit",25)),25)
    allow_remove_limits = tryparseint(adapter_obj.get("remove_limits", genparams.get("remove_limits",0)),0)

    prompt = genparams.get("prompt", "high quality")
    negative_prompt = genparams.get("negative_prompt", "")
    def build_prompt(prompt, forced):
        # truncate before the forced prompt to ensure it can't be "pushed away" by long inputs
        prompt = prompt.encode('utf-8')[:2048].decode("UTF-8", errors='ignore')
        return ", ".join([prompt, forced])
    prompt = build_prompt(prompt, forced_posprompt)
    negative_prompt = build_prompt(negative_prompt, forced_negprompt)
    init_images_arr = genparams.get("init_images", [])
    init_images = ("" if (not init_images_arr or len(init_images_arr)==0 or not init_images_arr[0]) else init_images_arr[0])
    init_images = strip_base64_prefix(init_images)
    mask = strip_base64_prefix(genparams.get("mask", ""))
    flip_mask = genparams.get("inpainting_mask_invert", 0)
    denoising_strength = tryparsefloat(genparams.get("denoising_strength", 0.6),0.6)
    cfg_scale = tryparsefloat(genparams.get("cfg_scale", 5),5)
    distilled_guidance = tryparsefloat(genparams.get("distilled_guidance", None), None)
    shifted_timestep = tryparseint(genparams.get("shifted_timestep", None), None)
    flow_shift = tryparsefloat(genparams.get("flow_shift", None), None)
    sample_steps = tryparseint(genparams.get("steps", 20),20)
    width = tryparseint(genparams.get("width", 512),512)
    height = tryparseint(genparams.get("height", 512),512)
    seed = tryparseint(genparams.get("seed", -1),-1)
    if seed < 0:
        seed = random.randint(100000, 999999)
    sample_method = (genparams.get("sampler_name") or "default")
    scheduler = (genparams.get("scheduler") or "default").lower()
    clip_skip = tryparseint(genparams.get("clip_skip", -1),-1)
    eta = tryparsefloat(genparams.get("eta", None), None)
    vid_req_frames = tryparseint(genparams.get("frames", 1),1)
    vid_req_frames = 1 if (not vid_req_frames or vid_req_frames < 1) else vid_req_frames
    vid_fps = tryparseint(genparams.get("fps", 16),16)
    video_output_type = genparams.get("video_output_type", 0)
    cache_mode = str(genparams.get("cache_mode", ""))
    cache_options = str(genparams.get("cache_options", ""))
    extra_images_arr = genparams.get("extra_images", [])
    extra_images_arr = ([] if not extra_images_arr else extra_images_arr)
    extra_images_arr = [img for img in extra_images_arr if img not in (None, "")]
    extra_images_arr = extra_images_arr[:extra_images_max]
    lora_filenames, lora_multipliers = prepare_lora_multipliers(genparams.get("lora", []))

    #clean vars
    cfg_scale = (1 if cfg_scale < 1 else (forced_maxcfg if cfg_scale > forced_maxcfg else cfg_scale))
    if distilled_guidance is not None and (distilled_guidance < 0 or distilled_guidance > 100):
        distilled_guidance = None # fall back to the default
    if shifted_timestep is not None and (shifted_timestep < 0 or shifted_timestep > 1000):
        shifted_timestep = None # fall back to the default
    if flow_shift is not None and flow_shift < 0:
        flow_shift = None # fall back to the default
    sample_steps = (1 if sample_steps < 1 else (forced_steplimit if sample_steps > forced_steplimit else sample_steps))
    vid_req_frames = (1 if vid_req_frames < 1 else (400 if vid_req_frames > 400 else vid_req_frames))
    vid_fps = (16 if vid_fps < 16 else (32 if vid_fps > 32 else vid_fps))

    swap_refimg = (True if tryparseint(genparams.get("send_as_refimg", 0),0) else False)
    reverse_refimg = (True if tryparseint(genparams.get("reverse_refimg", 0),0) else False)
    if len(extra_images_arr)==0 and swap_refimg and init_images and init_images!="" and not mask:
        extra_images_arr = [init_images]
        init_images = ""

    inputs = sd_generation_inputs()
    inputs.prompt = prompt.encode("UTF-8")
    inputs.negative_prompt = negative_prompt.encode("UTF-8")
    inputs.init_images = init_images.encode("UTF-8")
    inputs.mask = "".encode("UTF-8") if not mask else mask.encode("UTF-8")
    inputs.extra_images_len = len(extra_images_arr)
    inputs.extra_images = (ctypes.c_char_p * inputs.extra_images_len)()
    for n, estr in enumerate(extra_images_arr):
        extra_image = strip_base64_prefix(estr)
        inputs.extra_images[n] = extra_image.encode("UTF-8")
    inputs.reverse_refimg = reverse_refimg
    inputs.flip_mask = flip_mask
    inputs.cfg_scale = cfg_scale
    if distilled_guidance is not None:
        inputs.distilled_guidance = distilled_guidance
    inputs.denoising_strength = (0 if denoising_strength < 0 else (1 if denoising_strength > 1 else denoising_strength))
    if shifted_timestep is not None:
        inputs.shifted_timestep = shifted_timestep
    if flow_shift is not None:
        inputs.flow_shift = flow_shift
    inputs.sample_steps = sample_steps
    inputs.width = width
    inputs.height = height
    inputs.seed = ((seed + 2**31) % 2**32) - 2**31
    inputs.sample_method = sd_sampler_canonical_name(sample_method).encode("UTF-8")
    inputs.scheduler = scheduler.encode("UTF-8")
    inputs.eta = -1.0 if eta is None else eta
    inputs.clip_skip = clip_skip
    inputs.vid_req_frames = vid_req_frames
    inputs.vid_fps = vid_fps
    inputs.video_output_type = video_output_type
    inputs.remove_limits = allow_remove_limits
    inputs.circular_x = tryparseint(adapter_obj.get("circular_x", genparams.get("circular_x",0)),0)
    inputs.circular_y = tryparseint(adapter_obj.get("circular_y", genparams.get("circular_y",0)),0)
    inputs.cache_mode = cache_mode.encode("UTF-8")
    inputs.cache_options = cache_options.encode("UTF-8")
    inputs.upscale = (True if tryparseint(genparams.get("enable_hr", 0),0) else False)
    inputs.lora_len = len(lora_filenames)
    inputs.lora_filenames = (ctypes.c_char_p * inputs.lora_len)(*lora_filenames)
    inputs.lora_multipliers = (ctypes.c_float * inputs.lora_len)(*lora_multipliers)

    ret = handle.sd_generate(inputs)
    data_main = ""
    data_extra = ""
    final_frame = ""
    info = {}
    animated = False
    if ret.status==1:
        data_main = ret.data.decode("UTF-8","ignore")
        data_extra = ret.data_extra.decode("UTF-8","ignore")
        final_frame = ret.final_frame.decode("UTF-8","ignore")
        info = json.loads(ret.info.decode("UTF-8","ignore"))
        animated = True if ret.animated else False
    info["job_timestamp"] = job_timestamp
    return {"animated": animated, "data":data_main, "data_extra":data_extra, "final_frame":final_frame, "info": info}


def whisper_load_model(model_filename):
    global args
    inputs = whisper_load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs = set_backend_props(inputs)
    ret = handle.whisper_load_model(inputs)
    return ret

def whisper_generate(genparams):
    global args
    prompt = genparams.get("prompt", "")
    audio_data = genparams.get("audio_data", "")
    if audio_data.startswith("data:audio"):
        audio_data = audio_data.split(",", 1)[1]
    inputs = whisper_generation_inputs()
    inputs.prompt = prompt.encode("UTF-8")
    inputs.audio_data = audio_data.encode("UTF-8")
    lc = genparams.get("langcode", genparams.get("language", "auto"))
    lc = lc.strip().lower() if (lc and lc.strip().lower()!="") else "auto"
    inputs.langcode = lc.encode("UTF-8")
    inputs.suppress_non_speech = genparams.get("suppress_non_speech", False)
    ret = handle.whisper_generate(inputs)
    outstr = ""
    if ret.status==1:
        outstr = ret.data.decode("UTF-8","ignore")
    return outstr

def tts_load_model(ttc_model_filename,cts_model_filename):
    global args
    inputs = tts_load_model_inputs()
    inputs.ttc_model_filename = ttc_model_filename.encode("UTF-8") if ttc_model_filename else "".encode("UTF-8")
    inputs.cts_model_filename = cts_model_filename.encode("UTF-8") if cts_model_filename else "".encode("UTF-8")
    inputs.gpulayers = (999 if args.ttsgpu else 0)
    inputs.flash_attention = (False if args.noflashattention else True)
    thds = args.threads
    if args.ttsthreads and args.ttsthreads > 0:
        ttst = int(args.ttsthreads)
        if ttst > 0:
            thds = ttst
    inputs.threads = thds
    inputs.ttsmaxlen = args.ttsmaxlen if args.ttsmaxlen < 4096 else 4096
    inputs = set_backend_props(inputs)
    ret = handle.tts_load_model(inputs)
    return ret

def tts_prepare_voice_json(jsonstr):
    try:
        if not jsonstr:
            return None
        parsed_json = json.loads(jsonstr)
        txt = parsed_json.get("text","")
        items = parsed_json.get("words",[])
        processed = ""
        if txt=="" or not items or len(items)<1:
            return None
        for item in items:
            word = item.get("word","")
            duration = item.get("duration","")
            codes = item.get("codes",[])
            codestr = ""
            for c in codes:
                codestr += f"<|{c}|>"
            processed += f"{word}<|t_{duration:.2f}|><|code_start|>{codestr}<|code_end|>\n"
        return {"phrase":txt.strip()+".","voice":processed.strip()}
    except Exception:
        return None

def tts_extract_instruction(x):
    match = re.match(r'^\[([^\]]+)\]\s*(.+)$', x, re.DOTALL)
    if match:
        instruction = match.group(1)
        x1 = match.group(2)
        return x1, (instruction if instruction else "")
    return x, ""

def tts_generate(genparams):
    global args, voicebank, voicelist
    prompt = genparams.get("input", genparams.get("text", ""))
    prompt = prompt.strip()
    voice = 1
    speaker_json = tts_prepare_voice_json(genparams.get("speaker_json","")) #handle custom json voices
    voicestr = genparams.get("voice", genparams.get("speaker_wav", ""))
    oai_voicemap = ["alloy","onyx","echo","nova","shimmer"] # map to kcpp defaults
    voice_mapping = voicelist
    normalized_voice = voicestr.strip().lower() if voicestr else ""
    if normalized_voice.endswith(".wav"):
        normalized_voice = normalized_voice[:-4]
    if normalized_voice in voice_mapping:
        voice = voice_mapping.index(normalized_voice) + 1
    elif normalized_voice in oai_voicemap:
        voice = oai_voicemap.index(normalized_voice) + 1
    else:
        voice = simple_lcg_hash(voicestr.strip()) if voicestr else 1
    inputs = tts_generation_inputs()
    inputs.custom_speaker_voice = normalized_voice.encode("UTF-8")
    ttsinstruction = genparams.get("instruction", "")
    # if no instruction provided, extract from text
    if not genparams.get("instruction", ""):
        prompt, ttsinstruction = tts_extract_instruction(prompt)
    inputs.speaker_instruction = ttsinstruction.encode("UTF-8")
    inputs.prompt = prompt.encode("UTF-8")
    inputs.speaker_seed = voice
    aseed = -1
    try:
        aseed = int(genparams.get("seed", -1))
    except Exception:
        aseed = -1
    inputs.audio_seed = aseed
    if speaker_json:
        inputs.custom_speaker_text = speaker_json.get("phrase","").encode("UTF-8")
        inputs.custom_speaker_data = speaker_json.get("voice","").encode("UTF-8")
        inputs.speaker_seed = 100
    else:
        inputs.custom_speaker_text = "".encode("UTF-8")
        inputs.custom_speaker_data = "".encode("UTF-8")
    reference_audio = voicebank.get(voicestr,"") #for cloned voices in qwen3tts
    if reference_audio and reference_audio.startswith("data:audio"):
        reference_audio = reference_audio.split(",", 1)[1]
    inputs.reference_audio = reference_audio.encode("UTF-8")
    ret = handle.tts_generate(inputs)
    outstr = ""
    if ret.status==1:
        outstr = ret.data.decode("UTF-8","ignore")
    return outstr

def embeddings_load_model(model_filename):
    global args
    inputs = embeddings_load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs.gpulayers = (999 if args.embeddingsgpu else 0)
    inputs.flash_attention = (False if args.noflashattention else True)
    inputs.threads = args.threads
    inputs.use_mmap = args.usemmap
    inputs.embeddingsmaxctx = (args.embeddingsmaxctx if args.embeddingsmaxctx else args.contextsize) # for us to clamp to contextsize if embeddingsmaxctx unspecified
    inputs = set_backend_props(inputs)
    ret = handle.embeddings_load_model(inputs)
    return ret

def embeddings_generate(genparams):
    global args
    prompts = []
    if isinstance(genparams.get('input',[]), list):
        prompts = genparams.get('input',[])
    else:
        prompt = genparams.get("input", "")
        if prompt:
            prompts.append(prompt)

    tokarrs = []
    tokcnt = 0
    for prompt in prompts:
        tokarr = []
        tmpcnt = 0
        try:
            inputs = embeddings_generation_inputs()
            inputs.prompt = prompt.encode("UTF-8")
            inputs.truncate = genparams.get('truncate', True)
            ret = handle.embeddings_generate(inputs)
            if ret.status==1:
                outstr = ret.data.decode("UTF-8","ignore")
                tokarr = json.loads(outstr) if outstr else []
                tmpcnt = ret.count
        except Exception as e:
            tokarr = []
            tmpcnt = 0
            print(f"Error: {e}")
        tokarrs.append(tokarr)
        tokcnt += tmpcnt
    return {"count":tokcnt, "data":tokarrs}

def music_load_model(musicllm,musicembedding,musicdiffusion,musicvae):
    global args
    inputs = music_load_model_inputs()
    inputs.musicllm_filename = musicllm.encode("UTF-8")
    inputs.musicembedding_filename = musicembedding.encode("UTF-8")
    inputs.musicdiffusion_filename = musicdiffusion.encode("UTF-8")
    inputs.musicvae_filename = musicvae.encode("UTF-8")
    inputs.lowvram = True if args.musiclowvram else False
    inputs = set_backend_props(inputs)
    ret = handle.music_load_model(inputs)
    return ret

def music_generate_codes(genparams):
    global args
    input_json = json.dumps(genparams)
    inputs = music_generation_inputs()
    inputs.is_planner_mode = True
    inputs.stereo = genparams.get('stereo', True)
    inputs.use_mp3 = genparams.get('use_mp3', False)
    inputs.gen_codes =  genparams.get('gen_codes', False)
    inputs.rewrite_caption =  genparams.get('rewrite_caption', True)
    inputs.input_json = input_json.encode("UTF-8")
    inputs.music_reference_audio_data = "".encode("UTF-8")
    ret = handle.music_generate(inputs)
    outstr = ""
    if ret.status==1:
        outstr = ret.music_output_json.decode("UTF-8","ignore")
        outstr = json.dumps(json.loads(outstr))
    return outstr

def music_generate_audio(genparams):
    global args
    input_json = json.dumps(genparams)
    inputs = music_generation_inputs()
    inputs.is_planner_mode = False
    inputs.stereo = genparams.get('stereo', True)
    inputs.use_mp3 = genparams.get('use_mp3', False)
    inputs.gen_codes =  genparams.get('gen_codes', False)
    inputs.rewrite_caption =  genparams.get('rewrite_caption', True)
    inputs.input_json = input_json.encode("UTF-8")
    refaudio = genparams.get('music_reference_audio_data', None)
    inputs.music_reference_audio_data = (refaudio.encode("UTF-8") if (refaudio and refaudio!="") else "".encode("UTF-8"))
    ret = handle.music_generate(inputs)
    outstr = ""
    if ret.status==1:
        outstr = ret.data.decode("UTF-8","ignore")
    return outstr

def tokenize_ids(countprompt,tcaddspecial):
    rawcountdata = handle.token_count(countprompt.encode("UTF-8"),tcaddspecial)
    count = rawcountdata.count
    hardlimit = (2**31) - 1
    countlimit = count if (count>=0 and count<=hardlimit) else 0
    if count > hardlimit:
        utfprint("Warning: TokenCount exceeds max limit.")
    # the above protects the server in case the count limit got corrupted
    countdata = [rawcountdata.ids[i] for i in range(countlimit)]
    return countdata

def detokenize_ids(tokids,addspecial):
    tokidslen = len(tokids)
    detokstr = ""
    if tokidslen > 0 and tokidslen < 65536:
        inputs = detokenize_inputs()
        inputs.count = tokidslen
        inputs.special = addspecial
        inputs.ids = (ctypes.c_int * tokidslen)()
        for i, cid in enumerate(tokids):
            inputs.ids[i] = cid
        detok = handle.detokenize(inputs)
        detokstr = ctypes.string_at(detok).decode("UTF-8","ignore")
    return detokstr

# Performs a web search using DuckDuckGo and extracts text content from the top results.
def websearch(query):
    global websearch_lastquery
    global websearch_lastresponse
    global nocertify
    # sanitize query
    query = re.sub(r'[+\-\"\\/*^|<>~`]', '', query) # Remove blacklisted characters
    query = re.sub(r'\s+', ' ', query).strip() # Replace multiple spaces with a single space
    if not query or query=="":
        return []
    query = query[:300] # only search first 300 chars, due to search engine limits
    if query==websearch_lastquery:
        print("Returning cached websearch...")
        return websearch_lastresponse
    import difflib
    from html.parser import HTMLParser
    num_results = 3
    searchresults = []
    utfprint("Performing new websearch...",1)

    def fetch_searched_webpage(url, random_agent=False):
        from urllib.parse import quote, urlsplit, urlunsplit
        uagent = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
        if random_agent:
            agents = ["Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) Gecko/20100101 Firefox/114.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.1823.79 Safari/537.36 Edg/114.0.1823.79",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Safari/537.36"]
            uagent = random.choice(agents)
        if args.debugmode:
            utfprint(f"WebSearch URL: {url}")
        # Encode non-ASCII parts of the URL
        try:
            split_url = urlsplit(url)
            encoded_path = quote(split_url.path)
            encoded_url = urlunsplit((split_url.scheme, split_url.netloc, encoded_path, split_url.query, split_url.fragment))

            ssl_cert_dir = os.environ.get('SSL_CERT_DIR')
            if not ssl_cert_dir and not nocertify and os.name != 'nt':
                os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'

            req = urllib.request.Request(encoded_url, headers={'User-Agent': uagent})
            with urllib.request.urlopen(req, timeout=30) as response:
                html_content = response.read().decode('utf-8', errors='ignore')
                return html_content
        except urllib.error.HTTPError: #we got blocked? try 1 more time with a different user agent
            try:
                req = urllib.request.Request(encoded_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    html_content = response.read().decode('utf-8', errors='ignore')
                    return html_content
            except Exception as e:
                utfprint(f"Error fetching text from URL {url}: {e}",1)
                return ""
        except Exception as e:
            utfprint(f"Error fetching text from URL {url}: {e}",1)
            return ""
    def fetch_webpages_parallel(urls):
        with ThreadPoolExecutor() as executor:
            # Submit tasks and gather results
            results = list(executor.map(fetch_searched_webpage, urls))
        return results

    def normalize_page_text(text):
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove spaces before punctuation
        # text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text) # Ensure a single space follows punctuation, if not at the end of a line
        return text

    class VisibleTextParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.texts = []
            self.is_script_or_style = False
        def handle_starttag(self, tag, attrs):
            if tag in {'script', 'style'}:
                self.is_script_or_style = True
        def handle_endtag(self, tag):
            if tag in {'script', 'style'}:
                self.is_script_or_style = False
        def handle_data(self, data):
            if not self.is_script_or_style and data.strip():
                self.texts.append(data.strip())
        def get_text(self):
            return ' '.join(self.texts)

    class ExtractResultsParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.titles = []
            self.urls = []
            self.descs = []
            self.recordingTitle = False
            self.recordingUrl = False
            self.recordingDesc = False
            self.currsegmenttxt = ""

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                # Check if the "class" attribute matches the target class
                for attr_name, attr_value in attrs:
                    if not self.recordingTitle and attr_name == "class" and "result__a" in attr_value.split():
                        self.recordingTitle = True
                        self.currsegmenttxt = ""
                    if not self.recordingUrl and attr_name == "class" and "result__url" in attr_value.split():
                        self.recordingUrl = True
                        self.currsegmenttxt = ""
                    if not self.recordingDesc and attr_name == "class" and "result__snippet" in attr_value.split():
                        self.recordingDesc = True
                        self.currsegmenttxt = ""

        def handle_endtag(self, tag):
            if tag == "a" and self.recordingTitle:
                self.recordingTitle = False
                self.titles.append(self.currsegmenttxt.strip())
                self.currsegmenttxt = ""
            if tag == "a" and self.recordingUrl:
                self.recordingUrl = False
                self.urls.append(f"https://{self.currsegmenttxt.strip()}")
                self.currsegmenttxt = ""
            if tag == "a" and self.recordingDesc:
                self.recordingDesc = False
                self.descs.append(self.currsegmenttxt.strip())
                self.currsegmenttxt = ""

        def handle_data(self, data):
            if self.recordingTitle or self.recordingDesc or self.recordingUrl:
                self.currsegmenttxt += data

    encoded_query = urllib.parse.quote(query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    try:
        search_html = fetch_searched_webpage(search_url, random_agent=True)
        parser = ExtractResultsParser()
        parser.feed(search_html)
        titles = parser.titles[:num_results]
        searchurls = parser.urls[:num_results]
        descs = parser.descs[:num_results]

        if len(descs)==0 or len(titles)==0 or len(descs)==0:
            utfprint("No results found! Maybe something went wrong...",1)
            return []

        fetchedcontent = fetch_webpages_parallel(searchurls)
        for i in range(len(descs)):
            # dive into the results to try and get even more details
            title = titles[i]
            url = searchurls[i]
            desc = descs[i]
            pagedesc = ""
            try:
                desclen = len(desc)
                html_content = fetchedcontent[i]
                parser2 = VisibleTextParser()
                parser2.feed(html_content)
                scraped = parser2.get_text().strip()
                scraped = normalize_page_text(scraped)
                desc = normalize_page_text(desc)
                s = difflib.SequenceMatcher(None, scraped.lower(), desc.lower(), autojunk=False)
                matches = s.find_longest_match(0, len(scraped), 0, desclen)
                if matches.size > 100 and desclen-matches.size < 100: #good enough match
                    # expand description by some chars both sides
                    expandamtbefore = 200
                    expandamtafter = 800
                    startpt = matches.a - expandamtbefore
                    startpt = 0 if startpt < 0 else startpt
                    endpt =  matches.a + expandamtafter + desclen
                    pagedesc = scraped[startpt:endpt].strip()
            except Exception:
                pass
            searchresults.append({"title":title,"url":url,"desc":desc,"content":pagedesc})

    except Exception as e:
        utfprint(f"Error fetching URL {search_url}: {e}",1)
        return []
    if len(searchresults) > 0:
        websearch_lastquery = query
        websearch_lastresponse = searchresults
    return searchresults

def is_port_in_use(portNum):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(('localhost', portNum)) == 0
    except Exception:
        return True

def is_ipv6_supported():
    try:
        # Attempt to create an IPv6 socket
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.close()
        return True
    except Exception:
        return False

def coerce_tool_argtypes(tool_calls: list, tool_list: list) -> list:
    if not tool_calls or not tool_list:
        return tool_calls

    schema_map = {}
    for tool in tool_list:
        try:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "")
                props = func.get("parameters", {}).get("properties", {})
            else:
                name = tool.get("name", "")
                props = tool.get("parameters", {}).get("properties", {})
            if name:
                schema_map[name] = props
        except Exception:
            continue

    def coerce_value(val, prop_type):
        if val is None:
            return val
        try:
            if prop_type == "integer":
                return val if isinstance(val, int) else int(val)
            elif prop_type == "number":
                return val if isinstance(val, (int, float)) else float(val)
            elif prop_type == "boolean":
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    if val.lower() in ("true", "1", "yes"):
                        return True
                    if val.lower() in ("false", "0", "no"):
                        return False
                if isinstance(val, int):
                    return bool(val)
                return val
            elif prop_type == "string":
                return val if isinstance(val, str) else str(val) if val is not None else val
            elif prop_type == "array":
                if isinstance(val, list):
                    return val
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        return parsed if isinstance(parsed, list) else [parsed]
                    except Exception:
                        return [val]
                if isinstance(val, (set, tuple)):
                    return list(val)
                return [val]
            elif prop_type == "object":
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        return parsed if isinstance(parsed, dict) else val
                    except Exception:
                        return val
                return val
            elif prop_type == "null":
                return None
        except (ValueError, TypeError, AttributeError):
            pass
        return val

    result = []
    for call in tool_calls:
        try:
            if "function" in call:
                name = call["function"].get("name", "")
                arguments = call["function"].get("arguments", {})
            else:
                name = call.get("name", "")
                arguments = call.get("arguments", {})

            props = schema_map.get(name, {})
            if props and isinstance(arguments, dict):
                coerced = {}
                for key, val in arguments.items():
                    prop_schema = props.get(key, {})
                    prop_type = prop_schema.get("type")
                    # handle anyOf/oneOf for nullable types like {"anyOf": [{"type": "string"}, {"type": "null"}]}
                    if prop_type is None:
                        for combiner in ("anyOf", "oneOf"):
                            options = prop_schema.get(combiner, [])
                            for option in options:
                                t = option.get("type")
                                if t and t != "null":
                                    prop_type = t
                                    break
                            if prop_type: # Found a type, stop looking in other combiners
                                break
                    try:
                        coerced[key] = coerce_value(val, prop_type)
                    except Exception:
                        coerced[key] = val
                if "function" in call:
                    call = {**call, "function": {**call["function"], "arguments": coerced}}
                else:
                    call = {**call, "arguments": coerced}
        except Exception:
            pass
        result.append(call)

    return result

def toolcall_to_normalized_json(text,start_tag,end_tag): #convert weird formats into standard tool call json
    text = text.strip()
    def parse_qwen35(text: str) -> str:
        fn_match = re.search(r"<function=(.*?)>", text)
        if not fn_match:
            return text
        fn_name = fn_match.group(1).strip()
        params = {}
        param_blocks = re.findall(r"<parameter=(.*?)>(.*?)</parameter>", text, re.DOTALL)
        for key, value in param_blocks:
            params[key.strip()] = value.strip()
        return json.dumps({"name": fn_name, "arguments": params})
    def parse_glm(text: str) -> str:
        text = text.strip()
        # Extract function name: it's the first thing before any <arg_key>
        fn_match = re.match(r"^\s*([^\<\s]+)", text)
        if not fn_match:
            return text
        fn_name = fn_match.group(1).strip()
        # Extract all key/value pairs
        keys = re.findall(r"<arg_key>(.*?)</arg_key>", text)
        values = re.findall(r"<arg_value>(.*?)</arg_value>", text)
        params = {}
        for i in range(min(len(keys), len(values))):
            params[keys[i].strip()] = values[i].strip()
        return json.dumps({"name": fn_name, "arguments": params})
    def parse_deepseek_r1_sep(text: str) -> str:
        text = re.sub(r'<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>', r'\1',
                    text, flags=re.DOTALL).strip()
        sep = '<｜tool▁sep｜>'
        if sep not in text:
            return text
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        results = []
        for part in parts:
            lines = part.split('\n', 1)
            fn_name = lines[0].strip()
            args_block = lines[1] if len(lines) > 1 else '{}'
            args_block = re.sub(r'^```(?:json)?\s*', '', args_block.strip())
            args_block = re.sub(r'\s*```$', '', args_block.strip())
            try:
                results.append({"name": fn_name, "arguments": json.loads(args_block)})
            except Exception:
                pass
        if not results:
            return text
        return json.dumps(results) if len(results) > 1 else json.dumps(results[0])
    def parse_minimax(text: str) -> str:
        results = []
        for invoke in re.finditer(
            r'<invoke\s+name=["\']?([^"\'>\s]+)["\']?>(.*?)</invoke>',
            text, re.DOTALL
        ):
            fn_name = invoke.group(1).strip()
            params = {}
            for p in re.finditer(
                r'<parameter\s+name=["\']?([^"\'>\s]+)["\']?>(.*?)</parameter>',
                invoke.group(2), re.DOTALL
            ):
                val = p.group(2).strip()
                try:
                    params[p.group(1).strip()] = json.loads(val)
                except Exception:
                    params[p.group(1).strip()] = val
            results.append({"name": fn_name, "arguments": params})
        if not results:
            return text
        return json.dumps(results) if len(results) > 1 else json.dumps(results[0])
    def parse_gemma4(text: str) -> str:
        if text.startswith("call:"):
            text = text[len("call:"):]
        if '<|"|>' in text:
            text = text.replace('<|"|>', '!$$REAL_QUOTE$$!')
            text = text.replace('\\', '\\\\')
            text = text.replace('"', '\\"')
            text = text.replace('!$$REAL_QUOTE$$!','"')
        fn_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\{(.*)\}$', text.strip(), re.DOTALL) # extract fn name
        if not fn_match:
            return text
        fn_name = fn_match.group(1)
        body = fn_match.group(2).strip()
        body = '{' + body + '}'
        if not body:
            return json.dumps({"name": fn_name, "arguments": {}})
        try:   # Try to parse body as JSON object by wrapping it
            args = json.loads(body,strict=False)
            return json.dumps({"name": fn_name, "arguments": args})
        except Exception:
            pass
        normalized = fix_unquoted_keys(body)
        try:
            args = json.loads(normalized,strict=False)
            return json.dumps({"name": fn_name, "arguments": args})
        except Exception:
            pass
        return text

    def parse_gpt_oss(text: str) -> str:
        fn_match = re.search(r'functions\.([a-zA-Z_][a-zA-Z0-9_]*)', text)
        if not fn_match:
            return text
        fn_name = fn_match.group(1).strip()
        msg_split = text.split('<|message|>', 1)
        if len(msg_split) < 2:
            return text
        args_block = msg_split[1].strip()
        try:
            args = json.loads(args_block)
        except Exception:
            return text
        return json.dumps({"name": fn_name, "arguments": args})

    # gemma4 takes precedence, since it can contain valid json fragments
    if end_tag=="<tool_call|>":
        return parse_gemma4(text)

    #if we are already valid JSON, return
    check_ok = extract_json_from_string(text, True)
    if check_ok and len(check_ok)>0:
        return text #is valid JSON or parsable

    if "<arg_key>" in text and "<arg_value>" in text: # handle glm with args
        return parse_glm(text)

    if "<function=" in text: # handle qwen3.5
        return parse_qwen35(text)

    if "<invoke " in text: #minimax
        return parse_minimax(text)

    if '<｜tool▁sep｜>' in text: #deepseek
        return parse_deepseek_r1_sep(text)

    if ' ' not in text and '\n' not in text: # handle glm without args
        return parse_glm(text)

    if 'functions.' in text and "commentary" in start_tag:  # handle GPT-OSS
        return parse_gpt_oss(text)

    return text #fallback

def repack_toolcall_tags(text: str, original_tools:list):
    global thinkformats, tool_call_pairs
    tool_calls = []
    if not text:
        return tool_calls
    for fmt in thinkformats:
        pattern = f"{re.escape(fmt['start'])}.*?{re.escape(fmt['end'])}"
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    text = text.strip()
    found = False
    for start, end, streamhandled in tool_call_pairs:
        pattern=""
        if end:
            pattern = re.escape(start) + r"(.*?)" + re.escape(end)
        else:
            pattern = re.escape(start) + r"(.*)$"  # match to end of string
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            found = True
            for match in matches:
                normalizedtc = toolcall_to_normalized_json(match.strip(),start,end)
                sub_tool_calls = extract_json_from_string(normalizedtc)
                tool_calls.extend(sub_tool_calls)
            break
    # fallback ONLY if no tags were found at all
    if not found:
        tool_calls = extract_json_from_string(text,True)
    tool_calls = coerce_tool_argtypes(tool_calls, original_tools)
    return tool_calls

def format_jinja(messages_orig, tools, chat_template_kwargs=None):
    try:
        def strftime_now(format='%Y-%m-%d %H:%M:%S'):
            return datetime.now().strftime(format)
        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        def raise_exception(msg):
            print(f"Warning: Jinja template raised an exception: {msg}")
            return ""
        global cached_chat_template
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        # sanitize messages to remove none types
        messages = json.loads(json.dumps(messages_orig))
        for m in messages:
            if m.get("content") is None:
                m["content"] = ""
        # fix image placeholders, erase them and slap a reference onto the turn text message
        mediacount = 1
        for m in messages:
            if isinstance(m.get("content"), list):
                normalized = []
                turn_text = ""
                media_text = ""
                for item in m["content"]:
                    if item.get("type")=="text":
                        turn_text += item.get("text","")
                for item in m["content"]:
                    if item.get("type")=="text":
                        pass
                    elif item.get("type")=="image_url" or item.get("type")=="image":
                        media_text += f"\n(Attached Image {mediacount})\n"
                        mediacount += 1
                    elif item.get("type")=="input_audio":
                        media_text += f"\n(Attached Audio {mediacount})\n"
                        mediacount += 1
                    else:
                        normalized.append(item)
                turn_text = media_text + turn_text
                if turn_text:
                    normalized.append({"type": "text","text": turn_text})
                m["content"] = normalized
        for m in messages: # Fix tool_calls arguments and content if parsable
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except Exception:
                            pass
        jinja_env.globals['strftime_now'] = strftime_now
        jinja_env.globals['raise_exception'] = raise_exception
        jinja_env.filters["tojson"] = tojson
        jinja_compiled_template = jinja_env.from_string(cached_chat_template)
        text = None
        messages_for_render = []
        assist_should_prefill = False
        chat_template_kwargs = chat_template_kwargs or {}
        last_assist_msg = ""
        if messages:
            last_assist_msg = messages[-1]["content"]
            assist_should_prefill = (messages and messages[-1]["role"].lower() == "assistant" and last_assist_msg and isinstance(last_assist_msg, str) and len(last_assist_msg.strip())>0) #avoid single character newline or space content
            last_assist_msg = "" if not assist_should_prefill else last_assist_msg
            messages_for_render = messages[:-1] if len(messages) > 1 and assist_should_prefill else messages
        if tools and len(tools)>0:
            text = jinja_compiled_template.render(messages=messages_for_render, tools=tools, add_generation_prompt=True, bos_token="", eos_token="", **chat_template_kwargs)
        else:
            text = jinja_compiled_template.render(messages=messages_for_render, add_generation_prompt=True, bos_token="", eos_token="", **chat_template_kwargs)
        if assist_should_prefill and text and last_assist_msg: # handle prefill continuations
            text = text + last_assist_msg
        return text if text else None
    except Exception as e:
        print(f"Jinja formatting failed: {e}")
        return None

def remove_outer_tags(inputstr):
    try:
        stripped = inputstr.strip()
        match = re.match(r'^<([^\s<>]+)>(.*?)</\1>\s*$', stripped, re.DOTALL) # Try angle brackets first
        if match:
            return match.group(2).strip()
        match = re.match(r'^\[([^\s<>]+)\](.*?)\[/\1]\s*$', stripped, re.DOTALL) # Then try square brackets
        if match:
            return match.group(2).strip()
        return stripped # If no match, return original string
    except Exception:
        return stripped

def normalize_tool_call(obj): # Normalize various tool call formats to OpenAI format
    if "type" in obj and "function" in obj: # Already in OpenAI format
        return obj
    if "name" in obj and ("arguments" in obj or "parameters" in obj):
        args = obj.get("arguments", obj.get("parameters", {}))
        return {
            "type": "function",
            "function": {
                "name": obj["name"],
                "arguments": args
            }
        }
    if "function" in obj and isinstance(obj["function"], dict):
        func = obj["function"]
        if "name" in func:
            return {
                "type": "function",
                "function": {
                    "name": func["name"],
                    "arguments": func.get("arguments", func.get("parameters", {}))
                }
            }

    return obj

# Used to parse json for openai tool calls
def extract_json_from_string(input_string, check_strict=False):
    parsed_json = None
    input_string = remove_outer_tags(input_string) #if we detected wrapper tags, remove them

    try: # First check if model exported perfect json
        parsed_json = json.loads(input_string)
        if not isinstance(parsed_json, list):
            parsed_json = [parsed_json]
        return parsed_json
    except Exception:
        pass
    try: # Next check if all we need is to add brackets to make it perfect json
        parsed_json = json.loads(f"[{input_string}]")
        return parsed_json
    except Exception:
        pass
    try:
        if not check_strict: #only allow when not strict mode
            # Now use regular expression to match JSON objects or arrays in case part is valid json and part is not
            json_pattern = r'(\{.*?\}|\[.*?\])'  # was json_pattern = r'(\{.*\}|\[.*\])'
            potential_jsons = re.findall(json_pattern, input_string, re.DOTALL)
            for potential_json in potential_jsons:
                try:
                    parsed_json = json.loads(potential_json, strict=False)
                    if not isinstance(parsed_json, list):
                        parsed_json = [parsed_json]
                    return parsed_json
                except Exception:
                    continue
    except Exception:
        pass
    return []

def parse_last_logprobs(lastlogprobs):
    if not lastlogprobs:
        return None
    logprobsdict = {}
    logprobsdict['content'] = []
    logprobsdict['tokens'] = []
    logprobsdict['token_ids'] = []
    logprobsdict['token_logprobs'] = []
    logprobsdict['top_logprobs'] = []
    logprobsdict['text_offset'] = []
    text_offset_counter = 0
    for i in range(lastlogprobs.count):
        lp_content_item = {}
        logprob_item = lastlogprobs.logprob_items[i]
        toptoken = ctypes.string_at(logprob_item.selected_token).decode("UTF-8","ignore")
        logprobsdict['tokens'].append(toptoken)
        logprobsdict['token_ids'].append(logprob_item.selected_token_id)
        lp_content_item['token'] = toptoken
        lp_content_item['token_id'] = logprob_item.selected_token_id
        logprobsdict['token_logprobs'].append(logprob_item.selected_logprob)
        lp_content_item['logprob'] = logprob_item.selected_logprob
        lp_content_item['bytes'] = list(toptoken.encode('utf-8'))
        lp_content_item['top_logprobs'] = []
        logprobsdict['text_offset'].append(text_offset_counter)
        text_offset_counter += len(toptoken)
        tops = {}
        for j in range(min(logprob_item.option_count,logprobs_max)):
            tl_item = {}
            tl_item['logprob'] = logprob_item.logprobs[j]
            tokstr = ctypes.string_at(logprob_item.tokens[j]).decode("UTF-8","ignore")
            tops[tokstr] = logprob_item.logprobs[j]
            tl_item['token'] = tokstr
            tl_item['token_id'] = logprob_item.token_ids[j]
            tl_item['bytes'] = list(tokstr.encode('utf-8'))
            lp_content_item['top_logprobs'].append(tl_item)
        logprobsdict['top_logprobs'].append(tops)
        logprobsdict['content'].append(lp_content_item)
    return logprobsdict

def extract_tool_info_from_tool_array(chosen_tool, tools_array):
    found_function = ""
    found_tooljson = None
    try:
        if isinstance(chosen_tool, str):
            found_function = chosen_tool
        elif isinstance(chosen_tool, dict): #if we can match the tool name, we must use that tool, remove all other tools
            found_function = chosen_tool.get('function').get('name')
        #if we find the function in tools, remove all other tools except the one matching the function name
        for tool in tools_array:
            if found_function and tool.get('type') == "function" and tool.get('function').get('name').lower() == found_function.lower():
                found_tooljson = tool
                break
    except Exception:
        # In case of any issues, just revert back to no specified function
        print("Tools parsing not valid - discarded")
        pass
    return found_tooljson

def extract_all_names_from_tool_array(tools_array):
    toolnames = []
    for tool in tools_array:
        try:
            if tool.get('type') == "function" and tool.get('function').get('name'):
                toolnames.append(tool.get('function').get('name'))
        except Exception:
            pass
    return toolnames

def strip_oaicontent_of_media(oaicontent):
    if isinstance(oaicontent, list):
        outarr = []
        for x in oaicontent:
            if not isinstance(x, dict):
                outarr.append({"type": "unknown", "data": "(base64 data attached)"})
                continue
            xtype = x.get("type","data")
            if xtype=="text":
                outarr.append(x)
            else:
                outarr.append({"type":xtype, "data":"(base64 data attached)"})
        return outarr
    return oaicontent

def strip_mcpcontent_of_media(mcpcontentstr):
    try:
        if isinstance(mcpcontentstr, str):
            #we try to strip out the b64 of MCP type tool responses with images for past turns
            mcp_pl = json.loads(mcpcontentstr)
            pl_modified = False
            if isinstance(mcp_pl, dict) and isinstance(mcp_pl.get("content",None),list):
                pl_arr = mcp_pl.get("content",[])
                for idx in range(len(pl_arr)):
                    if pl_arr[idx].get("type","")=="image" and pl_arr[idx].get("data","")!="":
                        pl_arr[idx]["data"] = "(base64 data attached)"
                        pl_modified = True
                if pl_modified:
                    mcpcontentstr = json.dumps(mcp_pl)
    except Exception:
        pass
    return mcpcontentstr

#returns the found JSON of the correct tool to use, or None if no tool is suitable
def determine_tool_json_to_use(genparams, curr_ctx, assistant_message_start, is_followup_tool):
    # tools handling: Check if user is passing a openai tools array, if so add to end of prompt before assistant prompt unless tool_choice has been set to None
    tools_array = genparams.get('tools', [])
    chosen_tool = genparams.get('tool_choice', "auto")
    messages = genparams.get('messages',[])
    toolmem = genparams.get("memory","")

    # first handle auto mode, determine whether a tool is needed
    used_tool_json = None
    if not curr_ctx:
        return None

    # get user's last message and last tool call results
    last_user_message = ""
    tool_call_results = ""

    images_added = [] #sometimes images are needed to make a decision too
    audio_added = []

    if messages:
        images_added, audio_added = sweep_media_from_messages(messages)
        reversed_messages = list(reversed(messages))
        for message in reversed_messages:
            if message["role"] == "user":
                last_user_message = message["content"]
                last_user_message = strip_oaicontent_of_media(last_user_message)
                last_user_message = f"\n\nUser's current request: {last_user_message}"
                break
        tool_call_chunk = []
        for message in reversed_messages:
            if message["role"] == "tool":
                toolrespstr = message["content"]
                # toolrespstr = strip_mcpcontent_of_media(toolrespstr)
                tool_call_chunk.append(toolrespstr)
            else:
                break
        tmp_tool_replies = list(reversed(tool_call_chunk))
        if tmp_tool_replies and len(tmp_tool_replies)>0:
            tool_call_results = f"\n\nTool call responses: {tmp_tool_replies}"

    if tools_array and len(tools_array) > 0 and chosen_tool is not None and chosen_tool!="none":
        should_use_tools = True
        if chosen_tool=="auto" or chosen_tool=="required":
            # note: message string already contains the instruct start tag!
            temptoolnames = extract_all_names_from_tool_array(tools_array)
            tempjson = {}
            if chosen_tool=="required":
                custom_tools_prompt_json_format = "Respond with a JSON object using this structure:\r\n{\r\n    \"tool_name\": \"exact_tool_name_here\"\r\n}\r\n\r\nRules:\r\n- You must pick one of the tools to use, pick the most suitable tool."
                tempjson = {"type":"object","properties":{"tool_name":{"type":"string","enum":temptoolnames}},"required":["tool_name"],"additionalProperties":False}
            else:
                temptoolnames.append("null")
                custom_tools_prompt_json_format = "Respond with a JSON object using this structure:\r\n{\r\n    \"reasoning\": \"Your reasoning here\",\r\n    \"final_decision\": \"yes\" or \"no\",\r\n    \"tool_name\": \"exact_tool_name_here\" or \"null\"\r\n}\r\n\r\nRules:\r\n- Output only the JSON object. Do NOT add anything before or after the json object.\r\n- final_decision must be exactly \"yes\" or \"no\"\r\n- tool_name must be either an exact tool name, or if no tool is required, an empty string: \"\"\r\n- Keep reasoning short, maximum one or two sentences.\r\n- No unnecessary comments"
                tempjson = {"type":"object","properties":{"reasoning":{"type":"string"},"final_decision":{"type":"string","enum":["yes","no","Yes","No","YES","NO"," yes"," no"," Yes"," No"," YES"," NO"]},"tool_name":{"type":"string","enum":temptoolnames}},"required":["reasoning","final_decision","tool_name"],"additionalProperties":False}
            toolquerygrammar = convert_json_to_gbnf(tempjson)

            if not is_followup_tool:
                custom_tools_prompt = "Is calling one of the tools listed above absolutely essential to answer user's current request, or is a tool call optional?"
                custom_tools_prompt_processed = f"{curr_ctx}{last_user_message}\n\n{custom_tools_prompt} {custom_tools_prompt_json_format}{assistant_message_start}"
            else:
                custom_tools_prompt = "Given the tool call response to the user's current request, is another tool call needed to further answer user's message?"
                custom_tools_prompt_processed = f"{curr_ctx}{last_user_message}{tool_call_results}\n\n{custom_tools_prompt} {custom_tools_prompt_json_format}{assistant_message_start}"

            # first, prompt to see if a tool call is needed using the prompt above.
            # the result is a short explanation by the LLM on why a tool call is or is not needed, along with it's final decision at the end.
            temp_poll = {
                "prompt": custom_tools_prompt_processed,
                "memory": toolmem,
                "max_length":300,
                "temperature":0.1,
                "top_k":1,
                "rep_pen":1,
                "ban_eos_token":False,
                "grammar":toolquerygrammar
            }
            if len(images_added)>0:
                temp_poll["images"] = images_added
            if len(audio_added)>0:
                temp_poll["audio"] = audio_added
            temp_poll_result = generate(genparams=temp_poll)
            temp_poll_text = temp_poll_result['text'].strip().rstrip('.')
            temp_poll_data_arr = extract_json_from_string(temp_poll_text)
            temp_poll_data = temp_poll_data_arr[0] if (temp_poll_data_arr and len(temp_poll_data_arr)>0) else None

            if temp_poll_data:
                if chosen_tool!="required" and ("yes" not in temp_poll_data.get("final_decision","").lower() or "null" in temp_poll_data.get("tool_name","").lower()):
                    should_use_tools = False
                elif (chosen_tool=="auto" or chosen_tool=="required") and "null" not in temp_poll_data.get("tool_name","").lower():
                    chosen_tool = temp_poll_data.get("tool_name","").lower().strip()

            if not args.quiet:
                print(f"\n[TOOLCALL REASONING]: {temp_poll_text}")

        if should_use_tools:
            #first, try and extract a specific tool if selected
            used_tool_json = extract_tool_info_from_tool_array(chosen_tool, tools_array)
            if used_tool_json: #already found the tool we want, remove all others
                pass
            elif len(tools_array)==1:
                used_tool_json = tools_array[0]
            else: # we have to find the tool we want the old fashioned way
                toolnames = extract_all_names_from_tool_array(tools_array)
                if len(toolnames) == 1:
                    used_tool_json = extract_tool_info_from_tool_array(toolnames[0], tools_array)

    return used_tool_json

def compress_tools_array(tools_array):
    tools_array_filtered = []
    for tool_dict in tools_array:
        tool_data = tool_dict
        if 'function' in tool_dict:
            tool_data = tool_dict['function']
        tool_props = {}
        params = tool_data.get("parameters", {})
        props = params.get("properties", {})
        for prop_name, prop_data in props.items():
            prop_type = prop_data.get("type")
            if prop_type is None and "anyOf" in prop_data:
                for option in prop_data["anyOf"]:
                    option_type = option.get("type")
                    if option_type and option_type != "null":
                        prop_type = option_type
                        break
            if prop_type is None:
                prop_type = "string"
            tool_props[prop_name] = prop_type
        tools_array_filtered.append({
            "name": tool_data['name'],
            "description": tool_data.get("description", ""),
            "properties": tool_props
        })

    return tools_array_filtered

def sweep_media_from_messages(messages_array):
    images = []
    audio = []
    for message in messages_array:
        curr_content = message.get("content", None)
        if isinstance(curr_content, list):
            for item in curr_content:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        images.append(url.split(",", 1)[1])
                elif item.get("type") == "input_audio":
                    data = item.get("input_audio", {}).get("data")
                    if data:
                        audio.append(data)
        elif message.get("role", "")=="tool" and isinstance(curr_content, str): #handle mcp returned images
            try:
                mcp_pl = json.loads(curr_content)
                if isinstance(mcp_pl, dict) and isinstance(mcp_pl.get("content",None),list):
                    pl_arr = mcp_pl.get("content",[])
                    if len(pl_arr)>0 and pl_arr[0].get("type","")=="image" and pl_arr[0].get("data","")!="":
                        images.append(pl_arr[0].get("data",""))
            except Exception:
                pass
        imgs_ollama = message.get("images", None)
        if imgs_ollama:
            for img in imgs_ollama:
                images.append(img)
    return images, audio


def transform_genparams(genparams, api_format, use_jinja):
    global chatcompl_adapter, maxctx, thinkformats, cached_jinja_kwargs

    if api_format < 0: #not text gen, do nothing
        return

    jsongrammar = r"""
root   ::= arr
value  ::= object | array | string | number | ("true" | "false" | "null") ws
arr  ::=
  "[\n" ws (
            value
    (",\n" ws value)*
  )? "]"
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [1-9] [0-9]{0,15})? ws
ws ::= | " " | "\n" [ \t]{0,20}
"""

    used_tool_json = None
    #api format 1=basic,2=kai,3=oai,4=oai-chat,5=interrogate,6=ollama,7=ollamachat,8=oai-responses,9=anthropic-messages
    #alias all nonstandard alternative names for rep pen.
    rp1 = float(genparams.get('repeat_penalty', 1.0))
    rp2 = float(genparams.get('repetition_penalty', 1.0))
    rp3 = float(genparams.get('rep_pen', 1.0))
    rp_max = max(rp1,rp2,rp3)
    genparams["rep_pen"] = rp_max
    if "use_default_badwordsids" in genparams and "ban_eos_token" not in genparams:
        genparams["ban_eos_token"] = genparams.get('use_default_badwordsids', False)

    if api_format==1:
        genparams["prompt"] = genparams.get('text', "")
        genparams["top_k"] = int(genparams.get('top_k', 100))
        genparams["max_length"] = int(genparams.get('max', args.defaultgenamt))

    elif api_format==2: #note: kobold api does not support tool calling
        adapter_obj = {} if chatcompl_adapter is None else chatcompl_adapter
        assistant_message_start = adapter_obj.get("assistant_start", "\n### Response:\n")
        assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)

    elif api_format==3 or api_format==4 or api_format==7:
        default_adapter = {} if chatcompl_adapter is None else chatcompl_adapter
        adapter_obj = genparams.get('adapter', default_adapter)
        default_max_tok = (adapter_obj.get("max_length", args.defaultgenamt) if (api_format==4 or api_format==7) else args.defaultgenamt)
        oaiml = tryparseint(genparams.get('max_tokens', genparams.get('max_completion_tokens', default_max_tok)),default_max_tok)
        genparams["max_length"] = genparams.get('max_length', oaiml)
        if genparams["max_length"] <= 0:
            genparams["max_length"] = default_max_tok
        presence_penalty = genparams.get('presence_penalty', genparams.get('frequency_penalty', 0.0))
        genparams["presence_penalty"] = tryparsefloat(presence_penalty,0.0)
        # openai allows either a string or a list as a stop sequence
        if genparams.get('stop',[]) is not None:
            if isinstance(genparams.get('stop',[]), list):
                genparams["stop_sequence"] = genparams.get('stop', [])
            else:
                genparams["stop_sequence"] = [genparams.get('stop')]

        genparams["sampler_seed"] = tryparseint(genparams.get('seed', -1),-1)
        genparams["mirostat"] = genparams.get('mirostat_mode', 0)

        if api_format==4 or api_format==7: #handle ollama chat here too
            # translate openai chat completion messages format into one big string.
            messages_array = genparams.get('messages', [])
            messages_string = adapter_obj.get("chat_start", "")
            system_message_start = adapter_obj.get("system_start", "\n### Instruction:\n")
            system_message_end = adapter_obj.get("system_end", "")
            user_message_start = adapter_obj.get("user_start", "\n### Instruction:\n")
            user_message_end = adapter_obj.get("user_end", "")
            assistant_message_start = adapter_obj.get("assistant_start", "\n### Response:\n")
            assistant_message_end = adapter_obj.get("assistant_end", "")
            assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
            tools_message_start = adapter_obj.get("tools_start", "")
            tools_message_end = adapter_obj.get("tools_end", "")
            images_added = []
            audio_added = []
            continue_assistant_turn = genparams.get('continue_assistant_turn', True)
            latest_turn_was_assistant = False
            latest_turn_was_tool = False

            # handle structured outputs
            respformat = genparams.get('response_format', None)
            if respformat:
                try:
                    rt = respformat.get('type')
                    if rt.lower() == "json_schema":
                        schema = respformat.get('json_schema').get('schema')
                        decoded = convert_json_to_gbnf(schema)
                        if decoded:
                            genparams["grammar"] = decoded
                    elif rt.lower() == "json_object":
                        genparams["grammar"] = jsongrammar
                except Exception:
                    # In case of any issues, just do normal gen
                    print("Structured Output not valid - discarded")
                    pass
            elif 'json_schema' in genparams:
                try:
                    schema = genparams.get('json_schema')
                    decoded = convert_json_to_gbnf(schema)
                    if decoded:
                        genparams["grammar"] = decoded
                except Exception:
                    print("Structured Output (old format) not valid - discarded")
                    pass

            message_index = 0
            attachedimgid = 0
            attachedaudid = 0
            jinja_output = None
            jinjatools = genparams.get('tools', [])
            if use_jinja and cached_chat_template:
                copied_jinja_kwargs = dict(cached_jinja_kwargs or {})
                # Merge user-provided chat_template_kwargs into our defaults
                user_chat_template_kwargs = genparams.get("chat_template_kwargs")
                if user_chat_template_kwargs is not None and isinstance(user_chat_template_kwargs, dict):
                    # User kwargs override cached/default kwargs
                    copied_jinja_kwargs.update(user_chat_template_kwargs)
                if "reasoning_effort" in genparams and genparams["reasoning_effort"] is not None:
                    copied_jinja_kwargs["reasoning_effort"] = genparams["reasoning_effort"]
                jinja_output = format_jinja(messages_array,jinjatools,copied_jinja_kwargs)
            if jinja_output:
                messages_string = jinja_output
                for pair in thinkformats:
                    starter = pair['start']
                    if jinja_output.rstrip().endswith(starter): #the prompt template already forced a start think.
                        genparams["already_started_thinking"] = True
                        break
                if jinjatools and len(jinjatools)>0:
                    genparams["using_openai_tools"] = True
                # handle media
                images_added, audio_added = sweep_media_from_messages(messages_array)
            else:
                if jinjatools:
                    # inject the tools list at the top of the context window, even if context has shifted
                    # uses koboldcpp's special memory parameter
                    tools_string = f"{system_message_start}### Available Tools:\n{json.dumps(compress_tools_array(jinjatools), indent=0)}{system_message_end}\n"
                    exist_mem = genparams.get('memory', "")
                    genparams["memory"] = tools_string + exist_mem

                for message in messages_array:
                    message_index += 1
                    latest_turn_was_assistant = False
                    latest_turn_was_tool = False
                    if message['role'] == "system":
                        messages_string += system_message_start
                    elif message['role'] == "user":
                        messages_string += user_message_start
                    elif message['role'] == "assistant":
                        messages_string += assistant_message_start
                        latest_turn_was_assistant = True
                    elif message['role'] == "tool":
                        latest_turn_was_tool = True
                        messages_string += tools_message_start
                        tcid = message.get("tool_call_id","")
                        tcid = ("" if not tcid else f" {tcid}")
                        messages_string += f"\nReceived results of function call{tcid}:\n"

                    # content can be a string or an array of objects
                    curr_content = message.get("content",None)
                    if api_format==7: #ollama handle vision
                        imgs = message.get("images",None)
                        if imgs and len(imgs) > 0:
                            for img in imgs:
                                images_added.append(img)
                    if not curr_content:
                        if "tool_calls" in message:
                            try:
                                nlstart = True
                                for tc in message.get("tool_calls"):
                                    if nlstart:
                                        nlstart = False
                                        messages_string += "\n"
                                    tcid = tc.get("id","")
                                    tcfnname = tc.get("function").get("name")
                                    tcfnargs = tc.get("function").get("arguments","")
                                    tcfnargs = (f" with arguments={tcfnargs}" if tcfnargs else "")
                                    messages_string += f"(Made a function call {tcid} to {tcfnname}{tcfnargs})\n"
                            except Exception:
                                messages_string += "\n(Made a function call)\n"
                        pass  # do nothing
                    elif isinstance(curr_content, str):
                        if latest_turn_was_tool and message_index < len(messages_array):
                            curr_content = strip_mcpcontent_of_media(curr_content)
                        messages_string += curr_content
                    elif isinstance(curr_content, list): #is an array
                        for item in curr_content:
                            if isinstance(item, dict):
                                if item['type']=="text":
                                        messages_string += item['text']
                                elif item['type']=="image_url":
                                    if 'image_url' in item and item['image_url'] and item['image_url']['url'] and item['image_url']['url'].startswith("data:image"):
                                        images_added.append(item['image_url']['url'].split(",", 1)[1])
                                        attachedimgid += 1
                                        messages_string += f"\n(Attached Image {attachedimgid})\n"
                                elif item['type']=="input_audio":
                                    if 'input_audio' in item and item['input_audio'] and item['input_audio']['data']:
                                        audio_added.append(item['input_audio']['data'])
                                        attachedaudid += 1
                                        messages_string += f"\n(Attached Audio {attachedaudid})\n"
                            elif isinstance(item, str):
                                messages_string += item # If item is just a string, append it directly

                    # If last message, add any tools calls after message content and before message end token if any
                    if message_index == len(messages_array):
                        is_followup = (message['role'] == "tool")
                        #small hack: if the current turn is assistant, but its short (e.g. a prefilled name), and the previous turn was tool, consider it followup as well
                        if(not is_followup and message_index>1 and message['role'] == "assistant" and messages_array[message_index-2]['role']=="tool" and message['content'] and len(message['content']) < 100): #100 char limit
                            is_followup = True
                        used_tool_json = determine_tool_json_to_use(genparams, messages_string, assistant_message_start, is_followup)

                        if used_tool_json:
                            toolparamjson = None
                            toolname = None
                            # Set temperature lower automatically if function calling, cannot exceed 0.5
                            genparams["temperature"] = (1.0 if genparams.get("temperature", 0.5) > 1.0 else genparams.get("temperature", 0.5))
                            genparams["using_openai_tools"] = True
                            # Set grammar to llamacpp example grammar to force json response (see https://github.com/ggerganov/llama.cpp/blob/master/grammars/json_arr.gbnf)
                            genparams["grammar"] = jsongrammar
                            try:
                                toolname = used_tool_json.get('function').get('name')
                                toolparamjson = used_tool_json.get('function').get('parameters')
                                bettergrammarjson = {"type":"array","items":{"type":"object","properties":{"id":{"type":"string","enum":["call_001"]},"type":{"type":"string","enum":["function"]},"function":{"type":"object","properties":{"name":{"type":"string"},"arguments":{}},"required":["name","arguments"],"additionalProperties":False}},"required":["id","type","function"],"additionalProperties":False}}
                                bettergrammarjson["items"]["properties"]["function"]["properties"]["arguments"] = toolparamjson
                                decoded = convert_json_to_gbnf(bettergrammarjson)
                                if decoded:
                                    genparams["grammar"] = decoded
                            except Exception:
                                pass
                            tool_json_formatting_instruction = f"\nPlease use the provided schema to fill the parameters to create a function call for {toolname}, in the following format: " + json.dumps([{"id": "call_001", "type": "function", "function": {"name": f"{toolname}", "arguments": {"first property key": "first property value", "second property key": "second property value"}}}], indent=0)
                            messages_string += f"\n\nJSON Schema:\n{used_tool_json}\n\n{tool_json_formatting_instruction}{assistant_message_start}"

                    if message['role'] == "system":
                        messages_string += system_message_end
                    elif message['role'] == "user":
                        messages_string += user_message_end
                    elif message['role'] == "assistant":
                        messages_string += assistant_message_end
                    elif message['role'] == "tool":
                        messages_string += tools_message_end
                messages_string += assistant_message_gen
                if (latest_turn_was_assistant and continue_assistant_turn): #allow continue a prefill, chop off end
                    messages_string = messages_string[:-(len(assistant_message_gen)+len(assistant_message_end))]
            genparams["prompt"] = messages_string
            for pair in thinkformats:
                starter = pair['start']
                if messages_string.rstrip().endswith(starter): #the prompt template already forced a start think.
                    genparams["already_started_thinking"] = True
                    break
            if len(images_added)>0:
                genparams["images"] = images_added
            if len(audio_added)>0:
                genparams["audio"] = audio_added
            if len(genparams.get('stop_sequence', []))==0: #only set stop seq if it wont overwrite existing
                genparams["stop_sequence"] = [user_message_start.strip(),assistant_message_start.strip()]
            else:
                genparams["stop_sequence"].append(user_message_start.strip())
                genparams["stop_sequence"].append(assistant_message_start.strip())
            if not used_tool_json and jinjatools and latest_turn_was_tool:
                genparams["stop_sequence"].append("(Made a function call") # qol prevent fake toolcalls
            genparams["trim_stop"] = True


    elif api_format==5:
        firstimg = genparams.get('image', "")
        genparams["images"] = [firstimg]
        genparams["max_length"] = 150
        adapter_obj = {} if chatcompl_adapter is None else chatcompl_adapter
        user_message_start = adapter_obj.get("user_start", "### Instruction:")
        assistant_message_start = adapter_obj.get("assistant_start", "### Response:")
        assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
        genparams["prompt"] = f"{user_message_start} In one sentence, write a descriptive caption for this image.\n{assistant_message_gen}"

    elif api_format==6:
        detokstr = ""
        tokids = genparams.get('context', [])
        adapter_obj = {} if chatcompl_adapter is None else chatcompl_adapter
        user_message_start = adapter_obj.get("user_start", "\n\n### Instruction:\n")
        assistant_message_start = adapter_obj.get("assistant_start", "\n\n### Response:\n")
        assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
        try:
            detokstr = detokenize_ids(tokids,True)
        except Exception as e:
            utfprint("Ollama Context Error: " + str(e))
        ollamasysprompt = genparams.get('system', "")
        ollamabodyprompt = f"{detokstr}{user_message_start}{genparams.get('prompt', '')}{assistant_message_gen}"
        ollamaopts = genparams.get('options', {})
        if genparams.get('stop',[]) is not None:
            genparams["stop_sequence"] = genparams.get('stop', [])
        if "num_predict" in ollamaopts:
            genparams["max_length"] = ollamaopts.get('num_predict', args.defaultgenamt)
        if "num_ctx" in ollamaopts:
            genparams["max_context_length"] = ollamaopts.get('num_ctx', maxctx)
        if "temperature" in ollamaopts:
            genparams["temperature"] = ollamaopts.get('temperature', 0.75)
        if "top_k" in ollamaopts:
            genparams["top_k"] = ollamaopts.get('top_k', 100)
        if "top_p" in ollamaopts:
            genparams["top_p"] = ollamaopts.get('top_p', 0.92)
        if "seed" in ollamaopts:
            genparams["sampler_seed"] = tryparseint(ollamaopts.get('seed', -1),-1)
        if "stop" in ollamaopts:
            genparams["stop_sequence"] = ollamaopts.get('stop', [])
        genparams["stop_sequence"].append(user_message_start.strip())
        genparams["stop_sequence"].append(assistant_message_start.strip())
        genparams["trim_stop"] = True
        genparams["ollamasysprompt"] = ollamasysprompt
        genparams["ollamabodyprompt"] = ollamabodyprompt
        genparams["prompt"] = ollamasysprompt + ollamabodyprompt
    elif api_format==8: # OpenAI Responses API, oai-responses
        raw_input = genparams.get('input', '')
        raw_instructions = genparams.get('instructions', '')
        if isinstance(raw_input, str):
            genparams['messages'] = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list): # Convert Responses API input items to chat messages format
            converted = []
            for item in raw_input:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    # content can itself be a list of typed parts
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if part.get("type") == "input_text":
                                parts.append({"type": "text", "text": part.get("text", "")})
                            elif part.get("type") == "input_image":
                                img = part.get("image_url", part.get("source", {}))
                                parts.append({"type": "image_url", "image_url": {"url": img}})
                        content = parts
                    converted.append({"role": role, "content": content})
                elif isinstance(item, str):
                    converted.append({"role": "user", "content": item})
            genparams['messages'] = converted
        else:
            genparams['messages'] = []
        if raw_instructions and isinstance(raw_instructions, str):
            genparams['messages'].insert(0, {"role": "system", "content": raw_instructions})
        transform_genparams(genparams, 4, use_jinja) # Delegate to the chat-completions transform by re-running as format 4
        return genparams
    elif api_format==9: # Anthropic Messages API
        genparams["max_length"] = genparams.get("max_tokens", args.defaultgenamt)
        sys_prompt = genparams.get("system", "")
        messages = genparams.get("messages", [])
        if sys_prompt:
            if isinstance(sys_prompt, list): # Handle array-style system prompts
                sys_prompt = "".join([s.get("text","") for s in sys_prompt if s.get("type") == "text"])
            messages.insert(0, {"role": "system", "content": sys_prompt})
        genparams["messages"] = messages
        transform_genparams(genparams, 4, use_jinja) # Delegate to oai chat completions
        return genparams

    #final transformations (universal template replace)
    replace_instruct_placeholders = genparams.get('replace_instruct_placeholders', True)
    stop_sequence = (genparams.get('stop_sequence', []) if genparams.get('stop_sequence', []) is not None else [])
    stop_sequence = stop_sequence[:stop_token_max]
    if replace_instruct_placeholders:
        prompt = genparams.get('prompt', "")
        memory = genparams.get('memory', "")
        adapter_obj = {} if chatcompl_adapter is None else chatcompl_adapter
        system_message_start = adapter_obj.get("system_start", "\n### Instruction:\n")
        system_message_end = adapter_obj.get("system_end", "")
        user_message_start = adapter_obj.get("user_start", "\n### Instruction:\n")
        user_message_end = adapter_obj.get("user_end", "")
        assistant_message_start = adapter_obj.get("assistant_start", "\n### Response:\n")
        assistant_message_end = adapter_obj.get("assistant_end", "")
        assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
        if isinstance(prompt, str): #needed because comfy SD uses same field name
            if assistant_message_gen and assistant_message_gen!=assistant_message_start: #replace final output tag with unspaced (gen) version if exists
                if "{{[OUTPUT]}}" in prompt:
                    prompt = replace_last_in_string(prompt,"{{[OUTPUT]}}",assistant_message_gen)
                elif "{{[OUTPUT]}}" in memory:
                    memory = replace_last_in_string(memory,"{{[OUTPUT]}}",assistant_message_gen)
                elif assistant_message_start and prompt.rstrip().endswith(assistant_message_start):
                    prompt = replace_last_in_string(prompt, assistant_message_start, assistant_message_gen)
            if "{{[INPUT_END]}}" in prompt or "{{[OUTPUT_END]}}" in prompt:
                prompt = prompt.replace("{{[INPUT]}}", user_message_start)
                prompt = prompt.replace("{{[OUTPUT]}}", assistant_message_start)
                prompt = prompt.replace("{{[SYSTEM]}}", system_message_start)
                prompt = prompt.replace("{{[INPUT_END]}}", user_message_end)
                prompt = prompt.replace("{{[OUTPUT_END]}}", assistant_message_end)
                prompt = prompt.replace("{{[SYSTEM_END]}}", system_message_end)
                memory = memory.replace("{{[INPUT]}}", user_message_start)
                memory = memory.replace("{{[OUTPUT]}}", assistant_message_start)
                memory = memory.replace("{{[SYSTEM]}}", system_message_start)
                memory = memory.replace("{{[INPUT_END]}}", user_message_end)
                memory = memory.replace("{{[OUTPUT_END]}}", assistant_message_end)
                memory = memory.replace("{{[SYSTEM_END]}}", system_message_end)
            else:
                if "{{[INPUT]}}" in memory:
                    memory = memory.replace("{{[INPUT]}}", user_message_start, 1)
                else:
                    prompt = prompt.replace("{{[INPUT]}}", user_message_start, 1)
                prompt = prompt.replace("{{[INPUT]}}", assistant_message_end + user_message_start)
                prompt = prompt.replace("{{[OUTPUT]}}", user_message_end + assistant_message_start)
                prompt = prompt.replace("{{[SYSTEM]}}", system_message_start)
                prompt = prompt.replace("{{[INPUT_END]}}", "")
                prompt = prompt.replace("{{[OUTPUT_END]}}", "")
                prompt = prompt.replace("{{[SYSTEM_END]}}", "")
                memory = memory.replace("{{[INPUT]}}", assistant_message_end + user_message_start)
                memory = memory.replace("{{[OUTPUT]}}", user_message_end + assistant_message_start)
                memory = memory.replace("{{[SYSTEM]}}", system_message_start)
                memory = memory.replace("{{[INPUT_END]}}", "")
                memory = memory.replace("{{[OUTPUT_END]}}", "")
                memory = memory.replace("{{[SYSTEM_END]}}", "")
        for i in range(len(stop_sequence)):
            if stop_sequence[i] == "{{[INPUT]}}":
                stop_sequence[i] = user_message_start.strip()
            elif stop_sequence[i] == "{{[OUTPUT]}}":
                stop_sequence[i] = assistant_message_start.strip()
            elif stop_sequence[i] == "{{[INPUT_END]}}":
                stop_sequence[i] = (user_message_end.strip() if user_message_end.strip()!="" else "")
            elif stop_sequence[i] == "{{[OUTPUT_END]}}":
                stop_sequence[i] = (assistant_message_end.strip() if assistant_message_end.strip()!="" else "")
        stop_sequence = list(filter(None, stop_sequence))
        genparams["prompt"] = prompt
        genparams["memory"] = memory
    genparams["stop_sequence"] = stop_sequence
    return genparams

def LaunchWebbrowser(target_url, failedmsg):
    try:
        if os.name == "posix" and "DISPLAY" in os.environ:  # UNIX-like systems
            clean_env = os.environ.copy()
            clean_env.pop("LD_LIBRARY_PATH", None)
            clean_env["PATH"] = "/usr/bin:/bin"
            result = subprocess.run(["/usr/bin/env", "xdg-open", target_url], check=True, env=clean_env)
            if result.returncode == 0:
                return  # fallback successful
        raise RuntimeError("no xdg-open")
    except Exception:
        try:
            import webbrowser as wb
            if wb.open(target_url, autoraise=True):
                return  # If successful, exit the function
            raise RuntimeError("wb.open failed")
        except Exception:
            print(failedmsg)
            print(f"Please manually open your browser to {target_url}")

def get_my_epurl():
    global sslvalid
    httpsaffix = ("https" if sslvalid else "http")
    displayedport = (args.port if not args.proxy_port else args.proxy_port)
    epurl = f"{httpsaffix}://localhost:{displayedport}"
    if args.host!="":
        epurl = f"{httpsaffix}://{args.host}:{displayedport}"
    return epurl

proxy_reload_lock = threading.Lock()
###########################################################
###   A simple reverse proxy used in Kcpp Router mode   ###
###########################################################
class KcppProxyHandler(http.server.BaseHTTPRequestHandler):
    sys_version = "1"
    server_version = "KoboldCppServer"
    protocol_version = "HTTP/1.1"
    HOP_BY_HOP = { "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade" }
    STREAM_CHUNK = 512

    def log_message(self, fmt, *args):
        global showdebug
        if showdebug:
            print(f"[proxy] {self.address_string()} {fmt % args}", flush=True)
        pass

    def wait_for_upstream_ready(self, port, timeout, interval):
        start = time.time()
        while time.time() - start < timeout:
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=5)
                conn.request("GET", "/api/v1/info/version")
                resp = conn.getresponse()
                if resp.status == 200:
                    data = resp.read()
                    try:
                        json.loads(data.decode("utf-8"))
                        return True
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            time.sleep(interval)
        return False  # timeout

    def _handle(self):
        upstream_port = self.server.upstream_port
        length = self.headers.get("Content-Length") #  read request body
        body = None
        if length:
            body = self.rfile.read(int(length))
        headers = {} # forward headers
        for k, v in self.headers.items():
            if k.lower() not in self.HOP_BY_HOP:
                headers[k] = v
        headers["Connection"] = "close"

        global global_memory
        #specifically look for generation requests from completions or chat completions to handle hotswap
        is_post = self.command.upper() == "POST"
        is_completions_path = (self.path.endswith('/v1/completions') or self.path.endswith('/v1/completion') or self.path=='/completions')
        is_chat_completions_path = (self.path.endswith('/v1/chat/completions') or self.path=='/chat/completions')

        #any requests to the following endpoints is capable of waking the server
        wake_requests = ["/api/extra/generate/stream","/api/extra/tokencount","/api/v1/generate","/sdapi/v1/interrogate","/v1/completions","/v1/chat/completions","/v1/responses","/completions","/chat/completions","/responses","/api/extra/transcribe","/v1/audio/transcriptions","/api/extra/tts","/v1/audio/speech","/api/extra/embeddings","/v1/embeddings","/api/extra/music/prepare","/api/extra/music/generate","/sdapi/v1/txt2img","/sdapi/v1/img2img","/sdapi/v1/upscale"]
        is_wake_request = self.path in wake_requests

        autoswapEnabled = global_memory["autoswapmode"] is not None and global_memory["autoswapmode"]
        model_switch_pass = False

        with proxy_reload_lock:
            if is_post and (is_completions_path or is_chat_completions_path or (not autoswapEnabled and is_wake_request)):
                model_name = ""
                if body:
                    try:
                        request_json = json.loads(body.decode("utf-8"))
                        model_name = request_json.get("model")
                    except Exception:
                        pass

                was_auto_unloaded = (global_memory["triggered_sleeping"] and global_memory["current_model"]=="unload_model")

                is_different_model = False
                # we only need to check if the currently loaded model is different from the requested model. everything else is handled downstream in the stack
                if model_name and model_name != global_memory["current_model"]:
                    is_different_model = True

                if is_different_model or was_auto_unloaded:
                    model_switch_pass = True
                    whitelist = get_current_admindir_list() # see if its an allowed swap
                    if was_auto_unloaded and not model_name:
                        model_name = "initial_model"
                    if is_different_model and (model_name in whitelist):
                        global_memory["last_active_timestamp"] = datetime.now()
                        global_memory["triggered_sleeping"] = False
                        reqbody = json.dumps({"filename":model_name})
                        reqheaders = {
                            'Content-Type': 'application/json',
                            'Content-Length': str(len(reqbody)),
                        }
                        if args.adminpassword:
                            reqheaders["Authorization"] = f"Bearer {args.adminpassword}"
                        conn = http.client.HTTPConnection('localhost', upstream_port, timeout=args.reqtimeout)
                        conn.request("POST", "/api/admin/reload_config", body=reqbody, headers=reqheaders)
                        resp = conn.getresponse()
                        time.sleep(3)
                        global_memory["last_active_timestamp"] = datetime.now()
                        global_memory["triggered_sleeping"] = False
                        if not self.wait_for_upstream_ready(upstream_port,120,0.5):
                            self.send_error(504, "KoboldCpp model swap reload timed out")
                            return
                        time.sleep(0.1)
            if autoswapEnabled and not model_switch_pass:
                textReqs = ["/api/extra/generate/stream","/api/extra/tokencount","/api/v1/generate","/sdapi/v1/interrogate","/v1/completions","/v1/chat/completions","/v1/responses","/completions","/chat/completions","/responses"]
                sttReqs = ["/api/extra/transcribe","/v1/audio/transcriptions"]
                ttsReqs = ["/api/extra/tts", "/v1/audio/speech"]
                embedReqs = ["/api/extra/embeddings", "/v1/embeddings"]
                musicReqs = ["/api/extra/music/prepare","/api/extra/music/generate"]
                imageReqs = ["/sdapi/v1/txt2img", "/sdapi/v1/img2img", "/sdapi/v1/upscale"] # "/sdapi/v1/sd-models", "/sdapi/v1/options", "/sdapi/v1/samplers"

                swapModeChanged = False
                if any(self.path.endswith(e) for e in textReqs) and (global_memory["swapReqType"] is None or global_memory["swapReqType"] != "text"):
                    global_memory["swapReqType"] = "text"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in sttReqs) and (global_memory["swapReqType"] is None or global_memory["swapReqType"] != "stt"):
                    global_memory["swapReqType"] = "stt"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in ttsReqs) and (global_memory["swapReqType"] is None or global_memory["swapReqType"] != "tts"):
                    global_memory["swapReqType"] = "tts"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in embedReqs) and (global_memory["swapReqType"] is None or global_memory["swapReqType"] != "embed"):
                    global_memory["swapReqType"] = "embed"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in musicReqs) and (global_memory["swapReqType"] is None or global_memory["swapReqType"] != "music"):
                    global_memory["swapReqType"] = "music"
                    swapModeChanged = True
                elif any(self.path.endswith(e) for e in imageReqs) and (global_memory["swapReqType"] is None or global_memory["swapReqType"] != "image"):
                    global_memory["swapReqType"] = "image"
                    swapModeChanged = True

                if (global_memory["swapReqType"] is not None and swapModeChanged):
                    reqbody = json.dumps({"filename":global_memory["current_model"], "baseconfig": global_memory["base_config"]})
                    reqheaders = {
                        'Content-Type': 'application/json',
                        'Content-Length': str(len(reqbody)),
                    }
                    if args.adminpassword:
                        reqheaders["Authorization"] = f"Bearer {args.adminpassword}"
                    conn = http.client.HTTPConnection('localhost', upstream_port, timeout=args.reqtimeout)
                    conn.request("POST", "/api/admin/reload_config", body=reqbody, headers=reqheaders)
                    resp = conn.getresponse()
                    time.sleep(3)
                    global_memory["last_active_timestamp"] = datetime.now()
                    if not self.wait_for_upstream_ready(upstream_port,120,0.5):
                        self.send_error(504, "KoboldCpp model swap reload timed out")
                        return
                    time.sleep(0.1)

        try:  # connect upstream
            conn = http.client.HTTPConnection('localhost', upstream_port, timeout=args.reqtimeout)
            conn.request( self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            if args.debugmode:
                print(f"OSError has occurred: {e}")
            html_502 = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>502 - KoboldCpp</title>
                <style>
                *,dialog{padding:0;margin:0}*{box-sizing:border-box}body{background-color:#0a0e14;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,sans-serif;min-height:100vh;display:flex;justify-content:center;align-items:center}dialog{background-color:#1a222e;border:1px solid #3a4a5a;border-radius:4px;width:90%;max-width:550px;box-shadow:0 8px 32px rgba(0,0,0,.6);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%)}dialog::backdrop{background-color:rgba(0,0,0,.7)}.dialog-header{background-color:#3a506b;padding:14px 18px;border-bottom:1px solid #2a3a4a}.dialog-header h2{color:#e8e8e8;font-size:15px;font-weight:600;margin:0}.dialog-content{padding:24px 20px;text-align:center}.dialog-content p{color:#d0d0d0;font-size:15px;line-height:1.7;margin:0 0 16px}.dialog-content p:last-child{margin-bottom:0;font-style:italic;color:#a0a0a0}
                </style>
            </head>
            <body>
                <dialog open>
                    <div class="dialog-header"><h2>KoboldCpp is not available.</h2></div>
                    <div class="dialog-content">
                        <p>It may take some time during a model (re)load before it is ready to use.</p>
                        <p>Taking a long time for this message to go away?<br>It may have crashed, check the logs.</p>
                        <p>Your browser should automatically refresh when KoboldCpp is back online.</p>
                    </div>
                </dialog>
            </body>
            <script>
                setInterval(async () => {
                    try {
                        const response = await fetch(window.location.href, { cache: "no-store" });
                        if (response.ok) {
                            window.location.reload();
                        }
                    } catch (err) {
                        // Ignore network errors and try again on next interval
                    }
                }, 2000);
            </script>
            </html>
            """
            self.send_response(502)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_502.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(html_502.encode("utf-8"))
            return


        self.send_response(resp.status, resp.reason) # forward response headers
        for k, v in resp.getheaders():
            lk = k.lower()
            if lk in self.HOP_BY_HOP:
                continue
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True

        try:  # stream response
            while True:
                chunk = resp.read(self.STREAM_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
                conn.close()

    # proxy all HTTP methods
    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_PATCH = _handle
    do_OPTIONS = _handle
    do_HEAD = _handle

class KcppProxyHttpServer(http.server.HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, upstream_port):
        self.upstream_port = upstream_port
        super().__init__(server_address, RequestHandlerClass)
    def process_request(self, request, client_address):
        thread = threading.Thread(target=self._worker,args=(request, client_address),daemon=True)
        thread.start()
    def _worker(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        finally:
            self.shutdown_request(request)

def run_router_proxy(proxy_port, upstream_port):
    server = KcppProxyHttpServer(("", proxy_port), KcppProxyHandler, upstream_port)
    global args, sslvalid
    if args.ssl and sslvalid:
        import ssl
        if args.nocertify:
            ssl._create_default_https_context = ssl._create_unverified_context
        certpath = os.path.abspath(args.ssl[0])
        keypath = os.path.abspath(args.ssl[1])
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=certpath, keyfile=keypath)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print(f"KoboldCpp Proxy starting on port {proxy_port} (SSL/HTTPS), forwarding to port {upstream_port}",flush=True)
    else:
        print(f"KoboldCpp Proxy starting on port {proxy_port}, forwarding to port {upstream_port}",flush=True)
    proxy_thread = threading.Thread(target=server.serve_forever, daemon=True)
    proxy_thread.start()
    return server  # Return the server object in case you need to shut it down later


#################################################################
### A hacky simple HTTP server simulating a kobold api by Concedo
### we are intentionally NOT using flask, because we want MINIMAL dependencies
#################################################################
class KcppServerRequestHandler(http.server.SimpleHTTPRequestHandler):
    sys_version = "1"
    server_version = "KoboldCppServer"

    def __init__(self, addr, port):
        self.addr = addr
        self.port = port

    def __call__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        global showdebug
        if showdebug:
            super().log_message(format, *args)
        pass

    def extract_formdata_from_file_upload(self, body):
        result = {"file": None, "prompt": None, "language": None}
        try:
            if 'content-type' in self.headers and self.headers['content-type']:
                boundary = self.headers['content-type'].split("=")[1].encode()
                if boundary:
                    fparts = body.split(boundary)
                    for fpart in fparts:
                        detected_upload_filename = re.findall(r'Content-Disposition[^;]*;\s*name=(?:"file"|file)\s*;\s*filename=(?:"([^"]+)"|([^\s";]+))', fpart.decode('utf-8',errors='ignore'),flags=re.IGNORECASE)
                        detected_upload_filename_comfy = re.findall(r'Content-Disposition[^;]*;\s*name=(?:"image"|image)\s*;\s*filename=(?:"([^"]+)"|([^\s";]+))', fpart.decode('utf-8',errors='ignore'),flags=re.IGNORECASE)
                        if detected_upload_filename and len(detected_upload_filename)>0:
                            utfprint(f"Detected uploaded file: {detected_upload_filename[0]}")
                            file_content_start = fpart.find(b'\r\n\r\n') + 4  # Position after headers
                            file_content_end = fpart.rfind(b'\r\n')  # Ending boundary
                            if file_content_start != -1 and file_content_end != -1:
                                if "file" in result and result["file"] is None:
                                    file_data = fpart[file_content_start:file_content_end]
                                    file_data_base64 = base64.b64encode(file_data).decode('utf-8',"ignore")
                                    base64_string = f"data:audio/wav;base64,{file_data_base64}"
                                    result["file"] = base64_string
                        elif detected_upload_filename_comfy and len(detected_upload_filename_comfy)>0:
                            utfprint(f"Detected uploaded image: {detected_upload_filename_comfy[0]}")
                            file_content_start = fpart.find(b'\r\n\r\n') + 4  # Position after headers
                            file_content_end = fpart.rfind(b'\r\n')  # Ending boundary
                            if file_content_start != -1 and file_content_end != -1:
                                if "file" in result and result["file"] is None:
                                    file_data = fpart[file_content_start:file_content_end]
                                    file_data_base64 = base64.b64encode(file_data).decode('utf-8',"ignore")
                                    base64_string = f"{file_data_base64}"
                                    result["file"] = base64_string

                        # Check for fields
                        detected_prompt_field = re.findall(r'Content-Disposition.*name="prompt"\r\n\r\n(.*)\r\n', fpart.decode('utf-8', errors='ignore'))
                        if detected_prompt_field and len(detected_prompt_field)>0:
                            result["prompt"] = detected_prompt_field[0].strip()  # Extract and strip whitespace

                        detected_lang_field = re.findall(r'Content-Disposition.*name="language"\r\n\r\n(.*)\r\n', fpart.decode('utf-8', errors='ignore'))
                        if detected_lang_field and len(detected_lang_field)>0:
                            result["language"] = detected_lang_field[0].strip()  # Extract and strip whitespace

            if not ("file" in result and result["file"]):
                print("Uploaded file not found.")
            return result
        except Exception as e:
            print(f"File Upload Process Error: {e}")
            return result

    def prepare_basic_responses_body(self,resp_id,genparams):
        global friendlymodelname
        global autoswapmode, textName, sttName, ttsName, embedName, musicName, imageName, mmprojName

        modelNameToReturn = friendlymodelname
        if autoswapmode and textName is not None:
            modelNameToReturn = textName
        ret = {
            "id": resp_id,
            "object": "response",
            "created_at": int(time.time()),
            "completed_at": None,
            "incomplete_details": None,
            "previous_response_id": None,
            "truncation": "disabled",
            "parallel_tool_calls": False,
            "text": {"format": {"type": "text"},"verbosity": "medium"},
            "instructions": genparams.get('instructions', None),
            "model": modelNameToReturn,
            "error": None,
            "metadata": {},
            "tools": genparams.get('tools', []),
            "tool_choice": "auto",
            "background": False,
            "service_tier": "default",
            "safety_identifier": None,
            "prompt_cache_key": None,
            "max_tool_calls": None,
            "store": False,
            "top_p": genparams.get("top_p", 0.92),
            "max_output_tokens":genparams.get("max_length", None),
            "presence_penalty": genparams.get("presence_penalty", 0),
            "frequency_penalty": genparams.get("frequency_penalty", 0),
            "top_logprobs": 0,
            "temperature": genparams.get("temperature", 1),
            "reasoning": {"effort": None, "summary": None},
            "usage": None
        }
        return ret

    async def generate_text(self, genparams, api_format, stream_flag):
        global friendlymodelname, chatcompl_adapter, currfinishreason, thinkformats
        global autoswapmode, textName, sttName, ttsName, embedName, musicName, imageName, mmprojName

        currfinishreason = None
        req_id_suffix = genparams.get('oai_uniqueid',1)
        chatcmpl_id = f"chatcmpl-A{req_id_suffix}"
        cmpl_id = f"cmpl-A{req_id_suffix}"

        def run_blocking():  # api format 1=basic,2=kai,3=oai,4=oai-chat
            # flag instance as non-idle for a while
            washordereq = genparams.get('genkey', '').startswith('HORDEREQ_')
            if not washordereq:
                global last_non_horde_req_time
                last_non_horde_req_time = time.time()

            return generate(genparams=genparams,stream_flag=stream_flag)

        genout = {"text": "", "status": -1, "stopreason": -1, "prompt_tokens":0, "completion_tokens": 0, "total_tokens": 0}
        if stream_flag:
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor()
            genout = await loop.run_in_executor(executor, run_blocking)
        else:
            genout = run_blocking()

        recvtxt = genout['text']
        prompttokens = genout['prompt_tokens'] if genout['prompt_tokens'] > 0 else 0
        comptokens = genout['completion_tokens'] if genout['completion_tokens'] > 0 else 0
        currfinishreason = "error" if (genout['stopreason'] == -2) else ("length" if (genout['stopreason'] != 1) else "stop")

        # grab logprobs if not streaming
        logprobsdict = None
        if not stream_flag and ("logprobs" in genparams and genparams["logprobs"]):
            lastlogprobs = handle.last_logprobs()
            logprobsdict = parse_last_logprobs(lastlogprobs)

        # flag instance as non-idle for a while
        washordereq = genparams.get('genkey', '').startswith('HORDEREQ_')
        if not washordereq:
            global last_non_horde_req_time
            last_non_horde_req_time = time.time()

        utfprint("\nOutput: " + recvtxt,1)

        #tool calls resolution
        tool_calls = []
        if api_format == 4 or api_format == 2 or api_format == 8:
            using_openai_tools = genparams.get('using_openai_tools', False)
            if using_openai_tools:
                # first, check and potentially segment multiple tags for multi-tool calls
                tool_calls = repack_toolcall_tags(recvtxt,genparams.get('tools', []))
                if tool_calls and len(tool_calls)>0:
                    flat = []
                    for obj in tool_calls:
                        if isinstance(obj, list):
                            flat.extend(obj)
                        else:
                            flat.append(obj)
                    tool_calls = [normalize_tool_call(obj) for obj in flat]
                    for tc in tool_calls:
                        tcarg = tc.get("function",{}).get("arguments",None)
                        tc["id"] = f"call_{random.randint(10000, 99999)}"
                        if tcarg is not None and not isinstance(tcarg, str):
                            tc["function"]["arguments"] = json.dumps(tcarg)
                    recvtxt = None
                    currfinishreason = "tool_calls"
                    if args.debugmode >= 1:
                        print(f"\nDebug ToolCall Response: {json.dumps(tool_calls)}")

        modelNameToReturn = friendlymodelname
        if autoswapmode and textName is not None:
            modelNameToReturn = textName

        #handle potential think tags, but only chat completions will return them. the others just drop them
        reasoningtxt = ""
        if api_format==4 or api_format==8 or api_format==9: #chat completions, responses and anthropic messages, but only chat has reasoning returned
            if recvtxt:
                for pair in thinkformats:
                    starter = pair['start']
                    ender = pair['end']
                    start_idx = recvtxt.find(starter)
                    end_idx = recvtxt.find(ender, start_idx + len(starter))
                    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                        reasoningtxt = recvtxt[start_idx + len(starter):end_idx]
                        recvtxt = recvtxt[:start_idx] + recvtxt[end_idx + len(ender):]
                        break
                    elif starter not in recvtxt and ender in recvtxt:
                        parts = recvtxt.split(ender, 1)
                        reasoningtxt = parts[0]
                        recvtxt = parts[1]
                        break
        if api_format == 1:
            res = {"data": {"seqs": [recvtxt]}}
        elif api_format == 3:
            res = {"id": cmpl_id, "object": "text_completion", "created": int(time.time()), "model": modelNameToReturn,
                   "usage": {"prompt_tokens": prompttokens, "completion_tokens": comptokens, "total_tokens": (prompttokens+comptokens)},
                   "choices": [{"text": recvtxt, "index": 0, "finish_reason": currfinishreason, "logprobs":logprobsdict}]}
        elif api_format == 4: #chat completions
            ccmsg = {"role": "assistant", "content": recvtxt, "tool_calls": tool_calls}
            if reasoningtxt and genparams.get('encapsulate_thinking', True):
                ccmsg["reasoning_content"] = reasoningtxt
            else:
                ccmsg["content"] = reasoningtxt + (recvtxt if recvtxt else "")
            res = {"id": chatcmpl_id, "object": "chat.completion", "created": int(time.time()), "model": modelNameToReturn,
                   "usage": {"prompt_tokens": prompttokens, "completion_tokens": comptokens, "total_tokens": (prompttokens+comptokens)},
                   "choices": [{"index": 0, "message": ccmsg, "finish_reason": currfinishreason, "logprobs":logprobsdict}]}
        elif api_format == 5:
            res = {"caption": end_trim_to_sentence(recvtxt)}
        elif api_format == 6:
            oldprompt = genparams.get('ollamabodyprompt', "")
            tokarr = tokenize_ids(oldprompt+recvtxt,False)
            res = {"model": modelNameToReturn,"created_at": str(datetime.now(timezone.utc).isoformat()),"response":recvtxt,"done": True,"done_reason":currfinishreason,"context": tokarr,"total_duration": 1,"load_duration": 1,"prompt_eval_count": prompttokens,"prompt_eval_duration": 1,"eval_count": comptokens,"eval_duration": 1}
        elif api_format == 7:
            res = {"model": modelNameToReturn,"created_at": str(datetime.now(timezone.utc).isoformat()),"message":{"role":"assistant","content":recvtxt},"done": True,"done_reason":currfinishreason,"total_duration": 1,"load_duration": 1,"prompt_eval_count": prompttokens,"prompt_eval_duration": 1,"eval_count": comptokens,"eval_duration": 1}
        elif api_format == 8: #oai-responses
            resp_id = f"resp-A{genparams.get('oai_uniqueid', 1)}"
            output_item_id = f"msg_0{genparams.get('oai_uniqueid', 1)}"
            output_items = []
            if recvtxt is not None:    # Add text message if there's content
                output_items.append({
                    "type": "message",
                    "id": output_item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": recvtxt, "annotations": [], "logprobs": []}]
                })
            if tool_calls and len(tool_calls) > 0:  # Add function call items if tool calls exist
                for tc in tool_calls:
                    output_items.append({"type": "function_call", "id": tc.get("id", ""), "call_id": tc.get("id", ""), "name": tc.get("function", {}).get("name", ""), "arguments": tc.get("function", {}).get("arguments", "{}"), "status": "completed"})
            res = self.prepare_basic_responses_body(resp_id,genparams)
            res["completed_at"] = int(time.time())
            res["status"] = "completed" if currfinishreason != "error" else "failed"
            res["output"] = output_items
            res["usage"] = {"input_tokens": prompttokens, "output_tokens": comptokens, "total_tokens": prompttokens + comptokens, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}
        elif api_format == 9: # Anthropic Format
            anthropic_reason = "end_turn" if currfinishreason == "stop" else ("max_tokens" if currfinishreason == "length" else "stop_sequence")
            res = {
                "id": f"msg_A{req_id_suffix}",
                "type": "message",
                "role": "assistant",
                "model": modelNameToReturn,
                "content": [{"type": "text", "text": recvtxt}],
                "stop_reason": anthropic_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": prompttokens, "output_tokens": comptokens}
            }
        else: #kcpp format
            res = {"results": [{"text": recvtxt, "tool_calls": tool_calls, "finish_reason": currfinishreason, "logprobs":logprobsdict, "prompt_tokens": prompttokens, "completion_tokens": comptokens}]}

        try:
            return res
        except Exception as e:
            print(f"Generate: Error while generating: {e}")

    async def send_oai_sse_event(self, data):
        if data and data.strip()=="[DONE]":
            self.wfile.write(f'data: {data.strip()}\n\n'.encode())
        else:
            self.wfile.write(f'data: {data}\n\n'.encode())
        self.wfile.flush()

    async def send_oai_responses_sse_event(self, eventname, data):
        self.wfile.write(f'event: {eventname}\ndata: {data}\n\n'.encode())
        self.wfile.flush()

    async def send_anthropic_sse_event(self, eventname, data):
        self.wfile.write(f'event: {eventname}\ndata: {data}\n\n'.encode())
        self.wfile.flush()

    async def send_kai_sse_event(self, data):
        self.wfile.write('event: message\n'.encode())
        self.wfile.write(f'data: {data}\n\n'.encode())
        self.wfile.flush()

    async def handle_sse_stream(self, genparams, api_format):
        global friendlymodelname, currfinishreason, thinkformats, tool_call_pairs, cached_chat_template
        global autoswapmode, textName, sttName, ttsName, embedName, musicName, imageName, mmprojName

        modelNameToReturn = friendlymodelname
        if autoswapmode and textName is not None:
            modelNameToReturn = textName

        using_openai_tools = genparams.get('using_openai_tools', False)
        req_id_suffix = genparams.get('oai_uniqueid',1)
        chatcmpl_id = f"chatcmpl-A{req_id_suffix}"
        cmpl_id = f"cmpl-A{req_id_suffix}"
        self.send_response(200)
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.end_headers(content_type='text/event-stream')

        # if tools, do not send anything else - OAI tool calls will be handled with fakestreaming!
        # only exception is if we know the exact toolcall tag to segment!
        tool_segment_tag = ""
        for start, end, streamhandled in tool_call_pairs:
            if streamhandled and cached_chat_template and start in cached_chat_template:
                tool_segment_tag = start
                break
        jinjatools = (args.jinja and args.jinja_tools)
        if api_format == 4 and using_openai_tools:
            if not jinjatools or not tool_segment_tag:
                genparams['sync_toolcall_stream_ineligible'] = True
                return

        think_tag_buf = ""
        encap_in_thinking = False
        if genparams.get('already_started_thinking', False):
            encap_in_thinking = True
        encap_first_loop = True
        thinkpairs = json.loads(json.dumps(thinkformats))
        responses_first_loop = True
        anthropic_first_loop = True
        rseq_num = 0
        current_token = 0
        prompttokens = 0
        incomplete_token_buffer = bytearray()
        async_sleep_short = 0.02
        await asyncio.sleep(0.35) #anti race condition, prevent check from overtaking generate
        batch_request_id = genparams.get('_batch_request_id', -1)
        batch_final_result = None

        try:
            tokenReserve = "" #keeps fully formed tokens that we cannot send out yet
            while True:
                if batch_request_id < 0:
                    batch_request_id = genparams.get('_batch_request_id', -1)
                    if genparams.get('_batch_expected', False) and batch_request_id < 0 and not genparams.get('_batch_fallback', False):
                        await asyncio.sleep(async_sleep_short)
                        continue
                using_batch_stream = batch_request_id >= 0
                streamDone = handle.batch_generate_has_finished(batch_request_id) if using_batch_stream else handle.has_finished() #exit next loop on done
                if streamDone:
                    if using_batch_stream and batch_final_result is None:
                        batch_final_result = handle.batch_generate_result(batch_request_id)
                    sr = batch_final_result.stopreason if using_batch_stream else handle.get_last_stop_reason()
                    currfinishreason = "error" if sr==-2 else ("length" if (sr!=1) else "stop")
                    prompttokens = batch_final_result.prompt_tokens if using_batch_stream else handle.get_last_input_count()
                tokenStr = ""
                streamcount = handle.batch_generate_stream_count(batch_request_id) if using_batch_stream else handle.get_stream_count()
                while current_token < streamcount:
                    token = handle.batch_generate_new_token(batch_request_id, current_token) if using_batch_stream else handle.new_token(current_token)

                    if token is None: # Token isnt ready yet, received nullpointer
                        break

                    current_token += 1
                    newbyte = ctypes.string_at(token)
                    incomplete_token_buffer += bytearray(newbyte)
                    tokenSeg = incomplete_token_buffer.decode("UTF-8","ignore")
                    incseq = is_incomplete_utf8_sequence(incomplete_token_buffer)
                    badFragment = (tokenSeg==" " and len(incomplete_token_buffer)>1) or incseq #partial incomplete unicode
                    if tokenSeg!="" and not badFragment:
                        incomplete_token_buffer.clear()
                        tokenStr += tokenSeg

                if tokenStr!="" or streamDone:
                    # split think tag handling
                    tokenStr = think_tag_buf + tokenStr
                    think_tag_buf = ""
                    if not streamDone and genparams.get('encapsulate_thinking', True):
                        tail = ""
                        for pair in thinkpairs:
                            for tag in (pair["start"], pair["end"]):
                                for n in range(1, len(tag)):
                                    if tokenStr.endswith(tag[:n]) and len(tag[:n]) > len(tail):
                                        tail = tag[:n]
                        if tail:
                            think_tag_buf = tail
                            tokenStr = tokenStr[:-len(tail)]
                    # end split think tag handling

                    sseq = genparams.get('stop_sequence', [])
                    trimstop = genparams.get('trim_stop', True)
                    if trimstop and not streamDone and string_contains_or_overlaps_sequence_substring(tokenStr,sseq):
                        tokenReserve += tokenStr
                        await asyncio.sleep(async_sleep_short) #if a stop sequence could trigger soon, do not send output
                    else:
                        if tokenStr!="" or tokenReserve!="":
                            tokenStr = tokenReserve + tokenStr
                            tokenReserve = ""

                            #apply trimming if needed
                            if trimstop:
                                for trim_str in sseq:
                                    sindex = tokenStr.find(trim_str)
                                    if sindex != -1 and trim_str!="":
                                        tokenStr = tokenStr[:sindex]

                        sync_potential_toolcall_splitmatch = ""
                        if tokenStr!="" or streamDone:
                            # Tool boundary detection for tool-capable chat completions.
                            # if triggered, stop real streaming, and let the buffered fakestreaming take over
                            if api_format == 4 and using_openai_tools:
                                tokenStr = tokenReserve + tokenStr
                                tokenReserve = ""
                                if tool_segment_tag in tokenStr:
                                    if not genparams.get("sync_toolcall_potential_triggered",False):
                                        sync_potential_toolcall_splitmatch = tool_segment_tag
                                        genparams['sync_toolcall_potential_triggered'] = True #if tool calls is triggered, rest will be sync fake streaming. we'll buffer it for later

                            need_split_final_msg = True if (currfinishreason is not None and streamDone and tokenStr!="") else False

                            # Hack for lcppui reasoning_content for thinking models
                            delta = {'role': 'assistant'}
                            if genparams.get('encapsulate_thinking', True):
                                if encap_in_thinking:
                                    foundend = False
                                    for pair in thinkpairs:
                                        if pair["end"] in tokenStr:
                                            encap_in_thinking = False
                                            foundend = True
                                            out1, out2 = tokenStr.split(pair["end"], 1)
                                            # Swallow any extraneous start tags that appear while already thinking
                                            if pair["start"] in out1:
                                                out1 = out1.replace(pair["start"], "")
                                            if out1:
                                                delta['reasoning_content'] = out1
                                            # Swallow any extraneous end tags that appear after already ending
                                            if pair["end"] in out2:
                                                out2 = out2.replace(pair["end"], "")
                                            if out2:
                                                delta['content'] = out2
                                            break
                                    if not foundend:
                                        # Still thinking - swallow extraneous start tags from THIS pair only
                                        cleaned = tokenStr
                                        if pair["start"] in cleaned:
                                            cleaned = cleaned.replace(pair["start"], "")
                                        delta['reasoning_content'] = cleaned
                                else:
                                    # Not thinking. Let's see if a start tag appears in this chunk.
                                    matched_start = False
                                    for pair in thinkpairs:
                                        # Condition A: The prompt ended exactly with the start tag
                                        if encap_first_loop and genparams.get("prompt", "").endswith(pair["start"]):
                                            encap_in_thinking = True
                                            thinkpairs = [pair] # lock in this pair
                                            delta['reasoning_content'] = tokenStr
                                            matched_start = True
                                            break
                                        # Condition B: The start tag is inside this chunk
                                        elif pair["start"] in tokenStr:
                                            encap_in_thinking = True
                                            thinkpairs = [pair] # lock in this pair
                                            out1, out2 = tokenStr.split(pair["start"], 1)
                                            # Preserve text that came BEFORE the start tag
                                            if out1:
                                                delta['content'] = out1
                                            if out2:
                                                delta['reasoning_content'] = out2
                                            # Edge Case: The end tag is ALSO in this exact same chunk (in out2)
                                            if pair["end"] in out2:
                                                encap_in_thinking = False
                                                out2_think, out2_content = out2.split(pair["end"], 1)
                                                # Overwrite reasoning with the exact thinking part
                                                delta['reasoning_content'] = out2_think
                                                # Append anything after the end tag to the content part
                                                if out2_content:
                                                    delta['content'] = delta.get('content', '') + out2_content
                                            matched_start = True
                                            break
                                    # Condition C: No start tag found, just normal text
                                    # Swallow any extraneous end tags that appear while not thinking
                                    if not matched_start:
                                        cleaned = tokenStr
                                        # Only swallow stray end tags if we've already locked in a pair
                                        if len(thinkpairs) == 1 and thinkpairs[0]["end"] in cleaned:
                                            cleaned = cleaned.replace(thinkpairs[0]["end"], "")
                                        delta['content'] = cleaned
                                encap_first_loop = False
                            else:
                                delta['content'] = tokenStr

                            if genparams.get("sync_toolcall_potential_triggered",False) and delta: # if sync_toolcall_potential_triggered, buffer up the impending content chunk for tools in fakestreaming, in case toolcalls fail
                                ec = genparams.get("sync_toolcall_extra_content","")
                                erc = genparams.get("sync_toolcall_extra_reasoning_content","")
                                ec += delta.get("content","")
                                erc += delta.get("reasoning_content","")
                                if erc and sync_potential_toolcall_splitmatch and sync_potential_toolcall_splitmatch in erc:
                                    parts = erc.split(sync_potential_toolcall_splitmatch,1)
                                    erc = sync_potential_toolcall_splitmatch + parts[1]
                                    delta["reasoning_content"] = parts[0]
                                elif ec and sync_potential_toolcall_splitmatch and sync_potential_toolcall_splitmatch in ec:
                                    parts = ec.split(sync_potential_toolcall_splitmatch,1)
                                    ec = sync_potential_toolcall_splitmatch + parts[1]
                                    delta["content"] = parts[0]
                                genparams['sync_toolcall_extra_content'] = ec
                                genparams['sync_toolcall_extra_reasoning_content'] = erc
                                if not sync_potential_toolcall_splitmatch:
                                    if not streamDone:
                                        await asyncio.sleep(async_sleep_short)
                                        continue
                                    await asyncio.sleep(async_sleep_short)
                                    return

                            if need_split_final_msg: #we need to send one message without the finish reason, then send a finish reason with no msg to follow standards
                                if api_format == 4:  # if oai chat, set format to expected openai streaming response
                                    event_str = json.dumps({"id":chatcmpl_id,"object":"chat.completion.chunk","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":None,"delta":delta}]})
                                    await self.send_oai_sse_event(event_str)
                                elif api_format == 3:  # non chat completions
                                    event_str = json.dumps({"id":cmpl_id,"object":"text_completion","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":None,"text":tokenStr}]})
                                    await self.send_oai_sse_event(event_str)
                                else:
                                    event_str = json.dumps({"token": tokenStr, "finish_reason":None})
                                    await self.send_kai_sse_event(event_str)
                                tokenStr = "" # now the final finish reason can be sent alone
                                if delta and 'role' in delta:
                                    delta = {'role':delta["role"],'content':''}
                            if api_format == 4:  # if oai chat, set format to expected openai streaming response
                                if streamDone and ("logprobs" in genparams and genparams["logprobs"]): # this is a hack that sends an extra message containing ALL the logprobs
                                    lastlogprobs = handle.last_logprobs()
                                    logprobsdict = parse_last_logprobs(lastlogprobs)
                                    addonstr = json.dumps({"id":chatcmpl_id,"object":"chat.completion.chunk","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":None,"delta":{'role':'assistant','content':''},"logprobs":logprobsdict}]})
                                    await self.send_oai_sse_event(addonstr)
                                event_str = json.dumps({"id":chatcmpl_id,"object":"chat.completion.chunk","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":currfinishreason,"delta":delta}]})
                                genparams['sync_toolcall_first_role_sent'] = True
                                await self.send_oai_sse_event(event_str)
                            elif api_format == 3:  # non chat completions
                                if streamDone and ("logprobs" in genparams and genparams["logprobs"]): # this is a hack that sends an extra message containing ALL the logprobs
                                    lastlogprobs = handle.last_logprobs()
                                    logprobsdict = parse_last_logprobs(lastlogprobs)
                                    addonstr = json.dumps({"id":cmpl_id,"object":"text_completion","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":None,"text":"","logprobs":logprobsdict}]})
                                    await self.send_oai_sse_event(addonstr)
                                event_str = json.dumps({"id":cmpl_id,"object":"text_completion","created":int(time.time()),"model":modelNameToReturn,"choices":[{"index":0,"finish_reason":currfinishreason,"text":tokenStr}]})
                                await self.send_oai_sse_event(event_str)
                            elif api_format == 8: #oai-responses
                                resp_id = f"resp-A{genparams.get('oai_uniqueid', 1)}"
                                item_id = f"msg_0{genparams.get('oai_uniqueid', 1)}"
                                # Send response.created once at the start (only on first iteration)
                                if responses_first_loop:
                                    res = self.prepare_basic_responses_body(resp_id, genparams)
                                    res["status"] = "in_progress"
                                    res["output"] = []
                                    created_event = json.dumps({"type": "response.created", "response": res, "sequence_number":rseq_num})
                                    rseq_num += 1
                                    await self.send_oai_responses_sse_event("response.created",created_event)
                                    # response.output_item.added
                                    item_added = json.dumps({"type": "response.output_item.added", "output_index": 0, "sequence_number":rseq_num, "item": { "type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": []}})
                                    rseq_num += 1
                                    await self.send_oai_responses_sse_event("response.output_item.added",item_added)
                                    # content_part.added
                                    part_added = json.dumps({"type": "response.content_part.added", "item_id": item_id, "output_index": 0, "sequence_number":rseq_num, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})
                                    rseq_num += 1
                                    await self.send_oai_responses_sse_event("response.content_part.added",part_added)
                                    responses_first_loop = False
                                if tokenStr != "" or streamDone:
                                    if tokenStr != "":
                                        delta_event = json.dumps({"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "sequence_number":rseq_num, "logprobs":[], "content_index": 0, "delta": tokenStr})
                                        rseq_num += 1
                                        await self.send_oai_responses_sse_event("response.output_text.delta",delta_event)
                                    if streamDone:
                                        # content_part.done, reply full text
                                        await asyncio.sleep(async_sleep_short)
                                        finalraw = handle.batch_generate_pending_output(batch_request_id) if using_batch_stream else handle.get_pending_output()
                                        finaltxt = finalraw.decode("UTF-8", "ignore")
                                        await asyncio.sleep(async_sleep_short)
                                        done_event = json.dumps({"type": "response.output_text.done", "item_id": item_id, "output_index": 0, "sequence_number":rseq_num, "content_index": 0, "text": finaltxt})
                                        rseq_num += 1
                                        await self.send_oai_responses_sse_event("response.output_text.done",done_event)
                                        # response.output_item.done
                                        item_done = json.dumps({"type": "response.output_item.done", "output_index": 0, "sequence_number":rseq_num, "item": { "type": "message", "id": item_id, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": finaltxt, "annotations": [], "logprobs": []}]}})
                                        rseq_num += 1
                                        await self.send_oai_responses_sse_event("response.output_item.done",item_done)
                                        usage_pp = batch_final_result.prompt_tokens if using_batch_stream else handle.get_last_input_count()
                                        usage_gen = current_token
                                        res = self.prepare_basic_responses_body(resp_id,genparams)
                                        res["completed_at"] = int(time.time())
                                        res["status"] = "completed" if currfinishreason != "error" else "failed"
                                        res["output"] = [{"type": "message", "id": item_id, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": finaltxt, "annotations": [], "logprobs": []}]}]
                                        res["usage"] = {"input_tokens": usage_pp,"input_tokens_details":{"cached_tokens":0}, "output_tokens": usage_gen, "output_tokens_details":{"reasoning_tokens":0}, "total_tokens": usage_pp + usage_gen}
                                        completed_event = json.dumps({"type": "response.completed", "response": res, "sequence_number":rseq_num})
                                        rseq_num += 1
                                        await self.send_oai_responses_sse_event("response.completed",completed_event)
                            elif api_format == 9: # Anthropic Streaming Format
                                if anthropic_first_loop:
                                    start_msg = json.dumps({"type":"message","id":f"msg_A{req_id_suffix}","role":"assistant","model":modelNameToReturn,"usage":{"input_tokens":prompttokens,"output_tokens":0}})
                                    await self.send_anthropic_sse_event("message_start", json.dumps({"type": "message_start", "message": json.loads(start_msg)}))
                                    await self.send_anthropic_sse_event("content_block_start", json.dumps({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}))
                                    anthropic_first_loop = False
                                if tokenStr != "":
                                    await self.send_anthropic_sse_event("content_block_delta", json.dumps({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":tokenStr}}))
                                if streamDone:
                                    anthropic_reason = "end_turn" if currfinishreason == "stop" else ("max_tokens" if currfinishreason == "length" else "stop_sequence")
                                    await self.send_anthropic_sse_event("content_block_stop", json.dumps({"type":"content_block_stop","index":0}))
                                    await self.send_anthropic_sse_event("message_delta", json.dumps({"type":"message_delta","delta":{"stop_reason":anthropic_reason,"stop_sequence":None},"usage":{"output_tokens":current_token}}))
                                    await self.send_anthropic_sse_event("message_stop", json.dumps({"type":"message_stop"}))
                            else:
                                event_str = json.dumps({"token": tokenStr, "finish_reason":currfinishreason})
                                await self.send_kai_sse_event(event_str)
                            tokenStr = ""
                        else:
                            await asyncio.sleep(async_sleep_short)
                else:
                    await asyncio.sleep(async_sleep_short) #this should keep things responsive

                if streamDone:
                    if api_format == 4 or api_format == 3:  # if oai chat, send last [DONE] message consistent with openai format
                        strop = genparams.get("stream_options",None)
                        if (strop and strop.get("include_usage",False)):  # Send a final chunk with usage info, only if requested
                            usage_obj = {"prompt_tokens": prompttokens, "completion_tokens": current_token, "total_tokens": (prompttokens + current_token)}
                            if api_format == 4:
                                usage_str = json.dumps({"id":chatcmpl_id,"object":"chat.completion.chunk","created":int(time.time()),"model":modelNameToReturn,"choices":[],"usage":usage_obj})
                            else:
                                usage_str = json.dumps({"id":cmpl_id,"object":"text_completion","created":int(time.time()),"model":modelNameToReturn,"choices":[],"usage":usage_obj})
                            await self.send_oai_sse_event(usage_str)
                        await self.send_oai_sse_event('[DONE]')
                        await asyncio.sleep(async_sleep_short)
                    break
        except Exception as ex:
            print("Token streaming was interrupted or aborted!")
            print(ex)
            if batch_request_id >= 0:
                handle.batch_generate_abort(batch_request_id)
            else:
                handle.abort_generate()
            await asyncio.sleep(0.2) #short delay
        finally:
            if batch_request_id >= 0:
                handle.batch_generate_release(batch_request_id)
                genparams.pop('_batch_request_id', None)
            genparams.pop('_batch_expected', None)
            genparams.pop('_batch_fallback', None)

        # flush buffers, sleep a bit to make sure all data sent, and then force close the connection
        self.wfile.flush()
        await asyncio.sleep(0.1)
        self.close_connection = True
        await asyncio.sleep(0.05)

    async def monitor_connection(self): #Poll the socket to detect client disconnection during prompt processing
        import select
        loop = asyncio.get_event_loop()
        def check_connection_closed():
            try:
                sock = self.connection
                readable, _, exceptional = select.select([sock], [], [sock], 0)
                if exceptional:
                    return True
                if readable:
                    data = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                    if len(data) == 0:
                        return True
                return False
            except (OSError, Exception):
                return True  # Treat any error as disconnected
        while True:
            try:
                await asyncio.sleep(0.5)
                disconnected = await loop.run_in_executor(None, check_connection_closed)
                if disconnected:
                    if args.debugmode:
                        print("\nClient disconnected unexpectedly, aborting...")
                    handle.abort_generate()
                    return
            except Exception:
                return

    async def handle_request(self, genparams, api_format, stream_flag):
        tasks = []
        genparams["oai_uniqueid"] = random.randint(100000, 999999)
        monitor_task = None
        try:
            if stream_flag:
                tasks.append(self.handle_sse_stream(genparams, api_format))
            generate_task = asyncio.create_task(self.generate_text(genparams, api_format, stream_flag))
            tasks.append(generate_task)
            if stream_flag:
                monitor_task = asyncio.create_task(self.monitor_connection())
            await asyncio.gather(*tasks)
            generate_result = generate_task.result()
            return generate_result
        except (BrokenPipeError, ConnectionAbortedError) as cae: # attempt to abort if connection lost
            print("An ongoing connection was aborted or interrupted!")
            print(cae)
            handle.abort_generate()
            await asyncio.sleep(0.2) #short delay
        except Exception as e:
            print(e)
        finally:
            if monitor_task and not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

    def get_multiplayer_idle_state(self,userid):
        if modelbusy.locked() or batched_request_runner_count>0:
            return False
        for key, value in multiplayer_lastactive.items():
            if key!=userid and time.time()-value<6: #6s to idle
                return False
        return True

    def check_header_password(self, target_password, target_alt_password=None):
        auth_ok = True
        if target_password and target_password !="":
            auth_header = None
            auth_ok = False
            if 'Authorization' in self.headers:
                auth_header = self.headers['Authorization']
            elif 'authorization' in self.headers:
                auth_header = self.headers['authorization']
            if auth_header is not None and auth_header.startswith('Bearer '):
                token = auth_header[len('Bearer '):].strip()
                if token==target_password or (target_alt_password and target_alt_password!="" and token==target_alt_password):
                    auth_ok = True
        return auth_ok

    def secure_endpoint(self): #returns false if auth fails. caller should exit
        #handle password stuff
        auth_ok = self.check_header_password(password, args.adminpassword)
        if auth_ok is False:
            self.send_response(401)
            self.end_headers(content_type='application/json')
            self.wfile.write(json.dumps({"detail": {
                    "error": "Unauthorized",
                    "msg": "Authentication key is missing or invalid.",
                    "type": "unauthorized",
                }}).encode())
            return False
        return True

    def noscript_webui(self):
        global modelbusy, sslvalid, batched_request_runner_count
        parsed_url = urllib.parse.urlparse(self.path)
        parsed_dict = urllib.parse.parse_qs(parsed_url.query)
        reply = ""
        status = str(parsed_dict['status'][0]) if 'status' in parsed_dict else "Ready To Generate"
        prompt = str(parsed_dict['prompt'][0]) if 'prompt' in parsed_dict else ""
        chatmsg = str(parsed_dict['chatmsg'][0]) if 'chatmsg' in parsed_dict else ""
        imgprompt = str(parsed_dict['imgprompt'][0]) if 'imgprompt' in parsed_dict else ""
        max_length = int(parsed_dict['max_length'][0]) if 'max_length' in parsed_dict else 100
        temperature = float(parsed_dict['temperature'][0]) if 'temperature' in parsed_dict else 0.75
        top_k = int(parsed_dict['top_k'][0]) if 'top_k' in parsed_dict else 100
        top_p = float(parsed_dict['top_p'][0]) if 'top_p' in parsed_dict else 0.9
        rep_pen = float(parsed_dict['rep_pen'][0]) if 'rep_pen' in parsed_dict else 1.0
        ban_eos_token = int(parsed_dict['ban_eos_token'][0]) if 'ban_eos_token' in parsed_dict else 0
        steps = int(parsed_dict['steps'][0]) if 'steps' in parsed_dict else 25
        cfg = int(parsed_dict['cfg'][0]) if 'cfg' in parsed_dict else 7
        genbtnval = (parsed_dict['generate'][0] if 'generate' in parsed_dict else "")
        gencommand = (genbtnval=="Generate" or genbtnval=="Send")
        chatmode = int(parsed_dict['chatmode'][0]) if 'chatmode' in parsed_dict else 0
        imgmode = int(parsed_dict['imgmode'][0]) if 'imgmode' in parsed_dict else 0
        human_name = str(parsed_dict['human_name'][0]) if 'human_name' in parsed_dict else "User"
        bot_name = str(parsed_dict['bot_name'][0]) if 'bot_name' in parsed_dict else "Assistant"
        stops = []
        prefix = ""
        if chatmode:
            ban_eos_token = False
            prompt = prompt.replace("1HdNl1","\n")
            if chatmsg:
                prompt += f"\n{human_name}: {chatmsg}\n{bot_name}:"
            else:
                gencommand = False
            stops = [f"\n{human_name}:",f"\n{bot_name}:"]
            prefix = f"[This is a chat conversation log between {human_name} and {bot_name}.]\n"
        elif imgmode:
            if imgprompt:
                prompt = imgprompt
                max_length = 1
            else:
                gencommand = False

        if modelbusy.locked() or batched_request_runner_count>0:
            status = "Model is currently busy, try again later."
        elif gencommand:
            if prompt=="" or max_length<=0:
                status = "Need a valid prompt and length to generate."
            else:
                if max_length>512:
                    max_length = 512
                epurl = get_my_epurl()
                if imgmode and imgprompt:
                    gen_payload = {"prompt":{"3":{"class_type": "KSampler","inputs":{"cfg":cfg,"steps":steps,"latent_image":["5", 0],"positive": ["6", 0]}},"5":{"class_type": "EmptyLatentImage","inputs":{"height":512,"width":512}},"6":{"class_type": "CLIPTextEncode","inputs":{"text":imgprompt}}}}
                    respjson = make_url_request(f'{epurl}/prompt', gen_payload)
                else:
                    gen_payload = {"prompt": prefix+prompt,"max_length": max_length,"temperature": temperature,"top_k": top_k,"top_p": top_p,"rep_pen": rep_pen,"ban_eos_token":ban_eos_token, "stop_sequence":stops}
                    respjson = make_url_request(f'{epurl}/api/v1/generate', gen_payload)
                    reply = html.escape(respjson["results"][0]["text"])
                    if chatmode:
                        reply = " "+reply.strip()
                status = "Generation Completed"

            if "generate" in parsed_dict:
                del parsed_dict["generate"]
            if "chatmsg" in parsed_dict:
                del parsed_dict["chatmsg"]
            if "imgprompt" in parsed_dict:
                del parsed_dict["imgprompt"]
            parsed_dict["prompt"] = prompt + reply
            parsed_dict["status"] = status
            parsed_dict["chatmode"] = ("1" if chatmode else "0")
            parsed_dict["imgmode"] = ("1" if imgmode else "0")
            updated_query_string = urllib.parse.urlencode(parsed_dict, doseq=True)
            updated_path = parsed_url._replace(query=updated_query_string).geturl()
            self.path = updated_path
            time.sleep(0.5) #short delay
            self.send_response(302)
            self.send_header("location", self.path)
            self.end_headers(content_type='text/html')
            return

        imgbtn = '''<form action="/noscript" style="display: inline;">
        <input type="hidden" name="imgmode" value="1">
        <input type="submit" value="Image Mode">
        </form>'''

        bodycontent = f'''<b><u>{"Image Mode" if imgmode else ("Chat Mode" if chatmode else "Story Mode")}</u></b><br>'''
        optionscontent = ""
        if imgmode:
            randimg = f'<img src="view_image{random.randint(100, 999)}.png" width="320" width="320">'
            bodycontent += f'''<p>Generated Image: {prompt if prompt else "None"}</p>
            {randimg if prompt else ""}<br>
            <label>Image Prompt: </label><input type="text" size="40" value="" name="imgprompt">
            <input type="hidden" name="generate" value="Generate" />
            <input type="submit" value="Generate"> (Be patient)'''
        elif chatmode:
            oldconvo = prompt.strip().replace(f"{human_name}:",f"<b>{human_name}:</b>").replace(f"{bot_name}:",f"<b>{bot_name}:</b>").replace("\n","<br>")
            oldconvo += f'''<input type="hidden" name="human_name" value="{human_name}"><input type="hidden" name="bot_name" value="{bot_name}">'''
            newconvo = '''Start a new conversation.<br>
            <label>Your Name: </label> <input type="text" size="10" value="User" name="human_name"><br>
            <label>Bot Name: </label> <input type="text" size="10" value="Assistant" name="bot_name"><br>'''
            clnprompt = prompt.replace("\n","1HdNl1")
            bodycontent += f'''<p>{newconvo if prompt=="" else oldconvo}</p>
            <input type="hidden" name="prompt" value="{clnprompt}">
            <label>Say: </label><input type="text" size="40" value="" name="chatmsg">
            <input type="hidden" name="generate" value="Send" />
            <input type="submit" value="Send"> (Be patient)'''
        else:
            bodycontent += f'''
<textarea name="prompt" cols="60" rows="8" wrap="soft" placeholder="Enter Prompt Here">{prompt}</textarea><br>
<input type="hidden" name="generate" value="Generate" />
<input type="submit" value="Generate"> (Be patient)
'''
        if not imgmode:
            optionscontent = f'''<label>Gen. Amount</label> <input type="text" size="4" value="{max_length}" name="max_length"><br>
            <label>Temperature</label> <input type="text" size="4" value="{temperature}" name="temperature"><br>
            <label>Top-K</label> <input type="text" size="4" value="{top_k}" name="top_k"><br>
            <label>Top-P</label> <input type="text" size="4" value="{top_p}" name="top_p"><br>
            <label>Rep. Pen</label> <input type="text" size="4" value="{rep_pen}" name="rep_pen"><br>
            <label>Prevent EOS</label> <input type="checkbox" name="ban_eos_token" value="1" {"checked" if ban_eos_token else ""}><br>'''
        else:
            optionscontent = f'''<label>Steps</label> <input type="text" size="4" value="{steps}" name="steps"><br>
            <label>Cfg. Scale</label> <input type="text" size="4" value="{cfg}" name="cfg"><br>'''

        caps = get_capabilities()
        finalhtml = f'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KoboldCpp NoScript Mode</title></head><body>
<h2>KoboldCpp NoScript Mode</h2>
<div>
<p>KoboldCpp can be used without Javascript enabled, however this is not recommended.
<br>If you have Javascript, please use <a href="/">KoboldAI Lite WebUI</a> instead.</p><hr>
<form action="/noscript">
{bodycontent}
<hr>
<b>{status}</b><br>
<hr>
{optionscontent}
<input type="hidden" name="chatmode" value="{chatmode}">
<input type="hidden" name="imgmode" value="{imgmode}">
</form>
<hr>
<div style="display: inline-block;">
Change Mode<br>
<form action="/noscript" style="display: inline;">
<input type="submit" value="Story Mode">
</form>
<form action="/noscript" style="display: inline;">
<input type="hidden" name="chatmode" value="1">
<input type="submit" value="Chat Mode">
</form>
{imgbtn if ("txt2img" in caps and caps["txt2img"]) else ""}
</div>
</div>
</body></html>'''
        finalhtml = finalhtml.encode('utf-8')
        self.send_response(200)
        self.send_header('content-length', str(len(finalhtml)))
        self.end_headers(content_type='text/html')
        self.wfile.write(finalhtml)

    def do_GET(self):
        global embedded_kailite, embedded_kcpp_docs, embedded_kcpp_sdui, embedded_kailite_gz, embedded_kcpp_docs_gz, embedded_kcpp_sdui_gz, embedded_lcpp_ui_gz, embedded_musicui, embedded_musicui_gz
        global last_req_time, start_time, cached_chat_template, cached_sd_info, has_vision_support, has_audio_support, has_whisper, friendlymodelname
        global savedata_obj, has_multiplayer, multiplayer_turn_major, multiplayer_turn_minor, multiplayer_story_data_compressed, multiplayer_dataformat, multiplayer_lastactive, maxctx, maxhordelen, friendlymodelname, lastuploadedcomfyimg, lastgeneratedcomfyimg, KcppVersion, totalgens, preloaded_story, exitcounter, currentusergenkey, friendlysdmodelname, fullsdmodelpath, password, friendlyembeddingsmodelname, voicelist
        global autoswapmode, textName, sttName, ttsName, embedName, musicName, imageName, mmprojName

        clean_path = self.path.split("?")[0] #for cases where we do not want query params
        if clean_path=="/lcpp": #fix for svelte redirect issues, browser path needs to end with slash
            clean_path = "/lcpp/"
            self.send_response(302)
            self.send_header("location", clean_path)
            self.end_headers(content_type='text/html')
            return None

        clean_path = clean_path.rstrip('/')
        response_body = None
        content_type = 'application/json'
        content_encoding = None

        # Check if browser supports gzip
        accept_encoding = self.headers.get('Accept-Encoding', '')
        supports_gzip = 'gzip' in accept_encoding.lower()

        if clean_path!="/lcpp" and clean_path.startswith("/lcpp/"):
            clean_path = clean_path[5:] #adapt lcpp paths to the root

        if clean_path in [""]: # the root url is lite
            content_type = 'text/html'
            if supports_gzip and embedded_kailite_gz is not None:
                response_body = embedded_kailite_gz
                content_encoding = 'gzip'
            elif embedded_kailite is not None:
                response_body = embedded_kailite
            else:
                response_body = (f"Embedded KoboldAI Lite is not found.<br>You will have to connect via the main KoboldAI client, or <a href='https://lite.koboldai.net?local=1&port={self.port}'>use this URL</a> to connect.").encode()


        elif clean_path in ["/noscript","noscript"]: #noscript webui
            self.noscript_webui()
            return

        elif clean_path.endswith(('/manifest.json')):
            response_body = (json.dumps({"name":"KoboldAI Lite","short_name":"KoboldAI Lite","description":"Progressive Web App for KoboldAI Lite","start_url":"./","scope":".","display":"standalone","background_color":"#303030","theme_color":"#337ab7","orientation":"portrait-primary","icons":[{"src":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAMAAAAL34HQAAAAAXNSR0IB2cksfwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAJZQTFRFAAAA+3F0nlRTBAMD+9Oq9HR2DwoKHBIT+Pj3s1dY5WttlUtL8MOhMhsaSygngENC8YB+ZjY1JyEmOTI3AAAA0l5gzUlKTENIdG3SAgEBAQEAAgIBAQAAAQEBraup8KCWY1tarZqQ3tvdamOriYaG3nR0kGRf1ayUdWxto3909WRovby9x673UEp1x4R9lIfPs57jv6znVipSqwAAADJ0Uk5TAP///f////7///////7+/v/+//8G/////35O1yCv///////////+///////////////GhbwlAAAU9klEQVR4nO2ca3ejug6GT6Dg4R5wCW1zgzTckkna/v8/d17JQEhCLjNN96fRWjO7003MY1mWJdnO//73T/7JP/knD5Xn2VzX9enr7PnKQ2/zqa7PZ/8V0+x1Ti8krvkVrhk/NJ3O5/PXn2Z7nr29KiYv9IWuX37h7HVq+qGu8F/frqn1+1Cvrxg9M/KE7juOb+rTC+97Zqog1IXve3hs/np9wL8F9fYKRZnCD4LQ80Jr4pO6ht4GKh3gofDCQgt8YdKAv739xFg2UJ4fBlq913wvmEzEBbMnKs8JPL/Y15oT+h6B0bMPp3ruNOU4jpXnhfScWBOD7yIq0wJVneeW4wSQUJCRPZwLtgJNJSGoNIiV74PIn8TBIBeeFdrEk8U+t+hpkIVsY/qDJyVRiaij0hyMTiLCzSTwdHIBx4JnAzvwgv1+r6nHgzAMPTWQj7T8N+jK9EIeQX6PZu2LyIs3EzU6JyJCOyfDygut0VZAhm/Cp0zfHoml64kQxEVkPIxW4QvLJi78DxYfAwXBv8PYiEVS75shBJYDZYnQgm5fH4plLtKo4eKBtCwtFI69wTiGpEPIZDJhhDC0RrYDLEVFH4DJQ9nx5PFY5bKgDve4QjOwR5tBGdmBGYC80NqJaHrow+Thg2hmy6Xl66YM6EXggrYIyzBGQ2IHgrD4WTgwXY80wyCsRypLYY3zOFCGryz/CpYBrFCZlcM+K7RyKPYHsOrl2DBszRfCbyw/NB3COhfC0kQCkyIouHhfi/HLTez8DNZolMN5i8j3Q6B5whoeQWAZsQc3B0E3vMDK6cENed+fwSKFof/Ck74v/PwyluGLKCJv4YWa3ajw8VhTsyrH3PjIyAFG7/MDYzTMRc8FHju0UMsZ6Wew5nqqsLjjhu1gFGHFlyyehlvDEDo2f+LHsKaMNVLa4XfkeT4eqXcOYtEDRsvNfz3atp5BpQf52Oi9daT0dkXUiBvGeLl0WZbj4YDjb6koqIHjGfMrIOOG5zqWeoKZnlhct0wfx0VUWGUVUdPve7HGywap4dpjAbuRx90pFGp5xf7QaWq/Gc9bWMa4TwUstywuJgB/RgXfENVHrTNXO0hXNWaMTz4GqZEwfZtrhkjLO6N6emrN/zrWeHnaHUjGXN+iIrvysnMqGsbblnWOpQzsu1ygMgepnp6Wd2ENdYi5vjMfSVfndtVgjf8Si7hq+R0ueNGoGKZ6gnO8STWMRWC1/0c50OytL1eoSF2jmx5iPPxROL4CUf3Ru64VT56p8tHIdAovWnUOuv3hMVjwX0JvylFKrtSknmf8oGlSokl/grJpJMuyksjcg6W53cp9F1bXP/4ZTTrqTUquFgJAZfoBJVYIios0DPdP6FbCqR/Fc7Vb+n72V1hlVaOL+yJNZJLgZ1ofQ6coCpXHhebllIhcJ6isOI4ncV6S1In00BeT9Wx6abb2/EZhf4Ll7kNZlzWFtaYpTETdYVDzC5Y23hVPHORGl6xrNseS7Fg2clKb1lg3S6VK5c02iU/WacvF6/WdWGUo0rriOpfOI0a1n2JPVrGktDJ2sFLOh9XFpbJAi8l/20RVJ2hHyCRNoXqhWpNVEvn1Aes6l8KieCYp0kj1MIJQYybSbuKiYDe3Qqhr2GMgMDYDzYo5tFyCysPoRcl6QbJOo4YrTQTb1/1YTJUIUjraS6uq7SUGVq1iRuwgmRxUFy00vkZYI8bKqHciBVKWrSGpUCMALi9gD3EnFvoXZXgr9IT21pjUqpcAI7fIjoYqUuagujCGwrGANWJtZZIYSFcYNtHYPE9l4ipbrDtsK5NRRbqKMIxmJGkmkvobfdHiymE+1u+hIj7GMCqgLZvetdyn0HgSRRJIrcE3YiapCN27sdzUS1JqQ5AZ0LTGZJQVuNgm9ktOxO2Aal8D+cfrFBagaXFOMRQtDrq5TqITJDUfZSIQVtyJVfsyPZ7PbPipsgqRqADciMPBIhN5h6A1rXEdUhupNJvJo7P6o6byF6WiKu+0LTeVUg71Lam4dVHnRDWyHS4VnloXRVYOjyE1FzCA0hUmjwTUbrerdolo1CWzO2diBiqzUXKa7iRMPeJ5KCQ3b/qxwc3EtP0wP8UiV6phDG3yI7HfKRyealFF0e795eXl4/Njp7hSsb4TK41IWeQadh9fX2hAkk+Wh4qrqXHZZ2THVB4+MfpnBMeBFiCBz3M7P9RphcRcTEQCqJf394+vD27QTEVaLu8IA59cP6FPmET1+fn79+/P90hki3Vy4Apj1dBmEtIKdKQvxFZwWoEfhj6km3xmgrlcRZJ09bKLkq/fHzwknkTecU90WotEsnmihXf5xVwmnGGWduMhgngMMTYbhxbs2dkY1oUn2jhDUZE3bZX1Ik2Z/f69pjEx06i4R1tu4El2yzvqlViTvj4iGoEsbScn/AXXqYOJE9GGQh8Ly2Gxz6nYd+SheOWR0Y7HMNKjCr2tIp6kxf52kjF2fZ7NJkzzPYGd8jDCMBdH+uJ3+U5MozifH2PBWMaGbQUHa9cVVdaMIbBMjOLvj5RdfRHfTvbHpVdF3RBCaR88iqLidpO+I/Mx22KEhq+vp1iuixGmmlozipFapSuBvn58fABLj6jdNX4Q1T6/neyP3YqwWNs0h82PLx5F1d+D3ZteaMHTxwGwjtZrZBPkIcEFMNo2wtzhPm1h8buXj9Vq9blDmPOu1GXq63LMWNez6icXKjElfMvLjjr7zqOYyPV2uyUu7r/wfK5jsoc4NnmqyiQcZoyMcY6h9LCg4rPb1aoij/MLspBQEtr9omU2ze7D8llZn1+fO9FifaVy8evXyt1u15LsPYTnGimqs4iealjS5fCHKnl7KGr1iyWFaX3iv6sFjDahdrdQl6zvwYKDQCjEvVpVHVZFWPSrbVZVNHXo2RgdOPfyMwQ2cACuSkuNZbnY/jpgfXx+QWsmz6UVAPGTcxdWgMhdKXtFJslYK4zBr4YLlr/nAM/i7cbzQPCZudal4hofY8E4vthmI8L6tcKsD+07TL7EvKZerVYLDm/f+eOdAKvcU6pia545nb8N5Yq0G4+Qq+TFbryvj7FefvexECxhzbjDQUSMBW2/cxhxggW9l3mzHEJVl08J6F5B/gtKTfu2BbejtCVVu1tp+tYdWHtyceS0Pl6SIaztOsuNTYw1eHqJqjF8rCrE5UTrPtb7+yeFbWTy/OtURBpee8O2DAtLC3189/GxI/9EttXHWic1sMgxXC8QIkYNuJTseGmDRX6rxWIHodQlgttYdkA73Nyrj7TF6qlrVXkFYYnzKXiGlZLNjx0RfTU9Yif9/kKJnuCp9PvXCrB3YMWhXiU7hbUzG6we1yLxNGO0QU59EwuDSG9zTNGM4lZyDPie8uIDL//58S63iyQwjFslrtgX6zThT7+rxQcfxkKWNWRrz7dGjDW9CwsRv2kmzYcbLMSVHAKg4Z3Yrmrt1jYGaStdpFLFHztKx7C00vJYqVm+ShGo58Dyb2IhdcPKaFG6KRoXofpLXPL98xNU7xLEJe183cCyrWxRSRWtvSdN6I1QpBkHrGaY0PlmchNrrstsObbo5ACWY/XpLGma21EowR1f//rlGrexDKxhaxm9v3QfV3rbtpMJ1oqI5h6sZL8cI3jWHKHLVtdqGMlC3mm2k7J+LUeGcTPiGiMESSOlrlbeo2aOrxLT9HzPn9yLxQkSls5afX4tlX3whMQQygxtLlkdt9RVLrDCR7s+1U4s2imu+zWiFcsK78DKEElR+u0g5MraBqK01dcOoTh+1Wz7dFZ/2D4/HsVydcwFE6saZSGqCegjd9kWtGVvNgbNRUp8mqUrgVtkkUjR0CxvxbY7hobNEsf0t9psbTHzcstcnXlK2VKlCC4C8kV3Yfm1HU8myHEF1bcSZfarLdVtIBGiw9Vqnaax2sK0LTq84vclDMPAQXrOfOOiyjh8QBYLiaKk9YaVYHu378WyYmsyAZXkqD5K1xnC1NV2XaVJktK/KClO8xFiby1oT7L1SsZU6PNAR1XYzbjwqSPbNZUU8afiAID+HQEK/XHiOL7tTue6pxFW4PuFsadyBRVFkrSq67qq0opKeZHwgn0eW07IaaVoTiOxqLNJgtNNzw8dqw4FlY0W2XpdrdEnzi2gepMOKCGg8amufZeXj6kkHThIuOBWvX42ZypVIPi2+f8QE96tjpqRFEVBR5BCUiIVxujoC37Egi2VZUZRU5tC12rDsEKfyt/3YKXxZjKxoLERnxHwOiYlPnLz3OIqheklaaFdECdNVL3CD6Aws9exJi/kGgyV5e/DCgiLZYPVI6CqDw0VOhwGRY3U27aolG1GSeBcYtLaU25ScK1b46M2B61HYUFlLZpcbMd3BDZhHCsqmkyx09V8hE9nZQysvzQTyKSvQ7WnAj06xTSJY0sL2+mh8tXRRr2IC1vXj0fzacNGWRNKItCaow4gYWIRJx3yA+E9TC2YLwQdtIFDx9SLInIi3MNmVLgOeJnrefZGhRvLiLtBVP678ZLkHbXAA1R4N1QHBk8Aj2LRuTzL4g4a3XswQS5uRTVng8kUW3VtNt36YvAOT65FYgDKGpTeA0XgYebl/RNe1G47KhOH0/xBdVHaowszhPsebVptbbr1mLKhmJJRGZzwqDWn1adSKi9FfTQMPbh6mSWtby0V5lgA83qbDQwkUemOz1g0R2yKIiZxt+jyKml6QZ8pjg8Hb46XaKxLEPjc7vHU95w2QOPxizuszWZDZyyn0/P9ApgVHHzut5XMUaO0uNMXsjkv6caPiqx5t1afHsI7xBQNGf7CnLTaQ3vsGajbBIUPsLoG7gXQbrA/sQ3f7x2s467wGyiXcMKwdZ4wWfskirkU13dgZPp5c/5L8YzU6VC0PjZIXULQ3vCRwmjfVYP75FSklbivLcPC0nKAGmQ4sPT/ATtrwLpDe5u2w+oJWMImxFwS59VAnzQce31twWzjTfdZdZqboY4Q7CE5wjaMhstqueBIDy2rblieFyLEO62dBlQJ1Ly+bZHHOmna4qnQqKMJ/njOdRKztHStnSmPpcXtL+yTvmGqh54PA5tOj7FSiuED0cfqf05ZSBPcNcxx3zedCXvNvPVRtrL8gxp7r1HxLNIt/7QAPqX8cDn2zWDYbIxY61PdZOrYYrsda24hHmpeYdl0z+Vkd2U+FZW7HHumMzyjqLOW1Q7KvVCss/zoM0NVMYU1ov2vE1fPRyWXe0FVqwvKsuLWYO+F4mMO3cAZNk2armvnWJZ/vhVF6bRbmGE8hEWW8edUnHtgAeVxNBoXZg28QP1mow1kGq80iurw9ABWjmWmXWfUlHS6g/MXxKFoGQG2w1unzXpASdsFrM1gmQujmJTCtAbrCv1FpfFBIWUT8nIsSLccENJytGF1VYGhFLfFmoQDh6dmtIOuC2u4fNxrSs10WuEomO4t3ScjSGmgp8INK+43Ndy6wVWu85iLdoX15Pr+Eh9nbhy2E/q06+8Pc1EgLHzZUA8dYO97bcZCzDU/o+IQQucthaMY4Ky1ztxDjzadeSt5AMtDaoGQ0TnFOlQIjpVlTyiiP8d6fnvV9WzZj5+GogK7w0JkR7mgP4hV0GaWaFVp2VfbJGGDHzzN8gYsd3mEda76AxamGiX33qDRO5TvHBKR4RWtx7ghZV04+6Pr63LZFYiGJ03fZwVkXRewfK8f8x9hdSYyHrdv2Uyci3dt4CMkbfn05ZTLOPKkBdzEsG1Bl/182+rHC82yOh4v23cZV5TFB7hq92nZl1Mu49618FiGsPpvsR39yik8qCsjrPG+xnIWpLQtZfwMlpEHVCyo93s6imddThRZXWYkm1IQSXXG9QAsRcXJHdXxSKisffkkpbou3Qn8fuWenMb9S6z4dH7bdEjxznOnsPpXugbdCE2BurtK0Djjv8fqT25F1b7ola5eXy3ZzN7eDikR1iOytYPL+A5WP4s0EFr1LmU8z2Z/cu6a8n+ZHU3Hv8Zqo22uQtOJp29cyaA54JfL3iH0B2DRToR56ZTpH3BlT4f5+C3bitXhUUX1ndsYz90xhNbC/h4LZu7RhbLvU7VcVV1oew6a/w6LSn88+Twnt5jqu1eQ+EsBIhl5YZH/LRalFUxFX4wQPoLqf009TqfqPhVcTrDUhdtOgsMN514SAh+fI9KJ1Hm8q0cL/pDLjCJwUfGkufvbiN/sC7Sle+F5fkPX7ADR/g8F6jAFN5MP/UIBii0SydWcWPOF2b8VcPSPA1533JiKzPlE6HQnwBGPGcBGnunaZJZ6voZ5RCfxRbcvgemQsZR0MyKri7SpvUdJVVUJ1/PDQIiifHpyA/HgO/tzs9qWqaCv8qBNMxY6jJfSHQ0WdRPLdcuyoKwn5W2nNagoUvBqvtXx8JvCwFo9ZSneIfkg5GKx3WZCF/XQxZQ6MelUIm1w6qKqpJc0FxK0x1/JXaPlskjSdSLpHNwK7xRYmQZvzNQ+sOiQ1tb0yrIumvtebv14LNXhvftUeOmCz7lsTb0YxnILYNF+5gLZweFCjbuPHn85vm29EFKdodpy6D8omeBt1tVaT34Ya9GNkGxOfSzAegnL4+Nfq1Sv+ljlD2K5qeDzFVCFNzyGMMJE0PGerTzS549iPdURHyfZpmYwqCzMhm1hkka3QvTBXTe6lAz+NVa56lSh1LWQglRR1uRQy7JxWvCo6/V2tTCTLfRppsc3hf3HfpcTsPCWtvXaoyHKoAoXVJJvo6RKkoQOO1TbVYRHFkmUPR1hye9FpQNYi9VBXYUpF9vK9F11AbKfWNENQiRy20okGf4cX2AupflgLORm2w7M9c0klV7huntJ3xc1nx5kTrulmKsSa448UhYUO3yX5++xKK/lwzXNG3zEBrIkKqqVPfdFXU5drOkOVHMhkbuSJfqjvyaMj6SaQq6/yBspsxfJUznc/WcsCnKxlknZ0xTdGXrMvfgjLiTcdPahgrEzV11nZXIh1KQEIKG1uoWqFNTg6eBvCoPRbZ31glwT3e+7lMJwYpJirWY7rwFFO6s/9c1g6hvUWGWLLa3HV84hz2FZ25WbZQXf+LpwjvpB8tx8tRsC86ySp3cBzrjwDB0doun50986NyN30HioaylMV5T6D5hInp9ns7fpLarmBMOczPwnvwbvjOzWt+7N3t5eZ7P/Dqp9662hAdd/QvJP/skfyP8BnWh46M1E/qoAAAAASUVORK5CYII=","type":"image/png","sizes":"150x150"}]}).encode())

        elif clean_path.endswith(('/api/v1/model', '/api/latest/model')):
            auth_ok = self.check_header_password(password, args.adminpassword)
            modelNameToReturn = friendlymodelname
            if autoswapmode and textName is not None:
                modelNameToReturn = textName
            response_body = (json.dumps({'result': (modelNameToReturn if auth_ok else "koboldcpp/protected-model") }).encode())

        elif clean_path.endswith(('/api/v1/config/max_length', '/api/latest/config/max_length')):
            response_body = (json.dumps({"value": maxhordelen}).encode())

        elif clean_path.endswith(('/api/v1/config/max_context_length', '/api/latest/config/max_context_length')):
            response_body = (json.dumps({"value": min(maxctx,(maxctx if maxhordectx==0 else maxhordectx))}).encode())

        elif clean_path.endswith(('/api/v1/config/soft_prompt', '/api/latest/config/soft_prompt')):
            response_body = (json.dumps({"value":""}).encode())

        elif clean_path.endswith(('/api/v1/config/soft_prompts_list', '/api/latest/config/soft_prompts_list')):
            response_body = (json.dumps({"values": []}).encode())

        elif clean_path.endswith(('/api/v1/info/version', '/api/latest/info/version')):
            response_body = (json.dumps({"result":"1.2.5"}).encode())

        elif clean_path.endswith(('/api/extra/true_max_context_length')): #do not advertise this to horde
            response_body = (json.dumps({"value": maxctx}).encode())

        elif clean_path.endswith(('/api/extra/version')):
            caps = get_capabilities()
            response_body = (json.dumps(caps).encode())

        elif clean_path.endswith(('/api/admin/list_options')):  # used by admin to get info about a kcpp instance
            opts = []
            if args.admin and args.admindir and os.path.exists(args.admindir) and self.check_header_password(args.adminpassword):
                opts = get_current_admindir_list()
            response_body = (json.dumps(opts).encode())

        elif clean_path.endswith(('/api/extra/perf')):
            lastp = handle.get_last_process_time()
            laste = handle.get_last_eval_time()
            lastc = handle.get_last_token_count()
            lastic = handle.get_last_input_count()
            totalgens = handle.get_total_gens()
            totalimggens = handle.get_total_img_gens()
            totalttsgens = handle.get_total_tts_gens()
            totaltranscribegens = handle.get_total_transcribe_gens()
            stopreason = handle.get_last_stop_reason()
            lastseed = handle.get_last_seed()
            lastdraftsuccess = handle.get_last_draft_success()
            lastdraftfailed = handle.get_last_draft_failed()
            t_pp = float(lastp)*float(lastic)*0.001
            t_gen = float(laste)*float(lastc)*0.001
            s_pp = float(lastic)/t_pp if t_pp>0 else 0
            s_gen = float(lastc)/t_gen if t_gen>0 else 0
            uptime = time.time() - start_time
            idletime = time.time() - last_req_time
            is_quiet = True if (args.quiet and args.debugmode != 1) else False
            response_body = json.dumps(
                {
                    "last_process": lastp,
                    "last_eval": laste,
                    "last_token_count": lastc,
                    "last_input_count": lastic,
                    "last_process_time": t_pp,
                    "last_eval_time": t_gen,
                    "last_process_speed": s_pp,
                    "last_eval_speed": s_gen,
                    "last_seed": lastseed,
                    "last_draft_success": lastdraftsuccess,
                    "last_draft_failed": lastdraftfailed,
                    "total_gens": totalgens,
                    "stop_reason": stopreason,
                    "total_img_gens": totalimggens,
                    "total_tts_gens": totalttsgens,
                    "total_transcribe_gens": totaltranscribegens,
                    "queue": requestsinqueue,
                    "idle": (0 if (modelbusy.locked() or batched_request_runner_count>0) else 1),
                    "hordeexitcounter": exitcounter,
                    "uptime": uptime,
                    "idletime": idletime,
                    "quiet": is_quiet,
                }
            ).encode()

        elif clean_path.endswith('/api/extra/generate/check'):
            if not self.secure_endpoint():
                return
            pendtxtStr = ""
            if requestsinqueue==0 and totalgens>0 and currentusergenkey=="":
                pendtxt = handle.get_pending_output()
                pendtxtStr = ctypes.string_at(pendtxt).decode("UTF-8","ignore")
            response_body = (json.dumps({"results": [{"text": pendtxtStr}]}).encode())

        elif clean_path.endswith('/api/extra/last_logprobs'):
            if not self.secure_endpoint():
                return
            logprobsdict = None
            if requestsinqueue==0 and totalgens>0 and currentusergenkey=="":
                lastlogprobs = handle.last_logprobs()
                logprobsdict = parse_last_logprobs(lastlogprobs)
            response_body = (json.dumps({"logprobs":logprobsdict}).encode())

        elif clean_path.endswith('/v1/models') or clean_path=='/models':
            modelNameToReturn = friendlymodelname
            if autoswapmode and textName is not None:
                modelNameToReturn = textName

            mlist = [{"id":modelNameToReturn,"object":"model","created":int(time.time()),"owned_by":"koboldcpp","permission":[],"root":"koboldcpp"}]
            if args.routermode:
                alist = get_current_admindir_list()
                for itm in alist:
                    mlist.append({"id":itm,"object":"model","created":int(time.time()),"owned_by":"koboldcpp","permission":[],"root":"koboldcpp"})
            response_body = (json.dumps({"object":"list","data":mlist}).encode())

        elif clean_path.endswith('/sdapi/v1/loras'):
            response_body = (json.dumps(mk_sdapi_lora_list(imglora_bypath))).encode()

        elif clean_path.endswith('/sdapi/v1/upscalers'):
            if args.sdupscaler:
                response_body = (json.dumps([{"name":"ESRGAN_4x","model_name":"ESRGAN_4x","model_path":"upscaler_model.gguf","model_url":None,"scale":4}]).encode())
            else:
                response_body = (json.dumps([]).encode())

        elif clean_path.endswith('/sdapi/v1/sd-models'):
            if autoswapmode and imageName is not None:
                response_body = (json.dumps([{"title":imageName,"model_name":imageName,"hash":"8888888888","sha256":"8888888888888888888888888888888888888888888888888888888888888888","filename":imageName,"config": None}]).encode())
            elif friendlysdmodelname=="inactive" or fullsdmodelpath=="":
                response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps([{"title":friendlysdmodelname,"model_name":friendlysdmodelname,"hash":"8888888888","sha256":"8888888888888888888888888888888888888888888888888888888888888888","filename":fullsdmodelpath,"config": None}]).encode())
        elif clean_path.endswith('/sdapi/v1/options'):
            modelNameToReturn = friendlysdmodelname
            if autoswapmode and imageName is not None:
                modelNameToReturn = imageName
            response_body = (json.dumps({"samples_format":"png","sd_model_checkpoint":modelNameToReturn}).encode())
        elif clean_path.endswith('/sdapi/v1/samplers'):
            if (friendlysdmodelname=="inactive" or fullsdmodelpath=="") and not(autoswapmode and imageName is not None):
                response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps(sd_sdapi_samplers()).encode())
        elif clean_path.endswith('/sdapi/v1/schedulers'):
            if (friendlysdmodelname=="inactive" or fullsdmodelpath=="") and not(autoswapmode and imageName is not None):
                response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps([{"name":name,"label":name} for name in cached_sd_info.get('available_schedulers', [])]).encode())
        elif clean_path.endswith('/sdapi/v1/latent-upscale-modes'):
           response_body = (json.dumps([]).encode())
        elif clean_path.endswith('/sdapi/v1/upscalers'):
           response_body = (json.dumps([]).encode())

        #vits compatible
        elif clean_path=='/voice/check':
            response_body = (json.dumps({"id":4,"lang":["en"],"name":"KoboldCppTTS","status":"success"}).encode())
        elif clean_path=='/voice/speakers':
            response_body = (json.dumps({"VITS":[{"id":4,"lang":["en"],"name":"KoboldCppTTS"}]}).encode())
        elif clean_path=='/voice/vits':
            parsed_url = urllib.parse.urlparse(self.path)
            parsed_dict = urllib.parse.parse_qs(parsed_url.query)
            prompt = str(parsed_dict['text'][0]) if 'text' in parsed_dict else ""
            if prompt:
                epurl = get_my_epurl()
                content_type = 'audio/wav'
                response_body = make_url_request(f'{epurl}/api/extra/tts', {"input": prompt})
            pass

        elif clean_path.endswith('/speakers_list') or clean_path.endswith('/api/extra/speakers_list'): #xtts compatible
            response_body = (json.dumps(voicelist).encode()) #some random voices for them to enjoy
        elif clean_path.endswith('/speakers'): #xtts compatible
            tmplist = []
            for itm in voicelist:
                tmplist.append({"name":itm,"voice_id":itm,"preview_url":""})
            response_body = (json.dumps(tmplist).encode()) #some random voices for them to enjoy
        elif clean_path.endswith('/v1/audio/voices') or clean_path=='/audio/voices':
            response_body = (json.dumps({"status":"ok","voices":voicelist}).encode()) #some random voices for them to enjoy
        elif clean_path.endswith('/get_tts_settings'): #xtts compatible
            response_body = (json.dumps({"temperature":0.75,"speed":1,"length_penalty":1,"repetition_penalty":1,"top_p":1,"top_k":4,"enable_text_splitting":True,"stream_chunk_size":100}).encode()) #some random voices for them to enjoy

        elif clean_path.endswith('/api/tags') or clean_path.endswith('/api/ps'): #ollama compatible
            modelNameToReturn = friendlymodelname
            if autoswapmode and textName is not None:
                modelNameToReturn = textName
            response_body = (json.dumps({"models":[{"name":"koboldcpp","model":f"{modelNameToReturn}:latest","modified_at":"2024-07-19T15:26:55.6122841+08:00","expires_at": "2055-06-04T19:06:25.5433636+08:00","size":394998579,"size_vram":394998579,"digest":"b5dc5e784f2a3ee1582373093acf69a2f4e2ac1710b253a001712b86a61f88bb","details":{"parent_model":"","format":"gguf","family":"koboldcpp","families":["koboldcpp"],"parameter_size":"128M","quantization_level":"Q4_0"}},{"name":"koboldcpp","model":modelNameToReturn,"modified_at":"2025-01-01T01:00:00.0000000+00:00","expires_at": "2069-01-01T01:00:00.0000000+00:00","size":394998579,"size_vram":394998579,"digest":"b5dc5e784f2a3ee1582373093acf69a2f4e2ac1710b253a001712b86a61f88bb","details":{"parent_model":"","format":"gguf","family":"koboldcpp","families":["koboldcpp"],"parameter_size":"128M","quantization_level":"Q4_0"}}]}).encode())
        elif clean_path.endswith('/api/version'): #ollama compatible, NOT the kcpp version
            response_body = (json.dumps({"version":"0.7.0"}).encode())
        elif clean_path=='/ping':
            response_body = (json.dumps({"status": "healthy"}).encode())

        #comfyui compatible
        elif clean_path=='/system_stats':
            response_body = (json.dumps({"system":{"os":"posix","ram_total":12345678900,"ram_free":12345678900,"comfyui_version":"v0.3.4-3-g7126ecf","python_version":"3.10.12","pytorch_version":"2.5.1","embedded_python":False,"argv":[]},"devices":[{"name":"koboldcpp","type":"cuda","index":0,"vram_total":12345678900,"vram_free":12345678900,"torch_vram_total":12345678900,"torch_vram_free":12345678900}]}).encode())
        elif clean_path=='/object_info':
            modelNameToReturn = friendlysdmodelname
            if autoswapmode and imageName is not None:
                modelNameToReturn = imageName
            response_body = (json.dumps({"KSampler":{"input":{"required":{"model":["MODEL",{"tooltip":""}],"seed":["INT",{"default":0,"min":0,"max":512,"tooltip":""}],"steps":["INT",{"default":20,"min":1,"max":512,"tooltip":""}],"cfg":["FLOAT",{"default":8.0,"min":0.0,"max":100.0,"step":0.1,"round":0.01,"tooltip":"512"}],"sampler_name":[["euler"],{"tooltip":""}],"scheduler":[["normal"],{"tooltip":""}],"positive":["CONDITIONING",{"tooltip":""}],"negative":["CONDITIONING",{"tooltip":""}],"latent_image":["LATENT",{"tooltip":""}],"denoise":["FLOAT",{"default":1.0,"min":0.0,"max":1.0,"step":0.01,"tooltip":""}]}},"input_order":{"required":["model","seed","steps","cfg","sampler_name","scheduler","positive","negative","latent_image","denoise"]},"output":["LATENT"],"output_is_list":[False],"output_name":["LATENT"],"name":"KSampler","display_name":"KSampler","description":"KSampler","python_module":"nodes","category":"sampling","output_node":False,"output_tooltips":[""]},"CheckpointLoaderSimple":{"input":{"required":{"ckpt_name":[[modelNameToReturn],{"tooltip":""}]}},"input_order":{"required":["ckpt_name"]},"output":["MODEL","CLIP","VAE"],"output_is_list":[False,False,False],"output_name":["MODEL","CLIP","VAE"],"name":"CheckpointLoaderSimple","display_name":"Load","description":"","python_module":"nodes","category":"loaders","output_node":False,"output_tooltips":["","",""]},"CLIPTextEncode":{"input":{"required":{"text":["STRING",{"multiline":True,"dynamicPrompts":True,"tooltip":""}],"clip":["CLIP",{"tooltip":""}]}},"input_order":{"required":["text","clip"]},"output":["CONDITIONING"],"output_is_list":[False],"output_name":["CONDITIONING"],"name":"CLIPTextEncode","display_name":"CLIP","description":"","python_module":"nodes","category":"conditioning","output_node":False,"output_tooltips":[""]},"CLIPSetLastLayer":{"input":{"required":{"clip":["CLIP"],"stop_at_clip_layer":["INT",{"default":-1,"min":-24,"max":-1,"step":1}]}},"input_order":{"required":["clip","stop_at_clip_layer"]},"output":["CLIP"],"output_is_list":[False],"output_name":["CLIP"],"name":"CLIPSetLastLayer","display_name":"CLIPSLL","description":"","python_module":"nodes","category":"conditioning","output_node":False},"VAEDecode":{"input":{"required":{"samples":["LATENT",{"tooltip":""}],"vae":["VAE",{"tooltip":""}]}},"input_order":{"required":["samples","vae"]},"output":["IMAGE"],"output_is_list":[False],"output_name":["IMAGE"],"name":"VAEDecode","display_name":"VAE","description":"","python_module":"nodes","category":"latent","output_node":False,"output_tooltips":[""]},"VAEEncode":{"input":{"required":{"pixels":["IMAGE"],"vae":["VAE"]}},"input_order":{"required":["pixels","vae"]},"output":["LATENT"],"output_is_list":[False],"output_name":["LATENT"],"name":"VAEEncode","display_name":"VAE","description":"","python_module":"nodes","category":"latent","output_node":False},"VAEEncodeForInpaint":{"input":{"required":{"pixels":["IMAGE"],"vae":["VAE"],"mask":["MASK"],"grow_mask_by":["INT",{"default":6,"min":0,"max":64,"step":1}]}},"input_order":{"required":["pixels","vae","mask","grow_mask_by"]},"output":["LATENT"],"output_is_list":[False],"output_name":["LATENT"],"name":"VAEEncodeForInpaint","display_name":"VAE","description":"","python_module":"nodes","category":"latent/inpaint","output_node":False},"VAELoader":{"input":{"required":{"vae_name":[["kcpp_vae"]]}},"input_order":{"required":["vae_name"]},"output":["VAE"],"output_is_list":[False],"output_name":["VAE"],"name":"VAELoader","display_name":"Load VAE","description":"","python_module":"nodes","category":"loaders","output_node":False},"EmptyLatentImage":{"input":{"required":{"width":["INT",{"default":512,"min":16,"max":16384,"step":8,"tooltip":""}],"height":["INT",{"default":512,"min":16,"max":16384,"step":8,"tooltip":""}],"batch_size":["INT",{"default":1,"min":1,"max":1,"tooltip":""}]}},"input_order":{"required":["width","height","batch_size"]},"output":["LATENT"],"output_is_list":[False],"output_name":["LATENT"],"name":"EmptyLatentImage","display_name":"Empty Latent Image","description":"","python_module":"nodes","category":"latent","output_node":False,"output_tooltips":[""]}}).encode())
        elif clean_path.endswith('/api/models/checkpoints') or clean_path.endswith('/models/checkpoints'): #emulate comfyui, duplication is redundant but added for clarity
            if autoswapmode and imageName is not None:
                response_body = (json.dumps([imageName]).encode())
            elif friendlysdmodelname=="inactive" or fullsdmodelpath=="":
                response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps([friendlysdmodelname]).encode())
        elif clean_path=='/api/models/loras' or clean_path=='/models/loras':
            response_body = (json.dumps([]).encode())
        elif clean_path=='/view' or clean_path=='/view.png' or clean_path=='/api/view' or clean_path.startswith('/view_image'): #emulate comfyui
            content_type = 'image/png'
            response_body = lastgeneratedcomfyimg
        elif clean_path=='/history' or clean_path=='/api/history' or clean_path.startswith('/api/history/') or clean_path.startswith('/history/'): #emulate comfyui
            modelNameToReturn = friendlysdmodelname
            if autoswapmode and imageName is not None:
                modelNameToReturn = imageName
            imgdone = (False if lastgeneratedcomfyimg==b'' else True)
            response_body = (json.dumps({"12345678-0000-0000-0000-000000000001":{"prompt":[0,"12345678-0000-0000-0000-000000000001",{"3":{"class_type":"KSampler","inputs":{"cfg":5.0,"denoise":1.0,"latent_image":["5",0],"model":["4",0],"negative":["7",0],"positive":["6",0],"sampler_name":"euler","scheduler":"normal","seed":1,"steps":20}},"4":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":modelNameToReturn}},"5":{"class_type":"EmptyLatentImage","inputs":{"batch_size":1,"height":512,"width":512}},"6":{"class_type":"CLIPTextEncode","inputs":{"clip":["4",1],"text":"prompt"}},"7":{"class_type":"CLIPTextEncode","inputs":{"clip":["4",1],"text":""}},"8":{"class_type":"VAEDecode","inputs":{"samples":["3",0],"vae":["4",2]}},"9":{"class_type":"SaveImage","inputs":{"filename_prefix":"kliteimg","images":["8",0]}}},{},["9"]],"outputs":{"9":{"images":[{"filename":"kliteimg_00001_.png","subfolder":"","type":"output"}]}},"status":{"status_str":"success","completed":imgdone,"messages":[["execution_start",{"prompt_id":"12345678-0000-0000-0000-000000000001","timestamp":1}],["execution_cached",{"nodes":[],"prompt_id":"12345678-0000-0000-0000-000000000001","timestamp":1}],["execution_success",{"prompt_id":"12345678-0000-0000-0000-000000000001","timestamp":1}]]},"meta":{"9":{"node_id":"9","display_node":"9","parent_node":None,"real_node_id":"9"}}}}).encode())
        elif clean_path=='/ws' and ('Upgrade' in self.headers and self.headers['Upgrade'].lower() == 'websocket' and
            'Sec-WebSocket-Key' in self.headers):
            ws_key = self.headers['Sec-WebSocket-Key']
            ws_accept = base64.b64encode(hashlib.sha1((ws_key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
            self.protocol_version = "HTTP/1.1"
            self.send_response(101) #fake websocket response, Switching Protocols
            self.send_header('Upgrade', 'websocket')
            self.send_header('Connection', 'Upgrade')
            self.send_header('Sec-WebSocket-Accept', ws_accept)
            self.end_headers()
            try:
                # Send a dummy WebSocket text frame: empty string
                payload = json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}}, "sid": "ffff000012345678ffff000012345678"}}).encode("utf-8")
                header = struct.pack("!BB", 0x81, len(payload))  # FIN + text frame, no mask
                self.connection.sendall(header + payload)
                time.sleep(0.1) #short delay before replying
                # Send close frame with status code 1000 (Normal Closure)
                close_payload = struct.pack("!H", 1000)
                close_frame = struct.pack("!BB", 0x88, len(close_payload)) + close_payload
                self.connection.sendall(close_frame)
                time.sleep(0.1) #short delay before replying
            except Exception as e:
                print(f"WebSocket send error: {e}")
            self.connection.close()
            return
        elif clean_path.endswith(('/.well-known/serviceinfo')):
            response_body = (json.dumps({"version":"0.2","software":{"name":"KoboldCpp","version":KcppVersion,"repository":"https://github.com/LostRuins/koboldcpp","homepage":"https://github.com/LostRuins/koboldcpp","logo":"https://raw.githubusercontent.com/LostRuins/koboldcpp/refs/heads/concedo/niko.ico"},"api":{"koboldai":{"name":"KoboldAI API","rel_url":"/api","documentation":"https://lite.koboldai.net/koboldcpp_api","version":KcppVersion},"openai":{"name":"OpenAI API","rel_url ":"/v1","documentation":"https://openai.com/documentation/api","version":KcppVersion}}}).encode())

        elif clean_path=="/props":
            modelNameToReturn = friendlymodelname
            if autoswapmode and textName is not None:
                modelNameToReturn = textName
            mmprojOverride = False
            if autoswapmode and mmprojName is not None:
                mmprojOverride = True
            response_body = (json.dumps({
                "chat_template": cached_chat_template,
                "id": 0,
		        "id_task": -1,
                "total_slots": 1,
                "modalities": {
                    "vision": mmprojOverride or has_vision_support,
                    "audio": has_audio_support
                },
                "model_path": modelNameToReturn,
                "n_ctx": maxctx,
                "default_generation_settings": {
                    "n_ctx": maxctx,
                },
            }).encode())

        elif clean_path=="/slots":
            self.send_response(501)
            self.end_headers(content_type='application/json')
            self.wfile.write(json.dumps({"error":{"code":501,"message":"This server does not support slots endpoint.","type":"not_supported_error"}}).encode())
            return

        elif clean_path=="/api" or clean_path=="/docs":
            content_type = 'text/html'
            if supports_gzip and embedded_kcpp_docs_gz is not None:
                response_body = embedded_kcpp_docs_gz
                content_encoding = 'gzip'
            elif embedded_kcpp_docs is not None:
                response_body = embedded_kcpp_docs
            else:
                response_body = ("KoboldCpp API is running!\n\nAPI usage reference can be found at the wiki: https://github.com/LostRuins/koboldcpp/wiki").encode()

        elif clean_path=="/lcpp":
            content_type = 'text/html'
            if supports_gzip and embedded_lcpp_ui_gz is not None:
                response_body = embedded_lcpp_ui_gz
                content_encoding = 'gzip'
            else:
                response_body = ("Llama.cpp UI is not available. Please use the KoboldAI Lite UI instead.").encode()

        elif clean_path.startswith(("/sdui")):
            content_type = 'text/html'
            if supports_gzip and embedded_kcpp_sdui_gz is not None:
                response_body = embedded_kcpp_sdui_gz
                content_encoding = 'gzip'
            elif embedded_kcpp_sdui is not None:
                response_body = embedded_kcpp_sdui
            else:
                response_body = ("KoboldCpp API is running, but KCPP SDUI is not loaded").encode()

        elif clean_path.startswith(("/musicui")):
            content_type = 'text/html'
            if supports_gzip and embedded_musicui_gz is not None:
                response_body = embedded_musicui_gz
                content_encoding = 'gzip'
            elif embedded_musicui is not None:
                response_body = embedded_musicui
            else:
                response_body = ("KoboldCpp API is running, but KCPP MusicUI is not loaded").encode()

        elif clean_path=="/v1":
            content_type = 'text/html'
            response_body = ("KoboldCpp OpenAI compatible endpoint is running!<br>For usage reference, see <a href='https://platform.openai.com/docs/api-reference'>https://platform.openai.com/docs/api-reference</a><br>For other endpoints, see <a href='/api'>KoboldCpp API Documentation</a>").encode()

        elif clean_path=="/api/extra/preloadstory":
            if preloaded_story is None:
                response_body = (json.dumps({}).encode())
            else:
                response_body = preloaded_story
        elif clean_path.endswith(('/api')) or clean_path.endswith(('/api/v1')):
            self.send_response(302)
            self.send_header("location", "/api")
            self.end_headers(content_type='text/html')
            return None

        if response_body is None:
            self.send_response(404)
            self.end_headers(content_type='text/html')
            rp = f"Error: KoboldCpp HTTP Server is running, but this endpoint does not exist. Please check the URL and METHOD.<br><a href=\"/api\">[Read API documentation here]</a><br><br>Current path: {self.path}"
            self.wfile.write(rp.encode())
        else:
            self.send_response(200)
            self.send_header('content-length', str(len(response_body)))
            if content_encoding:
                self.send_header('Content-Encoding', content_encoding)
            self.end_headers(content_type=content_type)
            self.wfile.write(response_body)
        return

    def do_POST(self):
        global thinkformats
        global modelbusy, batched_request_runner_count, requestsinqueue, currentusergenkey, totalgens, pendingabortkey, lastuploadedcomfyimg, lastgeneratedcomfyimg, multiplayer_turn_major, multiplayer_turn_minor, multiplayer_story_data_compressed, multiplayer_dataformat, multiplayer_lastactive, net_save_slots, has_vision_support, savestate_limit, mcp_lock
        global autoswapmode, textName, sttName, ttsName, embedName, musicName, imageName, mmprojName
        contlenstr = self.headers['content-length']
        content_length = 0
        body = None
        if contlenstr:
            content_length = int(contlenstr)
            max_pl = int(args.maxrequestsize) if args.maxrequestsize else 32
            if content_length > (1024*1024*max_pl): #payload size limit
                self.send_response(500)
                self.end_headers(content_type='application/json')
                self.wfile.write(json.dumps({"detail": {
                "msg": f"Payload is too big. Max payload size is {max_pl}MB.",
                "type": "bad_input",
                }}).encode())
                return
            body = self.rfile.read(content_length)
        elif self.headers.get('transfer-encoding', '').lower()=="chunked":
            content_length = 0
            chunklimit = 0  # do not process more than 512 chunks, prevents bad actors
            body = b''
            try:
                while True:
                    chunklimit += 1
                    line = self.rfile.readline().strip()
                    if line:
                        chunk_length = max(0,int(line, 16))
                        content_length += chunk_length
                    if not line or chunklimit > 512 or content_length > (1024*1024*48): #48mb payload limit
                        self.send_response(500)
                        self.end_headers(content_type='application/json')
                        self.wfile.write(json.dumps({"detail": {
                        "msg": "Payload is too big. Max payload size is 48MB.",
                        "type": "bad_input",
                        }}).encode())
                        return
                    if chunk_length != 0:
                        chunk = self.rfile.read(chunk_length)
                        body += chunk
                    self.rfile.readline()
                    if chunk_length == 0:
                        break
            except Exception:
                self.send_response(500)
                self.end_headers(content_type='application/json')
                self.wfile.write(json.dumps({"detail": {
                "msg": "Failed to parse chunked request.",
                "type": "bad_input",
                }}).encode())
                return

        self.path = self.path.rstrip('/')
        response_body = None
        response_code = 200

        if self.path.endswith('/api/extra/tokencount') or self.path.endswith('/api/extra/tokenize'):
            if not self.secure_endpoint():
                return
            try:
                genparams = json.loads(body)
                countprompt = genparams.get('prompt', "")
                tcaddspecial = genparams.get('special', True)
                msgs = genparams.get('messages',[])
                if msgs and len(msgs) > 0 and not countprompt:
                    transform_genparams(genparams,4,args.jinja)
                    countprompt = genparams.get('prompt', "")
                countdata = tokenize_ids(countprompt,tcaddspecial)
                response_body = (json.dumps({"value": len(countdata),"ids": countdata, "prompt":countprompt}).encode())

            except Exception as e:
                utfprint("Count Tokens - Body Error: " + str(e))
                response_code = 400
                response_body = (json.dumps({"value": -1}).encode())

        elif self.path.endswith('/api/extra/detokenize'):
            if not self.secure_endpoint():
                return
            try:
                genparams = json.loads(body)
                tokids = genparams.get('ids', [])
                addspecial = genparams.get('special', True)
                detokstr = detokenize_ids(tokids,addspecial)
                response_body = (json.dumps({"result": detokstr,"success":True}).encode())
            except Exception as e:
                utfprint("Detokenize Error: " + str(e))
                response_code = 400
                response_body = (json.dumps({"result": "","success":False}).encode())

        elif self.path.endswith('/api/extra/json_to_grammar'):
            if not self.secure_endpoint():
                return
            try:
                genparams = json.loads(body)
                schema = genparams.get('schema', None)
                if not schema:
                    schema = genparams
                decoded = convert_json_to_gbnf(schema)
                response_body = (json.dumps({"result": decoded,"success":(True if decoded else False)}).encode())
            except Exception as e:
                utfprint("JSON to Grammar Error: " + str(e))
                response_code = 400
                response_body = (json.dumps({"result": "","success":False}).encode())

        elif self.path.endswith('/api/extra/abort'):
            if not self.secure_endpoint():
                return
            multiuserkey = ""
            try:
                tempbody = json.loads(body)
                if isinstance(tempbody, dict):
                    multiuserkey = tempbody.get('genkey', "")
            except Exception:
                multiuserkey = ""
                pass
            if (multiuserkey=="" and requestsinqueue==0) or (multiuserkey!="" and multiuserkey==currentusergenkey):
                ag = handle.abort_generate()
                time.sleep(0.1) #short delay before replying
                response_body = (json.dumps({"success": ("true" if ag else "false"), "done":"true"}).encode())
                print("\nGeneration Aborted")
            elif (multiuserkey!="" and requestsinqueue>0):
                pendingabortkey = multiuserkey
                response_body = (json.dumps({"success": "true", "done":"false"}).encode())
            else:
                response_body = (json.dumps({"success": "false", "done":"false"}).encode())

        elif self.path.endswith('/api/extra/generate/check'):
            if not self.secure_endpoint():
                return
            pendtxtStr = ""
            multiuserkey = ""
            try:
                tempbody = json.loads(body)
                if isinstance(tempbody, dict):
                    multiuserkey = tempbody.get('genkey', "")
            except Exception:
                multiuserkey = ""

            if totalgens>0:
                if (multiuserkey=="" and multiuserkey==currentusergenkey and requestsinqueue==0) or (multiuserkey!="" and multiuserkey==currentusergenkey): #avoid leaking prompts in multiuser
                    pendtxt = handle.get_pending_output()
                    pendtxtStr = ctypes.string_at(pendtxt).decode("UTF-8","ignore")
            response_body = (json.dumps({"results": [{"text": pendtxtStr}]}).encode())

        elif self.path.endswith('/api/extra/last_logprobs'):
            if not self.secure_endpoint():
                return
            logprobsdict = None
            multiuserkey = ""
            try:
                tempbody = json.loads(body)
                if isinstance(tempbody, dict):
                    multiuserkey = tempbody.get('genkey', "")
            except Exception:
                multiuserkey = ""

            if totalgens>0:
                if (multiuserkey=="" and multiuserkey==currentusergenkey and requestsinqueue==0) or (multiuserkey!="" and multiuserkey==currentusergenkey): #avoid leaking prompts in multiuser
                    lastlogprobs = handle.last_logprobs()
                    logprobsdict = parse_last_logprobs(lastlogprobs)
            response_body = (json.dumps({"logprobs":logprobsdict}).encode())

        elif self.path.endswith('/api/extra/multiplayer/status'):
            if not self.secure_endpoint():
                return
            if not has_multiplayer:
                response_body = (json.dumps({"error":"Multiplayer not enabled!"}).encode())
            else:
                sender = ""
                senderbusy = False
                try:
                    tempbody = json.loads(body)
                    if isinstance(tempbody, dict):
                        sender = tempbody.get('sender', "")
                        senderbusy = tempbody.get('senderbusy', False)
                except Exception:
                    pass
                if sender!="" and senderbusy:
                    multiplayer_lastactive[sender] = int(time.time())
                response_body = (json.dumps({"turn_major":multiplayer_turn_major,"turn_minor":multiplayer_turn_minor,"idle":self.get_multiplayer_idle_state(sender),"data_format":multiplayer_dataformat}).encode())

        elif self.path.endswith('/api/extra/data/list'):
            if not self.secure_endpoint():
                return
            if savedata_obj is None:
                response_body = (json.dumps([]).encode())
                return
            output = []
            for i in range (net_save_slots):
                if str(i) in savedata_obj:
                    output.append(savedata_obj[str(i)]["title"])
                else:
                    output.append("")
            response_body = (json.dumps(output).encode())

        elif self.path.endswith('/api/extra/data/load'):
            if not self.secure_endpoint():
                return
            if savedata_obj is None:
                response_body = (json.dumps({"success":False,"data":None}).encode())
            loadid = -1
            try:
                tempbody = json.loads(body)
                loadid = tryparseint(tempbody.get('slot', 0),0)
            except Exception:
                loadid = -1
            if loadid < 0 or str(loadid) not in savedata_obj:
                response_body = (json.dumps({"success":False,"data":None}).encode())
            else:
                response_body = (json.dumps({"success":True,"data":savedata_obj[str(loadid)]}).encode())

        elif self.path.endswith('/api/extra/data/save'):
            if not self.secure_endpoint():
                return
            if savedata_obj is None:
                response_code = 400
                response_body = (json.dumps({"success":False, "error":"SaveDataFile not enabled!"}).encode())
            else:
                try:
                    incoming_story = json.loads(body) # ensure submitted data is valid json
                    slotid = tryparseint(incoming_story.get('slot', -1),-1)
                    dataformat = incoming_story.get('format', "")
                    title = incoming_story.get('title', "")
                    if not title or title=="":
                        title = "Untitled Save"
                    storybody = incoming_story.get('data', None) #should be a compressed string
                    if slotid >= 0 and slotid < net_save_slots:  # we shall provide some fixed network save slots
                        saveneeded = False
                        if storybody and storybody!="":
                            storybody = str(storybody)
                            if len(storybody) > (1024*1024*10): #limit each story to 10mb
                                response_code = 400
                                response_body = (json.dumps({"success":False, "error":"Story is too long!"}).encode())
                            else:
                                savedata_obj[str(slotid)] = {"title":title, "format":dataformat, "data":storybody}
                                saveneeded = True
                        else: #erasing existing story
                            if str(slotid) in savedata_obj:
                                savedata_obj.pop(str(slotid))
                                saveneeded = True
                        if saveneeded:
                            if args.savedatafile and os.path.exists(os.path.abspath(args.savedatafile)):
                                with open(os.path.abspath(args.savedatafile), 'w+', encoding='utf-8', errors='ignore') as f:
                                    json.dump(savedata_obj, f)
                                    print(f"Data was saved to slot {slotid}")
                                response_body = (json.dumps({"success":True, "error":""}).encode())
                            else:
                                response_code = 400
                                response_body = (json.dumps({"success":False, "error":"SaveDataFile is missing!"}).encode())
                        else:
                            response_body = (json.dumps({"success":True, "error":""}).encode())
                    else:
                        response_code = 400
                        response_body = (json.dumps({"success":False, "error":"No story submitted or invalid slot!"}).encode())
                except Exception as e:
                    utfprint("Remote Save Story - Body Error: " + str(e))
                    response_code = 400
                    response_body = (json.dumps({"success": False, "error":"Submitted story invalid!"}).encode())

        elif self.path.endswith('/api/extra/multiplayer/getstory'):
            if not self.secure_endpoint():
                return
            if not has_multiplayer:
                response_body = ("".encode())
            elif multiplayer_story_data_compressed is None:
                response_body = ("".encode())
            else:
                response_body = multiplayer_story_data_compressed.encode()

        elif self.path.endswith('/api/extra/multiplayer/setstory'):
            if not self.secure_endpoint():
                return
            if not has_multiplayer:
                response_code = 400
                response_body = (json.dumps({"success":False, "error":"Multiplayer not enabled!"}).encode())
            else:
                try:
                    incoming_story = json.loads(body) # ensure submitted data is valid json
                    fullupdate = incoming_story.get('full_update', False)
                    dataformat = incoming_story.get('data_format', "")
                    sender = incoming_story.get('sender', "")
                    storybody = incoming_story.get('data', None) #should be a compressed string
                    if storybody:
                        storybody = str(storybody)
                        if len(storybody) > (1024*1024*3): #limit story to 3mb
                            response_code = 400
                            response_body = (json.dumps({"success":False, "error":"Story is too long!"}).encode())
                        else:
                            multiplayer_story_data_compressed = str(storybody) #save latest story
                            multiplayer_dataformat = dataformat
                            if sender!="":
                                multiplayer_lastactive[sender] = int(time.time())
                            if fullupdate:
                                multiplayer_turn_minor = 1
                                multiplayer_turn_major += 1
                            else:
                                multiplayer_turn_minor += 1
                            response_body = (json.dumps({"success":True,"turn_major":multiplayer_turn_major,"turn_minor":multiplayer_turn_minor,"idle":self.get_multiplayer_idle_state(sender),"data_format":multiplayer_dataformat}).encode())
                    else:
                        response_code = 400
                        response_body = (json.dumps({"success":False, "error":"No story submitted!"}).encode())
                except Exception as e:
                    utfprint("Multiplayer Set Story - Body Error: " + str(e))
                    response_code = 400
                    response_body = (json.dumps({"success": False, "error":"Submitted story invalid!"}).encode())

        elif self.path.startswith(("/api/extra/websearch")):
            if not self.secure_endpoint():
                return
            if args.websearch:
                try:
                    tempbody = json.loads(body)
                    searchstr = tempbody.get('q', "")
                    searchres = websearch(searchstr)
                    response_body = (json.dumps(searchres).encode())
                except Exception as e:
                    utfprint("WebSearch Parse Error: " + str(e))
                    response_code = 400
                    response_body = (json.dumps([]).encode())
            else:
                response_body = (json.dumps([]).encode())

        elif self.path.startswith(("/api/admin/reload_config")):
            resp = {"success": False}
            if global_memory and args.admin and args.admindir and os.path.exists(args.admindir) and self.check_header_password(args.adminpassword):
                targetfile = ""
                baseconfig = ""
                try:
                    tempbody = json.loads(body)
                    if isinstance(tempbody, dict):
                        targetfile = tempbody.get('filename', "")
                        baseconfig = tempbody.get('baseconfig', tempbody.get('overrideconfig', ""))
                except Exception:
                    targetfile = ""
                if targetfile and targetfile!="":
                    if targetfile=="unload_model" or targetfile=="initial_model": #special request to simply unload model or swap back top intial model
                        print("Admin: Received request to unload model")
                        global_memory["restart_target"] = targetfile
                        global_memory["restart_override_base_config"] = ""
                        resp = {"success": True}
                    else:
                        dirpath = os.path.abspath(args.admindir)
                        allowed_files = get_current_admindir_list()
                        # Normalize requested target path
                        targetfilepath = os.path.abspath(os.path.join(dirpath, targetfile))

                        if (targetfile in allowed_files and os.path.commonpath([dirpath, targetfilepath]) == dirpath and os.path.exists(targetfilepath)):
                            global_memory["restart_override_base_config"] = "" # Jail enforcement
                            if targetfile and baseconfig:
                                baseconfigfilepath = os.path.abspath(os.path.join(dirpath, baseconfig))
                                if (baseconfig in allowed_files and os.path.commonpath([dirpath, baseconfigfilepath]) == dirpath and os.path.exists(baseconfigfilepath)):
                                    print(f"Admin: Override base config set to {baseconfig}")
                                    global_memory["restart_override_base_config"] = baseconfig
                            print(f"Admin: Received request to reload config to {targetfile}")
                            global_memory["restart_target"] = targetfile
                            resp = {"success": True}
            response_body = (json.dumps(resp).encode())

        elif self.path.endswith('/set_tts_settings'): #return dummy response
            response_body = (json.dumps({"message": "Settings successfully applied"}).encode())

        elif self.path=="/api/show": #ollama compatible
            response_body = (json.dumps({"parameters":"temperature 1.0","license":"Ollama Emulation. Running on KoboldCpp","modelfile":"KoboldCpp","capabilities":["completion"],"modified_at":"2025-01-01T01:00:00.0000000+00:00","details":{},"model_info":{}}).encode())

        elif self.path=="/mcp": #simple mcp proxy
            if not self.secure_endpoint():
                return
            try:
                tempbody = json.loads(body)
                method = tempbody.get("method","")
                if method == "initialize":
                    reply = {
                        "jsonrpc": "2.0",
                        "id": random.randint(100000, 999999),
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {"listChanged": False}},
                            "serverInfo": {"name": "mcp-koboldcpp", "version": "1.0.0"},
                        },
                    }
                    response_body = (json.dumps(reply).encode())
                elif method == "tools/list":
                    reply = {
                        "jsonrpc": "2.0",
                        "id": random.randint(100000, 999999),
                        "result": {"tools": []},
                    }
                    with mcp_lock:
                        for conn in mcp_connections:
                            currtools = conn["tools"]
                            for tool in currtools:
                                reply["result"]["tools"].append(tool)
                    response_body = (json.dumps(reply).encode())
                elif method == "tools/call":
                    foundtool = False
                    callparams = tempbody.get("params",{})
                    callname = callparams.get("name","")
                    with mcp_lock:
                        for conn in mcp_connections:
                            currtools = conn["tools"]
                            currclient = conn["client"]
                            for tool in currtools:
                                if currclient and tool.get("name","")!="" and tool.get("name","")==callname:
                                    foundtool = True
                                    mcpresp = currclient.send(tempbody)
                                    response_body = (json.dumps(mcpresp).encode())
                                    break
                    if not foundtool:
                        response_code = 400
                        response_body = (json.dumps({"error": {"code": -32700, "message": "Tool not found"}}).encode())
                else: #probably a notify, send empty response
                    response_body = (json.dumps({}).encode())
            except Exception as e:
                print(f"MCP Call Error: {e}")
                response_code = 400
                response_body = (json.dumps({"error": {"code": -32700, "message": "Parse error"}}).encode())

        elif self.path=="/api/extra/shutdown":
            # if args.singleinstance:
            client_ip = self.client_address[0]
            is_local = client_ip in ('127.0.0.1', '::1', 'localhost')
            if is_local and args.singleinstance:
                response_body = (json.dumps({"success": True}).encode())
                self.send_response(response_code)
                self.send_header('content-length', str(len(response_body)))
                self.end_headers(content_type='application/json')
                self.wfile.write(response_body)
                print("\nReceived Shutdown Command! Shutting down...\n")
                time.sleep(1)
                global exitcounter
                exitcounter = 999
                sys.exit(0)
                return
            else:
                response_body = (json.dumps({"success": False}).encode())

        if response_body is not None:
            self.send_response(response_code)
            self.send_header('content-length', str(len(response_body)))
            self.end_headers(content_type='application/json')
            self.wfile.write(response_body)
            return

        reqblocking = False
        #handle rate limiting
        ratelimiter = int(args.ratelimit)
        if ratelimiter > 0:
            client_ip = self.client_address[0]
            lastdone = ratelimitlookup.get(client_ip, datetime.min)
            diff = (datetime.now() - lastdone).total_seconds()
            if diff < ratelimiter:
                self.send_response(503)
                self.end_headers(content_type='application/json')
                self.wfile.write(json.dumps({"detail": {
                        "msg": f"You are sending requests too quickly. Please try again in {int(ratelimiter-diff)} seconds.",
                        "type": "service_unavailable",
                    }}).encode())
                return
            ratelimitlookup[client_ip] = datetime.now()
        muint = int(args.multiuser)
        if muint<=0 and ((args.whispermodel and args.whispermodel!="") or (args.sdmodel and args.sdmodel!="") or (args.ttsmodel and args.ttsmodel!="") or (args.embeddingsmodel and args.embeddingsmodel!="")):
            muint = 2 # this prevents errors when using voice/img together with text
        if args.parallelrequests > 1 and muint <=0:
            muint = multiuser_concurrent_limit # multiuser required for batching
        multiuserlimit = ((muint-1) if muint > 1 else multiuser_concurrent_limit)
        #backwards compatibility for up to X concurrent requests, use default limit of X if multiuser set to 1
        if muint > 0 and requestsinqueue < multiuserlimit:
            reqblocking = True
            requestsinqueue += 1
        if not modelbusy.acquire(blocking=reqblocking):
            self.send_response(503)
            self.end_headers(content_type='application/json')
            self.wfile.write(json.dumps({"detail": {
                    "msg": "Server is busy; please try again later.",
                    "type": "service_unavailable",
                }}).encode())
            return
        is_batchable_req = False
        if reqblocking:
            requestsinqueue = (requestsinqueue - 1) if requestsinqueue > 0 else 0

        # handle endpoints that require mutex locking and handle actual gens
        try:
            sse_stream_flag = False
            api_format = 0 #1=basic,2=kai,3=oai,4=oai-chat,5=interrogate,6=ollama,7=ollamachat,8=oai-responses,9=anthropic-messages
            is_imggen = False
            is_comfyui_imggen = False
            is_oai_imggen = False
            is_img_upscale = False
            is_transcribe = False
            is_tts = False
            is_embeddings = False
            is_music_codes = False
            is_music_audio = False
            response_body = None
            use_jinja = args.jinja
            global_memory["last_active_timestamp"] = datetime.now()
            global_memory["triggered_sleeping"] = False
            if self.path.endswith('/api/admin/check_state'):
                if global_memory and args.admin and args.admindir and os.path.exists(args.admindir) and self.check_header_password(args.adminpassword):
                    cur_states = []
                    for sl in range(savestate_limit): #0,1,2,3
                        oldstate = handle.calc_old_state_kv(sl)
                        oldtokencnt = handle.calc_old_state_tokencount(sl)
                        cur_states.append({"tokens":oldtokencnt,"size":oldstate})
                    newstate = handle.calc_new_state_kv()
                    newtokencnt = handle.calc_new_state_tokencount()
                    response_body = (json.dumps({"success": True, "old_states":cur_states, "new_state_size":newstate, "new_tokens":newtokencnt}).encode())
                else:
                    response_body = (json.dumps({"success": False, "old_states":[], "new_state_size":0, "new_tokens":0}).encode())
            elif self.path.endswith('/api/admin/load_state'):
                if global_memory and savestate_limit>0 and args.admin and args.admindir and os.path.exists(args.admindir) and self.check_header_password(args.adminpassword):
                    targetslot = 0
                    try:
                        tempbody = json.loads(body)
                        if isinstance(tempbody, dict):
                            targetslot = tempbody.get('slot', 0)
                    except Exception:
                        pass
                    targetslot = (targetslot if targetslot<savestate_limit else 0)
                    result = handle.load_state_kv(targetslot)
                    tokencnt = handle.calc_new_state_tokencount()
                    response_body = (json.dumps({"success": result, "new_tokens":tokencnt}).encode())
                else:
                    response_body = (json.dumps({"success": False, "new_tokens":0}).encode())
            elif self.path.endswith('/api/admin/save_state'):
                if global_memory and savestate_limit>0 and args.admin and args.admindir and os.path.exists(args.admindir) and self.check_header_password(args.adminpassword):
                    targetslot = 0
                    try:
                        tempbody = json.loads(body)
                        if isinstance(tempbody, dict):
                            targetslot = tempbody.get('slot', 0)
                    except Exception:
                        pass
                    targetslot = (targetslot if targetslot<savestate_limit else 0)
                    result = handle.save_state_kv(targetslot)
                    tokencnt = handle.calc_new_state_tokencount()
                    response_body = (json.dumps({"success": (result>0), "new_state_size":result, "new_tokens":tokencnt}).encode())
                else:
                    response_body = (json.dumps({"success": False, "new_state_size":0, "new_tokens":0}).encode())
            elif self.path.endswith('/api/admin/clear_state'):
                if global_memory and savestate_limit>0 and args.admin and args.admindir and os.path.exists(args.admindir) and self.check_header_password(args.adminpassword):
                    result = handle.clear_state_kv()
                    response_body = (json.dumps({"success": result}).encode())
                else:
                    response_body = (json.dumps({"success": False}).encode())
            elif self.path.startswith('/api/upload/image') or self.path.startswith("/upload/image"): #comfyui compatible
                lastuploadedcomfyimg = b''
                formdata = self.extract_formdata_from_file_upload(body)
                if "file" in formdata and formdata["file"]:
                    lastuploadedcomfyimg = formdata["file"]
                response_body = (json.dumps({"name": "kcpp_img2img.jpg", "subfolder": "", "type": "input"}).encode())
            elif self.path.endswith('/request'):
                api_format = 1
            elif self.path.endswith(('/api/v1/generate', '/api/latest/generate')):
                api_format = 2
            elif self.path.endswith('/api/extra/generate/stream'):
                api_format = 2
                sse_stream_flag = True
            elif self.path.endswith('/v1/completions') or self.path.endswith('/v1/completion') or self.path=='/completions':
                api_format = 3
            elif self.path.endswith('/v1/chat/completions') or self.path=='/chat/completions':
                api_format = 4
            elif self.path.endswith('/sdapi/v1/interrogate'):
                mmprojOverride = False
                if autoswapmode and mmprojName is not None:
                    mmprojOverride = True
                if not mmprojOverride and not has_vision_support:
                    self.send_response(503)
                    self.end_headers(content_type='application/json')
                    self.wfile.write(json.dumps({"detail": {
                            "msg": "No Vision model loaded",
                            "type": "service_unavailable",
                        }}).encode())
                    return
                api_format = 5
            elif self.path.endswith('/api/generate'): #ollama
                api_format = 6
            elif self.path.endswith('/api/chat'): #ollama
                api_format = 7
            elif self.path.endswith('/v1/responses') or self.path=='/responses': #oai-responses
                api_format = 8
            elif self.path.endswith('/v1/messages') or self.path=='/messages': #anthropic
                api_format = 9
            elif self.path.endswith('/sdapi/v1/extra-single-image') or self.path.endswith('/sdapi/v1/upscale'):
                is_img_upscale = True
            elif self.path=="/prompt" or self.path=="/images/generations" or self.path.endswith('/v1/images/generations') or self.path.endswith('/sdapi/v1/txt2img') or self.path.endswith('/sdapi/v1/img2img'):
                is_imggen = True
                if self.path=="/prompt":
                    is_comfyui_imggen = True
                elif self.path.endswith('/v1/images/generations') or self.path=="/images/generations":
                    is_oai_imggen = True
            elif self.path.endswith('/api/extra/transcribe') or self.path.endswith('/v1/audio/transcriptions') or self.path=="/audio/transcriptions":
                is_transcribe = True
            elif self.path.endswith('/api/extra/tts') or self.path.endswith('/v1/audio/speech') or self.path=="/audio/speech" or self.path.endswith('/tts_to_audio'):
                is_tts = True
            elif self.path.endswith('/api/extra/embeddings') or self.path.endswith('/v1/embeddings'):
                is_embeddings = True
            elif self.path.endswith('/api/extra/music/prepare'):
                is_music_codes = True
            elif self.path.endswith('/api/extra/music/generate'):
                is_music_audio = True

            if response_body is not None:
                self.send_response(response_code)
                self.send_header('content-length', str(len(response_body)))
                self.end_headers(content_type='application/json')
                self.wfile.write(response_body)
            elif is_imggen or is_img_upscale or is_transcribe or is_tts or is_embeddings or is_music_codes or is_music_audio or api_format > 0:
                global last_req_time
                last_req_time = time.time()

                if not is_imggen and not is_img_upscale and not self.path.endswith('/tts_to_audio') and api_format!=5:
                    if not self.secure_endpoint():
                        return

                genparams = None
                try:
                    genparams = json.loads(body)
                except Exception:
                    genparams = None
                    if is_transcribe: #fallback handling of file uploads
                        formdata = self.extract_formdata_from_file_upload(body)
                        if "file" in formdata and formdata["file"]:
                            b64wav = formdata["file"]
                            genparams = {"audio_data":b64wav}
                            if "prompt" in formdata and formdata["prompt"]:
                                genparams["prompt"] = formdata["prompt"]
                            if "language" in formdata and formdata["language"]:
                                genparams["language"] = formdata["language"]

                    if not genparams:
                        utfprint("Body Err: " + str(body))
                        self.send_response(500)
                        self.end_headers(content_type='application/json')
                        self.wfile.write(json.dumps({"detail": {
                        "msg": "Error parsing input.",
                        "type": "bad_input",
                        }}).encode())
                        return

                gendefaults = gendefaults_parse_meta_field(args.gendefaults or '')
                gen_new_keys = {k: v for k, v in gendefaults.items() if k not in genparams}
                #special handling for some params that should be overwritten if equal to literal string default
                special_fields = ["sampler_name", "scheduler"]
                for field in special_fields:
                    if field in genparams and isinstance(genparams[field], str):
                        genparams[field] = genparams[field].lower()
                special_fields_overwrite = {}
                if not args.gendefaultsoverwrite:
                    for field in special_fields:
                        if genparams.get(field, "default") == "default" and field in gendefaults:
                            value = gendefaults.get(field, "default")
                            if isinstance(value, str):
                                value = value.lower()
                            special_fields_overwrite[field] = value
                genparams.update(gendefaults if args.gendefaultsoverwrite else gen_new_keys)
                genparams.update(special_fields_overwrite)

                trunc_len = 10000
                if args.debugmode >= 1:
                    trunc_len = 40000

                if use_jinja and not args.jinja_tools:
                    tmptools = genparams.get('tools', [])
                    if tmptools and len(tmptools) > 0:
                        use_jinja = False # not allowed to use tools with jinja

                # payload modifications for lcpp endpoint. we detect this by the timings_per_token field existing
                if "timings_per_token" in genparams:
                    genparams["continue_assistant_turn"] = True
                    genparams["encapsulate_thinking"] = True

                printablegenparams_raw = truncate_long_json(genparams,trunc_len)
                utfprint("\nInput: " + json.dumps(printablegenparams_raw,ensure_ascii=False),1)

                # transform genparams (only used for text gen) first
                genparams = transform_genparams(genparams, api_format, use_jinja)

                if args.debugmode >= 1:
                    printablegenparams = truncate_long_json(genparams,trunc_len)
                    utfprint("\nAdapted Input: " + json.dumps(printablegenparams),1)

                if args.foreground:
                    bring_terminal_to_foreground()

                #if it's a non-batchable request and we already have batching ongoing, stall this request
                if batched_request_runner_count > 0 and not continuous_batching_python_eligible(genparams, api_format):
                    with batched_cond:
                        while batched_request_runner_count > 0:
                            batched_cond.wait()

                if api_format > 0: #text gen
                    # Check if streaming chat completions, if so, set stream mode to true
                    if (api_format == 4 or api_format == 3 or api_format == 8 or api_format == 9) and "stream" in genparams and genparams["stream"]:
                        sse_stream_flag = True
                    if continuous_batching_python_eligible(genparams, api_format):
                        genparams['_batch_expected'] = True
                        modelbusy.release()
                        is_batchable_req = True
                        with batched_cond:
                            batched_request_runner_count += 1

                    gendat = asyncio.run(self.handle_request(genparams, api_format, sse_stream_flag))

                    try:
                        modelNameToReturn = friendlymodelname
                        if autoswapmode and textName is not None:
                            modelNameToReturn = textName
                        # Headers are already sent when streaming
                        if (api_format == 6 or api_format == 7) and genparams.get('stream', True):
                            #ollama fake streaming
                            self.send_response(200)
                            self.send_header("X-Accel-Buffering", "no")
                            self.send_header("cache-control", "no-cache")
                            self.send_header("connection", "keep-alive")
                            self.end_headers(content_type='text/event-stream')
                            if api_format == 6:
                                bodytxt = gendat.get("response","") # extract and erase the AI response from the sync payload.
                                gendat["response"] = ""
                                pl = {"model":modelNameToReturn,"created_at":str(datetime.now(timezone.utc).isoformat()),"response":bodytxt,"done":False}
                                self.wfile.write(f'{json.dumps(pl)}\n'.encode())
                                self.wfile.flush()
                                time.sleep(0.05) #short delay
                                self.wfile.write(f'{json.dumps(gendat)}\n'.encode()) # note: gendat already contains done=true and empty response
                                self.wfile.flush()
                                time.sleep(0.05) #short delay
                            else:
                                bodytxt = gendat.get("message",{}).get("content","") # extract and erase the AI response from the sync payload.
                                gendat["message"] = {"role":"assistant","content":""}
                                pl = {"model":modelNameToReturn,"created_at":str(datetime.now(timezone.utc).isoformat()),"message":{"role":"assistant","content":bodytxt},"done":False}
                                self.wfile.write(f'{json.dumps(pl)}\n'.encode())
                                self.wfile.flush()
                                time.sleep(0.05) #short delay
                                self.wfile.write(f'{json.dumps(gendat)}\n'.encode()) # note: gendat already contains done=true and empty response
                                self.wfile.flush()
                                time.sleep(0.05) #short delay
                            self.close_connection = True
                        elif not sse_stream_flag:
                            self.send_response(200)
                            genresp = (json.dumps(gendat).encode())
                            self.send_header('content-length', str(len(genresp)))
                            self.end_headers(content_type='application/json')
                            self.wfile.write(genresp)
                        elif api_format == 4 and genparams.get('using_openai_tools', False): #special case, fake streaming for openai tool calls
                            # we only send content_text and reasoning_text if tools aren't used. they contain the balance of the output after sync_toolcall_potential_triggered was triggered
                            content_text = genparams.get('sync_toolcall_extra_content', "") #populated by the sse call, we don't use gendat['choices'][0]['message'].get('content', None)
                            reasoning_text = genparams.get('sync_toolcall_extra_reasoning_content', "")
                            toolsdata_res = []
                            try:
                                toolsdata_res = gendat['choices'][0]['message']['tool_calls']
                                if toolsdata_res and len(toolsdata_res)>0:
                                    toolsdata_res[0]["index"] = 0 # need to add an index for OWUI
                            except Exception:
                                toolsdata_res = []

                           # Send role chunk first, if needed
                            if genparams.get('sync_toolcall_first_role_sent', False):
                                genparams['sync_toolcall_first_role_sent'] = True
                                chunk_role = json.dumps({
                                    "id": "koboldcpp",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": modelNameToReturn,
                                    "choices": [{"index": 0, "finish_reason": None, "delta": {"role": "assistant"}}]
                                })
                                self.wfile.write(f"data: {chunk_role}\n\n".encode())
                                self.wfile.flush()

                            # if no valid tool splitter, we have to do 100% synchronous
                            if not content_text and not reasoning_text and genparams.get('sync_toolcall_stream_ineligible', False):
                                temp_content = ""
                                temp_reasoning = ""
                                try:
                                    temp_content = gendat['choices'][0]['message'].get('content', None)
                                except Exception:
                                    temp_content = None
                                try:
                                    temp_reasoning = gendat['choices'][0]['message'].get('reasoning_content', None)
                                except Exception:
                                    temp_reasoning = None
                                if temp_content and not temp_reasoning: #fix incorrect reasoning sent as content
                                    thinkstrips = [item["start"] for item in thinkformats] #start thinking tags
                                    thinksplitters = [item["end"] for item in thinkformats] #end thinking tags
                                    for tsp in thinksplitters:
                                        if tsp in temp_content:
                                            parts = temp_content.split(tsp, 1)
                                            temp_reasoning = parts[0]
                                            temp_content = parts[1]
                                            for ts in thinkstrips:
                                                temp_reasoning = temp_reasoning.replace(ts, "")

                                if temp_reasoning:
                                    chunk_content = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"reasoning_content": temp_reasoning}}]
                                    })
                                    self.wfile.write(f"data: {chunk_content}\n\n".encode())
                                    self.wfile.flush()
                                if temp_content:
                                    chunk_content = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"content": temp_content}}]
                                    })
                                    self.wfile.write(f"data: {chunk_content}\n\n".encode())
                                    self.wfile.flush()

                            # Send tool calls incrementally in OpenAI format
                            if toolsdata_res and len(toolsdata_res) > 0:
                                for idx, tool_call in enumerate(toolsdata_res):
                                    tc_meta = {
                                        "index": idx,
                                        "id": tool_call.get("id", f"call_{idx}"),
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.get("function", {}).get("name", ""),
                                            "arguments": ""
                                        }
                                    }
                                    chunk_meta = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [tc_meta]}}]
                                    })
                                    self.wfile.write(f"data: {chunk_meta}\n\n".encode())
                                    self.wfile.flush()

                                    args_str = tool_call.get("function", {}).get("arguments", "{}")
                                    if isinstance(args_str, dict):
                                        args_str = json.dumps(args_str)
                                    tc_args = {
                                        "index": idx,
                                        "function": {"arguments": args_str}
                                    }
                                    chunk_args = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [tc_args]}}]
                                    })
                                    self.wfile.write(f"data: {chunk_args}\n\n".encode())
                                    self.wfile.flush()
                            else:
                                # Send remaining buffered content if no tool calls were made
                                if reasoning_text:
                                    chunk_content = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"reasoning_content": reasoning_text}}]
                                    })
                                    self.wfile.write(f"data: {chunk_content}\n\n".encode())
                                    self.wfile.flush()
                                if content_text:
                                    chunk_content = json.dumps({
                                        "id": "koboldcpp",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": modelNameToReturn,
                                        "choices": [{"index": 0, "finish_reason": None, "delta": {"content": content_text}}]
                                    })
                                    self.wfile.write(f"data: {chunk_content}\n\n".encode())
                                    self.wfile.flush()

                            # Final chunk
                            chunk_final = json.dumps({
                                "id": "koboldcpp",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": modelNameToReturn,
                                "choices": [{"index": 0, "finish_reason": "tool_calls" if (len(toolsdata_res) > 0) else currfinishreason, "delta": {}}]
                            })
                            self.wfile.write(f"data: {chunk_final}\n\n".encode())
                            self.wfile.write("data: [DONE]\n\n".encode())
                            self.wfile.flush()
                            self.close_connection = True
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Generate: The response could not be sent, maybe connection was terminated?")
                        handle.abort_generate()
                        time.sleep(0.2) #short delay
                    return

                elif is_img_upscale: #esrgan upscale
                    try:
                        gen = sd_upscale(genparams)
                        genresp = (json.dumps({"html_info":"<p>Postprocess upscale by: 2.0, Postprocess upscaler: ESRGAN_4x</p>","image":gen}).encode())
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Upscale Image: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_imggen: #image gen
                    try:
                        if is_comfyui_imggen:
                            lastgeneratedcomfyimg = b''
                            genparams = sd_comfyui_tranform_params(genparams)
                        elif is_oai_imggen:
                            genparams = sd_oai_transform_params(genparams)
                        if not genparams.get('lora'):
                            # process <lora:name:multiplier> syntax
                            prompt, loras = extract_loras_from_prompt(genparams['prompt'])
                            if loras:
                                genparams['prompt'] = prompt
                                genparams['lora'] = lora_map_name_to_path(loras)
                        gen = sd_generate(genparams)
                        gendat = gen["data"]
                        genanim = gen["animated"]
                        gendatextra = gen["data_extra"]
                        genfinalframe = gen["final_frame"]
                        geninfo = json.dumps(gen["info"]) # sdapi really expects a stringified JSON
                        genresp = None
                        if is_comfyui_imggen:
                            if gendat:
                                lastgeneratedcomfyimg = base64.b64decode(gendat)
                            else:
                                lastgeneratedcomfyimg = b''
                            genresp = (json.dumps({"prompt_id": "12345678-0000-0000-0000-000000000001","number": 0,"node_errors":{}}).encode())
                        elif is_oai_imggen:
                            genresp = (json.dumps({"created":int(time.time()),"data":[{"b64_json":gendat}],"background":"opaque","output_format":"png","size":"1024x1024","quality":"medium"}).encode())
                        else:
                            genresp = (json.dumps({"images":[gendat],"parameters":{},"info":geninfo,"animated":genanim,"extra_data":gendatextra, "final_frame":genfinalframe}).encode())
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Generate Image: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_transcribe:
                    try:
                        global fullwhispermodelpath, has_audio_support
                        gendat = None
                        if genparams.get("audio_data","") and fullwhispermodelpath=="" and has_audio_support: #if we have no whisper model but an audio-capable projector, use that instead
                            adapter_obj = {} if chatcompl_adapter is None else chatcompl_adapter
                            user_message_start = adapter_obj.get("user_start", "### Instruction:")
                            assistant_message_start = adapter_obj.get("assistant_start", "### Response:")
                            assistant_message_gen = adapter_obj.get("assistant_gen", assistant_message_start)
                            prompt = f"{user_message_start} Transcribe all speech in the audio.\n{assistant_message_gen}"
                            rawaudio = genparams.get("audio_data","").replace("data:audio/wav;base64,","")
                            temp_poll = {
                                "prompt": prompt,
                                "max_length":300,
                                "temperature":0.1,
                                "top_k":1,
                                "rep_pen":1,
                                "ban_eos_token":False,
                                "audio": [rawaudio]
                            }
                            temp_poll_result = generate(genparams=temp_poll)
                            gendat = temp_poll_result['text']
                        else:
                            gendat = whisper_generate(genparams)
                        genresp = (json.dumps({"text":gendat}).encode())
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Transcribe: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_tts:
                    try:
                        gendat = tts_generate(genparams)
                        wav_data = b''
                        if gendat:
                            wav_data = base64.b64decode(gendat) # Decode the Base64 string into binary data
                        self.send_response(200)
                        self.send_header('content-length', str(len(wav_data)))  # Set content length
                        self.send_header('Content-Disposition', 'attachment; filename="output.wav"')
                        self.end_headers(content_type='audio/wav')
                        self.wfile.write(wav_data) # Write the binary WAV data to the response
                    except Exception as ex:
                        utfprint(ex,1)
                        print("TTS: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_embeddings:
                    try:
                        modelNameToReturn = friendlyembeddingsmodelname
                        if autoswapmode and embedName is not None:
                            modelNameToReturn = embedName
                        gendat = embeddings_generate(genparams)
                        outdatas = []
                        odidx = 0
                        for od in gendat["data"]:
                            if genparams.get("encoding_format", "")=="base64":
                                binary_data = struct.pack('<' + 'f' * len(od), *od)
                                b64_string = base64.b64encode(binary_data).decode('utf-8')
                                outdatas.append({"object":"embedding","index":odidx,"embedding":b64_string})
                            else:
                                outdatas.append({"object":"embedding","index":odidx,"embedding":od})
                            odidx += 1
                        genresp = (json.dumps({"object":"list","data":outdatas,"model":modelNameToReturn,"usage":{"prompt_tokens":gendat["count"],"total_tokens":gendat["count"]}}).encode())
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Create Embeddings: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_music_codes:
                    try:
                        gendat = music_generate_codes(genparams)
                        genresp = (json.dumps({"error":"music code generation failed"}).encode())
                        if gendat:
                            genresp = gendat.encode()
                        self.send_response(200)
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Music Gen Codes: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return
                elif is_music_audio:
                    try:
                        gendat = music_generate_audio(genparams)
                        wav_data = b''
                        if gendat:
                            wav_data = base64.b64decode(gendat) # Decode the Base64 string into binary data
                        self.send_response(200)
                        self.send_header('content-length', str(len(wav_data)))  # Set content length
                        self.send_header('Content-Disposition', 'attachment; filename="output.wav"')
                        self.end_headers(content_type='audio/wav')
                        self.wfile.write(wav_data) # Write the binary WAV data to the response
                    except Exception as ex:
                        utfprint(ex,1)
                        print("Music Gen Audio: The response could not be sent, maybe connection was terminated?")
                        time.sleep(0.2) #short delay
                    return

        finally:
            time.sleep(0.05)
            if is_batchable_req:
                with batched_cond:
                    batched_request_runner_count -= 1
                    batched_cond.notify_all()
            else:
                modelbusy.release()

        self.send_response(404)
        self.end_headers(content_type='text/html')


    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers(content_type='text/html')

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers(content_type='text/html')

    def end_headers(self, content_type=None):
        self.send_header('access-control-allow-origin', '*')
        self.send_header('access-control-allow-methods', '*')
        self.send_header('access-control-allow-headers', '*, Accept, Content-Type, Content-Length, Cache-Control, Accept-Encoding, X-CSRF-Token, Client-Agent, X-Fields, Content-Type, Authorization, X-Requested-With, X-HTTP-Method-Override, apikey, genkey')
        self.send_header("cache-control", "no-store")
        if content_type is not None:
            self.send_header('content-type', content_type)
        return super(KcppServerRequestHandler, self).end_headers()

def RunServerMultiThreaded(addr, port, server_handler):
    global exitcounter, sslvalid, global_memory
    if is_port_in_use(port):
        print(f"Warning: Port {port} already appears to be in use by another program.")

    ipv4_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ipv4_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ipv6_sock = None
    if is_ipv6_supported():
        ipv6_sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        ipv6_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ipv6_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)

    if args.ssl and sslvalid and not args.routermode: #if routermode, ssl is already offloaded
        import ssl
        certpath = os.path.abspath(args.ssl[0])
        keypath = os.path.abspath(args.ssl[1])
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=certpath, keyfile=keypath)
        ipv4_sock = context.wrap_socket(ipv4_sock, server_side=True)
        if ipv6_sock:
            ipv6_sock = context.wrap_socket(ipv6_sock, server_side=True)

    numThreads = 24
    try:
        ipv4_sock.bind((addr, port))
        ipv4_sock.listen(numThreads)
    except Exception:
        ipv4_sock = None
        print("IPv4 Socket Failed to Bind.")

    if ipv6_sock:
        try:
            ipv6_sock.bind((addr, port))
            ipv6_sock.listen(numThreads)
        except Exception:
            ipv6_sock = None
            print("IPv6 Socket Failed to Bind. IPv6 will be unavailable.")

    class Thread(threading.Thread):
        def __init__(self, i):
            threading.Thread.__init__(self)
            self.i = i
            self.daemon = True
            self.start()

        def run(self):
            global exitcounter
            handler = server_handler(addr, port)
            with http.server.HTTPServer((addr, port), handler, False) as self.httpd:
                try:
                    if ipv4_sock and ipv6_sock:
                        self.httpd.socket = ipv4_sock if self.i < 16 else ipv6_sock
                    elif ipv6_sock:
                        self.httpd.socket = ipv6_sock
                    elif ipv4_sock:
                        self.httpd.socket = ipv4_sock
                    else:
                        print("ERROR: Both IPv4 and IPv6 cannot bind. Server features will not work.")
                        return

                    self.httpd.server_bind = self.server_close = lambda self: None
                    self.httpd.serve_forever()
                except (KeyboardInterrupt,SystemExit):
                    exitcounter = 999
                    self.httpd.server_close()
                    sys.exit(0)
                finally:
                    exitcounter = 999
                    self.httpd.server_close()
                    os._exit(0)
        def stop(self):
            global exitcounter
            exitcounter = 999
            self.httpd.server_close()

    threadArr = []
    for i in range(numThreads):
        threadArr.append(Thread(i))
    while 1:
        try:
            time.sleep(10)
        except (KeyboardInterrupt,SystemExit):
            global exitcounter
            exitcounter = 999
            for i in range(numThreads):
                try:
                    threadArr[i].stop()
                except Exception:
                    continue
            sys.exit(0)

# Based on https://github.com/mathgeniuszach/xdialog/blob/main/xdialog/zenity_dialogs.py - MIT license | - Expanded version by Henk717
def zenity(filetypes=None, initialdir="", initialfile="", multiple=False, **kwargs) -> Tuple[int, object]:
    global zenity_recent_dir, zenity_permitted

    if not zenity_permitted:
        raise Exception("Zenity disabled, attempting to use TK GUI.")
    if sys.platform != "linux":
        raise Exception("Zenity GUI is only usable on Linux, attempting to use TK GUI.")
    zenity_bin = shutil.which("yad")
    using_yad = True
    if not zenity_bin:
        zenity_bin = shutil.which("zenity")
        using_yad = False
    if not zenity_bin:
        using_yad = False
        raise Exception("Zenity not present, falling back to TK GUI.")

    def zenity_clean(txt: str):
        return txt.replace("\\", "\\\\").replace("$", "\\$").replace("!", "\\!").replace("*", "\\*")\
        .replace("?", "\\?").replace("&", "&amp;").replace("|", "&#124;").replace("<", "&lt;").replace(">", "&gt;")\
        .replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]").replace("{", "\\{").replace("}", "\\}")

    def zenity_sanity_check(zenity_bin): #make sure zenity is sane
        try: # Run `zenity --help` and pipe to grep
            sc_clean_env = os.environ.copy()
            sc_clean_env.pop("LD_LIBRARY_PATH", None)
            sc_clean_env["PATH"] = "/usr/bin:/bin"
            scargs = ['/usr/bin/env', zenity_bin, '--help']
            result = subprocess.run(scargs, env=sc_clean_env, capture_output=True, text=True, encoding="utf-8", timeout=10)

            if result.returncode == 0 and "--file" in result.stdout:
                return True
            else:
                utfprint(f"Zenity/YAD sanity check failed - ReturnCode={result.returncode}",0)
                return False
        except FileNotFoundError:
            utfprint(f"Zenity/YAD sanity check failed - {zenity_bin} not found",0)
            return False

    if not zenity_sanity_check(zenity_bin):
        raise Exception("Zenity not working correctly, falling back to TK GUI.")

    # Build args based on keywords
    args = ['/usr/bin/env', zenity_bin, ('--file' if using_yad else '--file-selection')]
    for k, v in kwargs.items():
        if v is True:
            args.append(f'--{k.replace("_", "-").strip("-")}')
        elif isinstance(v, str):
            cv = zenity_clean(v) if k != "title" else v
            args.append(f'--{k.replace("_", "-").strip("-")}={cv}')

    # Build filetypes specially if specified
    if filetypes:
        for name, globs in filetypes:
            if name:
                globlist = globs.split()
                args.append(f'--file-filter={name.replace("|", "")} ({", ".join(t for t in globlist)})|{globs}')

    # Default filename and folder
    if initialdir is None:
        initialdir=zenity_recent_dir
    if initialfile is None:
        initialfile=""
    initialpath = os.path.join(initialdir, initialfile)
    args.append(f'--filename={initialpath}')

    if multiple:
        args.append("--multiple")
        args.append("--separator=|")

    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    clean_env["PATH"] = "/usr/bin:/bin"

    procres = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=clean_env,
        check=False
    )
    result = procres.stdout.decode('utf-8').strip()
    if procres.returncode==0 and result:
        directory = result
        if multiple:
            result = tuple(result.split("|"))
            directory = result[0]
        if not os.path.isdir(directory):
            directory = os.path.dirname(directory)
        zenity_recent_dir = directory
    return (procres.returncode, result)

# note: In this section we wrap around file dialogues to allow for zenity
def zentk_askopenfilename(**options):
    try:
        result = zenity(filetypes=options.get("filetypes"), initialdir=options.get("initialdir"), multiple=False, title=options.get("title"))[1]
        if result and not os.path.isfile(result):
            print("A folder was selected while we need a file, ignoring selection.")
            return ''
    except Exception:
        from tkinter.filedialog import askopenfilename
        result = askopenfilename(**options)
    return result

def zentk_askopenfilenames(**options):
    try:
        result = zenity(filetypes=options.get("filetypes"), initialdir=options.get("initialdir"), multiple=True, title=options.get("title"))[1]
        for itm in result:
            if itm and not os.path.isfile(itm):
                print("A folder was selected while we need a file, ignoring selection.")
                return ''
    except Exception:
        from tkinter.filedialog import askopenfilenames
        result = askopenfilenames(**options)
    return result

def zentk_askdirectory(**options):
    try:
        result = zenity(initialdir=options.get("initialdir"), multiple=False, title=options.get("title"), directory=True)[1]
    except Exception:
        from tkinter.filedialog import askdirectory
        result = askdirectory(**options)
    return result

def zentk_asksaveasfilename(**options):
    try:
        result = zenity(filetypes=options.get("filetypes"), initialdir=options.get("initialdir"), initialfile=options.get("initialfile"), multiple=False, title=options.get("title"), save=True)[1]
    except Exception:
        from tkinter.filedialog import asksaveasfilename
        result = asksaveasfilename(**options)
    return result
### End of MIT license

def save_config_dict(filename, savdict, template):
    filenamestr = str(filename).strip()
    if not filenamestr.endswith(".kcpps") and not template:
        filenamestr += ".kcpps"
    if not filenamestr.endswith(".kcppt") and template:
        filenamestr += ".kcppt"
    do_not_save = {'analyze', 'config', 'exportconfig', 'exporttemplate', 'testmemory', 'unpack', 'version'}
    filtered = {k: v for k, v in savdict.items() if k not in do_not_save}
    if 'gendefaults' in filtered:
        gendefaults = parse_json_object(filtered['gendefaults'], 'gendefaults')
        if isinstance(gendefaults, dict):
            filtered['gendefaults'] = gendefaults
        # keep it as-is if it's a broken string
    with open(filenamestr, 'w') as file:
        file.write(json.dumps(filtered,indent=2))
    return filenamestr

splitmode_choices = ['layer','row','tensor']
def splitmode_choices_to_int(value): #layer=1, row=2, tensor=3
    if value=='layer':
        return 1
    elif value=='row':
        return 2
    elif value=='tensor':
        return 3
    return 1

# note: customtkinter-5.2.0
def show_gui():
    global using_gui_launcher
    using_gui_launcher = True

    #check for potential scaling issues
    def get_problematic_scaler():
        if sys.platform != "linux":
            return False
        xdg_curr_desk = os.environ.get("XDG_CURRENT_DESKTOP")
        if xdg_curr_desk and ("KDE" in xdg_curr_desk or "GNOME" in xdg_curr_desk or "Cinnamon" in xdg_curr_desk): # broad spectrum dpi handler
            dpi = 0
            try:
                output = subprocess.check_output(["xrdb", "-query"], text=True).strip()
                if output:
                    for line in output.splitlines():
                        if line.startswith("Xft.dpi:"):
                            dpi = float(line.split(":")[1].strip())
                            break
            except Exception:
                pass
            if dpi > 100:
                return True

        import xml.etree.ElementTree as ET
        from pathlib import Path
        fractional_enabled = False # Check if fractional scaling is enabled
        try:
            features = subprocess.check_output(
                ["gsettings", "get", "org.gnome.mutter", "experimental-features"],
                text=True
            ).strip()
            fractional_enabled = "scale-monitor-framebuffer" in features
        except Exception:
            return False
        xml_path = Path.home() / ".config" / "monitors.xml"
        if not xml_path.exists(): #monitors.xml not found. if we have fractional scaling on gnome, just trigger the fallback
            if fractional_enabled and "GNOME" in os.environ.get("XDG_CURRENT_DESKTOP") and os.environ.get("XDG_SESSION_TYPE") == "wayland":
                return True
            return False
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            configs = root.findall(".//configuration")
            if not configs:
                return False
            logical_confs = [c for c in configs if c.findtext(".//layoutmode") == "logical"]
            physical_confs = [c for c in configs if c.findtext(".//layoutmode") == "physical"]
            if fractional_enabled and logical_confs:
                chosen_conf = logical_confs[-1]
            elif not fractional_enabled and physical_confs:
                chosen_conf = physical_confs[-1]
            else:
                chosen_conf = configs[-1]
            scales = [float(s.text) for s in chosen_conf.findall(".//scale") if s.text]
            if scales:
                return max(scales)>1.0
        except Exception:
            pass
        return False

    corrupt_scaler = get_problematic_scaler()

    # if args received, launch
    if len(sys.argv) != 1 and not args.showgui:
        import tkinter as tk
        root = tk.Tk() #we dont want the useless window to be visible, but we want it in taskbar
        root.attributes("-alpha", 0)
        args.model_param = zentk_askopenfilename(title="Select ggml model .bin or .gguf file or .kcpps config")
        root.withdraw()
        root.quit()
        if args.model_param and args.model_param!="" and (args.model_param.lower().endswith('.kcpps') or args.model_param.lower().endswith('.kcppt') or args.model_param.lower().endswith('.kcpps?download=true') or args.model_param.lower().endswith('.kcppt?download=true')):
            dlfile = download_model_from_url(args.model_param,[".kcpps",".kcppt"]) # maybe download from url
            if dlfile:
                args.model_param = dlfile
            load_config_cli(args.model_param)
        if not has_valid_model():
            global exitcounter
            exitcounter = 999
            exit_with_error(2,"No gguf model or kcpps file was selected. Exiting.")
        return

    #dummy line to get darkdetect imported in pyinstaller
    try:
        import darkdetect as darkdt
        darkdt.isDark()
        pass
    except Exception:
        pass

    import customtkinter as ctk
    nextstate = 0 #0=exit, 1=launch
    original_windowwidth = int(590)
    original_windowheight = int(590)
    windowwidth = original_windowwidth
    windowheight = original_windowheight
    ctk.set_appearance_mode("dark")
    ctk.deactivate_automatic_dpi_awareness()
    root = ctk.CTk(fg_color="#2b2b2b")
    if corrupt_scaler:
        print("Adjusting tk scaling to try and fix scaling issues...")
        root.tk.call('tk','scaling', 2.25)
    root.geometry(str(windowwidth) + "x" + str(windowheight))
    root.title(f"KoboldCpp v{KcppVersion}")

    gtooltip_box = None
    gtooltip_label = None

    window_reference_width = None
    window_reference_height = None
    previous_event_width = None
    previous_event_height = None
    resizing = False
    resizing_id1 = None
    resizing_id2 = None
    def clearesizing():
        nonlocal resizing, resizing_id1
        resizing = False
        resizing_id1 = None
    def actually_resize(windowwidth,windowheight,lastpos,smallratio):
        root.geometry(str(windowwidth) + "x" + str(windowheight) + str(lastpos))
        ctk.set_widget_scaling(smallratio)
        update_runmode_gui()
        togglerope(1,1,1)
        toggleflashattn(1,1,1)
        togglectxshift(1,1,1)
        togglehorde(1,1,1)
        toggletaesd(1,1,1)
        togglesdlora(1,1,1)
        togglejinja(1,1,1)
        toggleadmin(1,1,1)
        tabbuttonaction(tabnames[curr_tab_idx])
        pass
    def on_resize(event):
        nonlocal resizing, resizing_id1, resizing_id2
        if not event.widget.master and event.widget == root:
            nonlocal window_reference_width, window_reference_height, previous_event_width,previous_event_height
            if resizing:
                previous_event_width = event.width
                previous_event_height = event.height
                return
            if not window_reference_width and not window_reference_height:
                window_reference_width = event.width
                window_reference_height = event.height
                previous_event_width = window_reference_width
                previous_event_height = window_reference_height
            else:
                new_width = event.width
                new_height = event.height
                incr_w = new_width/window_reference_width
                incr_h = new_height/window_reference_height
                smallratio = min(incr_w,incr_h)
                smallratio = round(smallratio,2)
                if new_width != previous_event_width or new_height!=previous_event_height:
                    resizing = True
                    lastpos = root.geometry()
                    lparr = lastpos.split('+', 1)
                    lastpos = ("+"+str(lparr[1])) if (len(lparr)==2) else ""
                    previous_event_width = new_width
                    previous_event_height = new_height
                    windowwidth = math.floor(original_windowwidth*smallratio)
                    windowwidth = max(256, min(1024, windowwidth))
                    windowheight = math.floor(original_windowheight*smallratio)
                    windowheight = max(256, min(1024, windowheight))
                    if resizing_id2:
                        root.after_cancel(resizing_id2)
                        resizing_id2 = None
                    resizing_id2 = root.after(100, lambda: actually_resize(windowwidth,windowheight,lastpos,smallratio))
                    if resizing_id1:
                        root.after_cancel(resizing_id1)
                        resizing_id1 = None
                    resizing_id1 = root.after(5, clearesizing)

    if sys.platform=="darwin":
        root.resizable(False,False)
    else:
        root.resizable(True,True)
        root.bind("<Configure>", on_resize)
    kcpp_exporting_template = False

    # trigger empty tooltip then remove it
    def show_tooltip(event, tooltip_text=None):
        nonlocal gtooltip_box, gtooltip_label
        if not gtooltip_box and not gtooltip_label:
            gtooltip_box = ctk.CTkToplevel(root)
            gtooltip_box.configure(fg_color="#ffffe0")
            gtooltip_box.withdraw()
            gtooltip_box.overrideredirect(True)
            gtooltip_label = ctk.CTkLabel(gtooltip_box, text=tooltip_text, text_color="#000000", fg_color="#ffffe0")
            gtooltip_label.pack(expand=True, ipadx=2, ipady=1)
        else:
            gtooltip_label.configure(text=tooltip_text)

        gtooltip_box.update_idletasks()
        x, y = root.winfo_pointerxy()
        gtooltip_box.wm_geometry(f"+{x + 10}+{y + 10}")
        gtooltip_box.deiconify()

    def hide_tooltip(event):
        nonlocal gtooltip_box
        if gtooltip_box:
            gtooltip_box.withdraw()
    show_tooltip(None,"") #initialize tooltip objects
    hide_tooltip(None)

    default_threads = get_default_threads()

    tabs = ctk.CTkFrame(root, corner_radius = 0, width=windowwidth, height=windowheight-50)
    tabs.grid(row=0, stick="nsew")
    tabnames= ["Quick Launch", "Hardware", "Context", "Loaded Files", "Network", "Horde Worker","Image Gen","Audio","Admin","Extra"]
    navbuttons = {}
    navbuttonframe = ctk.CTkFrame(tabs, width=int(104), height=int(tabs.cget("height")))
    navbuttonframe.grid(row=0, column=0, padx=2,pady=2)
    navbuttonframe.grid_propagate(False)

    tabcontentframe = ctk.CTkFrame(tabs, width=windowwidth - int(navbuttonframe.cget("width")), height=int(tabs.cget("height")),fg_color="transparent")
    tabcontentframe.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
    tabcontentframe.grid_propagate(False)

    tabcontent = {}
    # slider data
    batchsize_values = ["-1","16","32","64","128","256","512","1024","2048","4096"]
    batchsize_text = ["Don't Batch","16","32","64","128","256","512","1024","2048","4096"]
    contextsize_text = ["256", "512", "1024", "2048", "3072", "4096", "5120", "6144", "7168", "8192", "9216", "10240", "11264", "12288", "13312", "14336", "15360", "16384", "18432", "20480", "22528", "24576", "26624", "28672", "30720", "32768", "36864", "40960", "45056", "49152", "53248", "57344", "61440", "65536", "73728", "81920", "90112", "98304", "106496", "114688", "122880", "131072", "147456", "163840", "180224", "196608", "212992", "229376", "245760", "262144" ]
    quantkv_text = ["f16","bf16","q8_0","q5_1","q4_0"]

    if not any(runopts):
        exitcounter = 999
        exit_with_error(2,"KoboldCPP couldn't locate any backends to use (i.e Default, Vulkan, CUDA).\n\nTo use the program, please run the 'make' command from the directory.","No Backends Available!")

    # Vars - should be in scope to be used by multiple widgets
    gpulayers_var = ctk.StringVar(value="-1")
    threads_var = ctk.StringVar(value=str(default_threads))
    runopts_var = ctk.StringVar()
    gpu_choice_var = ctk.StringVar(value="0")
    autofit_padding_var = ctk.StringVar(value=str(default_autofit_padding))

    launchbrowser = ctk.IntVar(value=1)
    highpriority = ctk.IntVar()
    usemmap = ctk.IntVar(value=0)
    usemlock = ctk.IntVar()
    debugmode = ctk.IntVar()
    keepforeground = ctk.IntVar()
    terminalonly = ctk.IntVar()
    pipelineparallel = ctk.IntVar(value=1)
    quietmode = ctk.IntVar(value=0)
    nocertifymode = ctk.IntVar(value=0)

    lowvram_var = ctk.IntVar()
    mmq_var = ctk.IntVar(value=1)
    quantkv_var = ctk.IntVar(value=0)
    blas_threads_var = ctk.StringVar()
    blas_size_var = ctk.IntVar()
    autofit_var = ctk.IntVar()
    tensor_split_str_vars = ctk.StringVar(value="")
    splitmode_var = ctk.StringVar(value=splitmode_choices[0])
    maingpu_var = ctk.StringVar(value="-1")
    deviceoverride_var = ctk.StringVar(value="")

    contextshift_var = ctk.IntVar(value=1)
    fastforward_var = ctk.IntVar(value=1)
    swa_var = ctk.IntVar(value=1)
    swa_padding_var = ctk.StringVar(value=str(swa_padding_default))
    smartcache_var = ctk.IntVar(value=0)
    smartcacheslots_var = ctk.StringVar(value=str(savestate_limit_default))
    remotetunnel_var = ctk.IntVar(value=0)
    smartcontext_var = ctk.IntVar()
    flashattention_var = ctk.IntVar(value=1)
    context_var = ctk.IntVar()
    customrope_var = ctk.IntVar()
    manualrope_var = ctk.IntVar()
    customrope_scale = ctk.StringVar(value="1.0")
    customrope_base = ctk.StringVar(value="10000")
    customrope_nativectx = ctk.StringVar(value=str(default_native_ctx))
    chatcompletionsadapter_var = ctk.StringVar(value="AutoGuess")
    jinjatemplate_var = ctk.StringVar()
    jinja_var = ctk.IntVar(value=0)
    jinja_tools_var = ctk.IntVar(value=0)
    jinja_kwargs_var = ctk.StringVar()
    jinja_think_var = ctk.StringVar(value="default")
    moeexperts_var = ctk.StringVar(value=str(-1))
    moecpu_var = ctk.StringVar(value=str(0))
    defaultgenamt_var = ctk.StringVar(value=str(default_genlen))
    genlimit_var = ctk.StringVar(value=str(0))
    nobostoken_var = ctk.IntVar(value=0)
    override_kv_var = ctk.StringVar(value="")
    override_tensors_var = ctk.StringVar(value="")
    enableguidance_var = ctk.IntVar(value=0)

    model_var = ctk.StringVar()
    lora_var = ctk.StringVar()
    loramult_var = ctk.StringVar(value="1.0")
    preloadstory_var = ctk.StringVar()
    savedatafile_var = ctk.StringVar()
    mcpfile_var = ctk.StringVar()
    mmproj_var = ctk.StringVar()
    mmprojcpu_var = ctk.IntVar(value=0)
    visionmaxres_var = ctk.StringVar(value=str(default_visionmaxres))
    vision_min_tokens_var = ctk.StringVar(value="-1")
    vision_max_tokens_var = ctk.StringVar(value="-1")
    draftmodel_var = ctk.StringVar()
    draftamount_var = ctk.StringVar(value=str(default_draft_amount))
    draftgpulayers_var = ctk.StringVar(value=str(999))
    draftgpusplit_str_vars = ctk.StringVar(value="")
    nomodel = ctk.IntVar(value=0)
    download_dir_var = ctk.StringVar()

    port_var = ctk.StringVar(value=defaultport)
    host_var = ctk.StringVar(value="")
    multiuser_var = ctk.StringVar(value=str(multiuser_concurrent_limit))
    multiplayer_var = ctk.IntVar(value=has_multiplayer)
    websearch_var = ctk.IntVar(value=0)
    horde_name_var = ctk.StringVar(value="koboldcpp")
    horde_gen_var = ctk.StringVar(value=maxhordelen)
    horde_context_var = ctk.StringVar(value=maxhordectx)
    horde_apikey_var = ctk.StringVar(value="")
    horde_workername_var = ctk.StringVar(value="")
    usehorde_var = ctk.IntVar()
    ssl_cert_var = ctk.StringVar()
    ssl_key_var = ctk.StringVar()
    password_var = ctk.StringVar()
    maxrequestsize_var = ctk.StringVar(value=str(32))
    ratelimit_var = ctk.StringVar(value=str(0))
    reqtimeout_var = ctk.StringVar(value=str(default_reqtimeout))
    parallel_requests_var = ctk.StringVar(value=str(1))
    rpcmode_var =  ctk.StringVar(value=str("disabled"))
    rpcendpoints_var = ctk.StringVar(value="")
    rpc_host_var = ctk.StringVar(value="0.0.0.0")
    rpc_port_var = ctk.StringVar(value=str(default_rpc_port))
    rpc_device_var = ctk.StringVar()

    sd_model_var = ctk.StringVar()
    sd_lora_var = ctk.StringVar()
    sd_loramult_var = ctk.StringVar(value="1.0")
    sd_vae_var = ctk.StringVar()
    sd_audio_vae_var = ctk.StringVar()
    sd_t5xxl_var = ctk.StringVar()
    sd_clip1_var = ctk.StringVar()
    sd_clip2_var = ctk.StringVar()
    sd_photomaker_var = ctk.StringVar()
    sd_upscaler_var = ctk.StringVar()
    sd_flash_attention_var = ctk.IntVar(value=0)
    sd_offload_cpu_var = ctk.IntVar(value=0)
    sd_vae_device_var = ctk.StringVar(value="main")
    sd_clip_device_var = ctk.StringVar(value="CPU")
    sd_runtime_loras_var = ctk.IntVar(value=0)
    sd_vaeauto_var = ctk.IntVar(value=0)
    sd_tiled_vae_var = ctk.StringVar(value=str(default_vae_tile_threshold))
    sd_convdirect_var = ctk.StringVar(value=str(sd_convdirect_choices[0]))
    sd_clamped_var = ctk.StringVar(value="0")
    sd_clamped_soft_var = ctk.StringVar(value="0")
    sd_threads_var = ctk.StringVar(value=str(default_threads))
    sd_quant_var = ctk.StringVar(value=sd_quant_choices[0])
    sd_main_gpu_var = ctk.StringVar(value="main")
    sd_vram_limit_var = ctk.StringVar(value="")

    gen_defaults_var = ctk.StringVar()
    gen_defaults_overwrite_var = ctk.IntVar(value=0)

    whisper_model_var = ctk.StringVar()
    tts_model_var = ctk.StringVar()
    wavtokenizer_var = ctk.StringVar()
    ttsgpu_var = ctk.IntVar(value=0)
    tts_threads_var = ctk.StringVar(value=str(default_threads))
    ttsmaxlen_var = ctk.StringVar(value=str(default_ttsmaxlen))
    tts_dir_var = ctk.StringVar()

    musicllm_var = ctk.StringVar()
    musicembeddings_var = ctk.StringVar()
    musicdiffusion_var = ctk.StringVar()
    musicvae_var = ctk.StringVar()
    musiclowvram_var = ctk.IntVar(value=0)

    embeddings_model_var = ctk.StringVar()
    embeddings_ctx_var = ctk.StringVar(value=str(default_embeddingsmaxctx))
    embeddings_gpu_var = ctk.IntVar(value=0)

    admin_var = ctk.IntVar(value=0)
    admin_dir_var = ctk.StringVar()
    baseconfig_var = ctk.StringVar()
    admin_password_var = ctk.StringVar()
    singleinstance_var = ctk.IntVar(value=0)
    router_mode_var = ctk.IntVar(value=0)
    autoswap_mode_var = ctk.IntVar(value=0)
    admin_unload_timeout_var = ctk.StringVar(value=str(0))

    nozenity_var = ctk.IntVar(value=0)

    curr_tab_idx = 0

    def tabbuttonaction(name):
        nonlocal curr_tab_idx
        idx = 0
        for t in tabcontent:
            if name == t:
                tabcontent[t].grid(row=0, column=0)
                navbuttons[t].configure(fg_color="#6f727b")
                curr_tab_idx = idx
            else:
                tabcontent[t].grid_remove()
                navbuttons[t].configure(fg_color="transparent")
            idx += 1

    # Dynamically create tabs + buttons based on values of [tabnames]
    for idx, name in enumerate(tabnames):
        tabcontent[name] = ctk.CTkFrame(tabcontentframe, width=int(tabcontentframe.cget("width")), height=int(tabcontentframe.cget("height")), fg_color="transparent")
        tabcontent[name].grid_propagate(False)
        if idx == 0:
            tabcontent[name].grid(row=idx, sticky="nsew")
        ctk.CTkLabel(tabcontent[name], text= name, font=ctk.CTkFont(None, 14, 'bold')).grid(row=0, padx=12, pady = 5, stick='nw')

        navbuttons[name] = ctk.CTkButton(navbuttonframe, text=name, width = 100, corner_radius=0 , command = lambda d=name:tabbuttonaction(d), hover_color="#868a94" )
        navbuttons[name].grid(row=idx)

    tabbuttonaction(tabnames[0])
    # Quick Launch Tab
    quick_tab = tabcontent["Quick Launch"]

    # helper functions
    def makecheckbox(parent, text, variable=None, row=0, column=0, command=None, padx=8,tooltiptxt=""):
        temp = ctk.CTkCheckBox(parent, text=text,variable=variable, onvalue=1, offvalue=0)
        if command is not None and variable is not None:
            variable.trace_add("write", command)
        temp.grid(row=row,column=column, padx=padx, pady=1, stick="nw")
        if tooltiptxt!="":
            temp.bind("<Enter>", lambda event: show_tooltip(event, tooltiptxt))
            temp.bind("<Leave>", hide_tooltip)
        return temp

    def makelabelcombobox(parent, text, variable=None, row=0, width=50, command=None, padx=8,tooltiptxt="", values=[], labelpadx=8):
        label = makelabel(parent, text, row, 0, tooltiptxt, padx=labelpadx)
        combo = ctk.CTkComboBox(parent, variable=variable, width=width, values=values, state="readonly")
        if command is not None and variable is not None:
            variable.trace_add("write", command)
        combo.grid(row=row,column=0, padx=padx, sticky="nw")
        if tooltiptxt!="":
            combo.bind("<Enter>", lambda event: show_tooltip(event, tooltiptxt))
            combo.bind("<Leave>", hide_tooltip)
        return combo, label

    def makelabel(parent, text, row, column=0, tooltiptxt="", columnspan=1, padx=8):
        temp = ctk.CTkLabel(parent, text=text)
        temp.grid(row=row, column=column, padx=padx, pady=1, stick="nw", columnspan=columnspan)
        if tooltiptxt!="":
            temp.bind("<Enter>", lambda event: show_tooltip(event, tooltiptxt))
            temp.bind("<Leave>", hide_tooltip)
        return temp

    def makeslider(parent, label, options, var, row=0, width=160, height=10, set=0, tooltip=""):
        sliderLabel = makelabel(parent, options[set], row + 1, 0, columnspan=2, padx=(width+12))
        titleLabel = makelabel(parent, label, row,0,tooltip)
        from_ = 0
        to = len(options)-1
        def sliderUpdate(a,b,c):
            sliderLabel.configure(text = options[int(var.get())])
        var.trace_add("write", sliderUpdate)
        slider = ctk.CTkSlider(parent, from_=from_, to=to, variable = var, width = width, height=height, border_width=5,number_of_steps=len(options) - 1)
        slider.grid(row=row+1,  column=0, padx = 8, stick="w", columnspan=2)
        slider.set(set)
        return slider, sliderLabel, titleLabel


    def makelabelentry(parent, text, var, row=0, width=50, padx=8, singleline=False, tooltip="", labelpadx=8):
        label = makelabel(parent, text, row, 0, tooltip, padx=labelpadx)
        entry = ctk.CTkEntry(parent, width=width, textvariable=var)
        entry.grid(row=row, column=(0 if singleline else 1), padx=padx, pady=1, sticky="nw")
        return entry, label

    #file dialog types: 0=openfile,1=savefile,2=opendir
    def makefileentry(parent, text, searchtext, var, row=0, width=200, filetypes=[], onchoosefile=None, singlerow=False, singlecol=True, dialog_type=0, tooltiptxt="", multiple=False):
        label = makelabel(parent, text, row,0,tooltiptxt,columnspan=3)
        def getfilename(var, text):
            initialDir = os.path.dirname(var.get())
            initialDir = initialDir if os.path.isdir(initialDir) else None
            fnam = None
            if dialog_type==2:
                fnam = zentk_askdirectory(title=text, mustexist=True, initialdir=initialDir)
            elif dialog_type==1:
                fnam = zentk_asksaveasfilename(title=text, filetypes=filetypes, defaultextension=filetypes, initialdir=initialDir)
                if not fnam:
                    fnam = ""
                else:
                    fnam = str(fnam).strip()
                    fnam = f"{fnam}.jsondb" if ".jsondb" not in fnam.lower() else fnam
            else:
                if multiple:
                    fnam = zentk_askopenfilenames(title=text,filetypes=filetypes, initialdir=initialDir)
                    fnam = "|".join(fnam)
                else:
                    fnam = zentk_askopenfilename(title=text,filetypes=filetypes, initialdir=initialDir)
            if fnam:
                var.set(fnam)
                if onchoosefile:
                    onchoosefile(var.get())
        entry = ctk.CTkEntry(parent, width, textvariable=var)
        button = ctk.CTkButton(parent, 50, text="Browse", command= lambda a=var,b=searchtext:getfilename(a,b))
        if singlerow:
            if singlecol:
                entry.grid(row=row, column=0, padx=((94)+8), pady=2, stick="w")
                button.grid(row=row, column=0, padx=((94)+width+12), pady=2, stick="w")
            else:
                entry.grid(row=row, column=1, padx=8, pady=2, stick="w")
                button.grid(row=row, column=1, padx=(width+12), pady=2, stick="w")
        else:
            if singlecol:
                entry.grid(row=row+1, column=0, columnspan=3, padx=8, pady=2, stick="w")
                button.grid(row=row+1, column=0, columnspan=3, padx=(width+12), pady=2, stick="w")
            else:
                entry.grid(row=row+1, column=0, columnspan=1, padx=8, pady=2, stick="w")
                button.grid(row=row+1, column=1, columnspan=1, padx=8, pady=2, stick="w")
        return label, entry, button

    def model_searcher():
        searchbox1 = None
        searchbox2 = None
        modelsearch1_var = ctk.StringVar(value="")
        modelsearch2_var = ctk.StringVar(value="")
        fileinfotxt_var = ctk.StringVar(value="")
        # Create popup window
        popup = ctk.CTkToplevel(root)
        popup.title("Model File Browser")
        popup.geometry("400x400")
        searchedmodels = []
        searchedsizes = []

        def confirm_search_model_choice():
            nonlocal modelsearch1_var, modelsearch2_var, model_var, fileinfotxt_var
            if modelsearch1_var.get()!="" and modelsearch2_var.get()!="":
                model_var.set(f"https://huggingface.co/{modelsearch1_var.get()}/resolve/main/{modelsearch2_var.get()}")
            popup.destroy()
        def update_search_quant_file_size(a,b,c):
            nonlocal modelsearch1_var, modelsearch2_var, fileinfotxt_var, searchedmodels, searchedsizes, searchbox2
            try:
                selected_index = searchbox2.cget("values").index(modelsearch2_var.get())
                pickedsize = searchedsizes[selected_index]
                fileinfotxt_var.set(f"Size: {round(pickedsize/1024/1024/1024,2)} GB")
            except Exception:
                fileinfotxt_var.set("")
        def fetch_search_quants(a,b,c):
            nonlocal modelsearch1_var, modelsearch2_var, fileinfotxt_var, searchedmodels, searchedsizes
            try:
                if modelsearch1_var.get()=="":
                    return
                searchedmodels = []
                searchedsizes = []
                resp = make_url_request(f"https://huggingface.co/api/models/{modelsearch1_var.get()}/tree/main?recursive=true",None,'GET',{},10)
                for m in resp:
                    if m["type"]=="file" and ".gguf" in m["path"]:
                        if "-of-0" in m["path"] and "00001" not in m["path"]:
                            continue
                        searchedmodels.append(m["path"])
                        searchedsizes.append(m["size"])
                searchbox2.configure(values=searchedmodels)
                if len(searchedmodels)>0:
                    quants = ["q4k","q4_k","q4", "q3", "q5", "q6", "q8"] #autopick priority
                    chosen_model = searchedmodels[0]
                    found_good = False
                    for quant in quants:
                        for filename in searchedmodels:
                            if quant in filename.lower():
                                chosen_model = filename
                                found_good = True
                                break
                        if found_good:
                            break
                    modelsearch2_var.set(chosen_model)
                    update_search_quant_file_size(1,1,1)
                else:
                    modelsearch2_var.set("")
                    fileinfotxt_var.set("")
            except Exception as e:
                modelsearch1_var.set("")
                modelsearch2_var.set("")
                fileinfotxt_var.set("")
                print(f"Error: {e}")
        def fetch_search_models():
            from tkinter import messagebox
            nonlocal searchbox1, searchbox2, modelsearch1_var, modelsearch2_var, fileinfotxt_var
            try:
                modelsearch1_var.set("")
                modelsearch2_var.set("")
                fileinfotxt_var.set("")
                searchbox1.configure(values=[])
                searchbox2.configure(values=[])
                searchedmodels = []
                searchbase = model_search.get()
                if searchbase.strip()=="":
                    return
                urlcode = urllib.parse.urlencode({"search":( "GGUF " + searchbase),"limit":10}, doseq=True)
                urlcode2 = urllib.parse.urlencode({"search":searchbase,"limit":6}, doseq=True)
                resp = make_url_request(f"https://huggingface.co/api/models?{urlcode}",None,'GET',{},10)
                for m in resp:
                    searchedmodels.append(m["id"])
                if len(resp)<=3: #too few results, repeat search without GGUF in the string
                    resp2 = make_url_request(f"https://huggingface.co/api/models?{urlcode2}",None,'GET',{},10)
                    for m in resp2:
                        searchedmodels.append(m["id"])

                if len(searchedmodels)==0:
                    messagebox.showinfo("No Results Found", "Search found no results")
                searchbox1.configure(values=searchedmodels)
                if len(searchedmodels)>0:
                    modelsearch1_var.set(searchedmodels[0])
                else:
                    modelsearch1_var.set("")
            except Exception as e:
                modelsearch1_var.set("")
                modelsearch2_var.set("")
                fileinfotxt_var.set("")
                print(f"Error: {e}")

        ctk.CTkLabel(popup, text="Enter Search String:").pack(pady=(10, 0))
        model_search = ctk.CTkEntry(popup, width=300)
        model_search.pack(pady=5)
        model_search.insert(0, "")

        ctk.CTkButton(popup, text="Search Huggingface", command=fetch_search_models).pack(pady=5)

        ctk.CTkLabel(popup, text="Selected Model:").pack(pady=(10, 0))
        searchbox1 = ctk.CTkComboBox(popup, values=[], width=340, variable=modelsearch1_var, state="readonly")
        searchbox1.pack(pady=5)
        ctk.CTkLabel(popup, text="Selected Quant:").pack(pady=(10, 0))
        searchbox2 = ctk.CTkComboBox(popup, values=[], width=340, variable=modelsearch2_var, state="readonly")
        searchbox2.pack(pady=5)
        modelsearch1_var.trace_add("write", fetch_search_quants)
        modelsearch2_var.trace_add("write", update_search_quant_file_size)
        ctk.CTkLabel(popup, text="", textvariable=fileinfotxt_var, text_color="#ffff00").pack(pady=(10, 0))
        ctk.CTkButton(popup, text="Confirm Selection", command=confirm_search_model_choice).pack(pady=5)

        popup.transient(root)

    # decided to follow yellowrose's and kalomaze's suggestions, this function will automatically try to determine GPU identifiers
    # run in new thread so it doesnt block. does not return anything, instead overwrites specific values and redraws GUI
    def auto_set_backend_gui(manual_select=False):
        global exitcounter, runmode_untouched
        if manual_select:
            print("\nA .kcppt template was selected from GUI - automatically selecting your backend...")
            runmode_untouched = True
        fetch_gpu_properties(True,True)
        found_new_backend = False

        # check for avx2 and avx support
        is_oldpc_ver = "Use CPU" not in runopts #on oldcpu ver, default lib does not exist
        cpusupport = old_cpu_check() # 0 if has avx2, 1 if has avx, 2 if has nothing
        eligible_cuda = (cpusupport<1 and not is_oldpc_ver) or (cpusupport<2 and is_oldpc_ver)

        #autopick cublas if suitable, requires at least 3.5GB VRAM to auto pick
        #we do not want to autoselect hip/cublas if the user has already changed their desired backend!
        if eligible_cuda and exitcounter < 100 and MaxMemory[0]>3500000000 and runmode_untouched and (("Use CUDA" in runopts and CUDevicesNames[0]!="") or "Use hipBLAS (ROCm)" in runopts) and (any(CUDevicesNames)):
            if "Use CUDA" in runopts:
                runopts_var.set("Use CUDA")
                gpu_choice_var.set("0")
                print(f"Auto Selected CUDA Backend (flag={cpusupport})\n")
                found_new_backend = True
            elif "Use hipBLAS (ROCm)" in runopts:
                runopts_var.set("Use hipBLAS (ROCm)")
                gpu_choice_var.set("0")
                print(f"Auto Selected HIP Backend (flag={cpusupport})\n")
                found_new_backend = True
        elif exitcounter < 100 and (1 in VKIsDGPU) and runmode_untouched and ("Use Vulkan" in runopts or "Use Vulkan (Old CPU)" in runopts):
            for i in range(0,len(VKIsDGPU)):
                if VKIsDGPU[i]==1:
                    if cpusupport<1 and "Use Vulkan" in runopts:
                        runopts_var.set("Use Vulkan")
                    else:
                        runopts_var.set("Use Vulkan (Old CPU)")
                    gpu_choice_var.set(str(i))
                    print(f"Auto Selected Vulkan Backend (flag={cpusupport})\n")
                    found_new_backend = True
                    break
        else:
            if runopts_var.get()=="Use CPU" and cpusupport==1 and "Use CPU (Old CPU)" in runopts:
                runopts_var.set("Use CPU (Old CPU)")
            elif runopts_var.get()=="Use CPU" and cpusupport==2 and "Failsafe Mode (Older CPU)" in runopts:
                runopts_var.set("Failsafe Mode (Older CPU)")
        if not found_new_backend:
            print(f"Auto Selected Default Backend (flag={cpusupport})\n")
        changed_gpu_choice_var()

    def on_picked_model_file(filepath):
        if filepath and (filepath.lower().endswith('.kcpps') or filepath.lower().endswith('.kcppt')):
            #load it as a config file instead
            if filepath.lower().endswith('.kcpps'):
                global runmode_untouched
                runmode_untouched = False
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                dict = json.load(f)
                import_vars(dict)

    def gui_changed_modelfile(*args):
        global importvars_in_progress
        if not importvars_in_progress:
            filepath = model_var.get()
            sdfilepath = sd_model_var.get()
            whisperfilepath = whisper_model_var.get()
            mmprojfilepath = mmproj_var.get()
            draftmodelpath = draftmodel_var.get()
            ttsmodelpath = tts_model_var.get() if ttsgpu_var.get()==1 else ""
            embdmodelpath = embeddings_model_var.get() if embeddings_gpu_var.get()==1 else ""
            musicllmpath = musicllm_var.get()
            musicditpath = musicdiffusion_var.get()
            extract_modelfile_params(filepath,sdfilepath,whisperfilepath,mmprojfilepath,draftmodelpath,ttsmodelpath,embdmodelpath,musicllmpath,musicditpath)
            changed_gpulayers_estimate()
        pass

    def changed_autofit(*args):
        global runmode_untouched
        orig_rmu = runmode_untouched
        update_runmode_gui()
        runmode_untouched = orig_rmu
        changed_gpulayers_estimate()

    def changed_gpulayers_estimate(*args):
        autoset_gpu_layers(int(contextsize_text[context_var.get()]),sd_quant_option(sd_quant_var.get()),int(batchsize_values[int(blas_size_var.get())]),musiclowvram_var.get()==1)
        max_gpu_layers = (f"{modelfile_extracted_meta[1][0]+1}" if (modelfile_extracted_meta and modelfile_extracted_meta[1] and modelfile_extracted_meta[1][0]!=0) else "")
        index = runopts_var.get()
        gpu_be = (index == "Use Vulkan" or index == "Use Vulkan (Old CPU)" or index == "Use Vulkan (Older CPU)" or index == "Use CUDA" or index == "Use hipBLAS (ROCm)")
        layercounter_label.grid(row=6, column=0, padx=230, sticky="W")
        quick_layercounter_label.grid(row=6, column=1, padx=75, sticky="W")
        if sys.platform=="darwin" and gpulayers_var.get()=="-1" and max_gpu_layers:
            quick_layercounter_label.configure(text=f"(Auto) ({max_gpu_layers} Total Layers)")
            layercounter_label.configure(text=f"(Auto) ({max_gpu_layers} Total Layers)")
        elif gpu_be and gpulayers_var.get()=="-1" and max_gpu_layers:
            quick_layercounter_label.configure(text=f"(Auto) ({max_gpu_layers} Total Layers)")
            layercounter_label.configure(text=f"(Auto) ({max_gpu_layers} Total Layers)")
        elif gpu_be and gpulayers_var.get()=="":
            quick_layercounter_label.configure(text="(Set -1 for Auto)")
            layercounter_label.configure(text="(Set -1 for Auto)")
        else:
            layercounter_label.grid_remove()
            quick_layercounter_label.grid_remove()

        if autofit_var.get()==1:
            layercounter_label.grid_remove()
            quick_layercounter_label.grid_remove()

    def changed_gpu_choice_var(*args):
        global exitcounter
        if exitcounter > 100:
            return
        if gpu_choice_var.get()!="All":
            try:
                s = int(gpu_choice_var.get())
                v = runopts_var.get()
                if v == "Use Vulkan" or v == "Use Vulkan (Old CPU)" or v == "Use Vulkan (Older CPU)":
                    quick_gpuname_label.configure(text=VKDevicesNames[s])
                    gpuname_label.configure(text=VKDevicesNames[s])
                else:
                    quick_gpuname_label.configure(text=CUDevicesNames[s])
                    gpuname_label.configure(text=CUDevicesNames[s])
            except Exception:
                pass
        else:
            quick_gpuname_label.configure(text="(dGPUs only, tensor split sets ratio)")
            gpuname_label.configure(text="(dGPUs only, tensor split sets ratio)")

    gpu_choice_var.trace_add("write", changed_gpu_choice_var)
    gpulayers_var.trace_add("write", changed_gpulayers_estimate)

    def toggleswa(a,b,c):
        if swa_var.get()==1:
            swa_padding_entry.grid()
            swa_padding_label.grid()
        else:
            swa_padding_entry.grid_remove()
            swa_padding_label.grid_remove()

    def togglesmartcache(a,b,c):
        if smartcache_var.get()==1:
            fastforward_var.set(1)

    def togglefastforward(a,b,c):
        if fastforward_var.get()==0:
            contextshift_var.set(0)
            smartcontext_var.set(0)
            smartcache_var.set(0)

    def togglectxshift(a,b,c):
        if contextshift_var.get()==0:
            smartcontextbox.grid()
        else:
            fastforward_var.set(1)
            smartcontextbox.grid_remove()
        qkvslider.grid()
        qkvlabel.grid()
        if flashattention_var.get()==0 and (quantkv_var.get()>1):
            noqkvlabel.grid()
        else:
            noqkvlabel.grid_remove()


    def toggleflashattn(a,b,c):
        qkvslider.grid()
        qkvlabel.grid()
        if flashattention_var.get()==0 and (quantkv_var.get()>1):
            noqkvlabel.grid()
        else:
            noqkvlabel.grid_remove()
        changed_gpulayers_estimate()

    def guibench():
        args.benchmark = "stdout"
        launchbrowser.set(0)
        guilaunch()

    def changerunmode(a,b,c):
        global runmode_untouched
        runmode_untouched = False
        update_runmode_gui()

    def update_runmode_gui():
        index = runopts_var.get()
        if index == "Use Vulkan" or index == "Use Vulkan (Old CPU)" or index == "Use Vulkan (Older CPU)" or index == "Use CUDA" or index == "Use hipBLAS (ROCm)":
            quick_gpuname_label.grid(row=3, column=1, padx=75, sticky="W")
            gpuname_label.grid(row=3, column=0, padx=230, sticky="W")
            gpu_selector_label.grid(row=3, column=0, padx = 8, pady=1, stick="nw")
            quick_gpu_selector_label.grid(row=3, column=0, padx = 8, pady=1, stick="nw")
            CUDA_gpu_selector_box.grid(row=3, column=0, padx=160, pady=1, stick="nw")
            CUDA_quick_gpu_selector_box.grid(row=3, column=1, padx=8, pady=1, stick="nw")
            maingpu_label.grid(row=8, column=0, padx = 270, pady=1, stick="nw")
            maingpu_entry.grid(row=8, column=0, padx = 340, pady=1, stick="nw")
            lowvram_box.grid(row=4, column=0, padx=8, pady=1, stick="nw")
            splitmode_box.grid(row=4, column=0, padx=230, pady=1, stick="nw")
            splitmode_box_label.grid(row=4, column=0, padx=160, pady=1, stick="nw")
        else:
            quick_gpuname_label.grid_remove()
            gpuname_label.grid_remove()
            gpu_selector_label.grid_remove()
            CUDA_gpu_selector_box.grid_remove()
            quick_gpu_selector_label.grid_remove()
            CUDA_quick_gpu_selector_box.grid_remove()
            maingpu_label.grid_remove()
            maingpu_entry.grid_remove()
            lowvram_box.grid_remove()
            splitmode_box.grid_remove()
            splitmode_box_label.grid_remove()

        if index == "Use CUDA" or index == "Use hipBLAS (ROCm)":
            mmq_box.grid(row=4, column=0, padx=340, pady=1,  stick="nw")
            quick_mmq_box.grid(row=4, column=1, padx=8, pady=1,  stick="nw")
        else:
            mmq_box.grid_remove()
            quick_mmq_box.grid_remove()

        if index == "Use Vulkan" or index == "Use Vulkan (Old CPU)" or index == "Use Vulkan (Older CPU)" or index == "Use CUDA" or index == "Use hipBLAS (ROCm)" or rpcmode_var.get()=="connect":
            gpu_layers_label.grid(row=6, column=0, padx=8, pady=1, stick="nw")
            gpu_layers_entry.grid(row=6, column=0, padx=160, pady=1, stick="nw")
            quick_gpu_layers_label.grid(row=6, column=0, padx = 8, pady=1, stick="nw")
            quick_gpu_layers_entry.grid(row=6, column=1, padx=8, pady=1, stick="nw")
            tensor_split_label.grid(row=8, column=0, padx = 8, pady=1, stick="nw")
            tensor_split_entry.grid(row=8, column=0, padx = 160, pady=1, stick="nw")
        elif sys.platform=="darwin":
            gpu_layers_label.grid(row=6, column=0, padx=8, pady=1, stick="nw")
            gpu_layers_entry.grid(row=6, column=0, padx=160, pady=1, stick="nw")
            quick_gpu_layers_label.grid(row=6, column=0, padx = 8, pady=1, stick="nw")
            quick_gpu_layers_entry.grid(row=6, column=1, padx=8, pady=1, stick="nw")
            tensor_split_label.grid_remove()
            tensor_split_entry.grid_remove()
        else:
            gpu_layers_label.grid_remove()
            gpu_layers_entry.grid_remove()
            quick_gpu_layers_label.grid_remove()
            quick_gpu_layers_entry.grid_remove()
            tensor_split_label.grid_remove()
            tensor_split_entry.grid_remove()

        if autofit_var.get()==1:
            gpu_layers_label.grid_remove()
            gpu_layers_entry.grid_remove()
            quick_gpu_layers_label.grid_remove()
            quick_gpu_layers_entry.grid_remove()
            autofit_padding_label.grid(row=6, column=0, padx=8, pady=1, stick="nw")
            autofit_padding_entry.grid(row=6, column=0, padx=160, pady=1, stick="nw")
            moecpu_box.grid_remove()
            tenos_box.grid_remove()
            moecpu_box_lbl.grid_remove()
            tenos_box_lbl.grid_remove()
        else:
            autofit_padding_label.grid_remove()
            autofit_padding_entry.grid_remove()
            moecpu_box.grid()
            tenos_box.grid()
            moecpu_box_lbl.grid()
            tenos_box_lbl.grid()

        changed_gpulayers_estimate()
        changed_gpu_choice_var()

    # presets selector
    makelabel(quick_tab, "Backend:", 1,0,"Select a backend to use.\nCUDA runs on Nvidia GPUs, and is much faster.\nVulkan works on all GPUs but is somewhat slower.\nOtherwise, runs on CPU only.\nNoAVX2 and Failsafe modes support older PCs.")

    runoptbox = ctk.CTkComboBox(quick_tab, values=runopts, width=190,variable=runopts_var, state="readonly")
    runoptbox.grid(row=1, column=1,padx=8, stick="nw")
    runoptbox.set(runopts[0]) # Set to first available option

    # gpu options
    quick_gpu_selector_label = makelabel(quick_tab, "GPU ID:", 3,0,"Which GPU ID to load the model with.\nNormally your main GPU is #1, but it can vary for multi GPU setups.",padx=8)
    CUDA_quick_gpu_selector_box = ctk.CTkComboBox(quick_tab, values=CUDevices, width=60, variable=gpu_choice_var, state="readonly")
    CUDA_quick_gpu_selector_box.grid(row=3, column=1, padx=8, pady=1, stick="nw")
    quick_gpuname_label = ctk.CTkLabel(quick_tab, text="")
    quick_gpuname_label.grid(row=3, column=1, padx=75, sticky="W")
    quick_gpuname_label.configure(text_color="#ffff00")
    quick_gpu_layers_entry,quick_gpu_layers_label = makelabelentry(quick_tab,"GPU Layers:", gpulayers_var, 6, 50,tooltip="How many layers to offload onto the GPU.\nUsage varies based on model type and increases with model and context size.\nRequires some trial and error to find the best fit value.\n\nNote: The auto estimation is often inaccurate! Please set layers yourself for best results!")
    quick_gpu_layers_label.grid(row=6, column=0, padx = 8, pady=1, stick="nw")
    quick_gpu_layers_entry.grid(row=6, column=1, padx=8, pady=1, stick="nw")
    quick_layercounter_label = ctk.CTkLabel(quick_tab, text="")
    quick_layercounter_label.grid(row=6, column=1, padx=75, sticky="W")
    quick_layercounter_label.configure(text_color="#ffff00")
    quick_mmq_box = makecheckbox(quick_tab,  "Use MMQ", mmq_var, 4,1,tooltiptxt="Enable MMQ mode instead of CuBLAS for prompt processing. Read the wiki. Speed may vary.")

    # quick boxes
    quick_boxes = {
        "Launch Browser": [launchbrowser, "Launches your default browser after model loading is complete"],
        "Use MMAP": [usemmap,  "Use mmap to load models if enabled, model will not be unloadable"],
        "Use ContextShift": [contextshift_var, "Uses Context Shifting to reduce reprocessing.\nRecommended. Check the wiki for more info."],
        "Remote Tunnel": [remotetunnel_var,  "Creates a trycloudflare tunnel.\nAllows you to access koboldcpp from other devices over an internet URL."],
        "Use FlashAttention": [flashattention_var, "Enable flash attention for GGUF models."],
        "Force AutoFit": [autofit_var, "Automatically attempt to fit the model in the best possible way. Overrides everything else.\nNot recommended for multi model setups. Experimental."],
        "Quiet Mode": [quietmode, "Prevents all generation related terminal output from being displayed."]
    }

    for idx, (name, properties) in enumerate(quick_boxes.items()):
        makecheckbox(quick_tab, name, properties[0], int(idx/2) + 20, idx % 2, tooltiptxt=properties[1])

    # context size
    makeslider(quick_tab, "Context Size:", contextsize_text, context_var, 40, width=280, set=9, tooltip="What is the maximum context size to support. Model specific. You cannot exceed it.\nLarger contexts require more memory, and not all models support it.")

    # load model
    makefileentry(quick_tab, "GGUF Text Model:", "Select GGUF or GGML Model File", model_var, 50, 280, onchoosefile=on_picked_model_file,tooltiptxt="Select a GGUF or GGML model file on disk to be loaded.")
    model_var.trace_add("write", gui_changed_modelfile)
    ctk.CTkButton(quick_tab, width=70, text = "HF Search", command = model_searcher ).grid(row=51,column=1, stick="sw", padx=184, pady=2)

    # Hardware Tab
    hardware_tab = tabcontent["Hardware"]

    # presets selector
    makelabel(hardware_tab, "Backend:", 1,0,"Select a backend to use.\nCUDA runs on Nvidia GPUs, and is much faster.\nVulkan works on all GPUs but is somewhat slower.\nOtherwise, runs on CPU only.\nNoAVX2 and Failsafe modes support older PCs.")
    runoptbox = ctk.CTkComboBox(hardware_tab, values=runopts,  width=180,variable=runopts_var, state="readonly")
    runoptbox.grid(row=1, column=0,padx=160, stick="nw")
    runoptbox.set(runopts[0]) # Set to first available option

    # gpu options
    gpu_selector_label = makelabel(hardware_tab, "GPU ID:", 3,0,"Which GPU ID to load the model with.\nNormally your main GPU is #1, but it can vary for multi GPU setups.")
    CUDA_gpu_selector_box = ctk.CTkComboBox(hardware_tab, values=CUDevices, width=60, variable=gpu_choice_var, state="readonly")
    CUDA_gpu_selector_box.grid(row=3, column=0, padx=160, pady=1, stick="nw")
    gpuname_label = ctk.CTkLabel(hardware_tab, text="")
    gpuname_label.grid(row=3, column=0, padx=230, sticky="W")
    gpuname_label.configure(text_color="#ffff00")
    lowvram_box = makecheckbox(hardware_tab,  "No KV offload", lowvram_var, 4,0, tooltiptxt='Avoid offloading KV Cache or scratch buffers to VRAM.\nAllows more layers to fit, but may result in a large speed loss.')
    mmq_box = makecheckbox(hardware_tab,  "Use MMQ", mmq_var, 4,0,padx=340, tooltiptxt="Enable MMQ mode to use finetuned kernels instead of default CuBLAS/HipBLAS for prompt processing.\nRead the wiki. Speed may vary.")
    splitmode_box,splitmode_box_label = makelabelcombobox(hardware_tab, "SplitMode: ", splitmode_var, 4, width=(80), padx=(230), labelpadx=160, tooltiptxt="How to split the model across multiple GPUs. Layer split is default.", values=splitmode_choices)
    gpu_layers_entry,gpu_layers_label = makelabelentry(hardware_tab,"GPU Layers:", gpulayers_var, 6, 50, padx=160,singleline=True,tooltip="How many layers to offload onto the GPU.\nUsage varies based on model type and increases with model and context size.\nRequires some trial and error to find the best fit value.\n\nNote: The auto estimation is often inaccurate! Please set layers yourself for best results!")
    autofit_padding_entry,autofit_padding_label = makelabelentry(hardware_tab,"Autofit Padding (MB):", autofit_padding_var, 6, 50, padx=160,singleline=True,tooltip="How much spare allowance in MB should autofit reserve? If it's too little, the load might fail.")
    layercounter_label = ctk.CTkLabel(hardware_tab, text="")
    layercounter_label.grid(row=6, column=0, padx=230, sticky="W")
    layercounter_label.configure(text_color="#ffff00")
    tensor_split_entry,tensor_split_label = makelabelentry(hardware_tab, "Tensor Split:", tensor_split_str_vars, 8, 80, padx=160, singleline=True, tooltip='When using multiple GPUs this option controls how large tensors should be split across all GPUs.\nUses a comma-separated list of non-negative values that assigns the proportion of data that each GPU should get in order.\nFor example, "3,2" will assign 60% of the data to GPU 0 and 40% to GPU 1.')
    maingpu_entry,maingpu_label = makelabelentry(hardware_tab, "Main GPU:" , maingpu_var, 8, 50,padx=340,singleline=True,tooltip="Only for multi-gpu, which GPU ID to set as main?\nIf left blank or -1, uses default value.",labelpadx=270)

    # threads
    makelabelentry(hardware_tab, "Threads:" , threads_var, 11, 50, padx=160, singleline=True,tooltip="How many threads to use.\nRecommended value is your CPU core count, defaults are usually OK.")
    # blas thread specifier
    makelabelentry(hardware_tab, "Batch Threads:" , blas_threads_var, 11, 50,padx=340, singleline=True,tooltip="How many threads to use during batched processing.\nIf left blank, uses same value as regular thread count.",labelpadx=240)
    makelabelentry(hardware_tab, "Device Override", deviceoverride_var, 15, 120, padx=(160), singleline=True, tooltip="Set llama.cpp compatible device selection override. Comma separated (e.g. Vulkan0,Vulkan1). Overrides normal device choices.")

    # hardware checkboxes
    hardware_boxes = {
        "Launch Browser": [launchbrowser, "Launches your default browser after model loading is complete"],
        "High Priority": [highpriority, "Increases the koboldcpp process priority.\nMay cause lag or slowdown instead. Not recommended."],
        "Use MMAP": [usemmap, "Use mmap to load models if enabled, model will not be unloadable"],
        "Use mlock": [usemlock, "Enables mlock, preventing the RAM used to load the model from being paged out."],
        "Debug Mode": [debugmode, "Enables debug mode, with extra info printed to the terminal."],
        "Keep Foreground": [keepforeground, "Bring KoboldCpp to the foreground every time there is a new generation."],
        "CLI Terminal Only": [terminalonly, "Does not launch KoboldCpp HTTP server. Instead, enables KoboldCpp from the command line, accepting interactive console input and displaying responses to the terminal."],
        "Pipeline Parallel": [pipelineparallel, "Enable Pipeline Parallelism for faster multigpu speeds but using more memory, only active for multigpu."],
    }

    for idx, (name, properties) in enumerate(hardware_boxes.items()):
        makecheckbox(hardware_tab, name, properties[0], int(idx/2) + 30, 0, padx=(160 if idx % 2 else 8), tooltiptxt=properties[1])

    # blas batch size
    makeslider(hardware_tab, "Batch Size:", batchsize_text, blas_size_var, 16,width=200, set=6,tooltip="How many tokens to process at once per batch.\nLarger values use more memory.")
    blas_size_var.trace_add("write", changed_gpulayers_estimate)

    makecheckbox(hardware_tab, "Use FlashAttention", flashattention_var, 100, command=toggleflashattn,  tooltiptxt="Enable flash attention for GGUF models.")

    makecheckbox(hardware_tab, "Force AutoFit", autofit_var, 100,0,command=changed_autofit,padx=160, tooltiptxt="Automatically attempt to fit the model in the best possible way. Overrides everything else.\nNot recommended for multi model setups. Experimental.")
    ctk.CTkButton(hardware_tab , text = "Run Benchmark", command = guibench ).grid(row=110,column=0, stick="nw", padx= 8, pady=2)


    # Context Tab
    context_tab = tabcontent["Context"]
    # Context checkboxes
    smartcontextbox = makecheckbox(context_tab, "Use SmartContext", smartcontext_var, 3, padx=330,tooltiptxt="Uses SmartContext. Now considered outdated and not recommended.\nCheck the wiki for more info.")
    makecheckbox(context_tab, "Use ContextShift", contextshift_var, 3,padx=180,tooltiptxt="Uses Context Shifting to reduce reprocessing.\nRecommended. Check the wiki for more info.", command=togglectxshift)
    makecheckbox(context_tab, "Use FastForwarding", fastforward_var, 3,tooltiptxt="Use fast forwarding to recycle previous context (always reprocess if disabled).\nRecommended.", command=togglefastforward)
    makecheckbox(context_tab, "Allow SWA", swa_var, 4,tooltiptxt="SWA will be enabled automatically on models that support it. SWA saves memory but cannot be used with context shifting.", command=toggleswa)
    swa_padding_entry,swa_padding_label = makelabelentry(context_tab,"SWA Padding Tokens:", swa_padding_var, 4, 50, padx=300,singleline=True,tooltip="If the SWA is too small, you can expand it with padding, allowing for greater distance context rewinds.",labelpadx=160)
    makecheckbox(context_tab, "Use SmartCache", smartcache_var, 5,tooltiptxt="Enables intelligent context switching by saving KV cache snapshots to RAM. Requires fast forwarding.", command=togglesmartcache)
    makelabelentry(context_tab, "CacheSlots:", smartcacheslots_var, row=5, padx=(300), singleline=True, tooltip="Number of slots for smartcache",labelpadx=(220))

    # context size
    makeslider(context_tab, "Context Size:",contextsize_text, context_var, 18, width=280, set=9,tooltip="What is the maximum context size to support. Model specific. You cannot exceed it.\nLarger contexts require more memory, and not all models support it.")
    context_var.trace_add("write", changed_gpulayers_estimate)
    makelabelentry(context_tab, "Default Gen Amt:", defaultgenamt_var, row=20, padx=(120), singleline=True, tooltip="How many tokens to generate by default, if not specified. Must be smaller than context size. Usually, your frontend GUI will override this.")
    makelabelentry(context_tab, "Prompt Limit:", genlimit_var, row=20, padx=(300), singleline=True, tooltip="If set, restricts max output tokens to this limit regardless of API request. Set to 0 to disable.",labelpadx=(210))
    makelabelentry(context_tab, "Default Params:", gen_defaults_var, row=21, width=200, padx=(110), singleline=True, tooltip='Set default generation parameters for incoming API payloads.\nSpecified as JSON fields: {"KEY1":"VALUE1", "KEY2":"VALUE2"...}')
    makecheckbox(context_tab, "Override", gen_defaults_overwrite_var, row=21,padx=(330), tooltiptxt="Allow the gendefaults parameters to overwrite the original value in API payloads.")

    nativectx_entry, nativectx_label = makelabelentry(context_tab, "Override Native Context:", customrope_nativectx, row=23, padx=(146), singleline=True, tooltip="Overrides the native trained context of the loaded model with a custom value to be used for Rope scaling.")
    customrope_scale_entry, customrope_scale_label = makelabelentry(context_tab, "RoPE Scale:", customrope_scale, row=23, padx=(100), singleline=True, tooltip="For Linear RoPE scaling. RoPE frequency scale.")
    customrope_base_entry, customrope_base_label = makelabelentry(context_tab, "Base:", customrope_base, row=23, padx=(220), singleline=True, tooltip="For NTK Aware Scaling. RoPE frequency base.",labelpadx=(180))
    def togglerope(a,b,c):
        manualropebox.grid_remove()
        nativectx_label.grid_remove()
        nativectx_entry.grid_remove()
        customrope_scale_label.grid_remove()
        customrope_scale_entry.grid_remove()
        customrope_base_label.grid_remove()
        customrope_base_entry.grid_remove()
        if customrope_var.get() == 1:
            manualropebox.grid(row=22, column=0,padx=(200), pady=1, stick="nw")
            if manualrope_var.get() == 1:
                customrope_scale_label.grid(row=23, column=0, padx=8, pady=1, stick="nw")
                customrope_scale_entry.grid(row=23, column=0, padx=(100), pady=1, stick="nw")
                customrope_base_label.grid(row=23, column=0, padx=(180), pady=1, stick="nw")
                customrope_base_entry.grid(row=23, column=0,  padx=(220), pady=1, stick="nw")
            else:
                nativectx_label.grid(row=23, column=0, padx=8, pady=1, stick="nw")
                nativectx_entry.grid(row=23, column=0, padx=(146), pady=1, stick="nw")

    manualropebox = makecheckbox(context_tab, "Manual Rope Scale", variable=manualrope_var, row=22, command=togglerope, padx=(200), tooltiptxt="Set RoPE base and scale manually.")

    makecheckbox(context_tab, "Custom RoPE Config", variable=customrope_var, row=22, command=togglerope,tooltiptxt="Override the default RoPE configuration with custom RoPE scaling.")
    noqkvlabel = makelabel(context_tab,"(Note: QuantKV works best with flash attention)",30,0,"Only K cache can be quantized, and performance can suffer.\nIn some cases, it might even use more VRAM when doing a full offload.",padx=160)
    noqkvlabel.configure(text_color="#ff5555")
    qkvslider,qkvlabel,qkvtitle = makeslider(context_tab, "Quantize KV Cache:", quantkv_text, quantkv_var, 30, set=0,tooltip="Enable quantization of KV cache.\nRequires Flash Attention for full effect, otherwise only K cache is quantized.")
    quantkv_var.trace_add("write", toggleflashattn)
    makecheckbox(context_tab, "No BOS Token", nobostoken_var, 43, tooltiptxt="Prevents BOS token from being added at the start of any prompt. Usually NOT recommended for most models.")
    makecheckbox(context_tab, "Enable Guidance", enableguidance_var, 43,padx=(140), tooltiptxt="Enables the use of Classifier-Free-Guidance, which allows the use of negative prompts. Has performance and memory impact.")
    def togglejinja(a,b,c):
        if jinja_var.get()==1:
            jinjatoolsbox.grid()
            jinjakwargsbox.grid()
            jinjakwargsboxlbl.grid()
            jinjathinkbox.grid()
            jinjathinklbl.grid()
        else:
            jinja_tools_var.set(0)
            jinjatoolsbox.grid_remove()
            jinjakwargsbox.grid_remove()
            jinjakwargsboxlbl.grid_remove()
            jinjathinkbox.grid_remove()
            jinjathinklbl.grid_remove()
        changed_gpulayers_estimate()
    def togglejinjathink(a,b,c):
        curr = parse_json_object(jinja_kwargs_var.get(),"tempjinja")
        curr = (curr if curr else {})
        changed = False
        if jinja_think_var.get()=="true":
            curr["enable_thinking"] = True
            changed = True
        elif jinja_think_var.get()=="false":
            curr["enable_thinking"] = False
            changed = True
        elif "enable_thinking" in curr:
            del curr["enable_thinking"]
            changed = True
        if changed:
            jinja_kwargs_var.set(json.dumps(curr) if curr else "")
    def updatejinjathinktoggle(a,b,c):
        curr = parse_json_object(jinja_kwargs_var.get(),"tempjinja",True)
        curr = (curr if curr else {})
        if "enable_thinking" in curr:
            if curr["enable_thinking"] is True:
                jinja_think_var.set("true")
            elif curr["enable_thinking"] is False:
                jinja_think_var.set("false")
    makecheckbox(context_tab, "Use Jinja", jinja_var, row=45, command=togglejinja, tooltiptxt="Enables using jinja chat template formatting for chat completions endpoint. Other endpoints are unaffected.")
    jinjatoolsbox = makecheckbox(context_tab, "Jinja for Tools", jinja_tools_var, row=45 ,padx=(140), tooltiptxt="Allows jinja even with tool calls. If unchecked, jinja will be disabled when tools are used.")
    jinja_think_choices = ['default', 'true', 'false']
    jinjathinkbox, jinjathinklbl = makelabelcombobox(context_tab, "Jinja Thinking:", jinja_think_var, 45, command=togglejinjathink,labelpadx=(280), padx=370, width=100, tooltiptxt="Tries to enable or disable thinking in Jinja mode. This is a shortcut to setting Jinja Kwargs directly.", values=jinja_think_choices)
    jinjakwargsbox,jinjakwargsboxlbl = makelabelentry(context_tab, "Jinja Kwargs:", jinja_kwargs_var, row=47, width=200, padx=(100), singleline=True, tooltip='Set additiona fields for Jinja JSON template parser, must be a valid json object.\nSpecified as JSON fields: {"KEY1":"VALUE1", "KEY2":"VALUE2"...}')
    jinja_var.trace_add("write", togglejinja)
    jinja_kwargs_var.trace_add("write", updatejinjathinktoggle)
    makelabelentry(context_tab, "MoE Experts:", moeexperts_var, row=55, padx=(86), singleline=True, tooltip="Override number of MoE experts.")
    moecpu_box,moecpu_box_lbl = makelabelentry(context_tab, "MoE CPU Layers:", moecpu_var, row=55, padx=(334), singleline=True, tooltip="Force Mixture of Experts (MoE) weights of the first N layers to the CPU.\nSetting it higher than GPU layers has no effect.", labelpadx=(230))
    makelabelentry(context_tab, "Override KV:", override_kv_var, row=57, padx=(86), singleline=True, width=130, tooltip="Override metadata value by key. Separate multiple values with commas. Format is name=type:value. Types: int, float, bool, str")
    tenos_box,tenos_box_lbl = makelabelentry(context_tab, "Override Tensors:", override_tensors_var, row=57, padx=(334), singleline=True, width=130, tooltip="Override selected backend for specific tensors matching tensor_name_regex_pattern=buffer_type, same as in llama.cpp.", labelpadx=(230))

    # Model Tab
    model_tab = tabcontent["Loaded Files"]

    makefileentry(model_tab, "Text Model:", "Select GGUF or GGML Model File", model_var, 1,width=205,singlerow=True, onchoosefile=on_picked_model_file,tooltiptxt="Select a GGUF or GGML model file on disk to be loaded.")
    ctk.CTkButton(model_tab, width=70, text = "HF Search", command = model_searcher ).grid(row=1,column=0, stick="nw", padx=(370), pady=2)
    makefileentry(model_tab, "Text Lora:", "Select Lora File",lora_var, 3,width=160,singlerow=True,tooltiptxt="Select an optional GGML Text LoRA adapter to use.\nLeave blank to skip.")
    makelabelentry(model_tab, "Multiplier: ", loramult_var, 3, 50,padx=(390),singleline=True,tooltip="Scale multiplier for Text LoRA Strength. Default is 1.0", labelpadx=(330))
    makefileentry(model_tab, "Mmproj File:", "Select Audio or Vision mmproj File", mmproj_var, 7,width=280,singlerow=True,tooltiptxt="Select a mmproj file to use for multimodal models for vision and audio recognition.\nLeave blank to skip.")
    makelabelentry(model_tab, "Vision MaxRes:", visionmaxres_var, 9, width=40, padx=(100), singleline=True, tooltip=f"Clamp MMProj vision maximum allowed resolution. Allowed values are between 512 to 2048 px (default {default_visionmaxres}).")
    makelabelentry(model_tab, "V.Min/Max Tok:", vision_min_tokens_var, 9, width=36, padx=(244), singleline=True, tooltip="Override the minimum tokens for the MMProj embedding (default -1).", labelpadx=(150))
    makelabelentry(model_tab, "", vision_max_tokens_var, 9, padx=(284),width=36, singleline=True, tooltip="Override the maximum tokens for the MMProj embedding (default -1).", labelpadx=(260))
    makecheckbox(model_tab, "V.Force CPU", mmprojcpu_var, 9, padx=340, tooltiptxt="Force CLIP for Vision mmproj always on CPU.")
    makefileentry(model_tab, "Draft Model:", "Select Speculative Text Model File", draftmodel_var, 11,width=280,singlerow=True,tooltiptxt="Select a draft text model file to use for speculative decoding.\nLeave blank to skip.")
    makelabelentry(model_tab, "Draft Amount: ", draftamount_var, 13, 50,padx=(100),singleline=True,tooltip="How many tokens to draft per chunk before verifying results")
    makelabelentry(model_tab, "Splits: ", draftgpusplit_str_vars, 13, 50,padx=(210),singleline=True,tooltip="Distribution of draft model layers. Leave blank to follow main model's gpu split. Only works if multi-gpu (All) selected in main model.", labelpadx=(160))
    makelabelentry(model_tab, "Layers: ", draftgpulayers_var, 13, 50,padx=(320),singleline=True,tooltip="How many layers to GPU offload for the draft model", labelpadx=(270))
    makefileentry(model_tab, "Embeds Model:", "Select Embeddings Model File", embeddings_model_var, 15, width=130,singlerow=True, filetypes=[("*.gguf","*.gguf")], tooltiptxt="Select an embeddings GGUF model that can be used to generate embedding vectors.")
    makelabelentry(model_tab, "ECtx: ", embeddings_ctx_var, 15, 50,padx=(335),singleline=True,tooltip="If set above 0, limits max context for embedding model to save memory.", labelpadx=(302))
    makecheckbox(model_tab, "GPU", embeddings_gpu_var, 15, 0,padx=(390),tooltiptxt="Uses the GPU for Embeddings.")
    embeddings_gpu_var.trace_add("write", gui_changed_modelfile)
    makefileentry(model_tab, "Preload Story:", "Select Preloaded Story File", preloadstory_var, 17,width=280,singlerow=True,tooltiptxt="Select an optional KoboldAI JSON savefile \nto be served on launch to any client.")
    makefileentry(model_tab, "SaveData File:", "Select or Create New SaveData Database File", savedatafile_var, 19,width=280,filetypes=[("KoboldCpp SaveDB", "*.jsondb")],singlerow=True,dialog_type=1,tooltiptxt="Selecting a file will allow data to be loaded and saved persistently to this KoboldCpp server remotely. File is created if it does not exist.")
    makefileentry(model_tab, "MCP JSON:", "Select a mcp.json configuration file", mcpfile_var, 21,width=280,filetypes=[("MCP JSON", "*.json")],singlerow=True,tooltiptxt="Specify path to mcp.json which contains the Claude Desktop compatible MCP server config.")
    makefileentry(model_tab, "Chat Adapter:", "Select ChatCompletions Adapter File", chatcompletionsadapter_var, 24, width=184, filetypes=[("JSON Adapter", "*.json")], singlerow=True, tooltiptxt="Select an optional ChatCompletions Adapter JSON file to force custom instruct tags.")
    def pickpremadetemplate():
        initialDir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'kcpp_adapters')
        initialDir = initialDir if os.path.isdir(initialDir) else None
        fnam = zentk_askopenfilename(title="Pick Premade ChatCompletions Adapter",filetypes=[("JSON Adapter", "*.json")], initialdir=initialDir)
        if fnam:
            chatcompletionsadapter_var.set(fnam)
    ctk.CTkButton(model_tab, 64, text="Pick Premade", command=pickpremadetemplate).grid(row=24, column=0, padx=(350), pady=2, stick="nw")
    makefileentry(model_tab, "Jinja Template:", "Select a custom Jinja chat template", jinjatemplate_var, 30, width=280, filetypes=[("Jinja Template", "*.jinja")], singlerow=True, tooltiptxt="Select a custom Jinja chat template, will overwrite model jinja chat template")

    mmproj_var.trace_add("write", gui_changed_modelfile)
    draftmodel_var.trace_add("write", gui_changed_modelfile)
    makefileentry(model_tab, "Download Dir:", "Select directory to store all model downloads", download_dir_var, 35, width=280, singlerow=True, dialog_type=2, tooltiptxt="Specify a directory to store any downloaded models.")
    makecheckbox(model_tab, "Allow Launch Without Models", nomodel, 40, tooltiptxt="Allows running the WebUI with no model loaded.")

    # Network Tab
    network_tab = tabcontent["Network"]

    # interfaces
    makelabelentry(network_tab, "Host: ", host_var, row=1, width=150, padx=(50), singleline=True,tooltip="Select a specific host interface to bind to.\n(Defaults to all)")
    makelabelentry(network_tab, "Port: ", port_var, row=1, width=100, padx=(254), singleline=True,tooltip=f"Select the port to host the KoboldCPP webserver.\n(Defaults to {defaultport})",labelpadx=220)

    makecheckbox(network_tab, "Remote Tunnel", remotetunnel_var, 11,tooltiptxt="Creates a trycloudflare tunnel.\nAllows you to access koboldcpp from other devices over an internet URL.")
    makecheckbox(network_tab, "Quiet Mode", quietmode, 12,tooltiptxt="Prevents all generation related terminal output from being displayed.")
    makecheckbox(network_tab, "NoCertify Mode (Insecure)", nocertifymode, 12,padx=(200),tooltiptxt="Allows insecure SSL connections. Use this if you have cert errors and need to bypass certificate restrictions.")
    makecheckbox(network_tab, "Shared Multiplayer", multiplayer_var, 13,tooltiptxt="Hosts a shared multiplayer session that others can join.")
    makecheckbox(network_tab, "Enable WebSearch", websearch_var, 13,padx=(200),tooltiptxt="Enable the local search engine proxy so Web Searches can be done.")

    makefileentry(network_tab, "SSL Cert:", "Select SSL cert.pem file",ssl_cert_var, 20, width=200,filetypes=[("Unencrypted Certificate PEM", "*.pem")], singlerow=True,tooltiptxt="Select your unencrypted .pem SSL certificate file for https.\nCan be generated with OpenSSL.")
    makefileentry(network_tab, "SSL Key:", "Select SSL key.pem file", ssl_key_var, 22, width=200, filetypes=[("Unencrypted Key PEM", "*.pem")], singlerow=True, tooltiptxt="Select your unencrypted .pem SSL key file for https.\nCan be generated with OpenSSL.")
    makelabelentry(network_tab, "Password: ", password_var, 24, 200, padx=(100), singleline=True, tooltip="Enter a password required to use this instance.\nThis key will be required for all text endpoints.\nImage endpoints are not secured.")

    makelabelentry(network_tab, "Multiuser Queue:", multiuser_var, row=30, width=50, padx=(120), singleline=True, tooltip="Maximum queued incoming requests.")
    makelabelentry(network_tab, "Max Req. Size (MB):", maxrequestsize_var, row=30, width=50, padx=(340), singleline=True, tooltip="Specify a max request payload size. Any requests to the server larger than this size will be dropped. Do not change if unsure.",labelpadx=210)
    makelabelentry(network_tab, "IP Rate Limiter (s):", ratelimit_var, row=34, width=50, padx=(120), singleline=True, tooltip="Rate limits each IP to allow a new request once per X seconds. Do not change if unsure.")
    makelabelentry(network_tab, "Request Timeout (s):", reqtimeout_var, row=34, width=50, padx=(340), singleline=True, tooltip="Timeout in seconds for HTTP requests",labelpadx=210)
    makelabelentry(network_tab, "Parallel Requests:", parallel_requests_var, row=38, width=50, padx=(120), singleline=True, tooltip="Allows multiple requests to be batched and executed in parallel. Only works for basic text generation requests. Experimental!")


    def togglerpcmode(a,b,c):
        if rpcmode_var.get()=="connect":
            rpcepbox.grid()
            rpceplbl.grid()
            rpchostbox.grid_remove()
            rpchostlbl.grid_remove()
            rpcportbox.grid_remove()
            rpcportlbl.grid_remove()
            rpcdevicebox.grid_remove()
            rpcdevicelbl.grid_remove()
            rpcdesc.configure(text="Connect to one or more remote RPC endpoints (ip:port), comma seperated")
        elif rpcmode_var.get()=="host":
            rpcepbox.grid_remove()
            rpceplbl.grid_remove()
            rpchostbox.grid()
            rpchostlbl.grid()
            rpcportbox.grid()
            rpcportlbl.grid()
            rpcdevicebox.grid()
            rpcdevicelbl.grid()
            rpcdesc.configure(text="Run a RPC server that others can connect to. Disables local models and API.")
        else:
            rpcepbox.grid_remove()
            rpceplbl.grid_remove()
            rpchostbox.grid_remove()
            rpchostlbl.grid_remove()
            rpcportbox.grid_remove()
            rpcportlbl.grid_remove()
            rpcdevicebox.grid_remove()
            rpcdevicelbl.grid_remove()
            rpcdesc.configure(text="RPC is disabled and will not be used")
        update_runmode_gui()
    makelabelcombobox(network_tab, "RPC Mode:", rpcmode_var, row=40, padx=(100), width=90, command=togglerpcmode, tooltiptxt="RPC functionality to connect to remote RPC instances or host one, allowing GPUs to be shared over a network.", values=["disabled","connect","host"])
    rpcdesc = makelabel(network_tab, "RPC is disabled and will not be used", row=42)
    rpcepbox, rpceplbl = makelabelentry(network_tab, "RPC Endpoints: ", rpcendpoints_var, row=44, padx=(100), width=200, singleline=True,tooltip="Specify a comma separated list of remote RPC endpoints to connect to e.g. 127.0.0.1:5551,127.0.0.1:5552")
    rpchostbox, rpchostlbl = makelabelentry(network_tab, "RPC Host IP: ", rpc_host_var, row=46, width=150, padx=(100), singleline=True,tooltip="IP address for RPC server to listen on. Use 0.0.0.0 for all interfaces, 127.0.0.1 for localhost only.")
    rpcportbox, rpcportlbl = makelabelentry(network_tab, "RPC Host Port: ", rpc_port_var, row=46, width=100, padx=(360), singleline=True,tooltip=f"Port for RPC server to listen on. (default:{default_rpc_port})",labelpadx=260)
    rpcdevicebox, rpcdevicelbl = makelabelentry(network_tab, "RPC Devices: ", rpc_device_var, row=48, padx=(100), width=200, singleline=True,tooltip="Set specific devices to use for RPC server. Comma separated. Overrides normal RPC device choices.")


    # Horde Tab
    horde_tab = tabcontent["Horde Worker"]
    makelabel(horde_tab, "Horde:", 18,0,"Settings for embedded AI Horde worker").grid(pady=10)

    horde_name_entry,  horde_name_label = makelabelentry(horde_tab, "Horde Model Name:", horde_name_var, 20, 180,tooltip="The model name to be displayed on the AI Horde.")
    horde_gen_entry,  horde_gen_label = makelabelentry(horde_tab, "Gen. Length:", horde_gen_var, 21, 50,tooltip="The maximum amount to generate per request that this worker will accept jobs for.")
    horde_context_entry,  horde_context_label = makelabelentry(horde_tab, "Max Context:",horde_context_var, 22, 50,tooltip="The maximum context length that this worker will accept jobs for.\nIf 0, matches main context limit.")
    horde_apikey_entry,  horde_apikey_label = makelabelentry(horde_tab, "API Key (If Embedded Worker):",horde_apikey_var, 23, 180,tooltip="Your AI Horde API Key that you have registered.")
    horde_workername_entry,  horde_workername_label = makelabelentry(horde_tab, "Horde Worker Name:",horde_workername_var, 24, 180,tooltip="Your worker's name to be displayed.")

    def togglehorde(a,b,c):
        horde_items = zip([horde_name_entry, horde_gen_entry, horde_context_entry, horde_apikey_entry, horde_workername_entry],
                          [horde_name_label, horde_gen_label, horde_context_label, horde_apikey_label, horde_workername_label])

        for item, label in horde_items:
            if usehorde_var.get() == 1:
                item.grid()
                label.grid()
            else:
                item.grid_remove()
                label.grid_remove()
        if usehorde_var.get()==1 and (horde_name_var.get()=="koboldcpp" or horde_name_var.get()=="") and model_var.get()!="":
            basefile = os.path.basename(model_var.get())
            horde_name_var.set(sanitize_string(os.path.splitext(basefile)[0]))

    makecheckbox(horde_tab, "Configure for Horde", usehorde_var, 19, command=togglehorde,tooltiptxt="Enable the embedded AI Horde worker.")

    # Image Gen Tab

    images_tab = tabcontent["Image Gen"]
    makefileentry(images_tab, "Image Model:", "Select Image Gen Model File (safetensors/gguf)", sd_model_var, 1, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")], tooltiptxt="Select a .safetensors or .gguf Image Generation model file on disk to be loaded.")

    def togglesdlora(a,b,c):
        if sd_runtime_loras_var.get()==1:
            imglora1.grid_remove()
            imglora2.grid_remove()
            imglora3.grid_remove()
            imglora4.grid_remove()
            imglora5.grid_remove()
            imglora6.grid()
            imglora7.grid()
            imglora8.grid()
        else:
            imglora1.grid()
            imglora2.grid()
            imglora3.grid()
            imglora4.grid()
            imglora5.grid()
            imglora6.grid_remove()
            imglora7.grid_remove()
            imglora8.grid_remove()

    makelabelentry(images_tab, "Clamp Resolution (Hard):", sd_clamped_var, 14, 50, padx=(150),singleline=True,tooltip="Limit generation steps and output image size for shared use.\nSet to 0 to disable, otherwise value is clamped to the max size limit (min 512px).")
    makelabelentry(images_tab, "(Soft):", sd_clamped_soft_var, 14, 50, padx=(250),singleline=True,tooltip="Square image size restriction, to protect the server against memory crashes.\nAllows width-height tradeoffs, eg. 640 allows 640x640 and 512x768\nLeave at 0 for the default value: 832 for SD1.5/SD2, 1024 otherwise.",labelpadx=(210))
    makecheckbox(images_tab, "Runtime LoRAs", sd_runtime_loras_var, 14,command=togglesdlora, padx=(310), tooltiptxt="Allow using LoRAs in a directory dynamically (syntax is <lora:name:weight>)")

    makelabelentry(images_tab, "ImgThreads:" , sd_threads_var, 8, 40,padx=(275),singleline=True,tooltip="How many threads to use during image generation.\nIf left blank, uses same value as threads.",labelpadx=(195))
    makelabelcombobox(images_tab, "ImgGPU:" , sd_main_gpu_var, 8, width=70,padx=375,tooltiptxt="Which device to use for Image Gen?\nIf left blank or 'main', uses the main device.",labelpadx=320, values=sd_device_choices)
    sd_model_var.trace_add("write", gui_changed_modelfile)
    makelabelcombobox(images_tab, "Compress Weights: ", sd_quant_var, 8, width=(60), padx=(126), labelpadx=8, tooltiptxt="Quantizes the SD model weights to save memory.\nHigher levels save more memory, and cause more quality degradation.", values=sd_quant_choices)
    sd_quant_var.trace_add("write", changed_gpulayers_estimate)

    imglora1,imglora2,imglora3 = makefileentry(images_tab, "Image LoRAs:", "Select SD lora files to load",sd_lora_var, 20, width=160, singlerow=True, filetypes=[("*.safetensors *.gguf", "*.safetensors *.gguf")],tooltiptxt="Select multiple .safetensors or .gguf SD LoRA model files to be loaded. Should be unquantized!", multiple=True)
    imglora4,imglora5 = makelabelentry(images_tab, "Multiplier:" , sd_loramult_var, 20, 50,padx=(390),singleline=True,tooltip="What mutiplier value to apply the SD LoRA with.",labelpadx=(330))
    imglora6,imglora7,imglora8 = makefileentry(images_tab, "LoRA Dir:", "Select directory for runtime lora triggers",sd_lora_var, 20, width=280, singlerow=True, dialog_type=2,tooltiptxt="Select directory containing LoRAs that can be used at runtime.\nSyntax is <lora:name:weight>")

    makefileentry(images_tab, "T5-XXL File:", "Select T5-XXL model file (SD3, Flux, WAN)",sd_t5xxl_var, 24, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")],tooltiptxt="Select a .safetensors t5xxl file to be loaded.")
    makefileentry(images_tab, "Clip-1 File:", "Select First Clip model file (Clip-L for SD3 or Flux, or other vision encoder)",sd_clip1_var, 26, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")],tooltiptxt="Select a .safetensors Clip-1 file to be loaded.\nThis is Clip-L for SD3 and Flux, Clip Vision for WAN, and Qwen2.5VL for QwenImage")
    makefileentry(images_tab, "Clip-2 File:", "Select Second Clip model file (Clip-G for SD3)",sd_clip2_var, 28, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")],tooltiptxt="Select a .safetensors Clip-2 file to be loaded.\nThis is Clip-G for SD3")
    makefileentry(images_tab, "PhotoMaker:", "Select Optional PhotoMaker model file (SDXL)",sd_photomaker_var, 30, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")],tooltiptxt="PhotoMaker is a model that allows face cloning.\nSelect a .safetensors PhotoMaker file to be loaded (SDXL only).")
    makefileentry(images_tab, "Upscaler:", "Select Optional Upscaling model file (ESRGAN)",sd_upscaler_var, 32, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf *.pth","*.safetensors *.gguf *.pth")],tooltiptxt="Select an upscaler model file.\nCurrently only ESRGAN is supported.")


    sdvaeitem1,sdvaeitem2,sdvaeitem3 = makefileentry(images_tab, "Image VAE:", "Select Optional SD VAE file",sd_vae_var, 40, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf", "*.safetensors *.gguf")],tooltiptxt="Select a .safetensors or .gguf SD VAE file to be loaded.")
    makefileentry(images_tab, "Audio VAE:", "Select Audio VAE file for LTX video gen",sd_audio_vae_var, 42, width=280, singlerow=True, filetypes=[("*.safetensors *.gguf","*.safetensors *.gguf")],tooltiptxt="Select an Audio VAE file for LTX video generation")

    def toggletaesd(a,b,c):
        if sd_vaeauto_var.get()==1:
            sdvaeitem1.grid_remove()
            sdvaeitem2.grid_remove()
            sdvaeitem3.grid_remove()
        else:
            if not sdvaeitem1.grid_info() or not sdvaeitem2.grid_info() or not sdvaeitem3.grid_info():
                sdvaeitem1.grid()
                sdvaeitem2.grid()
                sdvaeitem3.grid()
    makecheckbox(images_tab, "Automatic VAE (TAE SD)", sd_vaeauto_var, 44,command=toggletaesd,tooltiptxt="Replace VAE with TAESD. May fix bad VAE.")
    makelabelcombobox(images_tab, "Conv2D Direct:", sd_convdirect_var, row=44, labelpadx=(220), padx=(310), width=90, tooltiptxt="Use Conv2D Direct operation. May save memory or improve performance.\nMight crash if not supported by the backend.\n", values=sd_convdirect_choices)
    makelabelentry(images_tab, "VAE Tiling Threshold:", sd_tiled_vae_var, 46, 50, padx=(144),singleline=True,tooltip="Enable VAE Tiling for images above this size, to save memory.\nSet to 0 to disable VAE tiling.")
    makecheckbox(images_tab, "SD Flash Attention", sd_flash_attention_var, 46,padx=(230), tooltiptxt="Enable Flash Attention for image diffusion. May save memory or improve performance.")
    makecheckbox(images_tab, "Model Offload", sd_offload_cpu_var, 50,padx=8, tooltiptxt="Offload image weights in RAM to save VRAM, swap into VRAM when needed.")
    makelabelcombobox(images_tab, "VAE dev:", sd_vae_device_var, 50,labelpadx=(140),padx=(200), width=70, tooltiptxt="Change VAE device for image generation.", values=sd_device_choices)
    makelabelcombobox(images_tab, "CLIP dev:", sd_clip_device_var, 50,labelpadx=(280),padx=340, width=70, tooltiptxt="Change CLIP / T5 / LLM device for image generation.", values=sd_device_choices)
    makelabelentry(images_tab, "VRAM Limiter (MB):", sd_vram_limit_var, 60, 50, padx=(144),singleline=True,tooltip="If set, prevent VRAM usage from image gen from using more than this many MB.")


    # audio tab
    audio_tab = tabcontent["Audio"]
    makefileentry(audio_tab, "Whisper Model (Speech-To-Text):", "Select Whisper .bin Model File", whisper_model_var, 1, width=280, filetypes=[("*.bin","*.bin")], tooltiptxt="Select a Whisper .bin model file on disk to be loaded for Voice Recognition.")
    whisper_model_var.trace_add("write", gui_changed_modelfile)
    makefileentry(audio_tab, "TTS Model (Text-To-Speech):", "Select TTS GGUF Model File", tts_model_var, 3, width=280, filetypes=[("*.gguf","*.gguf")], tooltiptxt="Select a TTS GGUF model file on disk to be loaded for Narration.")
    tts_model_var.trace_add("write", gui_changed_modelfile)
    makelabelentry(audio_tab, "TTS Threads:" , tts_threads_var, 5, 50,padx=100,singleline=True,tooltip="How many threads to use during TTS generation.\nIf left blank, uses same value as threads.")
    makelabelentry(audio_tab, "TTS Max Tokens:" , ttsmaxlen_var, 5, 50,padx=300,singleline=True,tooltip="Max allowed audiotokens to generate per TTS request.", labelpadx=190)
    makecheckbox(audio_tab, "TTS Use GPU", ttsgpu_var, 9, 0,tooltiptxt="Uses the GPU for TTS. Currently only works on certain models (OuteTTS/Q3TTS).")
    ttsgpu_var.trace_add("write", gui_changed_modelfile)
    makefileentry(audio_tab, "WavTokenizer Model (Required for some models):", "Select WavTokenizer GGUF Model File", wavtokenizer_var, 11, width=280, filetypes=[("*.gguf","*.gguf")], tooltiptxt="Select a WavTokenizer GGUF model file on disk to be loaded for Narration.")
    wavtokenizer_var.trace_add("write", gui_changed_modelfile)
    makefileentry(audio_tab, "TTS Voices Dir:", "Select directory containing voices for voice cloning", tts_dir_var, 20, width=280, singlerow=True, dialog_type=2, tooltiptxt="Select directory containing voices for voice cloning")

    makefileentry(audio_tab, "MusicLLM:", "Select music LLM model (e.g acestep-5Hz-lm-0.6B)", musicllm_var, 30, width=280, singlerow=True, dialog_type=0, tooltiptxt="Select music LLM model (e.g acestep-5Hz-lm)")
    makefileentry(audio_tab, "MusicEmbeds:", "Select music embedding model (e.g Qwen3-Embedding-0.6B)", musicembeddings_var, 32, width=280, singlerow=True, dialog_type=0, tooltiptxt="Select music embedding model (e.g Qwen3-Embedding-0.6B)")
    makefileentry(audio_tab, "MusicDiffuser:", "Select music diffusion (DiT) model (e.g acestep-v15-turbo)", musicdiffusion_var, 34, width=280, singlerow=True, dialog_type=0, tooltiptxt="Select music diffusion (DiT) model (e.g acestep-v15-turbo)")
    makefileentry(audio_tab, "MusicVAE:", "Select music VAE model", musicvae_var, 36, width=280, singlerow=True, dialog_type=0, tooltiptxt="Select music VAE model")
    makecheckbox(audio_tab, "Music Low VRAM", musiclowvram_var, 38, 0,tooltiptxt="Unload music models when not in use.")

    admin_tab = tabcontent["Admin"]
    def toggleadmin(a,b,c):
        if admin_var.get()==1 and admin_dir_var.get()=="":
            autopath = os.path.realpath(__file__)
            if getattr(sys, 'frozen', False):
                autopath = sys.executable
            autopath = os.path.dirname(autopath)
            admin_dir_var.set(autopath)
        if admin_var.get()==1:
            router_mode_box.grid()
        else:
            router_mode_box.grid_remove()
        togglerouter(1,1,1)

    def togglerouter(a,b,c):
        if router_mode_var.get()==1 and admin_var.get()==1:
            autoswap_mode_box.grid()
        else:
            autoswap_mode_box.grid_remove()

    makecheckbox(admin_tab, "Enable Model Administration", admin_var, 1, 0, command=toggleadmin,tooltiptxt="Enable a admin server, allowing you to remotely relaunch and swap models and configs.")
    makelabelentry(admin_tab, "Admin Password:" , admin_password_var, 3, 150,padx=(120),singleline=True,tooltip="Require a password to access admin functions. You are strongly advised to use one for publically accessible instances!")
    makefileentry(admin_tab, "Config Directory (Required):", "Select directory containing .gguf or .kcpps files to relaunch from", admin_dir_var, 5, width=280, dialog_type=2, tooltiptxt="Specify a directory to look for .kcpps configs in, which can be used to swap models.")
    makefileentry(admin_tab, "Base config .kcpps (For reloading):", "", baseconfig_var, 7, width=280, dialog_type=0, tooltiptxt="Specify a base .kcpps config to apply, if no custom base config is selected during a model swap.")
    makelabelentry(admin_tab, "Auto Unload Timeout:" , admin_unload_timeout_var, 17, 70,padx=(150),singleline=True,tooltip="Set an idle timeout in seconds after which KoboldCpp will automatically unload the current model.")
    makecheckbox(admin_tab, "SingleInstance Mode", singleinstance_var, 19, 0,tooltiptxt="Allows this server to be shut down by another KoboldCpp instance with singleinstance starting on the same port.")
    router_mode_box = makecheckbox(admin_tab, "Router Mode", router_mode_var, 21, 0, command=togglerouter, tooltiptxt="Router mode uses a reverse proxy router, allowing you to easily hotswap models and configs within a single request. Requires admin mode.")
    autoswap_mode_box = makecheckbox(admin_tab, "Autoswap Mode", autoswap_mode_var, 23, 0,tooltiptxt="Autoswap mode builds on router mode to allow switching of model types within the same config automatically. Requires admin mode and router mode. All models desired must be defined within the same config.")

    def kcpp_export_template():
        nonlocal kcpp_exporting_template
        kcpp_exporting_template = True
        export_vars()
        kcpp_exporting_template = False
        savdict = json.loads(json.dumps(args.__dict__,indent=2))
        for key in deprecated_keys:
            savdict.pop(key, None)  # avoids KeyError if missing
        file_type = [("KoboldCpp LaunchTemplate", "*.kcppt")]
        #remove blacklisted fields
        savdict = convert_args_to_template(savdict)
        filename = zentk_asksaveasfilename(filetypes=file_type, defaultextension=".kcppt")
        if not filename:
            return
        filenamestr = str(filename).strip()
        if not filenamestr.endswith(".kcppt"):
            filenamestr += ".kcppt"
        file = open(filenamestr, 'w')
        file.write(json.dumps(savdict,indent=2))
        file.close()
        pass

    # extra tab
    extra_tab = tabcontent["Extra"]
    makelabel(extra_tab, "Extract KoboldCpp Files", 3, 0,tooltiptxt="Unpack KoboldCpp to a local directory to modify its files. You can also launch via koboldcpp.py for faster startup.")
    ctk.CTkButton(extra_tab , text = "Unpack KoboldCpp To Folder", command = unpack_to_dir ).grid(row=3,column=0, stick="w", padx=(170), pady=2)
    makelabel(extra_tab, "Export as .kcppt template", 4, 0,tooltiptxt="Creates a KoboldCpp launch template for others to use.\nEmbeds JSON files directly into exported file when saving.\nWhen loaded, forces the backend to be automatically determined.\nWarning! Not recommended for beginners!")
    ctk.CTkButton(extra_tab , text = "Generate LaunchTemplate", command = kcpp_export_template ).grid(row=4,column=0, stick="w", padx=(170), pady=2)
    makelabel(extra_tab, "Analyze GGUF Metadata", 6, 0,tooltiptxt="Reads the metadata, weight types and tensor names in any GGUF file.")
    ctk.CTkButton(extra_tab , text = "Analyze GGUF", command = analyze_gguf_model_wrapper ).grid(row=6,column=0, stick="w", padx=(170), pady=2)
    if os.name == 'nt':
        makelabel(extra_tab, "File Extensions Handler", 10, 0,tooltiptxt="Makes KoboldCpp the default handler for .kcpps, .kcppt, .ggml and .gguf files.")
        ctk.CTkButton(extra_tab , text = "Register", width=90, command = register_koboldcpp ).grid(row=10,column=0, stick="w", padx= (170), pady=2)
        ctk.CTkButton(extra_tab , text = "Unregister", width=90, command = unregister_koboldcpp ).grid(row=10,column=0, stick="w", padx= (264), pady=2)
    if sys.platform == "linux":
        def togglezenity(a,b,c):
            global zenity_permitted
            zenity_permitted = (nozenity_var.get()==0)
        makecheckbox(extra_tab, "Use Classic FilePicker", nozenity_var, 20, tooltiptxt="Use the classic TKinter file picker instead.")
        nozenity_var.trace_add("write", togglezenity)

    extra_terminal_process = None
    def stdout_has_terminal():
        try:
            return bool(sys.stdout and sys.stdout.isatty())
        except Exception:
            return False
    def showtermlogs():
        nonlocal extra_terminal_process
        try:
            if extra_terminal_process and extra_terminal_process.poll() is None:
                print("Error: Secondary terminal already running.")
                return
            if sys.platform == "linux":
                # Create an unnamed pipe, launch a terminal that reads from the read-end FD
                r, w = os.pipe()
                extra_terminal_process = subprocess.Popen(["xterm", "-hold","-e", f"bash -c 'cat <&{r}'"], pass_fds=[r])
                writer = os.fdopen(w, "w", buffering=1)
                redirector = StdoutRedirector(writer)
                sys.stdout = redirector
                print("--- Secondary Linux Terminal Active ---")
            else:
                print("Error: Secondary Terminal Not Supported on this Platform")
        except Exception as e:
            print(f"Spawn Extra Terminal Failed: {e}")
    if sys.platform == "linux":
        makelabel(extra_tab, "Spawn Terminal Logs", 12, 0,tooltiptxt="A simple terminal logger that duplicates the command line output.")
        ctk.CTkButton(extra_tab , text = "Spawn Terminal", command = showtermlogs ).grid(row=12,column=0, stick="w", padx= 170, pady=2)

    # refresh
    runopts_var.trace_add("write", changerunmode)
    update_runmode_gui()
    togglerope(1,1,1)
    toggleflashattn(1,1,1)
    togglectxshift(1,1,1)
    togglehorde(1,1,1)
    toggletaesd(1,1,1)
    togglesdlora(1,1,1)
    togglejinja(1,1,1)
    toggleadmin(1,1,1)
    updatejinjathinktoggle(1,1,1)
    togglerpcmode(1,1,1)
    global runmode_untouched
    runmode_untouched = True

    # launch
    def guilaunch():
        if model_var.get() == "" and sd_model_var.get() == "" and whisper_model_var.get() == "" and tts_model_var.get() == "" and embeddings_model_var.get() == "" and musicdiffusion_var.get() == "" and musicllm_var.get() == "" and nomodel.get()!=1 and rpcmode_var.get()!="host":
            # prevent launch without at least one valid model
            givehelp = show_gui_yesnobox("No Models Selected","Error: You need to load at least one AI model to continue.\n\nDo you want help finding a model?")
            if givehelp == 'yes':
                display_help()
            return
        nonlocal nextstate
        nextstate = 1
        root.withdraw()
        root.quit()
        pass

    def export_vars():
        nonlocal kcpp_exporting_template
        args.threads =  (get_default_threads() if threads_var.get()=="" else int(threads_var.get()))
        args.usemlock   = usemlock.get() == 1
        args.debugmode  = debugmode.get()
        args.launch     = launchbrowser.get()==1
        args.highpriority = highpriority.get()==1
        args.usemmap = usemmap.get()==1
        args.smartcontext = smartcontext_var.get()==1
        args.noflashattention = flashattention_var.get()==0
        args.noshift = contextshift_var.get()==0
        args.nofastforward = fastforward_var.get()==0
        args.noswa = swa_var.get()==0
        args.swapadding = int(swa_padding_var.get()) if swa_padding_var.get()!="" else swa_padding_default
        args.smartcache = (0 if smartcache_var.get()!=1 else int(smartcacheslots_var.get()))
        args.remotetunnel = remotetunnel_var.get()==1
        args.foreground = keepforeground.get()==1
        args.cli = terminalonly.get()==1
        args.nopipelineparallel = pipelineparallel.get()==0
        args.quiet = quietmode.get()==1
        args.nocertify = nocertifymode.get()==1
        args.nomodel = nomodel.get()==1
        qkvopt = quantkv_text[quantkv_var.get()].lower() if (quantkv_var.get()>=0 and quantkv_var.get() < len(quantkv_text)) else "f16"
        args.quantkv = qkvopt
        args.lowvram = lowvram_var.get()==1
        args.nommq = mmq_var.get()==0
        args.splitmode = splitmode_var.get() if splitmode_var.get() in splitmode_choices else splitmode_choices[0]

        gpuchoiceidx = 0
        args.usecpu = False
        args.usevulkan = None
        args.usecuda = None
        args.noavx2 = False
        args.failsafe = False
        if gpu_choice_var.get()!="All":
            gpuchoiceidx = int(gpu_choice_var.get())
        if runopts_var.get() == "Use CUDA" or runopts_var.get() == "Use hipBLAS (ROCm)":
            if gpu_choice_var.get()=="All":
                args.usecuda = ["normal"]
            else:
                args.usecuda = ["normal",str(gpuchoiceidx)]
        if runopts_var.get() == "Use Vulkan" or runopts_var.get() == "Use Vulkan (Old CPU)" or runopts_var.get() == "Use Vulkan (Older CPU)":
            if gpu_choice_var.get()=="All":
                args.usevulkan = []
            else:
                args.usevulkan = [int(gpuchoiceidx)]
            if runopts_var.get() == "Use Vulkan (Old CPU)":
                args.noavx2 = True
            elif runopts_var.get() == "Use Vulkan (Older CPU)":
                args.noavx2 = True
                args.failsafe = True

        args.gpulayers = (-1 if gpulayers_var.get()=="" else int(gpulayers_var.get()))
        args.autofitpadding = (default_autofit_padding if autofit_padding_var.get()=="" else int(autofit_padding_var.get()))
        if runopts_var.get()=="Use CPU":
            args.usecpu = True
        if runopts_var.get()=="Use CPU (Old CPU)":
            args.noavx2 = True
            args.usecpu = True
        if runopts_var.get()=="Failsafe Mode (Older CPU)":
            args.noavx2 = True
            args.usecpu = True
            args.usemmap = False
            args.failsafe = True
        args.tensor_split = None
        if tensor_split_str_vars.get()!="":
            tssv = tensor_split_str_vars.get()
            if "," in tssv:
                args.tensor_split = [float(x) for x in tssv.split(",")]
            else:
                args.tensor_split = [float(x) for x in tssv.split(" ")]
        args.draftgpusplit = None
        if draftgpusplit_str_vars.get()!="":
            tssv = draftgpusplit_str_vars.get()
            if "," in tssv:
                args.draftgpusplit = [float(x) for x in tssv.split(",")]
            else:
                args.draftgpusplit = [float(x) for x in tssv.split(" ")]

        args.maingpu = -1 if maingpu_var.get()=="" else int(maingpu_var.get())
        args.blasthreads = None if blas_threads_var.get()=="" else int(blas_threads_var.get())
        args.device = deviceoverride_var.get()
        args.batchsize = int(batchsize_values[int(blas_size_var.get())])
        args.autofit = autofit_var.get() == 1
        args.contextsize = int(contextsize_text[context_var.get()])
        if customrope_var.get()==1:
            if manualrope_var.get()==1:
                args.ropeconfig = [float(customrope_scale.get()),float(customrope_base.get())]
                args.overridenativecontext = 0
            else:
                args.ropeconfig = [0.0, 10000.0]
                args.overridenativecontext = int(customrope_nativectx.get())
        else:
            args.ropeconfig = [0.0, 10000.0]
            args.overridenativecontext = 0
        args.moeexperts = int(moeexperts_var.get()) if moeexperts_var.get()!="" else -1
        args.moecpu = int(moecpu_var.get()) if moecpu_var.get()!="" else 0
        args.defaultgenamt = int(defaultgenamt_var.get()) if defaultgenamt_var.get()!="" else default_genlen
        args.genlimit = int(genlimit_var.get()) if genlimit_var.get()!="" else 0
        args.nobostoken = (nobostoken_var.get()==1)
        args.jinja = (jinja_var.get()==1)
        args.jinja_tools = (jinja_tools_var.get()==1)
        args.jinja_kwargs = jinja_kwargs_var.get()  if jinja_kwargs_var.get() != "" else ""
        args.jinjatemplate = jinjatemplate_var.get() if jinjatemplate_var.get() != "" else ""
        args.enableguidance = (enableguidance_var.get()==1)
        args.overridekv = None if override_kv_var.get() == "" else override_kv_var.get()
        args.overridetensors = None if override_tensors_var.get() == "" else override_tensors_var.get()
        args.chatcompletionsadapter = "AutoGuess" if chatcompletionsadapter_var.get() == "" else chatcompletionsadapter_var.get()
        try:
            if kcpp_exporting_template and isinstance(args.chatcompletionsadapter, str) and args.chatcompletionsadapter!="" and os.path.exists(args.chatcompletionsadapter):
                print("Embedding chat completions adapter...")   # parse and save embedded preload story
                with open(args.chatcompletionsadapter, 'r', encoding='utf-8', errors='ignore') as f:
                    args.chatcompletionsadapter = json.load(f)
        except Exception:
            pass

        args.model_param = None if model_var.get() == "" else model_var.get()
        args.lora = None if lora_var.get() == "" else ([lora_var.get()])
        args.loramult = (float(loramult_var.get()) if loramult_var.get()!="" else 1.0)
        pls_str_or_obj = None if preloadstory_var.get() == "" else preloadstory_var.get()
        if pls_str_or_obj and isinstance(pls_str_or_obj,str):
            try:
                temp = json.loads(pls_str_or_obj)
                pls_str_or_obj = temp
            except Exception:
                pass
        args.preloadstory = pls_str_or_obj
        args.savedatafile = None if savedatafile_var.get() == "" else savedatafile_var.get()
        args.mcpfile = None if mcpfile_var.get() == "" else mcpfile_var.get()
        args.downloaddir = download_dir_var.get()
        try:
            if kcpp_exporting_template and isinstance(args.preloadstory, str) and args.preloadstory!="" and os.path.exists(args.preloadstory):
                print("Embedding preload story...")   # parse and save embedded preload story
                with open(args.preloadstory, 'r', encoding='utf-8', errors='ignore') as f:
                    args.preloadstory = json.load(f)
        except Exception:
            pass
        args.mmproj = None if mmproj_var.get() == "" else mmproj_var.get()
        args.mmprojcpu = (mmprojcpu_var.get()==1)
        args.visionmaxres = int(visionmaxres_var.get()) if visionmaxres_var.get()!="" else default_visionmaxres
        args.visionmintokens = int(vision_min_tokens_var.get()) if vision_min_tokens_var.get()!="" else -1
        args.visionmaxtokens = int(vision_max_tokens_var.get()) if vision_max_tokens_var.get()!="" else -1
        args.draftmodel = None if draftmodel_var.get() == "" else draftmodel_var.get()
        args.draftamount = int(draftamount_var.get()) if draftamount_var.get()!="" else default_draft_amount
        args.draftgpulayers = int(draftgpulayers_var.get()) if draftgpulayers_var.get()!="" else 999

        args.ssl = None if (ssl_cert_var.get() == "" or ssl_key_var.get() == "") else ([ssl_cert_var.get(), ssl_key_var.get()])
        args.password = None if (password_var.get() == "") else (password_var.get())

        args.port_param = defaultport if port_var.get()=="" else int(port_var.get())
        args.port = args.port_param
        args.host = host_var.get()
        args.multiuser = int(multiuser_var.get()) if multiuser_var.get()!="" else multiuser_concurrent_limit
        args.multiplayer = (multiplayer_var.get()==1)
        args.websearch = (websearch_var.get()==1)
        args.maxrequestsize = int(maxrequestsize_var.get()) if maxrequestsize_var.get()!="" else 32
        args.ratelimit = int(ratelimit_var.get()) if ratelimit_var.get()!="" else 0
        args.reqtimeout = int(reqtimeout_var.get()) if reqtimeout_var.get()!="" else 0
        if not args.reqtimeout:
            args.reqtimeout = default_reqtimeout
        args.parallelrequests = int(parallel_requests_var.get()) if parallel_requests_var.get()!="" else 1

        args.rpcmode = rpcmode_var.get() if rpcmode_var.get() else "disabled"
        args.rpchost = rpc_host_var.get() if (args.rpcmode=="host" and rpc_host_var.get()) else "0.0.0.0"
        args.rpcport = int(rpc_port_var.get()) if (args.rpcmode=="host" and rpc_port_var.get()) else default_rpc_port
        args.rpctargets = rpcendpoints_var.get() if (args.rpcmode=="connect" and rpcendpoints_var.get()) else ""
        args.rpcdevice = rpc_device_var.get() if (args.rpcmode=="host" and rpc_device_var.get()) else ""

        if usehorde_var.get() != 0:
            args.hordemodelname = horde_name_var.get()
            args.hordegenlen = int(horde_gen_var.get())
            args.hordemaxctx = int(horde_context_var.get())
            if horde_apikey_var.get()!="" and horde_workername_var.get()!="":
                args.hordekey = horde_apikey_var.get()
                args.hordeworkername = horde_workername_var.get()

        args.sdmodel = sd_model_var.get() if sd_model_var.get() != "" else ""
        args.sdflashattention = True if sd_flash_attention_var.get()==1 else False
        args.sdoffloadcpu = True if sd_offload_cpu_var.get()==1 else False
        args.sdvaedevice = sd_resolve_device(sd_vae_device_var.get())
        args.sdclipdevice = sd_resolve_device(sd_clip_device_var.get())
        args.sdthreads = (0 if sd_threads_var.get()=="" else int(sd_threads_var.get()))
        args.sdvramlimit = (0 if sd_vram_limit_var.get()=="" else int(sd_vram_limit_var.get()))
        args.sdclamped = (0 if int(sd_clamped_var.get())<=0 else int(sd_clamped_var.get()))
        args.sdclampedsoft = (0 if int(sd_clamped_soft_var.get())<=0 else int(sd_clamped_soft_var.get()))
        args.sdtiledvae = (default_vae_tile_threshold if sd_tiled_vae_var.get()=="" else int(sd_tiled_vae_var.get()))
        if sd_vaeauto_var.get()==1:
            args.sdvaeauto = True
            args.sdvae = ""
        else:
            args.sdvaeauto = False
            args.sdvae = ""
            if sd_vae_var.get() != "":
                args.sdvae = sd_vae_var.get()
        args.sdaudiovae = sd_audio_vae_var.get() if sd_audio_vae_var.get() != "" else ""
        args.sdconvdirect = sd_convdirect_option(sd_convdirect_var.get())
        args.sdt5xxl = sd_t5xxl_var.get() if sd_t5xxl_var.get() != "" else ""
        args.sdclip1 = sd_clip1_var.get() if sd_clip1_var.get() != "" else ""
        args.sdclip2 = sd_clip2_var.get() if sd_clip2_var.get() != "" else ""
        args.sdphotomaker = sd_photomaker_var.get() if sd_photomaker_var.get() != "" else ""
        args.sdupscaler = sd_upscaler_var.get()  if sd_upscaler_var.get() != "" else ""
        args.sdquant = sd_quant_option(sd_quant_var.get())
        args.sdlora = [item.strip() for item in sd_lora_var.get().split("|") if item]
        # XXX the user may have used '|' since it's used for the LoRAs
        args.sdloramult = sanitize_lora_multipliers(re.split(r"[ |]+", sd_loramult_var.get()))
        args.sdmaingpu = sd_resolve_device(sd_main_gpu_var.get())
        args.gendefaults = gen_defaults_var.get()  if gen_defaults_var.get() != "" else ""
        args.gendefaultsoverwrite = (gen_defaults_overwrite_var.get()==1)
        args.whispermodel = whisper_model_var.get() if whisper_model_var.get() != "" else ""
        args.embeddingsmodel = embeddings_model_var.get()  if embeddings_model_var.get() != "" else ""
        args.embeddingsmaxctx = (0 if embeddings_ctx_var.get()=="" else int(embeddings_ctx_var.get()))
        args.embeddingsgpu = (embeddings_gpu_var.get()==1)

        args.ttsthreads = (0 if tts_threads_var.get()=="" else int(tts_threads_var.get()))
        args.ttsmaxlen = (default_ttsmaxlen if ttsmaxlen_var.get()=="" else int(ttsmaxlen_var.get()))
        args.ttsgpu = (ttsgpu_var.get()==1)
        if tts_model_var.get() != "":
            args.ttsmodel = tts_model_var.get()
            args.ttswavtokenizer = wavtokenizer_var.get()
            args.ttsdir = tts_dir_var.get()
        else:
            args.ttsmodel = ""
            args.ttswavtokenizer = ""
            args.ttsdir = ""

        args.musicllm = musicllm_var.get()
        args.musicembeddings = musicembeddings_var.get()
        args.musicdiffusion = musicdiffusion_var.get()
        args.musicvae = musicvae_var.get()
        args.musiclowvram = musiclowvram_var.get()==1

        args.admin = (admin_var.get()==1 and not args.cli)
        args.admindir = admin_dir_var.get()
        args.adminpassword = admin_password_var.get()
        args.singleinstance = (singleinstance_var.get()==1)
        args.routermode = (router_mode_var.get()==1 and admin_var.get()==1)
        args.autoswapmode = (autoswap_mode_var.get()==1 and router_mode_var.get()==1 and admin_var.get()==1)
        args.baseconfig = baseconfig_var.get()
        args.adminunloadtimeout = (0 if admin_unload_timeout_var.get()=="" else int(admin_unload_timeout_var.get()))
        args.showgui = False #prevent showgui from leaking into configs, its cli only

    def import_vars(mydict):
        global importvars_in_progress
        importvars_in_progress = True
        mydict = convert_invalid_args(mydict)

        if "threads" in mydict:
            threads_var.set(mydict["threads"])
        usemlock.set(1 if "usemlock" in mydict and mydict["usemlock"] else 0)
        if "debugmode" in mydict:
            debugmode.set(mydict["debugmode"])
        launchbrowser.set(1 if "launch" in mydict and mydict["launch"] else 0)
        highpriority.set(1 if "highpriority" in mydict and mydict["highpriority"] else 0)
        usemmap.set(1 if "usemmap" in mydict and mydict["usemmap"] else 0)
        smartcontext_var.set(1 if "smartcontext" in mydict and mydict["smartcontext"] else 0)
        flashattention_var.set(0 if "noflashattention" in mydict and mydict["noflashattention"] else 1)
        contextshift_var.set(0 if "noshift" in mydict and mydict["noshift"] else 1)
        fastforward_var.set(0 if "nofastforward" in mydict and mydict["nofastforward"] else 1)
        swa_var.set(0 if "noswa" in mydict and mydict["noswa"] else 1)
        swa_padding_var.set(mydict["swapadding"] if ("swapadding" in mydict) else swa_padding_default)
        smartcache_var.set(1 if "smartcache" in mydict and mydict["smartcache"] else 0)
        smartcacheslots_var.set(mydict["smartcache"] if ("smartcache" in mydict and mydict["smartcache"] and int(mydict["smartcache"])>1) else savestate_limit_default)
        remotetunnel_var.set(1 if "remotetunnel" in mydict and mydict["remotetunnel"] else 0)
        keepforeground.set(1 if "foreground" in mydict and mydict["foreground"] else 0)
        terminalonly.set(1 if "cli" in mydict and mydict["cli"] else 0)
        pipelineparallel.set(0 if "nopipelineparallel" in mydict and mydict["nopipelineparallel"] else 1)
        quietmode.set(1 if "quiet" in mydict and mydict["quiet"] else 0)
        nocertifymode.set(1 if "nocertify" in mydict and mydict["nocertify"] else 0)
        nomodel.set(1 if "nomodel" in mydict and mydict["nomodel"] else 0)
        lowvram_var.set(1 if "lowvram" in mydict and mydict["lowvram"] else 0)
        if "quantkv" in mydict:
            qkvstr = str(mydict["quantkv"]).lower()
            qkvval = 0
            if qkvstr=="bf16" or qkvstr=="3": #migration for old index based values
                qkvval = 1
            elif qkvstr=="q8_0" or qkvstr=="1":
                qkvval = 2
            elif qkvstr=="q5_1":
                qkvval = 3
            elif qkvstr=="q4_0" or qkvstr=="2":
                qkvval = 4
            quantkv_var.set(qkvval)
        if "usecuda" in mydict and mydict["usecuda"] is not None:
            if cublas_option is not None or hipblas_option is not None:
                if cublas_option:
                    runopts_var.set(cublas_option)
                elif hipblas_option:
                    runopts_var.set(hipblas_option)
                gpu_choice_var.set("All")
                for g in range(4):
                    if str(g) in mydict["usecuda"]:
                        gpu_choice_var.set(str(g))
                        break
        elif "usevulkan" in mydict and mydict['usevulkan'] is not None:
            if "noavx2" in mydict and mydict["noavx2"]:
                if vulkan_noavx2_option is not None:
                    runopts_var.set(vulkan_noavx2_option)
                    gpu_choice_var.set("All")
                    for opt in range(0,4):
                        if opt in mydict["usevulkan"]:
                            gpu_choice_var.set(str(opt))
                            break
            elif "failsafe" in mydict and mydict["failsafe"]:
                if vulkan_failsafe_option is not None:
                    runopts_var.set(vulkan_failsafe_option)
                    gpu_choice_var.set("All")
                    for opt in range(0,4):
                        if opt in mydict["usevulkan"]:
                            gpu_choice_var.set(str(opt))
                            break
            else:
                if vulkan_option is not None:
                    runopts_var.set(vulkan_option)
                    gpu_choice_var.set("All")
                    for opt in range(0,4):
                        if opt in mydict["usevulkan"]:
                            gpu_choice_var.set(str(opt))
                            break

        elif ("noavx2" in mydict and "usecpu" in mydict and mydict["usecpu"] and mydict["noavx2"]) or ("failsafe" in mydict and mydict["failsafe"]):
            if failsafe_option is not None:
                runopts_var.set(failsafe_option)
        elif "noavx2" in mydict and mydict["noavx2"]:
            if noavx2_option is not None:
                runopts_var.set(noavx2_option)
        elif "usecpu" in mydict and mydict["usecpu"]:
            if default_option is not None:
                runopts_var.set(default_option)
        if "gpulayers" in mydict and mydict["gpulayers"]:
            gpulayers_var.set(mydict["gpulayers"])
        else:
            gpulayers_var.set("0")
        if "maingpu" in mydict:
            maingpu_var.set(mydict["maingpu"])
        else:
            maingpu_var.set("-1")
        if "tensor_split" in mydict and mydict["tensor_split"]:
            tssep = ','.join(map(str, mydict["tensor_split"]))
            tensor_split_str_vars.set(tssep)
        if "draftgpusplit" in mydict and mydict["draftgpusplit"]:
            tssep = ','.join(map(str, mydict["draftgpusplit"]))
            draftgpusplit_str_vars.set(tssep)
        if "blasthreads" in mydict and mydict["blasthreads"]:
            blas_threads_var.set(str(mydict["blasthreads"]))
        else:
            blas_threads_var.set("")
        if "device" in mydict and mydict["device"]:
            deviceoverride_var.set(str(mydict["device"]))
        else:
            deviceoverride_var.set("")
        if "contextsize" in mydict and mydict["contextsize"]:
            context_var.set(contextsize_text.index(str(mydict["contextsize"])))
        if "overridenativecontext" in mydict and mydict["overridenativecontext"]>0:
            customrope_var.set(1)
            manualrope_var.set(0)
            customrope_nativectx.set(str(mydict["overridenativecontext"]))
        elif "ropeconfig" in mydict and mydict["ropeconfig"] and len(mydict["ropeconfig"])>1:
            customrope_nativectx.set(default_native_ctx)
            if mydict["ropeconfig"][0]>0:
                customrope_var.set(1)
                manualrope_var.set(1)
                customrope_scale.set(str(mydict["ropeconfig"][0]))
                customrope_base.set(str(mydict["ropeconfig"][1]))
            else:
                customrope_var.set(0)
                manualrope_var.set(0)
        else:
            customrope_nativectx.set(default_native_ctx)
            customrope_var.set(0)
            manualrope_var.set(0)
        if "moeexperts" in mydict and mydict["moeexperts"]:
            moeexperts_var.set(mydict["moeexperts"])
        if "moecpu" in mydict and mydict["moecpu"]:
            moecpu_var.set(mydict["moecpu"])
        if "defaultgenamt" in mydict and mydict["defaultgenamt"]:
            defaultgenamt_var.set(mydict["defaultgenamt"])
        if "genlimit" in mydict and mydict["genlimit"]:
            genlimit_var.set(mydict["genlimit"])
        else:
            genlimit_var.set(str(0))
        nobostoken_var.set(mydict["nobostoken"] if ("nobostoken" in mydict) else 0)
        jinja_var.set(mydict["jinja"] if ("jinja" in mydict) else 0)
        jinja_tools_var.set(mydict["jinja_tools"] if ("jinja_tools" in mydict) else 0)
        jinja_think_var.set("default")
        jinja_kwargs = (mydict["jinja_kwargs"] if ("jinja_kwargs" in mydict and mydict["jinja_kwargs"]) else "")
        if isinstance(jinja_kwargs, type({})):
            jinja_kwargs = json.dumps(jinja_kwargs)
        if "jinjathink" in mydict:
            jinja_kwargs = (json.loads(jinja_kwargs) if jinja_kwargs else {})
            if mydict["jinjathink"]=="true":
                jinja_kwargs["enable_thinking"] = True
            if mydict["jinjathink"]=="false":
                jinja_kwargs["enable_thinking"] = False
            jinja_kwargs = json.dumps(jinja_kwargs) if jinja_kwargs else ""
        jinja_kwargs_var.set(jinja_kwargs if jinja_kwargs else "")
        jinjatemplate_var.set(mydict["jinjatemplate"] if ("jinjatemplate" in mydict and mydict["jinjatemplate"]) else "")

        enableguidance_var.set(mydict["enableguidance"] if ("enableguidance" in mydict) else 0)
        if "overridekv" in mydict and mydict["overridekv"]:
            override_kv_var.set(mydict["overridekv"])
        if "overridetensors" in mydict and mydict["overridetensors"]:
            override_tensors_var.set(mydict["overridetensors"])

        if "batchsize" in mydict and mydict["batchsize"]:
            blas_size_var.set(batchsize_values.index(str(mydict["batchsize"])))

        autofit_var.set(1 if "autofit" in mydict and mydict["autofit"] else 0)
        model_var.set(mydict["model_param"] if ("model_param" in mydict and mydict["model_param"]) else "")

        if "autofitpadding" in mydict and mydict["autofitpadding"]:
            autofit_padding_var.set(mydict["autofitpadding"])
        else:
            autofit_padding_var.set(str(default_autofit_padding))

        lora_var.set("")
        if "lora" in mydict and mydict["lora"]:
            if len(mydict["lora"]) > 1:
                lora_var.set(mydict["lora"][0])
            else:
                lora_var.set(mydict["lora"][0])
        loramult_var.set(str(mydict["loramult"]) if ("loramult" in mydict and mydict["loramult"]) else "1.0")

        splitmode_var.set(mydict["splitmode"] if ("splitmode" in mydict and mydict["splitmode"] in splitmode_choices) else splitmode_choices[0])
        mmq_var.set(0 if "nommq" in mydict and mydict["nommq"] else 1)

        mmproj_var.set(mydict["mmproj"] if ("mmproj" in mydict and mydict["mmproj"]) else "")
        mmprojcpu_var.set(1 if ("mmprojcpu" in mydict and mydict["mmprojcpu"]) else 0)
        if "visionmaxres" in mydict and mydict["visionmaxres"]:
            visionmaxres_var.set(mydict["visionmaxres"])
        if "visionmintokens" in mydict and mydict["visionmintokens"]:
            vision_min_tokens_var.set(mydict["visionmintokens"])
        if "visionmaxtokens" in mydict and mydict["visionmaxtokens"]:
            vision_max_tokens_var.set(mydict["visionmaxtokens"])
        draftmodel_var.set(mydict["draftmodel"] if ("draftmodel" in mydict and mydict["draftmodel"]) else "")
        if "draftamount" in mydict:
            draftamount_var.set(mydict["draftamount"])
        if "draftgpulayers" in mydict:
            draftgpulayers_var.set(mydict["draftgpulayers"])

        ssl_cert_var.set("")
        ssl_key_var.set("")
        if "ssl" in mydict and mydict["ssl"]:
            if len(mydict["ssl"]) == 2:
                ssl_cert_var.set(mydict["ssl"][0])
                ssl_key_var.set(mydict["ssl"][1])

        password_var.set(mydict["password"] if ("password" in mydict and mydict["password"]) else "")
        pls_obj = ""
        if ("preloadstory" in mydict and mydict["preloadstory"]):
            pls_obj = mydict["preloadstory"] if not isinstance(mydict["preloadstory"], dict) else json.dumps(mydict["preloadstory"])
        preloadstory_var.set(pls_obj)
        savedatafile_var.set(mydict["savedatafile"] if ("savedatafile" in mydict and mydict["savedatafile"]) else "")
        mcpfile_var.set(mydict["mcpfile"] if ("mcpfile" in mydict and mydict["mcpfile"]) else "")
        chatcompletionsadapter_var.set(mydict["chatcompletionsadapter"] if ("chatcompletionsadapter" in mydict and mydict["chatcompletionsadapter"]) else "")
        port_var.set(mydict["port_param"] if ("port_param" in mydict and mydict["port_param"]) else defaultport)
        host_var.set(mydict["host"] if ("host" in mydict and mydict["host"]) else "")
        multiuser_var.set(mydict["multiuser"] if ("multiuser" in mydict and mydict["multiuser"]>1) else multiuser_concurrent_limit)
        multiplayer_var.set(mydict["multiplayer"] if ("multiplayer" in mydict) else 0)
        websearch_var.set(mydict["websearch"] if ("websearch" in mydict) else 0)
        download_dir_var.set(mydict["downloaddir"] if ("downloaddir" in mydict and mydict["downloaddir"]) else "")
        parallel_requests_var.set(mydict["parallelrequests"] if ("parallelrequests" in mydict and mydict["parallelrequests"]>1) else 1)

        horde_name_var.set(mydict["hordemodelname"] if ("hordemodelname" in mydict and mydict["hordemodelname"]) else "koboldcpp")
        horde_context_var.set(mydict["hordemaxctx"] if ("hordemaxctx" in mydict and mydict["hordemaxctx"]) else maxhordectx)
        horde_gen_var.set(mydict["hordegenlen"] if ("hordegenlen" in mydict and mydict["hordegenlen"]) else maxhordelen)
        horde_apikey_var.set(mydict["hordekey"] if ("hordekey" in mydict and mydict["hordekey"]) else "")
        horde_workername_var.set(mydict["hordeworkername"] if ("hordeworkername" in mydict and mydict["hordeworkername"]) else "")
        usehorde_var.set(1 if ("hordekey" in mydict and mydict["hordekey"]) else 0)
        maxrequestsize_var.set(mydict["maxrequestsize"] if ("maxrequestsize" in mydict and mydict["maxrequestsize"]) else 32)
        ratelimit_var.set(mydict["ratelimit"] if ("ratelimit" in mydict and mydict["ratelimit"]) else 0)
        reqtimeout_var.set(mydict["reqtimeout"] if ("reqtimeout" in mydict and mydict["reqtimeout"]) else 0)
        rpcmode_var.set(mydict["rpcmode"] if ("rpcmode" in mydict and mydict["rpcmode"]) else "disabled")
        rpc_host_var.set(mydict["rpchost"] if ("rpchost" in mydict and mydict["rpchost"]) else "0.0.0.0")
        rpc_port_var.set(mydict["rpcport"] if ("rpcport" in mydict and mydict["rpcport"]) else str(default_rpc_port))
        rpcendpoints_var.set(mydict["rpctargets"] if ("rpctargets" in mydict and mydict["rpctargets"]) else "")
        rpc_device_var.set(mydict["rpcdevice"] if ("rpcdevice" in mydict and mydict["rpcdevice"]) else "")

        sd_model_var.set(mydict["sdmodel"] if ("sdmodel" in mydict and mydict["sdmodel"]) else "")
        sd_clamped_var.set(int(mydict["sdclamped"]) if ("sdclamped" in mydict and mydict["sdclamped"]) else 0)
        sd_clamped_soft_var.set(int(mydict["sdclampedsoft"]) if ("sdclampedsoft" in mydict and mydict["sdclampedsoft"]) else 0)
        sd_threads_var.set(str(mydict["sdthreads"]) if ("sdthreads" in mydict and mydict["sdthreads"]) else str(default_threads))
        sd_quant_var.set(sd_quant_choices[(mydict["sdquant"] if ("sdquant" in mydict and mydict["sdquant"]>=0 and mydict["sdquant"]<len(sd_quant_choices)) else 0)])
        sd_flash_attention_var.set(1 if ("sdflashattention" in mydict and mydict["sdflashattention"]) else 0)
        sd_offload_cpu_var.set(1 if ("sdoffloadcpu" in mydict and mydict["sdoffloadcpu"]) else 0)
        sd_vae_device_var.set(sd_get_device_name(sd_resolve_device(mydict.get("sdvaedevice"), default_sdvaedevice)))
        sd_clip_device_var.set(sd_get_device_name(sd_resolve_device(mydict.get("sdclipdevice"), default_sdclipdevice)))
        sd_convdirect_var.set(sd_convdirect_option(mydict.get("sdconvdirect")))
        sd_vae_var.set(mydict["sdvae"] if ("sdvae" in mydict and mydict["sdvae"]) else "")
        sd_audio_vae_var.set(mydict["sdaudiovae"] if ("sdaudiovae" in mydict and mydict["sdaudiovae"]) else "")
        sd_t5xxl_var.set(mydict["sdt5xxl"] if ("sdt5xxl" in mydict and mydict["sdt5xxl"]) else "")
        sd_clip1_var.set(mydict["sdclip1"] if ("sdclip1" in mydict and mydict["sdclip1"]) else "")
        sd_clip2_var.set(mydict["sdclip2"] if ("sdclip2" in mydict and mydict["sdclip2"]) else "")
        sd_photomaker_var.set(mydict["sdphotomaker"] if ("sdphotomaker" in mydict and mydict["sdphotomaker"]) else "")
        sd_upscaler_var.set(mydict["sdupscaler"] if ("sdupscaler" in mydict and mydict["sdupscaler"]) else "")
        sd_vaeauto_var.set(1 if ("sdvaeauto" in mydict and mydict["sdvaeauto"]) else 0)
        sd_tiled_vae_var.set(str(mydict["sdtiledvae"]) if ("sdtiledvae" in mydict and mydict["sdtiledvae"]) else str(default_vae_tile_threshold))
        sdl_sanitized = sanitize_lora_list(mydict.get('sdlora'))
        sd_lora_var.set("|".join(sdl_sanitized))
        sd_loramult_var.set(" ".join(f"{n:.3f}".rstrip('0').rstrip('.') for n in mydict.get("sdloramult", [])))
        sd_main_gpu_var.set(sd_get_device_name(sd_resolve_device(mydict.get("sdmaingpu"))))
        if sdl_sanitized and len(sdl_sanitized)==1 and os.path.isdir(sdl_sanitized[0]):
            sd_runtime_loras_var.set(1)
        else:
            sd_runtime_loras_var.set(0)
        sd_vram_limit_var.set(str(mydict["sdvramlimit"]) if ("sdvramlimit" in mydict and mydict["sdvramlimit"]) else "")

        gendefaults = (mydict["gendefaults"] if ("gendefaults" in mydict and mydict["gendefaults"]) else "")
        if isinstance(gendefaults, type({})):
            gendefaults = json.dumps(gendefaults)
        gen_defaults_var.set(gendefaults)
        gen_defaults_overwrite_var.set(1 if "gendefaultsoverwrite" in mydict and mydict["gendefaultsoverwrite"] else 0)

        whisper_model_var.set(mydict["whispermodel"] if ("whispermodel" in mydict and mydict["whispermodel"]) else "")

        tts_threads_var.set(str(mydict["ttsthreads"]) if ("ttsthreads" in mydict and mydict["ttsthreads"]) else str(default_threads))
        tts_model_var.set(mydict["ttsmodel"] if ("ttsmodel" in mydict and mydict["ttsmodel"]) else "")
        wavtokenizer_var.set(mydict["ttswavtokenizer"] if ("ttswavtokenizer" in mydict and mydict["ttswavtokenizer"]) else "")
        ttsgpu_var.set(mydict["ttsgpu"] if ("ttsgpu" in mydict) else 0)
        ttsmaxlen_var.set(str(mydict["ttsmaxlen"]) if ("ttsmaxlen" in mydict and mydict["ttsmaxlen"]) else str(default_ttsmaxlen))
        tts_dir_var.set(mydict["ttsdir"] if ("ttsdir" in mydict and mydict["ttsdir"]) else "")

        musicllm_var.set(mydict["musicllm"] if ("musicllm" in mydict and mydict["musicllm"]) else "")
        musicembeddings_var.set(mydict["musicembeddings"] if ("musicembeddings" in mydict and mydict["musicembeddings"]) else "")
        musicdiffusion_var.set(mydict["musicdiffusion"] if ("musicdiffusion" in mydict and mydict["musicdiffusion"]) else "")
        musicvae_var.set(mydict["musicvae"] if ("musicvae" in mydict and mydict["musicvae"]) else "")
        musiclowvram_var.set(mydict["musiclowvram"] if ("musiclowvram" in mydict) else 0)

        embeddings_model_var.set(mydict["embeddingsmodel"] if ("embeddingsmodel" in mydict and mydict["embeddingsmodel"]) else "")
        embeddings_ctx_var.set(str(mydict["embeddingsmaxctx"]) if ("embeddingsmaxctx" in mydict and mydict["embeddingsmaxctx"]) else "")
        embeddings_gpu_var.set(mydict["embeddingsgpu"] if ("embeddingsgpu" in mydict) else 0)

        admin_var.set(mydict["admin"] if ("admin" in mydict) else 0)
        router_mode_var.set(mydict["routermode"] if ("routermode" in mydict) else 0)
        autoswap_mode_var.set(mydict["autoswapmode"] if ("autoswapmode" in mydict) else 0)
        admin_dir_var.set(mydict["admindir"] if ("admindir" in mydict and mydict["admindir"]) else "")
        baseconfig_var.set(mydict["baseconfig"] if ("baseconfig" in mydict and mydict["baseconfig"]) else "")
        admin_password_var.set(mydict["adminpassword"] if ("adminpassword" in mydict and mydict["adminpassword"]) else "")
        admin_unload_timeout_var.set(mydict["adminunloadtimeout"] if ("adminunloadtimeout" in mydict and mydict["adminunloadtimeout"]) else 0)
        singleinstance_var.set(mydict["singleinstance"] if ("singleinstance" in mydict) else 0)

        importvars_in_progress = False
        gui_changed_modelfile()
        if "istemplate" in mydict and mydict["istemplate"]:
            auto_set_backend_gui(True)

    def save_config_gui():
        nonlocal kcpp_exporting_template
        kcpp_exporting_template = False
        export_vars()
        savdict = json.loads(json.dumps(args.__dict__,indent=2))
        for key in deprecated_keys:
            savdict.pop(key, None)  # avoids KeyError if missing
        savdict["istemplate"] = False
        file_type = [("KoboldCpp Settings", "*.kcpps")]
        filename = zentk_asksaveasfilename(filetypes=file_type, defaultextension=".kcpps",title="Save kcpps settings config file")
        if not filename:
            return
        save_config_dict(filename, savdict, False)
        pass

    def load_config_gui(): #this is used to populate the GUI with a config file, whereas load_config_cli simply overwrites cli args
        file_type = [("KoboldCpp Settings", "*.kcpps *.kcppt")]
        global runmode_untouched, zenity_permitted
        filename = zentk_askopenfilename(filetypes=file_type, defaultextension=".kcppt", initialdir=None, title="Select kcpps or kcppt settings config file")
        if not filename or filename=="":
            return
        if not os.path.exists(filename) or os.path.getsize(filename)<4 or os.path.getsize(filename)>50000000: #for sanity, check invaid kcpps
            print("The selected config file seems to be invalid.")
            if zenity_permitted:
                print("You can try using the legacy filepicker instead (in Extra).")
            return
        runmode_untouched = False
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            dict = json.load(f)
            import_vars(dict)
        pass

    def display_help():
        popup = ctk.CTkToplevel(root)
        popup.title("Help Menu")
        popup.geometry("380x440")
        templatedchoice_var = ctk.StringVar(value="")
        templatecatbox_var = ctk.StringVar(value="Newbie Templates")
        POPULAR_TEMPLATE_LBL = "Popular Templates"
        NEWB_TEMPLATE_LBL = "Newbie Templates"
        POPULAR_TEMPLATE_REPO = "popular-templates"
        NEWB_TEMPLATE_REPO = "newbie-templates"
        newbdesc1 = newbdesc2 = None
        def display_hf():
            popup.destroy()
            model_searcher()
        def fetch_easy_templates(a,b,c):
            nonlocal templatechoicebox, templatecatbox, newbdesc1, newbdesc2
            noobmodels = []
            resp = None
            if templatecatbox_var.get() == POPULAR_TEMPLATE_LBL:
                resp = make_url_request(f"https://huggingface.co/api/models/koboldcpp/{POPULAR_TEMPLATE_REPO}", None, 'GET', {}, 10)
                newbdesc1.pack_forget()
                newbdesc2.pack_forget()
                commdesc.pack(pady=(10, 0))
                for m in resp["siblings"]:
                    entry = m["rfilename"]
                    if entry.endswith(".kcppt"):
                        noobmodels.append(entry[:-6])
            else:
                resp = make_url_request(f"https://huggingface.co/api/models/koboldcpp/{NEWB_TEMPLATE_REPO}", None, 'GET', {}, 10)
                newbdesc1.pack(pady=(10, 0))
                newbdesc2.pack(pady=(10, 0))
                commdesc.pack_forget()
                for m in resp["siblings"]:
                    entry = m["rfilename"]
                    if entry.endswith(".kcppt") and "LowSpec" in entry:
                        noobmodels.append(entry[:-6])
                for m in resp["siblings"]:
                    entry = m["rfilename"]
                    if entry.endswith(".kcppt") and "MidSpec" in entry:
                        noobmodels.append(entry[:-6])
                for m in resp["siblings"]:
                    entry = m["rfilename"]
                    if entry.endswith(".kcppt") and "HighSpec" in entry:
                        noobmodels.append(entry[:-6])
            templatechoicebox.configure(values=noobmodels)
            if len(noobmodels)>0:
                templatedchoice_var.set(noobmodels[0])
        def load_easy_template():
            nonlocal templatechoicebox, templatecatbox, newbdesc1, newbdesc2
            cat = templatecatbox.get()
            repo = (POPULAR_TEMPLATE_REPO if cat==POPULAR_TEMPLATE_LBL else NEWB_TEMPLATE_REPO)
            fname = f"https://huggingface.co/koboldcpp/{repo}/resolve/main/{templatedchoice_var.get()}.kcppt"
            data = make_url_request(fname,data=None,method="GET")
            if data is not None:
                import_vars(data)
            popup.destroy()
        ctk.CTkLabel(popup, text="Helpful Newbie Resources").pack(pady=(5, 0))
        ctk.CTkButton(popup, text="Read the Wiki", command=display_wiki).pack(pady=5)
        ctk.CTkButton(popup, text="Read Starter Guides", command=display_starter_guides).pack(pady=5)
        ctk.CTkButton(popup, text="Search Model on Hugginface", command=display_hf).pack(pady=5)
        ctk.CTkLabel(popup, text="Or, Pick an Easy Template for Newbies").pack(pady=(12, 0))
        templatecatbox = ctk.CTkComboBox(popup, values=[NEWB_TEMPLATE_LBL,POPULAR_TEMPLATE_LBL], width=280, variable=templatecatbox_var, state="readonly")
        templatecatbox.pack(pady=5)
        templatechoicebox = ctk.CTkComboBox(popup, values=[], width=280, variable=templatedchoice_var, state="readonly")
        templatechoicebox.pack(pady=5)
        ctk.CTkButton(popup, text="Load Template", command=load_easy_template).pack(pady=5)
        newbdesc1 = ctk.CTkLabel(popup, text="LowSpec = Recommend 6GB VRAM\nMidSpec = Recommend 12GB VRAM\nHighSpec = Recommend 24GB VRAM")
        newbdesc1.pack(pady=(10, 0))
        newbdesc2 = ctk.CTkLabel(popup, text="Everything = All Features         Text = Text Generation\nImages = Image Generation         Vision = Image Recognition\nVoice = Speech Generation         Audio = Speech Recognition")
        newbdesc2.pack(pady=(10, 0))
        commdesc = ctk.CTkLabel(popup, text="Templates here are subject to change from time to time.\n\nFound a broken template? Want to contribute one?\nVisit https://huggingface.co/koboldcpp/popular-templates/")
        commdesc.pack_forget()
        templatecatbox_var.trace_add("write", fetch_easy_templates)
        fetch_easy_templates(1,1,1)
        popup.transient(root)

    def display_help_models():
        LaunchWebbrowser("https://github.com/LostRuins/koboldcpp/wiki#what-models-does-koboldcpp-support-what-architectures-are-supported","Cannot launch help in browser.")
    def display_starter_guides():
        LaunchWebbrowser("https://github.com/LostRuins/koboldcpp/wiki#step-by-step-guides","Cannot launch help in browser.")
    def display_wiki():
        LaunchWebbrowser("https://github.com/LostRuins/koboldcpp/wiki#the-koboldcpp-faq-and-knowledgebase","Cannot launch help in browser.")
    def display_updates():
        LaunchWebbrowser("https://github.com/LostRuins/koboldcpp/releases/latest","Cannot launch updates in browser.")

    ctk.CTkButton(tabs , text = "Launch", fg_color="#2f8d3c", hover_color="#2faa3c", command = guilaunch, width=100, height = 35 ).grid(row=1,column=1, stick="se", padx=(25), pady=5)

    ctk.CTkButton(tabs , text = "Update", fg_color="#9900cc", hover_color="#aa11dd", command = display_updates, width=90, height = 35 ).grid(row=1,column=0, stick="sw", padx= 5, pady=5)
    ctk.CTkButton(tabs , text = "Save Config", fg_color="#084a66", hover_color="#085a88", command = save_config_gui, width=60, height = 35 ).grid(row=1,column=1, stick="sw", padx= 5, pady=5)
    ctk.CTkButton(tabs , text = "Load Config", fg_color="#084a66", hover_color="#085a88", command = load_config_gui, width=60, height = 35 ).grid(row=1,column=1, stick="sw", padx= (92), pady=5)
    ctk.CTkButton(tabs , text = "Get Help", fg_color="#992222", hover_color="#bb3333", command = display_help, width=70, height = 35 ).grid(row=1,column=1, stick="sw", padx= (180), pady=5)

    # start a thread that tries to get actual gpu names and layer counts
    gpuinfo_thread = threading.Thread(target=auto_set_backend_gui)
    gpuinfo_thread.start() #submit job in new thread so nothing is waiting

    if args.showgui:
        if isinstance(args, argparse.Namespace):
            mydict = vars(args)
            import_vars(mydict)

    # runs main loop until closed or launch clicked
    try:
        root.mainloop()
    except (KeyboardInterrupt,SystemExit):
        exitcounter = 999
        print("Exiting by user request.")
        sys.exit(0)


    if nextstate==0:
        exitcounter = 999
        print("Exiting by user request.")
        sys.exit(0)
    else:
        # processing vars
        kcpp_exporting_template = False
        export_vars()

        if not has_valid_model():
            exitcounter = 999
            print("")
            time.sleep(0.5)
            print("Error: No valid model files were selected. Cannot continue.", flush=True)
            time.sleep(2)
            sys.exit(2)

def show_gui_msgbox(title,message):
    print(title + ": " + message, flush=True)
    try:
        from tkinter import messagebox
        import tkinter as tk
        root2 = tk.Tk()
        root2.attributes("-alpha", 0)
        messagebox.showerror(title=title, message=message)
        root2.withdraw()
        root2.destroy()
    except Exception:
        pass

def show_gui_yesnobox(title,message,icon='error'):
    print(title + ": " + message, flush=True)
    try:
        from tkinter import messagebox
        import tkinter as tk
        root2 = tk.Tk()
        root2.attributes("-alpha", 0)
        result = messagebox.askquestion(title=title, message=message,icon=icon)
        root2.withdraw()
        root2.destroy()
        return result
    except Exception:
        return False
        pass

def print_with_time(txt):
    print(f"{datetime.now().strftime('[%H:%M:%S]')} " + txt, flush=True)

def make_url_request(url, data, method='POST', headers={}, timeout=300):
    global nocertify
    try:
        request = None
        ssl_cert_dir = os.environ.get('SSL_CERT_DIR')
        if not ssl_cert_dir and not nocertify and os.name != 'nt':
            os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
        if method=='POST':
            json_payload = json.dumps(data).encode('utf-8')
            request = urllib.request.Request(url, data=json_payload, headers=headers, method=method)
            request.add_header('content-type', 'application/json')
        else:
            request = urllib.request.Request(url, headers=headers, method=method)
        response_data = ""
        with urllib.request.urlopen(request,timeout=timeout) as response:
            content_type = response.headers.get('Content-Type', '')
            response_data = response.read()
            if 'audio/wav' in content_type:
                return response_data #raw binary
            else:
                json_response = json.loads(response_data.decode('utf-8',"ignore"))
                return json_response
    except urllib.error.HTTPError as e:
        try:
            errmsg = e.read().decode('utf-8',"ignore")
            print_with_time(f"Error: {e} - {errmsg}")
        except Exception as e:
            print_with_time(f"Error: {e}")
        return None
    except Exception as e:
        print_with_time(f"Error: {e} - {response_data}")
        return None

#A very simple and stripped down embedded horde worker with no dependencies
def run_horde_worker(args, api_key, worker_name, worker_id, parallel_batching_threads):
    global friendlymodelname, maxhordectx, maxhordelen, exitcounter, punishcounter, modelbusy, session_starttime, sslvalid
    epurl = get_my_epurl()

    def submit_completed_generation(url, jobid, sessionstart, submit_dict):
        global exitcounter, punishcounter, session_kudos_earned, session_jobs, rewardcounter
        reply = make_url_request_horde(url, submit_dict)
        if not reply:
            punishcounter += 1
            print_with_time(f"Worker {worker_id} - Error, Job submit failed.")
        else:
            reward = reply["reward"]
            session_kudos_earned += reward
            session_jobs += 1
            curtime = datetime.now()
            elapsedtime = curtime - sessionstart
            hrs_float = elapsedtime.total_seconds() / 3600
            hrs = int(hrs_float)
            mins = elapsedtime.seconds // 60 % 60
            secs = elapsedtime.seconds % 60
            elapsedtimestr = f"{hrs:03d}h:{mins:02d}m:{secs:02d}s"
            earnrate = session_kudos_earned / hrs_float
            jobrate = session_jobs / hrs_float
            jobcost = session_kudos_earned / session_jobs
            print_with_time(f'Worker {worker_id} - Submitted {jobid} and earned {reward:.0f} kudos\n[Total:{session_kudos_earned:.0f} kudos, Time:{elapsedtimestr}, Jobs:{session_jobs}, EarnRate:{earnrate:.2f} kudos/hr, JobRate:{jobrate:.2f} jobs/hr, JobCost:{jobcost:.2f} kudos/job]')
            rewardcounter += 1
            if rewardcounter > 50:
                rewardcounter = 0
                if exitcounter > 1:
                    exitcounter -= 1

    def make_url_request_horde(url, data, method='POST',addmykey=False):
        global password
        headers = headers = {"apikey": api_key,'User-Agent':'KoboldCppEmbeddedWorkerV2','Client-Agent':'KoboldCppEmbedWorker:2'}
        if addmykey and password!="":
            headers["Authorization"] = f"Bearer {password}"
        ret = make_url_request(url, data, method, headers)
        if not ret:
            print(f"Worker {worker_id} - Make sure your Horde API key and worker name is valid!")
        return ret

    current_id = None
    current_payload = None
    current_generation = None
    session_starttime = datetime.now()
    sleepy_counter = 0 #if this exceeds a value, worker becomes sleepy (slower)
    exitcounter = 0
    print(f"===\nEmbedded Horde Worker '{worker_name}' Starting...\n(To use your own Horde Bridge/Scribe worker instead, don't set your API key)\n")
    BRIDGE_AGENT = "KoboldCppEmbedWorker:2:https://github.com/LostRuins/koboldcpp"
    cluster = "https://aihorde.net"
    while exitcounter < 10:
        time.sleep(3)
        readygo = make_url_request_horde(f'{epurl}/api/v1/info/version', None,'GET',addmykey=True)
        if readygo:
            print_with_time(f"Worker {worker_id} - Embedded Horde Worker '{worker_name}' is started.")
            break

    while exitcounter < 10:
        currentjob_attempts = 0
        current_generation = None

        if punishcounter >= 5:
            punishcounter = 0
            exitcounter += 1
            if exitcounter < 10:
                penaltytime = (2 ** exitcounter)
                print_with_time(f"Horde Worker Paused for {penaltytime} min - Too many errors. It will resume automatically, but you should restart it.")
                print_with_time("Caution: Too many failed jobs may lead to entering maintenance mode.")
                time.sleep(60 * penaltytime)
                print_with_time(f"Worker {worker_id} - Horde Worker Resumed")
            else:
                 print_with_time(f"Worker {worker_id} - Horde Worker Exit limit reached, too many errors.")

        global last_non_horde_req_time
        sec_since_non_horde = time.time() - last_non_horde_req_time
        no_recent_local_usage = sec_since_non_horde>20
        if not no_recent_local_usage:
            #print_with_time(f"Recent Local Usage - Horde Worker Waiting...")
            time.sleep(1)
            continue

        #first, make sure we are not generating (queue is empty)
        if modelbusy.locked():
            time.sleep(0.2)
            continue

        #pop new request
        gen_dict = {
            "name": worker_name,
            "models": [friendlymodelname],
            "max_length": maxhordelen,
            "max_context_length": min(maxctx,(maxctx if maxhordectx==0 else maxhordectx)),
            "priority_usernames": [],
            "softprompts": [],
            "bridge_agent": BRIDGE_AGENT,
        }
        if parallel_batching_threads>1:
            gen_dict["threads"] = parallel_batching_threads
        pop = make_url_request_horde(f'{cluster}/api/v2/generate/text/pop',gen_dict)
        if not pop:
            punishcounter += 1
            print_with_time(f"Worker {worker_id} - Failed to fetch job from {cluster}. Waiting 10 seconds...")
            time.sleep(10)
            continue
        if not pop["id"]:
            slp = (1 if sleepy_counter<10 else (2 if sleepy_counter<25 else 3))
            time.sleep(slp)
            sleepy_counter += 1
            if sleepy_counter==20:
                print_with_time(f"Worker {worker_id} - No recent jobs, entering low power mode...")
            continue

        sleepy_counter = 0
        current_id = pop['id']
        current_payload = pop['payload']
        print("") #empty newline
        print_with_time(f"Job {current_id} received from {cluster} for {current_payload.get('max_length',0)} tokens and {current_payload.get('max_context_length',0)} max context. Starting generation...")

        #do gen
        while exitcounter < 10:
            if parallel_batching_threads>1 or not modelbusy.locked():
                #horde gets a genkey to avoid KCPP overlap
                current_payload['genkey'] = f"HORDEREQ_{random.randint(100, 999)}"
                current_generation = make_url_request_horde(f'{epurl}/api/v1/generate', current_payload, method='POST',addmykey=True)
                if current_generation:
                    break
                else:
                    currentjob_attempts += 1
                    if currentjob_attempts>5:
                        break

            print_with_time(f"Worker {worker_id} - Server Busy - Not ready to generate...")
            time.sleep(5)

        #submit reply
        print("") #empty newline
        if current_generation:
            submit_dict = {
                "id": current_id,
                "generation": current_generation["results"][0]["text"],
                "state": "ok"
            }
            submiturl = cluster + '/api/v2/generate/text/submit'
            submit_thread = threading.Thread(target=submit_completed_generation, args=(submiturl, current_id, session_starttime, submit_dict))
            submit_thread.start() #submit job in new thread so nothing is waiting
        else:
            print_with_time(f"Worker {worker_id} - Error, Abandoned current job due to errors. Getting new job.")
        current_id = None
        current_payload = None
        time.sleep(0.1)

    if exitcounter<100:
        print_with_time(f"Worker {worker_id} - Horde Worker Shutdown - Too many errors.")
    else:
        print_with_time(f"Worker {worker_id} - Horde Worker Shutdown - Server Closing.")
    exitcounter = 999
    time.sleep(3)
    sys.exit(2)

def convert_invalid_args(args):
    dict = args
    if isinstance(args, argparse.Namespace):
        dict = vars(args)
    if "usecuda" not in dict and "usecublas" in dict and dict["usecublas"]:
        dict["usecuda"] = dict["usecublas"]
    if "usecuda" in dict and dict["usecuda"] and "lowvram" in dict["usecuda"]:
        dict["lowvram"] = True
    if "usecuda" in dict and dict["usecuda"] and "nommq" in dict["usecuda"]:
        dict["nommq"] = True
    if "usecuda" in dict and dict["usecuda"] and "rowsplit" in dict["usecuda"]:
        dict["splitmode"] = "row"
    if "batchsize" not in dict and "blasbatchsize" in dict and dict["blasbatchsize"]:
        dict["batchsize"] = dict["blasbatchsize"]
    if "sdconfig" in dict and dict["sdconfig"] and len(dict["sdconfig"])>0:
        dict["sdmodel"] = dict["sdconfig"][0]
        if dict["sdconfig"] and len(dict["sdconfig"]) > 1:
            dict["sdclamped"] = 512
        if dict["sdconfig"] and len(dict["sdconfig"]) > 2:
            dict["sdthreads"] = int(dict["sdconfig"][2])
        if dict["sdconfig"] and len(dict["sdconfig"]) > 3:
            dict["sdquant"] = (2 if dict["sdconfig"][3]=="quant" else 0)
    if "hordeconfig" in dict and dict["hordeconfig"] and dict["hordeconfig"][0]!="":
        dict["hordemodelname"] = dict["hordeconfig"][0]
        if len(dict["hordeconfig"]) > 1:
            dict["hordegenlen"] = int(dict["hordeconfig"][1])
        if len(dict["hordeconfig"]) > 2:
            dict["hordemaxctx"] = int(dict["hordeconfig"][2])
        if len(dict["hordeconfig"]) > 4:
            dict["hordekey"] = dict["hordeconfig"][3]
            dict["hordeworkername"] = dict["hordeconfig"][4]
    if "noblas" in dict and dict["noblas"]:
        dict["usecpu"] = True
    if "failsafe" in dict and dict["failsafe"]: #failsafe implies noavx2
        dict["noavx2"] = True
    if "skiplauncher" in dict and dict["skiplauncher"]:
        dict["showgui"] = False
    if ("model_param" not in dict or not dict["model_param"]) and ("model" in dict):
        model_value = dict["model"] #may be null, empty/non-empty string, empty/non empty array
        if isinstance(model_value, str) and model_value:  # Non-empty string
            dict["model_param"] = model_value
        elif isinstance(model_value, list) and model_value:  # Non-empty list
            dict["model_param"] = model_value[0]  # Take the first file in the list
    if ("port_param" in dict and dict["port_param"] and dict["port_param"]!=defaultport):
        dict["port"] = dict["port_param"]
    if "sdnotile" in dict and "sdtiledvae" not in dict:
        dict["sdtiledvae"] = (0 if (dict["sdnotile"]) else default_vae_tile_threshold) # convert legacy option
    if 'sdquant' in dict and type(dict['sdquant']) is bool:
        dict['sdquant'] = 2 if dict['sdquant'] else 0
    if "sdclipl" in dict and "sdclip1" not in dict:
        dict["sdclip1"] = dict["sdclipl"]
    if "sdclipg" in dict and "sdclip2" not in dict:
        dict["sdclip2"] = dict["sdclipg"]
    if "jinja_tools" in dict and dict["jinja_tools"]:
        dict["jinja"] = True
    if "jinja_kwargs" in dict and dict["jinja_kwargs"]:
        dict["jinja"] = True
    if "sdgendefaults" in dict and "gendefaults" not in dict:
        dict["gendefaults"] = dict["sdgendefaults"]
    if "flashattention" in dict and "noflashattention" not in dict:
        dict["noflashattention"] = not dict["flashattention"]
    if "useswa" in dict and "noswa" not in dict:
        dict["noswa"] = not dict["useswa"]
    if "sdlora" in dict:
        dict["sdlora"] = sanitize_lora_list(dict["sdlora"])
    if "sdloramult" in dict:
        dict["sdloramult"] = sanitize_lora_multipliers(dict["sdloramult"])
    if "sdclipgpu" in dict and dict.get("sdclipdevice") is None:
        dict["sdclipdevice"] = sd_get_device_number("main" if dict["sdclipgpu"] else "CPU")
    if "sdvaecpu" in dict and dict.get("sdvaedevice") is None:
        dict["sdvaedevice"] = sd_get_device_number("CPU" if dict["sdvaecpu"] else "main")
    return args

def setuptunnel(global_memory, has_sd, has_music):
    # This script will help setup a cloudflared tunnel for accessing KoboldCpp over the internet
    # It should work out of the box on both linux and windows
    try:
        global sslvalid
        httpsaffix = ("https" if sslvalid else "http")
        ssladd = (" --no-tls-verify" if sslvalid else "")
        def run_tunnel():
            tunnelproc = None
            tunneloutput = ""
            tunnelrawlog = ""
            time.sleep(0.2)
            tunnelbinary = ""
            if os.name == 'nt':
                print("Starting Cloudflare Tunnel for Windows, please wait...", flush=True)
                tunnelbinary = "cloudflared.exe"
            elif sys.platform=="darwin":
                print("Starting Cloudflare Tunnel for MacOS, please wait...", flush=True)
                tunnelbinary = "./cloudflared"
            elif sys.platform == "linux" and platform.machine().lower() == "aarch64":
                print("Starting Cloudflare Tunnel for ARM64 Linux, please wait...", flush=True)
                tunnelbinary = "./cloudflared-linux-arm64"
            else:
                print("Starting Cloudflare Tunnel for Linux, please wait...", flush=True)
                tunnelbinary = "./cloudflared-linux-amd64"

            tunnelproc = None
            displayedport = (args.port if not args.proxy_port else args.proxy_port)
            if sys.platform == "linux":
                clean_env = os.environ.copy()
                clean_env.pop("LD_LIBRARY_PATH", None)
                clean_env["PATH"] = "/usr/bin:/bin"
                tunnelproc = subprocess.Popen(f"{tunnelbinary} tunnel --url {httpsaffix}://localhost:{int(displayedport)}{ssladd}", text=True, encoding='utf-8', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=clean_env)
            else:
                tunnelproc = subprocess.Popen(f"{tunnelbinary} tunnel --url {httpsaffix}://localhost:{int(displayedport)}{ssladd}", text=True, encoding='utf-8', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(10)

            def tunnel_reader():
                nonlocal tunnelproc,tunneloutput,tunnelrawlog
                pattern = r'https://[\w\.-]+\.trycloudflare\.com'
                while True:
                    line = tunnelproc.stderr.readline() #cloudflare writes to stderr for some reason
                    tunnelrawlog += line+"\n"
                    if not line:
                        return
                    found = re.findall(pattern, line)
                    for x in found:
                        tunneloutput = x
                        if global_memory and global_memory["load_complete"]:
                            print(f"Your remote Kobold API can be found at {tunneloutput}/api")
                            print(f"Your remote OpenAI Compatible API can be found at {tunneloutput}/v1")
                            print(f"Your remote llama.cpp secondary WebUI at {tunneloutput}/lcpp/")
                            if has_sd:
                                print(f"StableUI is available at {tunneloutput}/sdui/")
                            if has_music:
                                print(f"MusicUI is available at {tunneloutput}/musicui/")
                            print("======\n")
                            print(f"Your remote tunnel is ready, please connect to {tunneloutput}", flush=True)
                        if global_memory:
                            global_memory["tunnel_url"] = tunneloutput
                        return

            tunnel_reader_thread = threading.Thread(target=tunnel_reader)
            tunnel_reader_thread.start()
            time.sleep(5)
            if tunneloutput=="":
                print(f"Error: Could not create cloudflare tunnel!\nMore Info:\n{tunnelrawlog}", flush=True)
            time.sleep(0.5)
            tunnelproc.wait()

        if os.name == 'nt':
            downloader_internal("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe", "cloudflared.exe", True, 500000)
        elif sys.platform=="darwin":
            downloader_internal("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz", "cloudflared-darwin-amd64.tgz", True, 500000)
            subprocess.run("tar -xzf cloudflared-darwin-amd64.tgz", shell=True)
            subprocess.run("chmod +x 'cloudflared'", shell=True)
        elif sys.platform == "linux" and platform.machine().lower() == "aarch64":
            downloader_internal("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64", "cloudflared-linux-arm64", True, 500000)
            subprocess.run("chmod +x 'cloudflared-linux-arm64'", shell=True)
        else:
            downloader_internal("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "cloudflared-linux-amd64", True, 500000)
            subprocess.run("chmod +x 'cloudflared-linux-amd64'", shell=True)
        print("Attempting to start tunnel thread...", flush=True)
        tunnel_thread = threading.Thread(target=run_tunnel)
        tunnel_thread.start()
    except Exception as ex:
        print("Remote Tunnel Failed!")
        print(str(ex))
        return None

def reload_from_new_args(newargs):
    try:
        args.istemplate = False
        newargs = convert_invalid_args(newargs)
        for key, value in newargs.items(): #do not overwrite certain values
            if key not in ["remotetunnel","showgui","port","host","port_param","admin","adminpassword","password","adminunloadtimeout","routermode","admindir","ssl","nocertify","benchmark","prompt","config","baseconfig","downloaddir","onready","rpcmode","rpchost","rpcport","rpcdevice"]:
                setattr(args, key, value)
        setattr(args,"showgui",False)
        setattr(args,"benchmark",False)
        setattr(args,"prompt","")
        setattr(args,"config",None)
        setattr(args,"launch",None)
        if "istemplate" in newargs and newargs["istemplate"]:
            auto_set_backend_cli()
    except Exception as e:
        print(f"Reload New Config Failed: {e}")

def reload_new_config(filename,defaultargs,overwrite_blank=False): #for changing config after launch
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            config = json.load(f)
            for key, value in defaultargs.items():   # Fill missing defaults directly into config
                if key not in config:
                    config[key] = value
                elif overwrite_blank and key in config and config[key] in (None, ""):
                    config[key] = value
            reload_from_new_args(config)
        except Exception as e:
            print(f"Reload New Config Failed: {e}")

def load_config_cli(filename):
    print(f"Loading configuration file {filename}...")
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        config = json.load(f)
        config = convert_invalid_args(config)
        if "onready" in config:
            config["onready"] = "" #do not allow onready commands from config
        args.istemplate = False
        raw_args = (sys.argv[1:]) #a lousy hack to allow for overriding kcpps
        # special: overriding model applies to model_param too
        if "--model" in raw_args:
            raw_args.append("--model_param")
        for key, value in config.items():
            if f"--{key}" in raw_args:
                if key!="config":
                    print(f"Overriding Config Value: {key}")
            else:
                setattr(args, key, value)
        if args.istemplate:
            print("\nA .kcppt template was selected from CLI...")
            if (args.usecuda is None) and (args.usevulkan is None):
                print("Automatically selecting your backend...")
                auto_set_backend_cli()

def convert_args_to_template(savdict):
    savdict["istemplate"] = True
    savdict["gpulayers"] = -1
    savdict["threads"] = -1
    savdict["hordekey"] = ""
    savdict["hordeworkername"] = ""
    savdict["sdthreads"] = 0
    savdict["maingpu"] = -1
    savdict["sdmaingpu"] = -1
    savdict["password"] = None
    savdict["adminpassword"] = None
    savdict["usemmap"] = False
    savdict["usemlock"] = False
    savdict["debugmode"] = 0
    savdict["ssl"] = None
    savdict["usecuda"] = None
    savdict["usevulkan"] = None
    savdict["usecpu"] = None
    savdict["tensor_split"] = None
    savdict["draftgpusplit"] = None
    savdict["config"] = None
    savdict["ttsthreads"] = 0
    savdict["nommq"] = False
    savdict["splitmode"] = splitmode_choices[0]
    return savdict

def save_config_cli(filename, template):
    savdict = json.loads(json.dumps(args.__dict__))
    if template:
        savdict = convert_args_to_template(savdict)
    else:
        savdict["istemplate"] = False
    if filename is None:
        return
    filenamestr = save_config_dict(filename, savdict, template)
    print(f"\nSaved configuration file as {filenamestr}\nIt can be loaded with --config [filename] in future.")
    pass

def delete_old_pyinstaller():
    try:
        base_path = sys._MEIPASS
    except Exception:
        return # not running from pyinstaller
    if not base_path:
        return

    selfdirpath = os.path.abspath(base_path)
    temp_parentdir_path = os.path.abspath(os.path.join(base_path, '..'))
    for dirname in os.listdir(temp_parentdir_path):
        absdirpath = os.path.abspath(os.path.join(temp_parentdir_path, dirname))
        if os.path.isdir(absdirpath) and os.path.basename(absdirpath).startswith('_MEI'): #only delete kobold pyinstallers
            if absdirpath!=selfdirpath and (time.time() - os.path.getctime(absdirpath)) > 14400: # remove if older than 4 hours
                kobold_itemcheck1 = os.path.join(absdirpath, 'koboldcpp_default.dll')
                kobold_itemcheck2 = os.path.join(absdirpath, 'koboldcpp_default.so')
                kobold_itemcheck3 = os.path.join(absdirpath, 'koboldcpp.py')
                kobold_itemcheck4 = os.path.join(absdirpath, 'cublasLt64_11.dll')
                kobold_itemcheck5 = os.path.join(absdirpath, 'cublas64_11.dll')
                if os.path.exists(kobold_itemcheck1) or os.path.exists(kobold_itemcheck2) or os.path.exists(kobold_itemcheck3) or (os.path.exists(kobold_itemcheck4) and os.path.exists(kobold_itemcheck5)):
                    try:
                        shutil.rmtree(absdirpath)
                        print(f"Deleted orphaned pyinstaller dir: {absdirpath}")
                    except Exception as e:
                        print(f"Error deleting orphaned pyinstaller dir: {absdirpath}: {e}")

def sanitize_string(input_string):
    # alphanumeric characters, dots, dashes, and underscores
    sanitized_string = re.sub( r'[^\w\d\.\-_]', '', input_string)
    return sanitized_string

def downloader_internal(input_url, output_filename, capture_output, min_file_size=64): # 64 bytes required by default
    download_dir_path = args.downloaddir
    if "https://huggingface.co/" in input_url and "/blob/main/" in input_url:
        input_url = input_url.replace("/blob/main/", "/resolve/main/")
    if download_dir_path:
        download_dir_path = os.path.abspath(download_dir_path)
        os.makedirs(download_dir_path, exist_ok=True)
    if output_filename != "auto" and download_dir_path and not os.path.isabs(output_filename):
        output_filename = os.path.join(download_dir_path, output_filename)
    if output_filename == "auto":
        filename = os.path.basename(input_url).split('?')[0].split('#')[0]
        if download_dir_path:
            output_filename = os.path.join(download_dir_path, filename)
        else:
            cwd = os.getcwd()
            non_writable = False
            if os.name == "nt":
                parts = [p.lower() for p in os.path.normpath(cwd).split(os.sep)]
                if "windows" in parts and ("system32" in parts or "syswow64" in parts):
                    non_writable = True
            if not non_writable:
                output_filename = filename
            else:
                exe_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)
                output_filename = os.path.join(exe_dir, filename)
    incomplete_dl_exist = (os.path.exists(output_filename+".aria2") and os.path.getsize(output_filename+".aria2") > 16)
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size and not incomplete_dl_exist:
        print(f"{output_filename} already exists, using existing file.")
        return output_filename
    print(f"Downloading {input_url}", flush=True)

    dl_success = False
    out_dir = os.path.dirname(os.path.abspath(output_filename)) or os.getcwd()
    out_name = os.path.basename(output_filename)
    try:
        if os.name == 'nt':
            basepath = os.path.abspath(os.path.dirname(__file__))
            a2cexe = os.path.join(basepath, "aria2c-win.exe")
            if os.path.exists(a2cexe):  # on windows try using embedded aria2c
                rc = subprocess.run([
                        a2cexe, "-x", "16", "-s", "16",
                        "--summary-interval=15", "--console-log-level=error", "--log-level=error",
                        "--download-result=default", "--continue=true", "--allow-overwrite=true",
                        "--file-allocation=none", "--max-tries=3",
                        "-d", out_dir, "-o", out_name, input_url
                    ], capture_output=capture_output, text=True, check=True, encoding='utf-8')
                dl_success = (rc.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size)
    except subprocess.CalledProcessError as e:
        print(f"aria2c-win failed: {e}")

    try:
        if not dl_success and shutil.which("aria2c") is not None:
            rc = subprocess.run([
                    "aria2c", "-x", "16", "-s", "16",
                    "--summary-interval=15", "--console-log-level=error", "--log-level=error",
                    "--download-result=default", "--allow-overwrite=true",
                    "--file-allocation=none", "--max-tries=3",
                    "-d", out_dir, "-o", out_name, input_url
                ], capture_output=capture_output, text=True, check=True, encoding='utf-8')
            dl_success = (rc.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size)
    except subprocess.CalledProcessError as e:
        print(f"aria2c failed: {e}")

    try:
        if not dl_success and shutil.which("curl") is not None:
            rc = subprocess.run(["curl", "-fLo", output_filename, input_url],
                capture_output=capture_output, text=True, check=True, encoding="utf-8")
            dl_success = (rc.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size)
    except subprocess.CalledProcessError as e:
        print(f"curl failed: {e}")

    try:
        if not dl_success and shutil.which("wget") is not None:
            rc = subprocess.run(["wget", "-O", output_filename, input_url],
                capture_output=capture_output, text=True, check=True, encoding="utf-8")
            dl_success = (rc.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size)
    except subprocess.CalledProcessError as e:
        print(f"wget failed: {e}")

    if not dl_success:
        print("Could not find suitable download software, or all download methods failed. Please install aria2, curl, or wget.")
        return None

    return output_filename


def download_model_from_url(url, permitted_types=[".gguf",".safetensors", ".ggml", ".bin"], min_file_size=64,handle_multipart=False):
    if url and url!="":
        if url.endswith("?download=true"):
            url = url.replace("?download=true","")
        end_ext_ok = False
        for t in permitted_types:
            if url.endswith(t):
                end_ext_ok = True
                break
        if ((url.startswith("http://") or url.startswith("https://")) and end_ext_ok):
            dlfile = downloader_internal(url, "auto", False, min_file_size)
            if handle_multipart and "-00001-of-00" in url: #handle multipart files up to 9 parts
                match = re.search(r'-(\d{5})-of-(\d{5})\.', url)
                if match:
                    total_parts = int(match.group(2))
                    if total_parts > 1 and total_parts <= 999:
                        current_part = 1
                        base_url = url
                        for part_num in range(current_part + 1, total_parts + 1):
                            part_str = f"-{part_num:05d}-of-{total_parts:05d}"
                            new_url = re.sub(r'-(\d{5})-of-(\d{5})', part_str, base_url)
                            downloader_internal(new_url, "auto", False, min_file_size)
            return dlfile
    return None

def analyze_gguf_model(args,filename):
    try:
        stime = datetime.now()
        dump_gguf_metadata(filename)
        atime = (datetime.now() - stime).total_seconds()
        print(f"---\nAnalyzing completed in {atime:.2f}s.\n---",flush=True)
    except Exception as e:
        print(f"Cannot Analyze File: {e}")
    return

def analyze_gguf_model_wrapper(filename=""):
    if not filename or filename=="":
        try:
            filename = zentk_askopenfilename(title="Select GGUF to analyze")
        except Exception as e:
            print(f"Cannot select file to analyze: {e}")
    if not filename or filename=="" or not os.path.exists(filename):
        print("Selected GGUF file not found. Please select a valid GGUF file to analyze.")
        return
    print("---")
    print(f"Analyzing {filename}, please wait...\n---",flush=True)
    dumpthread = threading.Thread(target=analyze_gguf_model, args=(args,filename))
    dumpthread.start()


def register_koboldcpp():
    try:
        exe_path = ""
        if getattr(sys, 'frozen', False):
            exe_path = sys.executable
        if os.name == 'nt' and exe_path!="":
            confirmyes = show_gui_yesnobox("Confirm Add File Extensions","Do you want to register KoboldCpp as the default file associations for .gguf, .kcpps, .kcppt and .ggml files?",icon="question")
            if confirmyes == 'yes':
                import winreg
                print(f"Registering file associations to {exe_path}")
                entries = [
                    (r"Software\Classes\KoboldCpp\DefaultIcon", "", f"{exe_path},0"),
                    (r"Software\Classes\KoboldCpp\shell\Open\command", "", f'"{exe_path}" "%1" --singleinstance'),
                    (r"Software\Classes\KoboldCpp\shell\Edit\command", "", f'"{exe_path}" "%1" --singleinstance --showgui'),
                    (r"Software\Classes\.gguf", "", "KoboldCpp"),
                    (r"Software\Classes\.kcpps", "", "KoboldCpp"),
                    (r"Software\Classes\.kcppt", "", "KoboldCpp"),
                    (r"Software\Classes\.ggml", "", "KoboldCpp"),
                ]
                for key_path, value_name, value_data in entries:
                    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                        winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, value_data)
                print("KoboldCpp file associations registered successfully.")
        else:
            show_gui_msgbox("Cannot Set File Association","File Associations only available for Windows standalone executables.")
    except Exception as e:
        print(f"Register Extensions: An error occurred: {e}")

def load_mcp_async(args):
    global mcp_connections, mcp_lock
    filepath = os.path.abspath(args.mcpfile)
    if not filepath.lower().endswith(".json"):
        filepath += ".json"
        args.mcpfile += ".json"
    try:
        print(f"MCP start loading json file at '{filepath}'...")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("MCP config must be a JSON object")
            servers = loaded.get("mcpServers")
            if not isinstance(servers, dict):
                serversVsCode = loaded.get("servers")
                if isinstance(serversVsCode, dict):
                    servers = serversVsCode
                else:
                    raise ValueError("MCP config missing 'mcpServers' object")
            for name, cfg in servers.items():
                try:
                    print(f"Connecting to MCP Server {name}...")
                    if not isinstance(cfg, dict):
                        raise ValueError(f"MCP server '{name}' must be an object")
                    mcpurl = cfg.get("url", "")
                    mcpcmd = cfg.get("command","")
                    if mcpcmd and not mcpurl:
                        mcpargs = cfg.get("args", [])
                        mcpenv = cfg.get("env", {})
                        client = MCPStdioClient(command=mcpcmd,largs=mcpargs,env=mcpenv)
                    elif mcpurl:
                        mcp_ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
                        headers = cfg.get("headers", {})
                        headers.setdefault('User-Agent', mcp_ua)
                        client = MCPHTTPClient(url=mcpurl, headers=headers)
                    else:
                        raise ValueError(f"MCP server '{name}' missing 'command' and 'url'")
                    with mcp_lock:
                        mcp_connections.append({"client":client,"tools":[],"name":name})
                except Exception as e:
                    print(f"MCP Init Error: {e}")
            for conn in list(mcp_connections):
                try:
                    init_payload = {
                        "jsonrpc": "2.0",
                        "id": random.randint(100000, 999999),
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "koboldcpp", "version": "1.0.0"}
                        }
                    }
                    notif_payload = {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    }
                    toolget_payload = {
                        "jsonrpc": "2.0",
                        "id": random.randint(100000, 999999),
                        "method": "tools/list",
                        "params": {}
                    }

                    resp1 = conn["client"].send(init_payload)
                    if "result" not in resp1:
                        continue

                    conn["client"].send(notif_payload, await_response=False)
                    resp2 = conn["client"].send(toolget_payload)

                    if "result" not in resp2 or "tools" not in resp2["result"]:
                        continue
                    with mcp_lock:
                        conn["tools"] = resp2["result"]["tools"]
                except Exception as e:
                    print(f"MCP Setup Error: {e}")
            print(f"Completed load of MCP json file at '{filepath}'.")
    except Exception as e:
        print(f"Failed to parse MCP json file at '{filepath}': {e}")


def unregister_koboldcpp():
    try:
        if os.name == 'nt':
            confirmyes = show_gui_yesnobox("Confirm Remove File Extensions","Do you want to unregister KoboldCpp as the default file associations for .gguf, .kcpps, .kcppt and .ggml files?",icon="question")
            if confirmyes == 'yes':
                import winreg
                keys_to_delete = [
                    r"Software\Classes\KoboldCpp\shell\Edit\command",
                    r"Software\Classes\KoboldCpp\shell\Edit",
                    r"Software\Classes\KoboldCpp\shell\Open\command",
                    r"Software\Classes\KoboldCpp\shell\Open",
                    r"Software\Classes\KoboldCpp\shell",
                    r"Software\Classes\KoboldCpp\DefaultIcon",
                    r"Software\Classes\KoboldCpp",
                    r"Software\Classes\.gguf",
                    r"Software\Classes\.kcpps",
                    r"Software\Classes\.kcppt",
                    r"Software\Classes\.ggml",
                ]
                for key_path in keys_to_delete:
                    try:
                        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
                    except Exception:
                        print(f"Failed to delete registry key: {key_path}")
                print("KoboldCpp file associations unregistered.")
        else:
            show_gui_msgbox("Cannot Set File Association","File Associations only available for Windows standalone executables.")
    except Exception as e:
        print(f"Unregister Extensions: An error occurred: {e}")

def main(launch_args, default_args):
    global args, showdebug, kcpp_instance, exitcounter, using_gui_launcher, sslvalid, global_memory
    args = launch_args #note: these are NOT shared with the child processes!

    if (args.version) and len(sys.argv) <= 2:
        print(f"{KcppVersion}") # just print version and exit
        return

    if args.testmemory:
        fetch_gpu_properties(True, True, testmemory=True)
        return

    #prevent disallowed combos
    if (args.nomodel or args.benchmark or args.launch or args.admin) and args.cli:
        exit_with_error(1, "Error: --cli cannot be combined with --launch, --nomodel, --admin or --benchmark")
    if (args.rpcmode!="connect" and args.rpctargets):
        exit_with_error(1, "Error: rpctargets can only be used in connect mode")

    args = convert_invalid_args(args)

    temp_hide_print = (args.model_param and (args.prompt and not args.cli) and not args.benchmark and not (args.debugmode >= 1))

    if not temp_hide_print:
        print(f"***\nWelcome to KoboldCpp - Version {KcppVersion}")
    if args.debugmode != 1:
        showdebug = False #not shared with child process!
    if args.debugmode >= 1:
        print("Debug Mode is Enabled!")
        args.quiet = False # verbose outputs

    # assign title to terminal on windows
    try:
        if os.name == 'nt':
            windowtitle = f"KoboldCpp {KcppVersion} Terminal"
            os.system(f'title {windowtitle}')
    except Exception:
        pass

    try:
        delete_old_pyinstaller()  #perform some basic cleanup of old temporary directories
    except Exception as e:
        print(f"Error cleaning up orphaned pyinstaller dirs: {e}")

    if args.unpack:
        unpack_to_dir(args.unpack)
        return

    if args.analyze:
        analyze_gguf_model_wrapper(args.analyze)
        return

    cfgname = ""
    if args.config and len(args.config)==1: #handle initial config loading for launch
        cfgname = args.config[0] #store first so baseconfig wont overwrite it

    if cfgname: #handle initial config loading for launch
        if isinstance(cfgname, str):
            dlfile = download_model_from_url(cfgname,[".kcpps",".kcppt"])
            if dlfile:
                cfgname = dlfile
        if isinstance(cfgname, str) and os.path.exists(cfgname):
           load_config_cli(cfgname)
        elif args.ignoremissing:
            print("Ignoring missing kcpp config file...")
        else:
            exitcounter = 999
            exit_with_error(2,"Specified kcpp config file invalid or not found.")
    args = convert_invalid_args(args)

    #positional handling for kcpps files (drag and drop)
    if args.model_param and args.model_param!="" and (args.model_param.lower().endswith('.kcpps') or args.model_param.lower().endswith('.kcppt') or args.model_param.lower().endswith('.kcpps?download=true') or args.model_param.lower().endswith('.kcppt?download=true')):
        dlfile = download_model_from_url(args.model_param,[".kcpps",".kcppt"]) # maybe download from url
        if dlfile:
            args.model_param = dlfile
        load_config_cli(args.model_param)

    if args.exportconfig:
        save_config_cli(args.exportconfig,False)
        return
    if args.exporttemplate:
        save_config_cli(args.exporttemplate,True)
        return

    # show the GUI launcher if a model was not provided
    if args.showgui or not has_valid_model():
        #give them a chance to pick a file
        print("For command line arguments, please run --help in the terminal.")
        print("Note: The GUI mode is not accessible to screen readers.")
        print("***")
        try:
            show_gui()
        except Exception as ex:
            exitcounter = 999
            ermsg = "Reason: " + str(ex) + "\nFile selection GUI unsupported.\ncustomtkinter python module required!\n\nYou must use the command line instead, e.g. python ./koboldcpp.py --help"
            show_gui_msgbox("Warning, GUI failed to start",ermsg)
            if args.skiplauncher:
                print("Note: In order to use --skiplauncher, you need to specify a model with --model")
            time.sleep(3)
            sys.exit(2)

    if args.ssl: #need to duplicate here for the tunnel
        if len(args.ssl)==2 and isinstance(args.ssl[0], str) and os.path.exists(args.ssl[0]) and isinstance(args.ssl[1], str) and os.path.exists(args.ssl[1]):
            sslvalid = True

    args.proxy_port = None #normally unused
    if args.autoswapmode:
        if not args.routermode:
            print("\nWARNING: Autoswap mode requires router, enabling router...")
            args.routermode = True
    if args.routermode:
        if not args.admin:
            print("\nWARNING: Router mode requires admin, enabling admin...")
            args.admin = True
        # setup router mode, find a usable high port swap the port
        newport = 15001
        for prt in range(15001,15011):
            if not is_port_in_use(prt):
                newport = prt
                break
        args.proxy_port = args.port_param
        args.port = args.port_param = newport
        if args.singleinstance and is_port_in_use(args.proxy_port):
            try:
                print(f"Warning: Port {args.proxy_port} already appears to be in use by another program.")
                print(f"Attempting to request shutdown of previous instance on port {args.proxy_port}...")
                shutdownreq = make_url_request(f'http://localhost:{args.proxy_port}/api/extra/shutdown',{},timeout=5)
                shutdownok = (shutdownreq and "success" in shutdownreq and shutdownreq["success"] is True)
                time.sleep(2)
                print("Shutdown existing successful!" if shutdownok else "Shutdown existing failed!")
                time.sleep(1)
            except Exception:
                pass
        run_router_proxy(args.proxy_port,newport)

    if args.admin and not args.admindir:
        print("\nWARNING: Admin was set without selecting an admin directory. Selecting current executable directory...")
        autopath = os.path.realpath(__file__)
        if getattr(sys, 'frozen', False):
            autopath = sys.executable
        autopath = os.path.dirname(autopath)
        args.admindir = autopath
        print(f"Admin Directory Set: {autopath}\n")

    if not args.admin: #run in single process mode
        if args.remotetunnel and not args.prompt and not args.benchmark and not args.cli:
            setuptunnel(global_memory, True if args.sdmodel else False, True if (args.musicdiffusion or args.musicllm or args.ttsmodel) else False)
        kcpp_main_process(args,global_memory,using_gui_launcher)
        if global_memory["input_to_exit"]:
            print("===")
            print("Press ENTER key to exit.", flush=True)
            input()
    else:  # manager command queue for admin mode
        with multiprocessing.Manager() as mp_manager:
            global_memory = mp_manager.dict({"tunnel_url": "", "restart_target":"", "input_to_exit":False, "load_complete":False, "restart_override_base_config":"", "last_active_timestamp":datetime.now(), "triggered_sleeping":False, "current_model":"initial_model", "base_config":"", "swapReqType": None, "autoswapmode": False})

            if args.remotetunnel and not args.prompt and not args.benchmark and not args.cli:
                setuptunnel(global_memory, True if args.sdmodel else False, True if (args.musicdiffusion or args.musicllm or args.ttsmodel) else False)

            # invoke the main koboldcpp process
            original_args = copy.deepcopy(args)

            kcpp_instance = multiprocessing.Process(target=kcpp_main_process,kwargs={"launch_args": args, "g_memory": global_memory, "gui_launcher": using_gui_launcher})
            kcpp_instance.daemon = True
            kcpp_instance.start()

            fault_recovery_mode = False #if a config reload fails, recover back to old settings

            while True: # keep the manager alive
                try:
                    restart_target = ""
                    restart_override_base_config = ""
                    if not kcpp_instance or not kcpp_instance.is_alive():
                        if fault_recovery_mode:
                            #attempt to recover
                            print("Attempting to recover to safe mode, launching known-good config...")
                            fault_recovery_mode = False
                            args = copy.deepcopy(original_args) #restore known good original launcher args
                            if kcpp_instance:
                                kcpp_instance.terminate()
                                kcpp_instance.join(timeout=10)  # Ensure process is stopped
                                kcpp_instance = None
                            kcpp_instance = multiprocessing.Process(target=kcpp_main_process,kwargs={"launch_args": args, "g_memory": global_memory, "gui_launcher": False})
                            kcpp_instance.daemon = True
                            kcpp_instance.start()
                            global_memory["restart_target"] = ""
                            global_memory["restart_override_base_config"] = ""
                            global_memory["swapReqType"] = None
                            time.sleep(3)
                        else:
                            break # kill the program
                    if fault_recovery_mode and global_memory["load_complete"]:
                        fault_recovery_mode = False
                    restart_target = global_memory["restart_target"]
                    restart_override_base_config = global_memory["restart_override_base_config"]
                    last_active = global_memory["last_active_timestamp"]
                    if last_active and args.adminunloadtimeout>0:
                        curtime = datetime.now()
                        elapsedtime = curtime - last_active
                        time_since_last_active = elapsedtime.total_seconds()
                        if time_since_last_active > args.adminunloadtimeout:
                            if args.autoswapmode:
                                if global_memory["swapReqType"] is not None and global_memory["swapReqType"] != "nomodel":
                                    print(f"[Unload Timeout] Inactive for over {time_since_last_active}s, unloading models via autoswap...")
                                    global_memory["swapReqType"] = "nomodel"
                                    global_memory["triggered_sleeping"] = True
                            elif global_memory["current_model"]!="unload_model":
                                print(f"[Unload Timeout] Inactive for over {time_since_last_active}s, unloading models...")
                                restart_target = "unload_model"
                                global_memory["triggered_sleeping"] = True
                    if restart_target!="":
                        overridetxt = ("" if not restart_override_base_config else f" with override config {restart_override_base_config}")
                        print(f"Reloading new model/config: {restart_target}{overridetxt}")
                        global_memory["restart_target"] = ""
                        global_memory["restart_override_base_config"] = ""
                        time.sleep(0.5) #sleep for 0.5s then restart
                        if args.admin and args.admindir:
                            dirpath = os.path.abspath(args.admindir)
                            maintarget_filepath = os.path.abspath(os.path.join(dirpath, restart_target))
                            basecfg_filepath = os.path.abspath(os.path.join(dirpath, restart_override_base_config)) if restart_override_base_config else ""
                            if os.path.commonpath([dirpath, maintarget_filepath]) != dirpath: # Enforce admindir jail
                                print("Security: Invalid restart target path.")
                                continue
                            if basecfg_filepath and os.path.commonpath([dirpath, basecfg_filepath]) != dirpath:
                                print("Security: Invalid override config path.")
                                continue
                            #if override config is not specified, AND baseconfig is, swap it as our override
                            if restart_target!="unload_model" and restart_target!="initial_model" and args.baseconfig and not basecfg_filepath:
                                my_basecfg_path = os.path.abspath(args.baseconfig)
                                if os.path.exists(my_basecfg_path):
                                    basecfg_filepath = my_basecfg_path
                                    print(f"No override config provided, using baseconfig {args.baseconfig}")
                            defaultargs = vars(default_args)
                            if (os.path.exists(maintarget_filepath) or restart_target=="unload_model" or restart_target=="initial_model") and (restart_override_base_config=="" or os.path.exists(basecfg_filepath)):
                                print("Terminating old process...")
                                global_memory["load_complete"] = False
                                kcpp_instance.terminate()
                                kcpp_instance.join(timeout=10)  # Ensure process is stopped
                                kcpp_instance = None
                                print("Restarting KoboldCpp...")
                                fault_recovery_mode = True
                                #then, apply the rest of the config stack
                                if restart_target=="unload_model":
                                    reload_from_new_args(defaultargs)
                                    args.model_param = None
                                    args.model = None
                                    args.nomodel = True
                                elif restart_target=="initial_model":
                                    reload_from_new_args(vars(original_args))
                                elif maintarget_filepath.endswith(".gguf") and basecfg_filepath=="":
                                    reload_from_new_args(defaultargs)
                                    args.model_param = maintarget_filepath
                                elif maintarget_filepath.endswith(".gguf") and basecfg_filepath!="":
                                    reload_from_new_args(defaultargs)
                                    reload_new_config(basecfg_filepath,vars(args),True)
                                    args.model_param = maintarget_filepath
                                elif maintarget_filepath and basecfg_filepath and basecfg_filepath!="":
                                    # stuff applied later overwrites stuff applied earlier, if they have the same field names
                                    reload_from_new_args(defaultargs)
                                    reload_new_config(basecfg_filepath,vars(args),True)
                                    reload_new_config(maintarget_filepath,vars(args),True)
                                else:
                                    reload_from_new_args(defaultargs)
                                    reload_new_config(maintarget_filepath,vars(args),True)
                                global_memory["autoswapmode"] = args.autoswapmode
                                kcpp_instance = multiprocessing.Process(target=kcpp_main_process,kwargs={"launch_args": args, "g_memory": global_memory, "gui_launcher": False})
                                kcpp_instance.daemon = True
                                kcpp_instance.start()
                                global_memory["restart_target"] = ""
                                if (restart_override_base_config and restart_override_base_config!=""):
                                    global_memory["base_config"] = restart_override_base_config
                                else:
                                    global_memory["base_config"] = ""
                                global_memory["restart_override_base_config"] = ""
                                global_memory["current_model"] = restart_target
                                time.sleep(3)
                    else:
                        time.sleep(0.2)
                except (KeyboardInterrupt,SystemExit):
                    break
            if global_memory["input_to_exit"]:
                print("===")
                print("Press ENTER key to exit.", flush=True)
                input()


def mk_lora_info(imgloras, multipliers):
    first_multiplier = multipliers[0] if len(multipliers) > 0 else 1.
    lora_files = []
    lora_dirs = []
    # identify files and dirs
    for i, lora_path in enumerate(imgloras):
        multiplier = multipliers[i] if i < len(multipliers) else first_multiplier
        if os.path.isfile(lora_path):
            lora_files.append(('', lora_path, multiplier))
        elif os.path.isdir(lora_path):
            lora_dirs.append(lora_path)
        elif os.path.exists(lora_path):
            print(f"Unexpected file type for SD LORA model file {lora_path}")
        else:
            print(f"Missing SD LORA model file {lora_path}...")
    # scan all dirs
    for lora_dir in lora_dirs:
        print(f'Scanning {lora_dir} for LoRAs...')
        files = scan_directory(lora_dir, ('.safetensors', '.gguf'), 1)
        print(f'  found {len(files)} files under {lora_dir}')
        for file in files:
            lora_files.append((lora_dir, file, 0.0))
    # dedup and map all files
    unique_lora_names = set()
    lora_fullmap = {}
    for i, (lora_dir, lora_path, multiplier) in enumerate(lora_files):
        if lora_dir:
            # lora_path is relative: we can show it on the interface and accept it
            lora_fullpath = os.path.join(lora_dir, lora_path)
            # NOTE: we are including the relative directory on the short name
            lora_file = lora_path
            preloaded = False
        else:
            lora_fullpath = lora_path
            # we don't know which portion of the path we can show, so omit it
            lora_file = os.path.basename(lora_path)
            preloaded = True
        lora_fullpath = os.path.abspath(lora_fullpath)
        # dedup paths (e.g. preloaded and on directory)
        info = lora_fullmap.get(lora_fullpath)
        if info:
            info["multiplier"] += multiplier
            if multiplier == 0.0 and 'fixed' in info:
                # allow changes if we see this lora again with weight 0
                del info['fixed']
            continue
        lora_name, lora_ext = os.path.splitext(lora_file)
        # ensure unique names
        i = 1
        lora_uname = lora_name
        while lora_uname in unique_lora_names:
            i += 1
            lora_uname = lora_name + '_' + str(i)
        unique_lora_names.add(lora_uname)
        lora_upath = lora_uname + lora_ext
        lora_entry = {
            'fullpath': lora_fullpath,  # where it is on disk
            'name': lora_uname,         # 'name' in api field and <lora:name:multiplier>
            'path': lora_upath,         # 'path' in api field (relative), + extension
            'multiplier': multiplier,   # preload multiplier
        }
        if preloaded:
            lora_entry['preloaded'] = preloaded
        if multiplier != 0.0 and imglora_initial_fixed:
            lora_entry['fixed'] = True
        lora_fullmap[lora_fullpath] = lora_entry
    # build the runtime tables
    preloaded_table = []
    lora_path_map = {}
    lora_name_map = {}
    for lora_entry in lora_fullmap.values():
        if not lora_entry.get("fixed"):  # only map LoRAs that can be changed
            lora_path_map[lora_entry["path"]] = lora_entry
            lora_name_map[lora_entry["name"]] = lora_entry["path"]
        if lora_entry.get("preloaded"):
            preloaded_table.append(lora_entry)
    return preloaded_table, lora_path_map, lora_name_map

def disableSwappedFieldsInConfig(args, swapReqType):
    print(f"Swapping to type: {swapReqType}")
    if swapReqType != "text":
        for e in ["model", "model_param", "lora", "mmproj"]:
            setattr(args, e, "")
    if swapReqType != "stt":
        for e in ["whispermodel"]:
            setattr(args, e, "")
    if swapReqType != "tts":
        for e in ["ttsmodel", "ttswavtokenizer"]:
            setattr(args, e, "")
    if swapReqType != "embed":
        for e in ["embeddingsmodel"]:
            setattr(args, e, "")
    if swapReqType != "music":
        for e in ["musicllm", "musicembeddings", "musicdiffusion", "musicvae"]:
            setattr(args, e, "")
    if swapReqType != "image":
        for e in ["sdmodel", "sdt5xxl", "sdclip1", "sdclip2", "sdphotomaker", "sdupscaler", "sdvae", "sdaudiovae", "sdlora"]:
            setattr(args, e, "")


def kcpp_main_process(launch_args, g_memory=None, gui_launcher=False):
    global embedded_kailite, embedded_kcpp_docs, embedded_kcpp_sdui, embedded_kailite_gz, embedded_kcpp_docs_gz, embedded_kcpp_sdui_gz, embedded_lcpp_ui_gz, embedded_musicui, embedded_musicui_gz, start_time, exitcounter, global_memory, using_gui_launcher
    global libname, args, friendlymodelname, friendlysdmodelname, fullsdmodelpath, password, fullwhispermodelpath, ttsmodelpath, embeddingsmodelpath, musicdiffusionmodelpath, musicllmmodelpath, friendlyembeddingsmodelname, has_audio_support, has_vision_support, cached_chat_template, cached_jinja_kwargs, cached_sd_info, preloaded_custom_jinja

    start_server = True

    args = launch_args
    global_memory = g_memory
    using_gui_launcher = gui_launcher
    start_time = time.time()

    if args.model_param and (args.prompt and not args.cli) and not args.benchmark and not (args.debugmode >= 1):
        suppress_stdout()


    global autoswapmode, textName, sttName, ttsName, embedName, musicName, imageName, mmprojName
    autoswapmode = False
    textName = None
    mmprojName = None
    sttName = None
    ttsName = None
    embedName = None
    musicName = None
    imageName = None
    if args.autoswapmode is not None and args.autoswapmode:
        autoswapmode = True
        global_memory["autoswapmode"] = True
        if args.model_param and args.model_param!="":
            tempName = os.path.basename(os.path.abspath(args.model_param))
            tempName = os.path.splitext(tempName)[0]
            textName = "koboldcpp/" + sanitize_string(tempName)
            if args.mmproj and args.mmproj!="": # multimodal vision and audio support is assumed to work with mmproj - this may be incorrect!
                tempName = os.path.basename(os.path.abspath(args.mmproj))
                tempName = os.path.splitext(tempName)[0]
                mmprojName = sanitize_string(tempName)
        if args.whispermodel and args.whispermodel!="":
            tempName = os.path.basename(os.path.abspath(args.whispermodel))
            tempName = os.path.splitext(tempName)[0]
            sttName = sanitize_string(tempName)
        if args.ttsmodel and args.ttsmodel!="":
            tempName = os.path.basename(os.path.abspath(args.ttsmodel))
            tempName = os.path.splitext(tempName)[0]
            ttsName = sanitize_string(tempName)
        if args.embeddingsmodel and args.embeddingsmodel!="":
            tempName = os.path.basename(os.path.abspath(args.embeddingsmodel))
            tempName = os.path.splitext(tempName)[0]
            embedName = sanitize_string(tempName)
        if args.musicdiffusion and args.musicdiffusion!="":
            tempName = os.path.basename(os.path.abspath(args.musicdiffusion))
            tempName = os.path.splitext(tempName)[0]
            musicName = sanitize_string(tempName)
        if args.sdmodel and args.sdmodel!="":
            tempName = os.path.basename(os.path.abspath(args.sdmodel))
            tempName = os.path.splitext(tempName)[0]
            imageName = sanitize_string(tempName)
        if global_memory["swapReqType"] is not None:
            disableSwappedFieldsInConfig(args, global_memory["swapReqType"])
        else:
            global_memory["swapReqType"] = "nomodel"
            setattr(args, "nomodel", True)
            disableSwappedFieldsInConfig(args, "nomodel")
    else:
        global_memory["autoswapmode"] = False

    if args.model_param and (args.benchmark or args.prompt or args.cli):
        start_server = False

    args.sdlora = sanitize_lora_list(args.sdlora)
    args.sdloramult = sanitize_lora_multipliers(args.sdloramult)

    #try to read story if provided
    if args.preloadstory:
        global preloaded_story
        canload = False
        if isinstance(args.preloadstory, str) and os.path.exists(args.preloadstory):
            print(f"Preloading saved story {args.preloadstory} into server...")
            with open(args.preloadstory, mode='rb') as f:
                preloaded_story = f.read()
                canload = True
        elif isinstance(args.preloadstory, str):
            print("Preloading saved story as JSON into server...")
            try:
                import ast
                parsed = ast.literal_eval(args.preloadstory)
                preloaded_story = json.dumps(parsed).encode()
                canload = True
            except Exception as ex:
                print(ex)
        elif isinstance(args.preloadstory, dict):
            try:
                preloaded_story = json.dumps(args.preloadstory).encode()
                canload = True
            except Exception as ex:
                print(ex)
        if canload:
            print("Saved story preloaded.")
        else:
            print("Warning: Saved story file invalid or not found. No story will be preloaded into server.")

    # try to read chat completions adapter
    if args.chatcompletionsadapter:
        global chatcompl_adapter, chatcompl_adapter_list
        ccadapter_path = None
        canload = False
        adapt_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'kcpp_adapters')
        adapt_dir = adapt_dir if os.path.isdir(adapt_dir) else None
        if isinstance(args.chatcompletionsadapter, str) and os.path.exists(args.chatcompletionsadapter):
            ccadapter_path = os.path.abspath(args.chatcompletionsadapter)
        elif isinstance(args.chatcompletionsadapter, str) and adapt_dir:
            filename = args.chatcompletionsadapter
            if not filename.endswith(".json"):
                filename += ".json"
            #strip to just the filename
            filename = os.path.basename(filename)
            # Case-insensitive match inside adapt_dir
            matched = None
            if adapt_dir:
                for f in os.listdir(adapt_dir):
                    if f.lower().strip() == filename.lower().strip():
                        matched = os.path.join(adapt_dir, f)
                        break
            if matched and os.path.exists(matched):
                ccadapter_path = os.path.abspath(matched)
        if ccadapter_path:
            print(f"Loading Chat Completions Adapter: {ccadapter_path}")
            with open(ccadapter_path, 'r', encoding='utf-8', errors='replace') as f:
                chatcompl_adapter = json.load(f)
                canload = True
        else:
            if isinstance(args.chatcompletionsadapter, str) and args.chatcompletionsadapter!="":
                try:
                    import ast
                    parsed = ast.literal_eval(args.chatcompletionsadapter)
                    chatcompl_adapter = json.loads(json.dumps(parsed))
                    canload = True
                except Exception as ex:
                    print(ex)
            elif isinstance(args.chatcompletionsadapter, dict):
                try:
                    chatcompl_adapter = json.loads(json.dumps(args.chatcompletionsadapter))
                    canload = True
                except Exception as ex:
                    print(ex)
        if canload:
            print("Chat Completions Adapter Loaded")
        else:
            print("Warning: Chat Completions Adapter invalid or not found.")
        if (chatcompl_adapter is not None and isinstance(chatcompl_adapter, list)):
            chatcompl_adapter_list = chatcompl_adapter
            chatcompl_adapter = None

    # handle model downloads if needed
    if args.model_param and args.model_param!="":
        dlfile = download_model_from_url(args.model_param,[".gguf",".bin", ".ggml"],min_file_size=500000,handle_multipart=True)
        if dlfile:
            args.model_param = dlfile
        if args.model and isinstance(args.model, list) and len(args.model)>1: #handle multi file downloading
            for extramodel in args.model[1:]:
                download_model_from_url(extramodel,[".gguf",".bin", ".ggml"],min_file_size=500000)
    if args.sdmodel and args.sdmodel!="":
        dlfile = download_model_from_url(args.sdmodel,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            args.sdmodel = dlfile
    if args.sdt5xxl and args.sdt5xxl!="":
        dlfile = download_model_from_url(args.sdt5xxl,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            args.sdt5xxl = dlfile
    if args.sdclip1 and args.sdclip1!="":
        dlfile = download_model_from_url(args.sdclip1,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            args.sdclip1 = dlfile
    if args.sdclip2 and args.sdclip2!="":
        dlfile = download_model_from_url(args.sdclip2,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            args.sdclip2 = dlfile
    if args.sdphotomaker and args.sdphotomaker!="":
        dlfile = download_model_from_url(args.sdphotomaker,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            args.sdphotomaker = dlfile
    if args.sdupscaler and args.sdupscaler!="":
        dlfile = download_model_from_url(args.sdupscaler,[".gguf",".safetensors",".pth"],min_file_size=500000)
        if dlfile:
            args.sdupscaler = dlfile
    if args.sdvae and args.sdvae!="":
        dlfile = download_model_from_url(args.sdvae,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            args.sdvae = dlfile
    if args.sdaudiovae and args.sdaudiovae!="":
        dlfile = download_model_from_url(args.sdaudiovae,[".gguf",".safetensors"],min_file_size=500000)
        if dlfile:
            args.sdaudiovae = dlfile
    if args.sdlora and len(args.sdlora)>0:
        for i in range(0,len(args.sdlora)):
            dlfile = download_model_from_url(args.sdlora[i],[".gguf",".safetensors"],min_file_size=500000)
            if dlfile:
                args.sdlora[i] = dlfile
    if args.mmproj and args.mmproj!="":
        dlfile = download_model_from_url(args.mmproj,[".gguf"],min_file_size=500000)
        if dlfile:
            args.mmproj = dlfile
    if args.whispermodel and args.whispermodel!="":
        dlfile = download_model_from_url(args.whispermodel,[".gguf",".bin"],min_file_size=500000)
        if dlfile:
            args.whispermodel = dlfile
    if args.draftmodel and args.draftmodel!="":
        dlfile = download_model_from_url(args.draftmodel,[".gguf"],min_file_size=500000)
        if dlfile:
            args.draftmodel = dlfile
    if args.ttsmodel and args.ttsmodel!="":
        dlfile = download_model_from_url(args.ttsmodel,[".gguf"],min_file_size=500000)
        if dlfile:
            args.ttsmodel = dlfile
    if args.ttswavtokenizer and args.ttswavtokenizer!="":
        dlfile = download_model_from_url(args.ttswavtokenizer,[".gguf"],min_file_size=500000)
        if dlfile:
            args.ttswavtokenizer = dlfile
    if args.embeddingsmodel and args.embeddingsmodel!="":
        dlfile = download_model_from_url(args.embeddingsmodel,[".gguf"],min_file_size=500000)
        if dlfile:
            args.embeddingsmodel = dlfile

    if args.musicllm and args.musicllm!="":
        dlfile = download_model_from_url(args.musicllm,[".gguf"],min_file_size=500000)
        if dlfile:
            args.musicllm = dlfile
    if args.musicembeddings and args.musicembeddings!="":
        dlfile = download_model_from_url(args.musicembeddings,[".gguf"],min_file_size=500000)
        if dlfile:
            args.musicembeddings = dlfile
    if args.musicdiffusion and args.musicdiffusion!="":
        dlfile = download_model_from_url(args.musicdiffusion,[".gguf"],min_file_size=500000)
        if dlfile:
            args.musicdiffusion = dlfile
    if args.musicvae and args.musicvae!="":
        dlfile = download_model_from_url(args.musicvae,[".gguf"],min_file_size=500000)
        if dlfile:
            args.musicvae = dlfile
    if args.mcpfile and args.mcpfile!="":
        dlfile = download_model_from_url(args.mcpfile,[".json"],min_file_size=64)
        if dlfile:
            args.mcpfile = dlfile
    if args.jinjatemplate and args.jinjatemplate!="":
        dlfile = download_model_from_url(args.jinjatemplate,[".jinja"],min_file_size=64)
        if dlfile:
            args.jinjatemplate = dlfile

    if args.jinjatemplate and os.path.exists(args.jinjatemplate):
        try:
            print(f"Using custom Jinja template: {args.jinjatemplate}")
            with open(args.jinjatemplate, 'r', encoding='utf-8', errors='ignore') as f:
                preloaded_custom_jinja = f.read()
        except Exception as e:
            print(f"Error loading jinja templat: {e}")
            preloaded_custom_jinja = ""

    # sanitize and replace the default vanity name. remember me....
    if args.model_param and args.model_param!="":
        newmdldisplayname = os.path.basename(args.model_param)
        newmdldisplayname = os.path.splitext(newmdldisplayname)[0]
        friendlymodelname = "koboldcpp/" + sanitize_string(newmdldisplayname)

    # horde worker settings
    global maxhordelen, maxhordectx, showdebug, has_multiplayer, savedata_obj
    if args.hordemodelname and args.hordemodelname!="":
        friendlymodelname = args.hordemodelname
        if args.debugmode == 1 or args.gendefaults:
            friendlymodelname = "debug-" + friendlymodelname
        if not friendlymodelname.startswith("koboldcpp/"):
            friendlymodelname = "koboldcpp/" + friendlymodelname
    if (args.hordemodelname and args.hordemodelname!="") or (args.hordeworkername and args.hordeworkername!="") or (args.hordekey and args.hordekey!=""):
        if args.debugmode == 0:
            args.debugmode = -1
    if args.hordegenlen and args.hordegenlen > 0:
        maxhordelen = int(args.hordegenlen)
    if args.hordemaxctx and args.hordemaxctx >= 0:
        maxhordectx = int(args.hordemaxctx)

    if args.debugmode != 1:
        showdebug = False
    else:
        showdebug = True

    if args.multiplayer:
        has_multiplayer = True

    if args.savedatafile and isinstance(args.savedatafile, str):
        filepath = os.path.abspath(args.savedatafile)  # Ensure it's an absolute path
        if not filepath.lower().endswith(".jsondb"):
            filepath += ".jsondb"
            args.savedatafile += ".jsondb"
        try:
            with open(filepath, 'r+', encoding='utf-8', errors='ignore') as f:
                loaded = json.load(f)
                savedata_obj = loaded
                print(f"Loaded existing savedatafile at '{filepath}'.")
        except FileNotFoundError:
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w+', encoding='utf-8', errors='ignore') as f:
                    savedata_obj = {}
                    print(f"File '{filepath}' did not exist. Created new savedatafile.")
                    json.dump(savedata_obj, f)
            except Exception as e:
                print(f"Failed to create savedatafile '{filepath}': {e}")
        except Exception as e:
            print(f"Failed to access savedatafile '{filepath}': {e}")

    if args.highpriority:
        print("Setting process to Higher Priority - Use Caution")
        try:
            import psutil
            os_used = sys.platform
            process = psutil.Process(os.getpid())  # Set high priority for the python script for the CPU
            oldprio = process.nice()
            if os.name == 'nt':  # Windows (either 32-bit or 64-bit)
                process.nice(psutil.REALTIME_PRIORITY_CLASS)
                print("High Priority for Windows Set: " + str(oldprio) + " to " + str(process.nice()))
            elif os_used == "linux":  # linux
                process.nice(psutil.IOPRIO_CLASS_RT)
                print("High Priority for Linux Set: " + str(oldprio) + " to " + str(process.nice()))
            else:  # MAC OS X or other
                process.nice(-18)
                print("High Priority for Other OS Set :" + str(oldprio) + " to " + str(process.nice()))
        except Exception as ex:
             print("Error, Could not change process priority: " + str(ex))

    if args.contextsize:
        global maxctx
        maxctx = args.contextsize

    args.defaultgenamt = max(64, min(args.defaultgenamt, 8192))
    args.defaultgenamt = min(args.defaultgenamt, maxctx / 2)

    #this uses the true port instead of the displayport, because we dont want to shut down a router
    if start_server and args.singleinstance and is_port_in_use(args.port):
        try:
            print(f"Warning: Port {args.port} already appears to be in use by another program.")
            print(f"Attempting to request shutdown of previous instance on port {args.port}...")
            shutdownreq = make_url_request(f'http://localhost:{args.port}/api/extra/shutdown',{},timeout=5)
            shutdownok = (shutdownreq and "success" in shutdownreq and shutdownreq["success"] is True)
            time.sleep(2)
            print("Shutdown existing successful!" if shutdownok else "Shutdown existing failed!")
            time.sleep(1)
        except Exception:
            pass

    if args.nocertify:
        import ssl
        global nocertify
        nocertify = True
        ssl._create_default_https_context = ssl._create_unverified_context

    if args.gpulayers:
        if args.autofit:
            args.gpulayers = -1
        shouldavoidgpu = False
        if args.usecpu and sys.platform!="darwin":
            shouldavoidgpu = True
            if args.gpulayers and args.gpulayers>0 and args.rpcmode!="connect":
                print("WARNING: GPU layers is set, but a GPU backend was not selected! GPU will not be used!")
                args.gpulayers = 0
        elif args.gpulayers==-1 and sys.platform=="darwin" and args.model_param and os.path.exists(args.model_param):
            print("MacOS detected: Auto GPU layers set to maximum")
            args.gpulayers = 200
        elif not shouldavoidgpu:
            if (args.usecuda is None) and (args.usevulkan is None):
                print("No GPU or CPU backend was selected. Trying to assign one for you automatically...")
                auto_set_backend_cli()
            if MaxMemory[0] == 0: #try to get gpu vram for cuda if not picked yet
                fetch_gpu_properties(True,True)
            if args.autofit:
                print("Forced autofit is selected, moecpu and overridetensors will be set automatically.")
                args.overridetensors = ""
                args.moecpu = 0
            if args.gpulayers==-1 and args.model_param and os.path.exists(args.model_param):
                if (not args.usecpu) and ((args.usecuda is not None) or (args.usevulkan is not None) or sys.platform=="darwin"):
                    if MaxMemory[0] > 0:
                        extract_modelfile_params(args.model_param,args.sdmodel,args.whispermodel,args.mmproj,args.draftmodel,args.ttsmodel if args.ttsgpu else "",args.embeddingsmodel if args.embeddingsgpu else "", args.musicllm, args.musicdiffusion)
                        layeramt = autoset_gpu_layers(args.contextsize,args.sdquant,args.batchsize,args.musiclowvram)
                        print(f"Auto Recommended GPU Layers: {layeramt}")
                        args.gpulayers = layeramt
                    else:
                        print("Unable to detect VRAM, but autofit may still be used if applicable.")
                        args.gpulayers = 0
                    # also enable autofit also if permissible
                    if not args.autofit and not args.tensor_split and not args.overridetensors and not args.moecpu:
                        args.autofit = True
                        args.autofitpadding = default_autofit_padding
                        print("GPU layers is default: Will enable AutoFit for increased estimation accuracy.")
                else:
                    print("No GPU backend found, or could not automatically determine GPU layers. You may prefer to set layers manually.")
                    args.gpulayers = 0

    if args.threads <= 0:
        args.threads = get_default_threads()
        print(f"Auto Set Threads: {args.threads}")

    print(f"System: {platform.system()} {platform.version()} {platform.machine()} {platform.processor()}")
    if MaxMemory[0]>0:
        print(f"Detected Available GPU Memory: {int(MaxMemory[0]/1024/1024)} MB")
    else:
        print("Unable to determine GPU Memory")
    try:
        import psutil
        vmem = psutil.virtual_memory()
        print(f"Detected Available RAM: {int(vmem.available/1024/1024)} MB")
    except Exception:
        print("Unable to determine available RAM")

    init_library() # Note: if blas does not exist and is enabled, program will crash.
    print("==========")
    time.sleep(1)

    if args.password and args.password!="":
        password = args.password.strip()

    print(args)
    print("==========")

    #if RPC Host mode is specified
    if args.rpcmode=="host":
        rpc_endpt = f"{args.rpchost}:{args.rpcport}"
        rpc_devices = args.rpcdevice
        print(f"Initialize RPC server as host mode at {rpc_endpt}.")
        handle.launch_rpc_server(rpc_endpt.encode("UTF-8"),rpc_devices.encode("UTF-8"))
        time.sleep(1)
        return

    #handle loading text model
    if args.model_param:
        if not os.path.exists(args.model_param):
            if args.ignoremissing:
                print(f"Ignoring missing model file: {args.model_param}")
                args.model_param = None
            else:
                exitcounter = 999
                exit_with_error(2,f"Cannot find text model file: {args.model_param}")

        if args.lora and args.lora[0]!="":
            if not os.path.exists(args.lora[0]):
                if args.ignoremissing:
                    print(f"Ignoring missing lora file: {args.lora[0]}")
                    args.lora = None
                else:
                    exitcounter = 999
                    exit_with_error(2,f"Cannot find lora file: {args.lora[0]}")
            else:
                args.lora[0] = os.path.abspath(args.lora[0])
                if len(args.lora) > 1:
                    if not os.path.exists(args.lora[1]):
                        if args.ignoremissing:
                            print(f"Ignoring missing lora base: {args.lora[1]}")
                            args.lora = None
                        else:
                            exitcounter = 999
                            exit_with_error(2,f"Cannot find lora base: {args.lora[1]}")

                    else:
                        args.lora[1] = os.path.abspath(args.lora[1])

        if args.mmproj and args.mmproj!="":
            if not os.path.exists(args.mmproj):
                if args.ignoremissing:
                    print(f"Ignoring missing mmproj file: {args.mmproj}")
                    args.mmproj = None
                else:
                    exitcounter = 999
                    exit_with_error(2,f"Cannot find mmproj file: {args.mmproj}")
            else:
                args.mmproj = os.path.abspath(args.mmproj)

        if not args.blasthreads or args.blasthreads <= 0:
            args.blasthreads = args.threads

        modelname = os.path.abspath(args.model_param)

        # Flush stdout for win32 issue with regards to piping in terminals,
        # especially before handing over to C++ context.
        print(f"Loading Text Model: {modelname}", flush=True)
        if not modelname.endswith(".bin") and not modelname.endswith(".gguf"):
            print("WARNING: Selected Text Model does not seem to be a GGUF file! Are you sure you picked the right file?")
        loadok = load_model(modelname)
        print("Load Text Model OK: " + str(loadok))
        if args.mmproj and args.mmproj!="": # multimodal vision and audio support is only known at runtime
            has_audio_support = handle.has_audio_support()
            has_vision_support = handle.has_vision_support()
        else:
            has_audio_support = False
            has_vision_support = False

        if not loadok:
            exitcounter = 999
            exit_with_error(3,"Could not load text model: " + modelname)

        # The chat completions adapter is a list that needs derivation from chat templates
        # Try to derive chat completions adapter from chat template, now that we have the model loaded
        if args.model_param:
            ctbytes = handle.get_chat_template()
            cached_chat_template = ctypes.string_at(ctbytes).decode("UTF-8","ignore")
            if cached_chat_template != "" and (chatcompl_adapter_list is not None and isinstance(chatcompl_adapter_list, list)):
                for entry in chatcompl_adapter_list:
                    if all(s in cached_chat_template for s in entry['search']):
                        print(f"Chat completion heuristic: {entry['name']}")
                        chatcompl_adapter = entry['adapter']
                        break
            cached_jinja_kwargs = None
            try:
                jinjakwargsstr = args.jinja_kwargs if args.jinja_kwargs else None
                if jinjakwargsstr and isinstance(jinjakwargsstr, str):
                    cached_jinja_kwargs = parse_json_object(jinjakwargsstr,"jinja_kwargs")
                    cached_jinja_kwargs = cached_jinja_kwargs if cached_jinja_kwargs else None
            except Exception:
                print("Jinja Kwargs not valid JSON dict!")
                pass

            if chatcompl_adapter is None:
                print("Chat template heuristics failed to identify chat completions format. Alpaca will be used.")

    #handle loading image model
    if args.sdmodel and args.sdmodel!="":
        imgmodel = args.sdmodel
        if not imgmodel or not os.path.exists(imgmodel):
            if args.ignoremissing:
                print(f"Ignoring missing img model file: {imgmodel}")
                args.sdmodel = None
            else:
                exitcounter = 999
                exit_with_error(2,f"Cannot find image model file: {imgmodel}")
        else:
            imgvae = ""
            imgt5xxl = ""
            imgclip1 = ""
            imgclip2 = ""
            imgphotomaker = ""
            imgupscaler = ""
            imgaudiovae = ""
            global imglora_preload, imglora_bypath, imglora_name2path
            imglora_preload, imglora_bypath, imglora_name2path = mk_lora_info(args.sdlora, args.sdloramult)
            if args.sdvae:
                if os.path.exists(args.sdvae):
                    imgvae = os.path.abspath(args.sdvae)
                else:
                    print("Missing SD VAE model file...")
            if args.sdaudiovae:
                if os.path.exists(args.sdaudiovae):
                    imgaudiovae = os.path.abspath(args.sdaudiovae)
                else:
                    print("Missing SD Audio VAE model file...")
            if args.sdt5xxl:
                if os.path.exists(args.sdt5xxl):
                    imgt5xxl = os.path.abspath(args.sdt5xxl)
                else:
                    print("Missing SD T5-XXL model file...")
            if args.sdclip1:
                if os.path.exists(args.sdclip1):
                    imgclip1 = os.path.abspath(args.sdclip1)
                else:
                    print("Missing SD Clip-1 model file...")
            if args.sdclip2:
                if os.path.exists(args.sdclip2):
                    imgclip2 = os.path.abspath(args.sdclip2)
                else:
                    print("Missing SD Clip-2 model file...")
            if args.sdphotomaker:
                if os.path.exists(args.sdphotomaker):
                    imgphotomaker = os.path.abspath(args.sdphotomaker)
                else:
                    print("Missing SD Photomaker model file...")
            if args.sdupscaler:
                if os.path.exists(args.sdupscaler):
                    imgupscaler = os.path.abspath(args.sdupscaler)
                else:
                    print("Missing SD Upscaler model file...")

            imgmodel = os.path.abspath(imgmodel)
            fullsdmodelpath = imgmodel
            friendlysdmodelname = os.path.basename(imgmodel)
            friendlysdmodelname = os.path.splitext(friendlysdmodelname)[0]
            friendlysdmodelname = sanitize_string(friendlysdmodelname)
            loadok = sd_load_model(imgmodel,imgvae,imgt5xxl,imgclip1,imgclip2,imgphotomaker,imgupscaler,imgaudiovae)
            cached_sd_info = sd_get_info()
            print("Load Image Model OK: " + str(loadok))
            if not loadok:
                exitcounter = 999
                exit_with_error(3,"Could not load image model: " + imgmodel)

    #handle whisper model
    if args.whispermodel and args.whispermodel!="":
        whispermodel = args.whispermodel
        if not whispermodel or not os.path.exists(whispermodel):
            if args.ignoremissing:
                print(f"Ignoring missing whisper model file: {whispermodel}")
                args.whispermodel = None
            else:
                exitcounter = 999
                exit_with_error(2,f"Cannot find whisper model file: {whispermodel}")
        else:
            whispermodel = os.path.abspath(whispermodel)
            fullwhispermodelpath = whispermodel
            loadok = whisper_load_model(whispermodel)
            print("Load Whisper Model OK: " + str(loadok))
            if not loadok:
                exitcounter = 999
                exit_with_error(3,"Could not load whisper model: " + whispermodel)

    #handle tts model
    if args.ttsmodel and args.ttsmodel!="":
        if not os.path.exists(args.ttsmodel) or (args.ttswavtokenizer and args.ttswavtokenizer!="" and not os.path.exists(args.ttswavtokenizer)):
            if args.ignoremissing:
                print("Ignoring missing TTS model files!")
                args.ttsmodel = None
                args.ttswavtokenizer = None
            else:
                exitcounter = 999
                exit_with_error(2,f"Cannot find tts model files: {args.ttsmodel} or {args.ttswavtokenizer}")
        else:
            ttsmodelpath = args.ttsmodel
            ttsmodelpath = os.path.abspath(ttsmodelpath)
            wavtokpath = args.ttswavtokenizer
            if wavtokpath:
                wavtokpath = os.path.abspath(wavtokpath)
            loadok = tts_load_model(ttsmodelpath,wavtokpath)
            print("Load TTS Model OK: " + str(loadok))
            if not loadok:
                exitcounter = 999
                exit_with_error(3,"Could not load TTS model!")

    #handle embeddings model
    if args.embeddingsmodel and args.embeddingsmodel!="":
        if not os.path.exists(args.embeddingsmodel):
            if args.ignoremissing:
                print("Ignoring missing TTS model files!")
                args.embeddingsmodel = None
            else:
                exitcounter = 999
                exit_with_error(2,f"Cannot find embeddings model files: {args.embeddingsmodel}")
        else:
            embeddingsmodelpath = args.embeddingsmodel
            embeddingsmodelpath = os.path.abspath(embeddingsmodelpath)
            loadok = embeddings_load_model(embeddingsmodelpath)
            print("Load Embeddings Model OK: " + str(loadok))
            friendlyembeddingsmodelname = os.path.basename(embeddingsmodelpath)
            friendlyembeddingsmodelname = os.path.splitext(friendlyembeddingsmodelname)[0]
            friendlyembeddingsmodelname = sanitize_string(friendlyembeddingsmodelname)
            if not loadok:
                exitcounter = 999
                exit_with_error(3,"Could not load Embeddings model!")

    #handle music model
    mu_has_llm = True if (args.musicllm and args.musicllm!="") else False
    mu_has_embed = True if  (args.musicembeddings and args.musicembeddings!="") else False
    mu_has_diff = True if (args.musicdiffusion and args.musicdiffusion!="") else False
    mu_has_vae = True if (args.musicvae and args.musicvae!="") else False
    if mu_has_llm or mu_has_embed or mu_has_diff or mu_has_vae:
        if mu_has_llm and not any([mu_has_embed, mu_has_diff, mu_has_vae]):
            if not os.path.exists(args.musicllm):
                if args.ignoremissing:
                    print("Ignoring missing Music LLM model file!")
                    args.musicllm = None
                else:
                    exitcounter = 999
                    exit_with_error(2, "Cannot find Music LLM model file!")
            else:
                musicllmpath = os.path.abspath(args.musicllm)
                loadok = music_load_model(musicllmpath, "", "", "")
                print("Load Music LLM Only OK: " + str(loadok))
                if not loadok:
                    exitcounter = 999
                    exit_with_error(3, "Could not load Music LLM model!")
        elif mu_has_diff:
            if not (mu_has_embed and mu_has_vae):
                exitcounter = 999
                exit_with_error(2,"Invalid config: Music Diffusion requires Music embedding and Music VAE models!")

            paths_to_check = [args.musicdiffusion,args.musicembeddings,args.musicvae]
            if mu_has_llm:
                paths_to_check.append(args.musicllm)

            if not all(os.path.exists(p) for p in paths_to_check):
                if args.ignoremissing:
                    print("Ignoring missing Music model files!")
                    args.musicllm = None
                    args.musicembeddings = None
                    args.musicdiffusion = None
                    args.musicvae = None
                else:
                    exitcounter = 999
                    exit_with_error(2,"Cannot find required music diffusion/embedding/VAE model files!")
            else:
                musicdiffusionmodelpath = os.path.abspath(args.musicdiffusion)
                musicembedpath = os.path.abspath(args.musicembeddings)
                musicvaepath = os.path.abspath(args.musicvae)
                musicllmpath = os.path.abspath(args.musicllm) if mu_has_llm else ""
                loadok = music_load_model(musicllmpath,musicembedpath,musicdiffusionmodelpath,musicvaepath)
                print("Load Music Models OK: " + str(loadok))
                if not loadok:
                    exitcounter = 999
                    exit_with_error(3, "Could not load Music models!")

    #load embedded lite
    embddir = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))),"embd_res")
    try:
        with open(os.path.join(embddir, "klite.embd"), mode='rb') as f:
            embedded_kailite = f.read()
            # patch it with extra stuff
            patches = [{"find":"Sorry, KoboldAI Lite requires Javascript to function.","replace":"Sorry, KoboldAI Lite requires Javascript to function.<br>You can use <a class=\"color_blueurl\" href=\"/noscript\">KoboldCpp NoScript mode</a> instead."},
                       {"find":"var localflag = urlParams.get('local');","replace":"var localflag = true;"},
                       {"find":"<p id=\"tempgtloadtxt\">Loading...</p>","replace":"<p id=\"tempgtloadtxt\">Loading...<br>(If load fails, try <a class=\"color_blueurl\" href=\"/noscript\">KoboldCpp NoScript mode</a> instead, or adding /noscript at this url.)</p>"}]
            embedded_kailite = embedded_kailite.decode("UTF-8","ignore")
            for p in patches:
                embedded_kailite = embedded_kailite.replace(p["find"], p["replace"])
            embedded_kailite = embedded_kailite.encode()
            embedded_kailite_gz = gzip.compress(embedded_kailite)
            print("Embedded KoboldAI Lite loaded.")
    except Exception:
        print("Could not find KoboldAI Lite. Embedded KoboldAI Lite will not be available.")

    try:
        with open(os.path.join(embddir, "kcpp_docs.embd"), mode='rb') as f:
            embedded_kcpp_docs = f.read()
            embedded_kcpp_docs_gz = gzip.compress(embedded_kcpp_docs)
            print("Embedded API docs loaded.")
    except Exception:
        print("Could not find Embedded KoboldCpp API docs.")

    try:
        with open(os.path.join(embddir, "kcpp_sdui.embd"), mode='rb') as f:
            embedded_kcpp_sdui = f.read()
            embedded_kcpp_sdui_gz = gzip.compress(embedded_kcpp_sdui)
            if args.sdmodel:
                print("Embedded SDUI loaded.")
    except Exception:
        print("Could not find Embedded SDUI.")

    try:
        with open(os.path.join(embddir, "lcpp.gz.embd"), mode='rb') as f:
            embedded_lcpp_ui_gz = f.read()
            print("Llama.cpp UI loaded.")
    except Exception:
        print("Could not find Embedded llama.cpp UI.")

    try:
        with open(os.path.join(embddir, "kcpp_musicui.embd"), mode='rb') as f:
            embedded_musicui = f.read()
            embedded_musicui_gz = gzip.compress(embedded_musicui)
            if args.musicllm or args.musicdiffusion or args.ttsmodel:
                print("Embedded MusicUI loaded.")
    except Exception:
        print("Could not find Embedded MusicUI.")

    # load all TTS audio files
    if args.ttsmodel or ttsName is not None:
        try:
            global voicebank, voicelist
            voicebank = {}
            voicecount = 0
            voicelist = []

            try:
                with open(os.path.join(embddir, "qwen3tts_voices_json.embd"), mode='r', encoding='utf-8', errors='ignore') as f:
                    vdict = json.load(f)
                    for key, value in vdict.items():
                        voicelist.append(key)
                        voicebank[key] = value
            except Exception:
                print("Could not find Embedded Qwen3TTS voices.")

            voicelist.append("random")
            voicebank["random"] = ""
            voicelist.append("instruct")
            voicebank["instruct"] = ""

            if args.ttsdir and os.path.isdir(args.ttsdir):
                for filename in os.listdir(args.ttsdir):
                    if filename.lower().endswith((".mp3", ".wav")):
                        full_path = os.path.join(args.ttsdir, filename)
                        with open(full_path, "rb") as f:
                            encoded = base64.b64encode(f.read()).decode("utf-8")
                            voicebank[filename] = encoded
                            voicecount += 1
                            voicelist.append(os.path.basename(filename))
            print(f"Loaded {voicecount} TTS voices.")
        except Exception:
            print("Could not load TTS voices.")

    if args.mcpfile and isinstance(args.mcpfile, str):
        threading.Thread(target=load_mcp_async, args=(args,), daemon=True).start()
        time.sleep(0.2) # short delay to allow get_capabilities to work

    # print enabled modules
    caps = get_capabilities()
    enabledmlist = []
    disabledmlist = []
    apimlist = ["KoboldCppApi"]
    if "llm" in caps and caps["llm"]:
        apimlist.append("OpenAiApi")
        apimlist.append("OllamaApi")
        apimlist.append("AnthropicApi")
    if "txt2img" in caps and caps["txt2img"]:
        apimlist.append("A1111ForgeApi")
        apimlist.append("ComfyUiApi")
    if "transcribe" in caps and caps["transcribe"]:
        apimlist.append("WhisperTranscribeApi")
    if "tts" in caps and caps["tts"]:
        apimlist.append("XttsApi")
        apimlist.append("OpenAiSpeechApi")
    enabledmlist.append("TextGeneration") if "llm" in caps and caps["llm"] else disabledmlist.append("TextGeneration")
    enabledmlist.append("ImageGeneration") if "txt2img" in caps and caps["txt2img"] else disabledmlist.append("ImageGeneration")
    enabledmlist.append("VoiceRecognition") if "transcribe" in caps and caps["transcribe"] else disabledmlist.append("VoiceRecognition")
    enabledmlist.append("MultimodalVision") if "vision" in caps and caps["vision"] else disabledmlist.append("MultimodalVision")
    enabledmlist.append("MultimodalAudio") if "audio" in caps and caps["audio"] else disabledmlist.append("MultimodalAudio")
    enabledmlist.append("NetworkMultiplayer") if "multiplayer" in caps and caps["multiplayer"] else disabledmlist.append("NetworkMultiplayer")
    enabledmlist.append("ApiKeyPassword") if "protected" in caps and caps["protected"] else disabledmlist.append("ApiKeyPassword")
    enabledmlist.append("WebSearchProxy") if "websearch" in caps and caps["websearch"] else disabledmlist.append("WebSearchProxy")
    enabledmlist.append("TextToSpeech") if "tts" in caps and caps["tts"] else disabledmlist.append("TextToSpeech")
    enabledmlist.append("VectorEmbeddings") if "embeddings" in caps and caps["embeddings"] else disabledmlist.append("VectorEmbeddings")
    enabledmlist.append("AdminControl") if "admin" in caps and caps["admin"]!=0 else disabledmlist.append("AdminControl")
    enabledmlist.append("MCPBridge") if "mcp" in caps and caps["mcp"] else disabledmlist.append("MCPBridge")
    enabledmlist.append("MusicGen") if "music" in caps and caps["music"] else disabledmlist.append("MusicGen")
    enabledmlist.append("RouterMode") if "router" in caps and caps["router"] else disabledmlist.append("RouterMode")

    print(f"======\nActive Modules: {' '.join(enabledmlist)}")
    print(f"Inactive Modules: {' '.join(disabledmlist)}")
    if not args.cli:
        print(f"Enabled APIs: {' '.join(apimlist)}")

    global sslvalid
    if args.ssl:
        if len(args.ssl)==2 and isinstance(args.ssl[0], str) and os.path.exists(args.ssl[0]) and isinstance(args.ssl[1], str) and os.path.exists(args.ssl[1]):
            sslvalid = True
            print("SSL configuration is valid and will be used.")
        else:
            print("Your SSL configuration is INVALID. SSL will not be used.")
    endpoint_url = ""
    remote_url = ""
    httpsaffix = ("https" if sslvalid else "http")
    displayedport = (args.port if not args.proxy_port else args.proxy_port)
    if args.host=="":
        endpoint_url = f"{httpsaffix}://localhost:{displayedport}"
    else:
        endpoint_url = f"{httpsaffix}://{args.host}:{displayedport}"

    if start_server:
        if not args.remotetunnel:
            if displayedport!=11434:
                print("Note: For third party Ollama API Emulation, you should set the port to 11434.")
            else:
                print("Ollama Emulation is now available at port 11434.")
            print(f"Starting Kobold API on port {displayedport} at {endpoint_url}/api/")
            print(f"Starting OpenAI Compatible API on port {displayedport} at {endpoint_url}/v1/")
            print(f"Starting llama.cpp secondary WebUI at {endpoint_url}/lcpp/")
            if args.sdmodel:
                print(f"StableUI is available at {endpoint_url}/sdui/")
            if args.musicdiffusion or args.musicllm or args.ttsmodel:
                print(f"MusicUI is available at {endpoint_url}/musicui/")
        elif global_memory:
            val = global_memory["tunnel_url"]
            if val:
                endpoint_url = val
                remote_url = val
                print(f"Your remote Kobold API can be found at {endpoint_url}/api")
                print(f"Your remote OpenAI Compatible API can be found at {endpoint_url}/v1")
                print(f"Starting llama.cpp secondary WebUI at {endpoint_url}/lcpp/")
                if args.sdmodel:
                    print(f"StableUI is available at {endpoint_url}/sdui/")
                if args.musicdiffusion or args.musicllm or args.ttsmodel:
                    print(f"MusicUI is available at {endpoint_url}/musicui/")
            global_memory["load_complete"] = True
        if args.launch:
            def launch_browser_thread():
                LaunchWebbrowser(endpoint_url,"--launch was set, but could not launch web browser automatically.")
            browser_thread = threading.Timer(2, launch_browser_thread) #2 second delay
            browser_thread.start()

        if args.hordekey and args.hordekey!="":
            if args.hordeworkername and args.hordeworkername!="":
                workers_to_use = 1
                if args.parallelrequests:
                    workers_to_use = args.parallelrequests if (args.parallelrequests>1) else 1
                for w in range(0,workers_to_use):
                    wid = (w+1)
                    print(f"Launching horde worker {wid}...")
                    horde_thread = threading.Thread(target=run_horde_worker,args=(args,args.hordekey,args.hordeworkername,wid,workers_to_use))
                    horde_thread.daemon = True
                    horde_thread.start()
            else:
                print("Horde worker could not start. You need to specify a horde worker name with --hordeworkername")

    #if post-ready script specified, execute it
    if args.onready:
        def onready_subprocess():
            print("Starting Post-Load subprocess...")
            subprocess.run(args.onready, shell=True)
        timer_thread = threading.Timer(1, onready_subprocess) #1 second delay
        timer_thread.start()

    if not start_server:
        if args.cli:
            print("\n===\nNow running KoboldCpp in Interactive Terminal Chat mode.\nType /quit or /exit to end session.\n")
            lastturns = []
            if args.prompt and args.prompt!="":
                lastturns.append({"role":"system","content":args.prompt})
                print(f"System Prompt:\n{args.prompt}\n")
            while True:
                lastuserinput = input("> ")
                if lastuserinput=="/quit" or lastuserinput=="/exit":
                    break
                if not lastuserinput:
                    continue
                lastturns.append({"role":"user","content":lastuserinput})
                payload = {"messages":lastturns,"rep_pen":1.07,"temperature":0.8}
                payload = transform_genparams(payload, 4, False) #to chat completions
                if args.debugmode < 1:
                    suppress_stdout()
                genout = generate(genparams=payload)
                if args.debugmode < 1:
                    restore_stdout()
                result = (genout["text"] if "text" in genout else "")
                if result:
                    lastturns.append({"role":"assistant","content":result})
                    print(result.strip() + "\n", flush=True)
                else:
                    print("(No Response Received)\n", flush=True)
        else:
            save_to_file = (args.benchmark and args.benchmark!="stdout" and args.benchmark!="")
            benchmaxctx = maxctx
            benchlen = args.genlimit if args.genlimit > 0 else 100
            benchtemp = 0.1
            benchtopk = 1
            benchreppen = 1
            benchbaneos = True
            benchmodel = sanitize_string(os.path.splitext(os.path.basename(modelname))[0])
            benchprompt = ""
            if args.prompt:
                benchprompt = args.prompt
                benchtopk = 100
                benchreppen = 1.07
                benchtemp = 0.8
                if not args.benchmark:
                    benchbaneos = False
            if args.benchmark:
                if os.path.exists(args.benchmark) and os.path.getsize(args.benchmark) > 1000000:
                    print("\nWarning: The benchmark CSV output file you selected exceeds 1MB. This is probably not what you want, did you select the wrong CSV file?\nFor safety, benchmark output will not be saved.")
                    save_to_file = False
                if save_to_file:
                    print(f"\nRunning benchmark (Save to File: {args.benchmark})...")
                else:
                    print("\nRunning benchmark (Not Saved)...")
                if benchprompt=="":
                    benchprompt = " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
                    for i in range(0,14): #generate massive prompt
                        benchprompt += benchprompt
            genp = {
                "prompt":benchprompt,
                "max_length":benchlen,
                "max_context_length":benchmaxctx,
                "temperature":benchtemp,
                "top_k":benchtopk,
                "rep_pen":benchreppen,
                "ban_eos_token":benchbaneos
            }
            genout = generate(genparams=genp)
            result = genout['text']
            if args.prompt and not args.benchmark:
                restore_stdout()
                print(result)
            if args.benchmark:
                result = (result[:8] if len(result)>8 else "") if not args.prompt else result
                t_pp = float(handle.get_last_process_time())*float(benchmaxctx-benchlen)*0.001
                t_gen = float(handle.get_last_eval_time())*float(benchlen)*0.001
                s_pp = float(benchmaxctx-benchlen)/t_pp
                s_gen = float(benchlen)/t_gen
                datetimestamp = datetime.now(timezone.utc)
                benchflagstr = f"{str(vars(args))}"
                print(f"\nBenchmark Completed - v{KcppVersion} Results:\n======")
                print(f"Flags: {benchflagstr}")
                print(f"Timestamp: {datetimestamp}")
                print(f"Backend: {libname}")
                print(f"Layers: {args.gpulayers if not args.autofit else 'Autofit'}")
                print(f"Model: {benchmodel}")
                print(f"MaxCtx: {benchmaxctx}")
                print(f"GenAmount: {benchlen}\n-----")
                print(f"ProcessingTime: {t_pp:.3f}s")
                print(f"ProcessingSpeed: {s_pp:.2f}T/s")
                print(f"GenerationTime: {t_gen:.3f}s")
                print(f"GenerationSpeed: {s_gen:.2f}T/s")
                print(f"TotalTime: {(t_pp+t_gen):.3f}s")
                print(f"Output: {result}\n-----")
                if save_to_file:
                    try:
                        with open(args.benchmark, "a") as file:
                            file.seek(0, 2)
                            if file.tell() == 0: #empty file
                                file.write("Timestamp,Backend,Layers,Model,MaxCtx,GenAmount,ProcessingTime,ProcessingSpeed,GenerationTime,GenerationSpeed,TotalTime,Output,Flags")
                            file.write(f"\n{datetimestamp},{libname},{args.gpulayers},{benchmodel},{benchmaxctx},{benchlen},{t_pp:.2f},{s_pp:.2f},{t_gen:.2f},{s_gen:.2f},{(t_pp+t_gen):.2f},{result},\"{benchflagstr}\"")
                    except Exception as e:
                        print(f"Error writing benchmark to file: {e}")
                if global_memory and using_gui_launcher and not save_to_file:
                    global_memory["input_to_exit"] = True
                    time.sleep(1)

    if start_server:
        if args.remotetunnel:
            if remote_url:
                print(f"======\nYour remote tunnel is ready, please connect to {remote_url}", flush=True)
        else:
            # Flush stdout for previous win32 issue so the client can see output.
            print(f"======\nPlease connect to custom endpoint at {endpoint_url}", flush=True)
        asyncio.run(RunServerMultiThreaded(args.host, args.port, KcppServerRequestHandler))
    else:
        # Flush stdout for previous win32 issue so the client can see output.
        if not args.prompt or args.benchmark or args.cli:
            print("Server was not started, main function complete. Idling.", flush=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    def check_range(value_type, min_value, max_value):
        def range_checker(arg: str):
            try:
                f = value_type(arg)
            except ValueError:
                raise argparse.ArgumentTypeError(f'must be a valid {value_type}')
            if f < min_value or f > max_value:
                raise argparse.ArgumentTypeError(f'must be within [{min_value}, {max_value}]')
            return f
        return range_checker

    parser = argparse.ArgumentParser(description=f'KoboldCpp Server - Version {KcppVersion}')

    compatgroup = parser.add_mutually_exclusive_group()
    compatgroup.add_argument("--usecuda", "--usecublas", "--usehipblas", help="Use CUDA for GPU Acceleration. Requires CUDA. Enter a number afterwards to select and use 1 GPU. Leaving no number will use all GPUs.", nargs='*',metavar=('[main GPU ID]'), choices=['0','1','2','3','all',  'mmq','nommq','normal','lowvram','rowsplit'])
    compatgroup.add_argument("--usevulkan", help="Use Vulkan for GPU Acceleration. Can optionally specify one or more GPU Device ID (e.g. --usevulkan 0), leave blank to autodetect.", metavar=('[Device IDs]'), nargs='*', type=int, default=None)
    compatgroup.add_argument("--usecpu", help="Do not use any GPU acceleration (CPU Only)", action='store_true')
    modelgroup = parser.add_mutually_exclusive_group() #we want to be backwards compatible with the unnamed positional args
    modelgroup.add_argument("--model","-m", metavar=('[filenames]'), help="Model file to load. Accepts multiple values if they are URLs.", type=str, nargs='+', default=[])
    modelgroup.add_argument("model_param", help="Model file to load (positional)", nargs="?")
    parser.add_argument("--config", metavar=('[filename]'), help="Load settings from a .kcpps file. Other arguments will be ignored", type=str, nargs=1)
    parser.add_argument("--contextsize","--ctx-size", "-c", help="Controls the memory allocated for maximum context size, only change if you need more RAM for big contexts. (default 8192).",metavar=('[256 to 262144]'), type=check_range(int,256,262144), default=8192)
    parser.add_argument("--gpulayers","--gpu-layers","--n-gpu-layers","-ngl", help="Set number of layers to offload to GPU (when using GPU). Set to -1 to enable autofit (default), set to 0 to disable GPU offload.",metavar=('[GPU layers]'), nargs='?', const=1, type=int, default=-1)
    parser.add_argument("--host", metavar=('[ipaddr]'), help="Host IP to listen on. If this flag is not set, all routable interfaces are accepted.", default="")
    parser.add_argument("--launch", help="Launches a web browser when load is completed.", action='store_true')
    parser.add_argument("--tensor_split","--tensorsplit","--tensor-split","-ts", help="For CUDA and Vulkan only, ratio to split tensors across multiple GPUs, space-separated list of proportions, e.g. 7 3", metavar=('[Ratios]'), type=float, nargs='+')
    parser.add_argument("--threads","-t", metavar=('[threads]'), help="Use a custom number of threads if specified. Otherwise, uses an amount based on CPU cores", type=int, default=get_default_threads())
    portgroup = parser.add_mutually_exclusive_group() #we want to be backwards compatible with the unnamed positional args
    portgroup.add_argument("--port", metavar=('[portnumber]'), help=f"Port to listen on. (Defaults to {defaultport})", default=defaultport, type=int, action='store')
    portgroup.add_argument("port_param", help="Port to listen on (positional)", default=defaultport, nargs="?", type=int, action='store')
    parser.add_argument("--version", help="Prints version and exits.", action='store_true')

    #more advanced params
    advparser = parser.add_argument_group('Advanced Commands')
    advparser.add_argument("--analyze", metavar=('[filename]'), help="Reads the metadata, weight types and tensor names in any GGUF file.", default="")
    advparser.add_argument("--autofit","--fit","-fit", help="Forces autofit, which attempts to fit the model in the best possible way. Overrides everything else.", action='store_true')
    advparser.add_argument("--autofitpadding", metavar=('[padding in MB]'), help="How much spare allowance in MB should autofit reserve? If it's too little, the load might fail.", type=int, default=default_autofit_padding)
    advparser.add_argument("--batchsize","--blasbatchsize","--batch-size","-b", help="Sets the batch size used in batched processing (default 512). Setting it to -1 disables batched mode, but keeps other benefits like GPU offload.", type=int,choices=[-1,16,32,64,128,256,512,1024,2048,4096], default=512)
    advparser.add_argument("--benchmark", help="Do not start server, instead run benchmarks. If filename is provided, appends results to provided file.", metavar=('[filename]'), nargs='?', const="stdout", type=str, default=None)
    advparser.add_argument("--blasthreads","--batchthreads","--threadsbatch","--threads-batch", help="Use a different number of threads during batching if specified. Otherwise, has the same value as --threads",metavar=('[threads]'), type=int, default=0)
    advparser.add_argument("--chatcompletionsadapter", metavar=('[filename]'), help="Select an optional ChatCompletions Adapter JSON file to force custom instruct tags.", default="AutoGuess")
    advparser.add_argument("--cli", help="Does not launch KoboldCpp HTTP server. Instead, enables KoboldCpp from the command line, accepting interactive console input and displaying responses to the terminal.", action='store_true')
    advparser.add_argument("--debugmode", help="Shows additional debug info in the terminal. Levels: -1 (Horde-quiet, suppresses non-essential prints; auto-applied when Horde args are set), 0 (default, normal output), 1 (verbose: extra slot/cache info, larger print buffers, retains horde-debug prefix). Passing the flag without a value implies 1.", nargs='?', const=1, type=int, default=0)
    advparser.add_argument("--defaultgenamt", help="How many tokens to generate by default, if not specified. Must be smaller than context size. Usually, your frontend GUI will override this.", type=check_range(int,64,8192), default=default_genlen)
    advparser.add_argument("--device", "-dev", metavar=('<dev1,dev2,..>'), help="Set llama.cpp compatible device selection override. Comma separated. Overrides normal device choices.", default="")
    advparser.add_argument("--downloaddir", metavar=('[directory]'), help="Specify a directory that models will be downloaded to or searched from, if unset uses the working directory.", default="")
    advparser.add_argument("--draftamount","--draft-max","--draft-n", metavar=('[tokens]'), help="How many tokens to draft per chunk before verifying results", type=int, default=default_draft_amount)
    advparser.add_argument("--draftgpulayers","--gpu-layers-draft","--n-gpu-layers-draft","-ngld", metavar=('[layers]'), help="How many layers to offload to GPU for the draft model (default=full offload)", type=int, default=999)
    advparser.add_argument("--draftgpusplit", help="GPU layer distribution ratio for draft model (default=same as main). Only works if multi-GPUs selected for MAIN model and tensor_split is set!", metavar=('[Ratios]'), type=float, nargs='+')
    advparser.add_argument("--draftmodel","--model-draft","-md", metavar=('[filename]'), help="Load a small draft model for speculative decoding. It will be fully offloaded. Vocab must match the main model.", default="")
    advparser.add_argument("--enableguidance", help="Enables the use of Classifier-Free-Guidance, which allows the use of negative prompts. Has performance and memory impact.", action='store_true')
    advparser.add_argument("--exportconfig", help="Exports the current selected arguments as a .kcpps settings file", metavar=('[filename]'), type=str, default="")
    advparser.add_argument("--exporttemplate", help="Exports the current selected arguments as a .kcppt template file", metavar=('[filename]'), type=str, default="")
    advparser.add_argument("--failsafe", help="Use failsafe mode, extremely old CPU compatibility mode that should work on all devices.", action='store_true')
    advparser.add_argument("--foreground", help="Windows only. Sends the terminal to the foreground every time a new prompt is generated. This helps avoid some idle slowdown issues.", action='store_true')
    advparser.add_argument("--gendefaults", metavar=('{"parameter":"value",...}'), help="Sets extra default parameters for some fields in API requests, as a JSON string.", default="")
    advparser.add_argument("--gendefaultsoverwrite", help="Allow the gendefaults parameters to overwrite the original value in API payloads.", action='store_true')
    advparser.add_argument("--genlimit","--promptlimit", help="Sets the maximum number of generated tokens, it will restrict all generations to this or lower. Also usable with --prompt or --benchmark.",metavar=('[token limit]'), type=int, default=0)
    advparser.add_argument("--highpriority", help="Experimental flag. If set, increases the process CPU priority, potentially speeding up generation. Use caution.", action='store_true')
    advparser.add_argument("--ignoremissing", help="Ignores all missing non-essential files, just skipping them instead.", action='store_true')
    advparser.add_argument("--jinja", help="Enables using jinja chat template formatting for chat completions endpoint. Other endpoints are unaffected. Tool calls are done without jinja.", action='store_true')
    advparser.add_argument("--jinja_kwargs","--jinja-kwargs","--jinjakwargs","--chat-template-kwargs", metavar=('{"parameter":"value",...}'), help="Set additional fields for Jinja JSON template parser, must be a valid JSON object.", default="")
    advparser.add_argument("--jinja_tools","--jinja-tools","--jinjatools", help="Enables using jinja chat template formatting for chat completions endpoint. Other endpoints are unaffected. Tool calls are done with jinja.", action='store_true')
    advparser.add_argument("--jinjatemplate","--chat-template-file", metavar=('[filename]'), help="Select a custom Jinja chat template, will overwrite model jinja chat template", default="")
    advparser.add_argument("--jinjathink", help="A quick way to enable or disable thinking in the jinja template.", type=str, choices=['default','true','false'], default="default")
    advparser.add_argument("--lora", help="GGUF models only, applies a lora file on top of model.", metavar=('[lora_filename]'), nargs='+')
    advparser.add_argument("--loramult", metavar=('[amount]'), help="Multiplier for the Text LORA model to be applied.", type=float, default=1.0)
    advparser.add_argument("--lowvram","-nkvo","--no-kv-offload", help="If supported by the backend, do not offload KV to GPU (lowvram mode). Not recommended, will be slow.", action='store_true')
    advparser.add_argument("--maingpu","--main-gpu","-mg", help="Only used in a multi-gpu setup. Sets the index of the main GPU that will be used.",metavar=('[Device ID]'), type=int, default=-1)
    advparser.add_argument("--maxrequestsize", metavar=('[size in MB]'), help="Specify a max request payload size. Any requests to the server larger than this size will be dropped. Do not change if unsure.", type=int, default=32)
    advparser.add_argument("--mcpfile", metavar=('[mcp json file]'), help="Specify path to mcp.json which contains the Cladue Desktop compatible MCP server config.", default="")
    advparser.add_argument("--mmproj", metavar=('[filename]'), help="Select a multimodal projector file for vision models like LLaVA.", default="")
    advparser.add_argument("--mmprojcpu","--no-mmproj-offload", help="Force CLIP for Vision mmproj always on CPU.", action='store_true')
    advparser.add_argument("--moecpu","--n-cpu-moe", "-ncmoe", metavar=('[layers affected]'), help="Keep the Mixture of Experts (MoE) weights of the first N layers in the CPU. If no value is provided, applies to all layers.", nargs='?', const=999, type=int, default=0)
    advparser.add_argument("--moeexperts", metavar=('[num of experts]'), help="How many experts to use for MoE models (default=follow gguf)", type=int, default=-1)
    advparser.add_argument("--multiuser", help="Set maximum number of queued incoming requests allowed.", metavar=('limit'), type=int, nargs='?', const=multiuser_concurrent_limit, default=multiuser_concurrent_limit)
    advparser.add_argument("--multiplayer", help="Hosts a shared multiplayer session that others can join.", action='store_true')
    advparser.add_argument("--nocertify", help="Allows insecure SSL connections. Use this if you have cert errors and need to bypass certificate restrictions.", action='store_true')
    advparser.add_argument("--nofastforward", help="If set, do not attempt to fast forward GGUF context (always reprocess). Will also enable noshift", action='store_true')
    advparser.add_argument("--noflashattention","--no-flash-attn","-nofa", help="Disables flash attention.", action='store_true')
    advparser.add_argument("--noavx2", help="Do not use AVX2 instructions, a slower compatibility mode for older devices.", action='store_true')
    advparser.add_argument("--nobostoken", help="Prevents BOS token from being added at the start of any prompt. Usually NOT recommended for most models.", action='store_true')
    advparser.add_argument("--nommq", help="Disables MMQ, only used for cuda backend. This flag may be removed in future.", action='store_true')
    advparser.add_argument("--nomodel", help="Allows you to launch the GUI alone, without selecting any model.", action='store_true')
    advparser.add_argument("--nopipelineparallel", help="Disable Pipeline Parallelism. Pipeline Parallelism provides faster multigpu speeds but using more memory, only active for multigpu.", action='store_true')
    advparser.add_argument("--noshift","--no-context-shift", help="If set, do not attempt to Trim and Shift the GGUF context.", action='store_true')
    advparser.add_argument("--noswa","--swa-full", help="If set, uses full-size SWA KV Cache. Otherwise, SWA will be enabled automatically on models that support it. SWA saves memory but cannot be used with context shifting.", action='store_true')
    advparser.add_argument("--onready", help="An optional shell command to execute after the model has been loaded.", metavar=('[shell command]'), type=str, default="")
    advparser.add_argument("--overridekv","--override-kv", metavar=('[name=type:value]'), help="Override metadata value by key. Separate multiple values with commas. Format is name=type:value. Types: int, float, bool, str", default="")
    advparser.add_argument("--overridenativecontext", help="Overrides the native trained context of the loaded model with a custom value to be used for Rope scaling.",metavar=('[trained context]'), type=int, default=0)
    advparser.add_argument("--overridetensors","--override-tensor","-ot", metavar=('[tensor name pattern=buffer type]'), help="Override selected backend for specific tensors matching tensor_name_regex_pattern=buffer_type, same as in llama.cpp.", default="")
    advparser.add_argument("--parallelrequests","--continuous-batching","--contbatch", help="Allows multiple requests to be batched and executed in parallel. Only works for basic text generation requests (Experimental, No media)", metavar=('[slots]'), type=check_range(int,0,32), default=1)
    advparser.add_argument("--password", metavar=('[API key]'), help="Enter a password required to use this instance. This key will be required for all text endpoints. Image endpoints are not secured. Can also be set with env var KCPP_PASSWORD", default=os.getenv('KCPP_PASSWORD',None))
    advparser.add_argument("--preloadstory", metavar=('[savefile]'), help="Configures a prepared story json save file to be hosted on the server, which frontends (such as KoboldAI Lite) can access over the API.", default="")
    advparser.add_argument("--prompt","-p", metavar=('[prompt]'), help="Passing a prompt string triggers a direct inference, loading the model, outputs the response to stdout and exits. Can be used alone or with benchmark.", type=str, default="")
    advparser.add_argument("--quantkv", help="Sets the KV cache data type quantization, options are f16/bf16/q8_0/q5_1/q4_0. Requires Flash Attention for full effect, otherwise only K cache is quantized.",metavar=('[quantization level f16/bf16/q8_0/q5_1/q4_0]'), type=str, choices=["f16","bf16","q8_0","q5_1","q4_0","0","1","2","3"], default="f16")
    advparser.add_argument("--quiet", help="Enable quiet mode, which hides generation inputs and outputs in the terminal. Quiet mode is automatically enabled when running a horde worker.", action='store_true')
    advparser.add_argument("--ratelimit", metavar=('[seconds]'), help="If enabled, rate limit generative request by IP address. Each IP can only send a new request once per X seconds.", type=int, default=0)
    advparser.add_argument("--remotetunnel", help="Uses Cloudflare to create a remote tunnel, allowing you to access koboldcpp remotely over the internet even behind a firewall.", action='store_true')
    advparser.add_argument("--ropeconfig", help="If set, uses customized RoPE scaling from configured frequency scale and frequency base (e.g. --ropeconfig 0.25 10000). Otherwise, uses NTK-Aware scaling set automatically based on context size. For linear rope, simply set the freq-scale and ignore the freq-base",metavar=('[rope-freq-scale]', '[rope-freq-base]'), default=[0.0, 10000.0], type=float, nargs='+')
    advparser.add_argument("--savedatafile", metavar=('[savefile]'), help="If enabled, creates or opens a persistent database file on the server, that allows users to save and load their data remotely. A new file is created if it does not exist.", default="")
    advparser.add_argument("--singleinstance", help="Allows this KoboldCpp instance to be shut down by any new instance requesting the same port, preventing duplicate servers from clashing on a port.", action='store_true')
    advparser.add_argument("--smartcache", help="Enables intelligent context switching by saving KV cache snapshots to RAM. Requires fast forwarding.", metavar=('limit'), nargs='?', const=1, type=int, default=0)
    advparser.add_argument("--smartcontext", help="Reserving a portion of context to try processing less frequently. Outdated. Not recommended.", action='store_true')
    advparser.add_argument("--splitmode","-sm","--split-mode", help="How to split the model across multiple GPUs", metavar=('[split mode]'), type=str, choices=splitmode_choices, default=splitmode_choices[0])
    advparser.add_argument("--ssl", help="Allows all content to be served over SSL instead. A valid UNENCRYPTED SSL cert and key .pem files must be provided", metavar=('[cert_pem]', '[key_pem]'), nargs='+')
    advparser.add_argument("--swapadding", help="How much extra to pad the SWA KV cache, this affects the rewind limit before reprocessing is forced.", type=int, default=swa_padding_default)
    advparser.add_argument("--unpack", help="Extracts the file contents of the KoboldCpp binary into a target directory.", metavar=('destination'), type=str, default="")
    advparser.add_argument("--usemlock","--mlock", help="Enables mlock, preventing the RAM used to load the model from being paged out. Not usually recommended.", action='store_true')
    compatgroup3 = advparser.add_mutually_exclusive_group()
    compatgroup3.add_argument("--usemmap", help="If set, uses mmap to load model.", action='store_true')
    advparser.add_argument("--visionmaxres", metavar=('[max px]'), help="Clamp MMProj vision maximum allowed resolution. Allowed values are between 512 to 2048 px (default 1024).", type=int, default=default_visionmaxres)
    advparser.add_argument("--visionmaxtokens","--image-max-tokens", metavar=('[tokens]'), help="Override the maximum tokens for the MMProj embedding (default -1).", type=int, default=-1)
    advparser.add_argument("--visionmintokens","--image-min-tokens", metavar=('[tokens]'), help="Override the minimum tokens for the MMProj embedding (default -1).", type=int, default=-1)
    advparser.add_argument("--websearch", help="Enable the local search engine proxy so Web Searches can be done.", action='store_true')
    compatgroup2 = advparser.add_mutually_exclusive_group()
    compatgroup2.add_argument("--showgui", help="Always show the GUI instead of launching the model right away when loading settings from a .kcpps file.", action='store_true')
    compatgroup2.add_argument("--skiplauncher", help="Doesn't display or use the GUI launcher. Overrides showgui.", action='store_true')

    hordeparsergroup = parser.add_argument_group('Horde Worker Commands')
    hordeparsergroup.add_argument("--hordemodelname", metavar=('[name]'), help="Sets your AI Horde display model name.", default="")
    hordeparsergroup.add_argument("--hordeworkername", metavar=('[name]'), help="Sets your AI Horde worker name.", default="")
    hordeparsergroup.add_argument("--hordekey", metavar=('[apikey]'), help="Sets your AI Horde API key.", default="")
    hordeparsergroup.add_argument("--hordemaxctx", metavar=('[amount]'), help="Sets the maximum context length your worker will accept from an AI Horde job. If 0, matches main context limit.", type=int, default=0)
    hordeparsergroup.add_argument("--hordegenlen", metavar=('[amount]'), help="Sets the maximum number of tokens your worker will generate from an AI horde job.", type=int, default=0)

    sdparsergroup = parser.add_argument_group('Image Generation Commands')
    sdparsergroup.add_argument("--sdaudiovae", metavar=('[filename]'), help="Specify an image generation audio VAE for LTX2.3 video generation.", default="")
    sdparsergroup.add_argument("--sdclamped", metavar=('[maxres]'), help="If specified, limit generation steps and image size for shared use. Accepts an extra optional parameter that indicates maximum resolution (eg. 768 clamps to 768x768, min 512px, disabled if 0).", nargs='?', const=512, type=int, default=0)
    sdparsergroup.add_argument("--sdclampedsoft", metavar=('[maxres]'), help="If specified, limit max image size to curb memory usage. Similar to --sdclamped, but less strict, allows trade-offs between width and height (e.g. 640 would allow 640x640, 512x768 and 768x512 images).", type=int, default=0)
    sdparsergroup.add_argument("--sdclip1", "--sdclipl", metavar=('[filename]'), help="Specify first safetensors Clip model (SD3 or Flux Clip-L, WAN or QwenImg vision). Leave blank if prebaked or unused.", default="")
    sdparsergroup.add_argument("--sdclip2", "--sdclipg", metavar=('[filename]'), help="Specify second safetensors Clip model (SD3 Clip-G). Leave blank if prebaked or unused.", default="")
    sdparsergroup.add_argument("--sdclipdevice", metavar=('[Device ID]'), help=f"CLIP / T5 / LLM device for image generation. GPU index, -1 or 'main' for the main GPU, or 'CPU' (default: {default_sdclipdevice}).", type=sd_get_device_number, default=None)
    sdparsergroup.add_argument("--sdconvdirect", help="Enables Conv2D Direct. May improve performance or reduce memory usage. Might crash if not supported by the backend. Can be 'off' (default) to disable, 'full' to turn it on for all operations, or 'vaeonly' to enable only for the VAE.", type=sd_convdirect_option, choices=sd_convdirect_choices, default=sd_convdirect_choices[0])
    sdparsergroup.add_argument("--sdflashattention", help="Enables Flash Attention for image generation.", action='store_true')
    sdparsergrouplora = sdparsergroup.add_mutually_exclusive_group()
    sdparsergrouplora.add_argument("--sdlora", metavar=('[filename]'), help="Specify image generation LoRAs safetensors models to be applied. Multiple LoRAs are accepted.", nargs='+')
    sdparsergroup.add_argument("--sdloramult", metavar=('[amounts]'), help="Multipliers for the image LoRA model to be applied.", type=float, nargs='+', default=[1.0])
    sdparsergroup.add_argument("--sdmaingpu", metavar=('[Device ID]'), help="If specified, Image Generation weights will be placed on the selected GPU index. GPU index, -1 or 'main' for the main GPU, or 'CPU' (default: 'main')", type=sd_get_device_number, default=None)
    sdparsergroup.add_argument("--sdmodel", metavar=('[filename]'), help="Specify an image generation safetensors or gguf model to enable image generation.", default="")
    sdparsergroup.add_argument("--sdoffloadcpu", help="Offload image weights in RAM to save VRAM, swap into VRAM when needed.", action='store_true')
    sdparsergroup.add_argument("--sdphotomaker", metavar=('[filename]'), help="PhotoMaker is a model that allows face cloning. Specify a PhotoMaker safetensors model which will be applied replacing img2img. SDXL models only. Leave blank if unused.", default="")
    sdparsergrouplora.add_argument("--sdquant",  metavar=('[quantization level 0/1/2]'), help="If specified, loads the model quantized to save memory. 0=off, 1=q8, 2=q4", type=int, choices=[0,1,2], nargs="?", const=2, default=0)
    sdparsergroup.add_argument("--sdt5xxl", metavar=('[filename]'), help="Specify a T5-XXL safetensors model. Leave blank if prebaked or unused.", default="")
    sdparsergroup.add_argument("--sdthreads", metavar=('[threads]'), help="Use a different number of threads for image generation if specified. Otherwise, has the same value as --threads.", type=int, default=0)
    sdparsergroup.add_argument("--sdtiledvae", metavar=('[maxres]'), help="Adjust the automatic VAE tiling trigger for images above this size. 0 disables vae tiling.", type=int, default=default_vae_tile_threshold)
    sdparsergroup.add_argument("--sdupscaler", metavar=('[filename]'), help="You can use ESRGAN as an upscaling model to resize images. Leave blank if unused.", default="")
    sdparsergroupvae = sdparsergroup.add_mutually_exclusive_group()
    sdparsergroupvae.add_argument("--sdvae", metavar=('[filename]'), help="Specify an image generation safetensors VAE which replaces the one in the model.", default="")
    sdparsergroupvae.add_argument("--sdvaeauto", help="Uses a built-in tiny VAE via TAE SD, which is very fast, and fixed bad VAEs.", action='store_true')
    sdparsergroup.add_argument("--sdvaedevice", metavar=('[Device ID]'), help=f"VAE device for image generation. GPU index, -1 or 'main' for the main GPU, or 'CPU' (default: {default_sdvaedevice}).", type=sd_get_device_number, default=None)
    sdparsergroup.add_argument("--sdvramlimit", metavar=('[limit MB]'), help="If set, prevent VRAM usage from image gen from using more than this many MB.", type=int, default=0)

    whisperparsergroup = parser.add_argument_group('Whisper Transcription Commands')
    whisperparsergroup.add_argument("--whispermodel", metavar=('[filename]'), help="Specify a Whisper .bin model to enable Speech-To-Text transcription.", default="")

    ttsparsergroup = parser.add_argument_group('TTS Narration Commands')
    ttsparsergroup.add_argument("--ttsmodel", metavar=('[filename]'), help="Specify the TTS Text-To-Speech GGUF model.", default="")
    ttsparsergroup.add_argument("--ttswavtokenizer", metavar=('[filename]'), help="Specify the WavTokenizer GGUF model.", default="")
    ttsparsergroup.add_argument("--ttsgpu", help="Use the GPU for TTS.", action='store_true')
    ttsparsergroup.add_argument("--ttsmaxlen", help="Limit number of audio tokens generated with TTS.",  type=int, default=default_ttsmaxlen)
    ttsparsergroup.add_argument("--ttsthreads", metavar=('[threads]'), help="Use a different number of threads for TTS if specified. Otherwise, has the same value as --threads.", type=int, default=0)
    ttsparsergroup.add_argument("--ttsdir", metavar=('[directory]'), help="Select directory containing voices for voice cloning.", default="")

    musicparsergroup = parser.add_argument_group('Music Gen Commands')
    musicparsergroup.add_argument("--musicllm", metavar=('[filename]'), help="Select music LLM model (e.g acestep-5Hz-lm-0.6B)", default="")
    musicparsergroup.add_argument("--musicembeddings", metavar=('[filename]'), help="Select music embedding model (e.g Qwen3-Embedding-0.6B)", default="")
    musicparsergroup.add_argument("--musicdiffusion", metavar=('[filename]'), help="Select music diffusion (DiT) model (e.g acestep-v15-turbo)", default="")
    musicparsergroup.add_argument("--musicvae", metavar=('[filename]'), help="Select music VAE model", default="")
    musicparsergroup.add_argument("--musiclowvram", help="Unload music models when not in use", action='store_true')

    embeddingsparsergroup = parser.add_argument_group('Embeddings Model Commands')
    embeddingsparsergroup.add_argument("--embeddingsmodel", metavar=('[filename]'), help="Specify an embeddings model to be loaded for generating embedding vectors.", default="")
    embeddingsparsergroup.add_argument("--embeddingsmaxctx", metavar=('[amount]'), help="Overrides the default maximum supported context of an embeddings model (defaults to trained context).", type=int, default=default_embeddingsmaxctx)
    embeddingsparsergroup.add_argument("--embeddingsgpu", help="Attempts to offload layers of the embeddings model to GPU. Usually not needed.", action='store_true')

    rpcgroup = parser.add_argument_group('RPC Commands')
    rpcgroup.add_argument("--rpcmode", help="RPC allows GPUs to be shared over the network. connect=access a remote GPU, host=share your GPU",metavar=('[disabled/connect/host]'), type=str, choices=["disabled","connect","host"], default="disabled")
    rpcgroupA = rpcgroup.add_mutually_exclusive_group()
    rpcgroupA.add_argument("--rpcport", metavar=('[portnumber]'), help=f"RPC host mode only. Port for RPC server to listen on (default: {default_rpc_port}).", default=default_rpc_port, type=int)
    rpcgroup.add_argument("--rpchost", metavar=('[IP address]'), help="RPC host mode only. IP address for RPC server to listen on. Use 0.0.0.0 for all interfaces, 127.0.0.1 for localhost only.", default="0.0.0.0", type=str)
    rpcgroup.add_argument("--rpcdevice", metavar=('<dev1,dev2,..>'), help="RPC host mode only. Set specific devices to use for RPC server. Comma separated. Overrides normal RPC device choices.", default="")
    rpcgroupA.add_argument("--rpctargets", metavar=('[remotehost1:port1,remotehost2:port2]'), help="RPC connect mode only. Specify a comma separated list of remote RPC endpoints to connect to e.g. 127.0.0.1:5551,127.0.0.1:5552", default="", type=str)

    admingroup = parser.add_argument_group('Administration Commands')
    admingroup.add_argument("--admin", help="Enables admin mode, allowing you to unload and reload different configurations or models.", action='store_true')
    admingroup.add_argument("--adminpassword", metavar=('[password]'), help="Require a password to access admin functions. You are strongly advised to use one for publically accessible instances! Can also be set with env var KCPP_ADMINPASSWORD", default=os.getenv('KCPP_ADMINPASSWORD',None))
    admingroup.add_argument("--admindir", metavar=('[directory]'), help="Specify a directory to look for .kcpps configs in, which can be used to swap models.", default="")
    admingroup.add_argument("--adminunloadtimeout", help="Set an idle timeout in seconds after which KoboldCpp will automatically unload the current model.", type=int, default=0)
    admingroup.add_argument("--routermode", help="Router mode uses a reverse proxy router, allowing you to easily hotswap models and configs within a single request. Requires admin mode.", action='store_true')
    admingroup.add_argument("--reqtimeout", metavar=('[seconds]'), help="Timeout in seconds for HTTP requests.", type=int, default=default_reqtimeout)
    admingroup.add_argument("--autoswapmode", help="Autoswap mode builds on router mode to allow switching of model types within the same config automatically. Requires admin mode and router mode. All models desired must be defined within the same config.", action='store_true')
    admingroup.add_argument("--baseconfig", help="Specify a base .kcpps config to apply, if no custom base config is selected during a model swap", default="")

    deprecatedgroup = parser.add_argument_group('Deprecated Commands, DO NOT USE!')
    deprecatedgroup.add_argument("--hordeconfig", help=argparse.SUPPRESS, nargs='+')
    deprecatedgroup.add_argument("--sdconfig", help=argparse.SUPPRESS, nargs='+')
    compatgroup.add_argument("--noblas", help=argparse.SUPPRESS, action='store_true')
    compatgroup3.add_argument("--nommap","--no-mmap", help=argparse.SUPPRESS, action='store_true')
    deprecatedgroup.add_argument("--pipelineparallel", help=argparse.SUPPRESS, action='store_true') #changed to nopipelineparallel
    deprecatedgroup.add_argument("--sdnotile", help=argparse.SUPPRESS, action='store_true') # legacy option, see sdtiledvae
    deprecatedgroup.add_argument("--sdvaecpu", help=argparse.SUPPRESS, action='store_true') # legacy option, see sdvaedevice
    deprecatedgroup.add_argument("--sdclipgpu", help=argparse.SUPPRESS, action='store_true') # legacy option, see sdclipgpu
    deprecatedgroup.add_argument("--forceversion", help=argparse.SUPPRESS, action='store_true') #no longer used
    deprecatedgroup.add_argument("--sdgendefaults", help=argparse.SUPPRESS, action='store_true') # legacy option, see gendefaults
    deprecatedgroup.add_argument("--flashattention","--flash-attn","-fa", help=argparse.SUPPRESS, action='store_true') #flash attention now default on
    deprecatedgroup.add_argument("--useswa", help=argparse.SUPPRESS, action='store_true')

    debuggroup = parser.add_argument_group('Debug Commands')
    debuggroup.add_argument("--testmemory", help=argparse.SUPPRESS, action='store_true')

    main(launch_args=parser.parse_args(),default_args=parser.parse_args([]))
