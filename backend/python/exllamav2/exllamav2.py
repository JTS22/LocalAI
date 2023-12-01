#!/usr/bin/env python3
import grpc
from concurrent import futures
import time
import backend_pb2
import backend_pb2_grpc
import argparse
import signal
import sys
import os, glob

from pathlib import Path
import torch
import torch.nn.functional as F
from torch import version as torch_version
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Lora

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))

# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))
    def LoadModel(self, request, context):
        try:
            model_directory = request.ModelFile

            # Create config, model, tokenizer and generator
            config = ExLlamaV2Config()               # create config from config.json
            config.model_dir = model_directory       # supply path to model
            config.debug_mode = True

            if request.LowVRAM:
                config.max_input_len = 512
                config.max_attention_size = 512 ** 2

            config.prepare()

            model = ExLlamaV2(config)                               # create ExLlama instance and load the weights
            model.load()

            tokenizer = ExLlamaV2Tokenizer(config)            # create tokenizer from tokenizer model file
            cache = ExLlamaV2Cache(model)             # create cache for inference

            # LoRA config
            self.lora = None

            if request.LoraAdapter:
                self.lora = ExLlamaV2Lora.from_directory(model, request.LoraAdapter)

            generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)   # create generator
            generator.warmup()

            self.generator= generator
            self.model = model
            self.tokenizer = tokenizer
            self.cache = cache
        except Exception as err:
            print(err, file=sys.stderr)
            return backend_pb2.Result(success=False, message=f"Unexpected {err=}, {type(err)=}")
        return backend_pb2.Result(message="Model loaded successfully", success=True)

    def Predict(self, request, context):
        penalty = 1.15
        if request.Penalty != 0.0:
            penalty = request.Penalty

        self.samplerConfig = ExLlamaV2Sampler.Settings()

        self.samplerConfig.temperature = request.Temperature
        self.samplerConfig.top_k = request.TopK
        self.samplerConfig.top_p = request.TopP
        self.samplerConfig.token_repetition_penalty = penalty

        if request.IgnoreEOS:
            self.samplerConfig.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        max_new_tokens = 512
        if request.Tokens != 0:
            max_new_tokens = request.Tokens

        input_ids = self.tokenizer.encode(request.Prompt)
        prompt_tokens = input_ids.shape[-1]

        if request.StopPrompts:
            self.generator.set_stop_conditions(list(request.StopPrompts))

        # Send prompt to generator to begin stream
        time_begin_prompt = time.time()
        self.generator.begin_stream(input_ids, self.samplerConfig, loras = self.lora)
        generated_tokens = 0
        t = ''

        time_begin_stream = time.time()
        while True:
            chunk, eos, _ = self.generator.stream()
            generated_tokens += 1
            t = t + chunk
            if eos or generated_tokens == max_new_tokens: break
        time_end = time.time()

        time_prompt = time_begin_stream - time_begin_prompt
        time_tokens = time_end - time_begin_stream

        print(f"Prompt processed in {time_prompt:.2f} seconds, {prompt_tokens} tokens, {prompt_tokens / time_prompt:.2f} tokens/second", file=sys.stderr)
        print(f"Response generated in {time_tokens:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_tokens:.2f} tokens/second", file=sys.stderr)

        return backend_pb2.Result(message=bytes(t, encoding='utf-8'))

    def PredictStream(self, request, context):
        # Implement PredictStream RPC
        #for reply in some_data_generator():
        #    yield reply
        # Not implemented yet
        return self.Predict(request, context)


def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)

    # Define the signal handler function
    def signal_handler(sig, frame):
        print("Received termination signal. Shutting down...")
        server.stop(0)
        sys.exit(0)

    # Set the signal handlers for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument(
        "--addr", default="localhost:50051", help="The address to bind the server to."
    )
    args = parser.parse_args()

    serve(args.addr)
