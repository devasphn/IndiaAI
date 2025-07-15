import torch
import transformers
import faster_whisper
import gradio as gr
import numpy as np
import soundfile as sf
import time
import os
import re
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# --- Configuration ---
STT_MODEL = "distil-large-v3"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TTS_MODEL = "parler-tts/parler-tts-mini-v1"  # Updated to compatible model
OUTPUT_WAV_FILE = "output.wav"

class RealTimeS2SAgent:
    def __init__(self):
        print("--- Initializing S2S Agent with Parler-TTS ---")
        if not torch.cuda.is_available():
            raise RuntimeError("This application requires a GPU to run.")
            
        self.device = "cuda"
        print(f"Using device: {self.device.upper()}")

        print(f"Loading STT model: {STT_MODEL}...")
        self.stt_model = faster_whisper.WhisperModel(
            STT_MODEL, 
            device=self.device, 
            compute_type="float16"
        )
        print("STT model loaded.")

        print(f"Loading LLM: {LLM_MODEL}...")
        self.llm_pipeline = transformers.pipeline(
            "text-generation",
            model=LLM_MODEL,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self.device,
        )
        print("LLM loaded.")

        print(f"Loading Parler-TTS model: {TTS_MODEL}...")
        self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            TTS_MODEL, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)
        print("Parler-TTS model loaded.")
        
        if os.path.exists(OUTPUT_WAV_FILE):
            os.remove(OUTPUT_WAV_FILE)
            
        print("\n--- Agent is Ready ---")

    def transcribe_audio(self, audio_filepath: str) -> str:
        if not audio_filepath: return ""
        print("Transcribing audio...")
        segments, _ = self.stt_model.transcribe(audio_filepath, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        print(f"User: {transcription}")
        return transcription

    def generate_response(self, chat_history: list) -> str:
        messages = [
            {"role": "system", "content": "You are Deva, a friendly AI with emotional intelligence. Respond naturally. If the user asks to laugh or express emotion, include tags like [laughs] or [sighs] in your response."}
        ]
        for msg in chat_history:
            if msg['role'] in ['user', 'assistant']:
                messages.append(msg)
        
        terminators = [
            self.llm_pipeline.tokenizer.eos_token_id,
            self.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.llm_pipeline(
            messages, max_new_tokens=256, eos_token_id=terminators, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
        )
        assistant_response = outputs[0]["generated_text"][-1]['content']
        print(f"Agent: {assistant_response}")
        return assistant_response

    def convert_text_to_speech(self, text: str) -> str:
        print("Speaking with Parler-TTS...")
        
        # Process for natural Indian voice with emotion
        description = "A female speaker with a natural Indian accent, speaking clearly and expressively."
        inputs = self.tts_tokenizer.apply_chat_template([{"role": "user", "content": description + " " + text}], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            audio = self.tts_model.generate(inputs, max_length=2048, temperature=0.8, top_p=0.95, top_k=50)
        
        sf.write(OUTPUT_WAV_FILE, audio.cpu().numpy().squeeze(), self.tts_model.config.sampling_rate)
        return OUTPUT_WAV_FILE

    def process_conversation_turn(self, audio_filepath: str, chat_history: list):
        if audio_filepath is None: return chat_history, None
        user_text = self.transcribe_audio(audio_filepath)
        if not user_text.strip(): return chat_history, None
            
        chat_history.append({"role": "user", "content": user_text})
        llm_response = self.generate_response(chat_history)
        chat_history.append({"role": "assistant", "content": llm_response})
        
        agent_audio_path = self.convert_text_to_speech(llm_response)
        return chat_history, agent_audio_path

def build_ui(agent: RealTimeS2SAgent):
    with gr.Blocks(theme=gr.themes.Soft(), title="S2S Agent with Parler-TTS") as demo:
        gr.Markdown("# Real-Time Speech-to-Speech AI Agent (Parler-TTS)")
        gr.Markdown("Tap the microphone, speak, and the agent will respond with natural voice.")

        chatbot = gr.Chatbot(label="Conversation", height=500, type="messages")
        
        with gr.Row():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Tap to Talk")
            audio_output = gr.Audio(label="Agent Response", autoplay=True, visible=True)

        def handle_interaction(audio_filepath, history):
            history = history or []
            return agent.process_conversation_turn(audio_filepath, history)

        mic_input.stop_recording(
            fn=handle_interaction,
            inputs=[mic_input, chatbot],
            outputs=[chatbot, audio_output]
        )
        
        clear_button = gr.Button("Clear Conversation")
        clear_button.click(lambda: ([], None), None, [chatbot, audio_output])

    return demo

if __name__ == "__main__":
    os.environ['GRADIO_SERVER_NAME'] = '127.0.0.1'
    
    agent = RealTimeS2SAgent()
    ui = build_ui(agent)
    
    ui.launch(server_name="0.0.0.0", server_port=7860, share=True)
