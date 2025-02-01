import gradio as gr
from threading import Thread
from transformers import TextIteratorStreamer


if gr.NO_RELOAD:
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        BitsAndBytesConfig
    )

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    cache_dir = "deepseek"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        cache_dir=cache_dir
    )

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )

def respond(message, chat_history):
    messages = [
        {"role": "user", "content": message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt", 
        return_dict=True        
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 512,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 0.9,
        "streamer": streamer
    }
       
    thread = Thread(
        target=model.generate, 
        kwargs=generation_kwargs
    )

    thread.start()
    
    response = ""
    for new_token in streamer:
        response += new_token
        current_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response.strip()}
        ]
        yield current_history

with gr.Blocks(title="Simple ChatBot") as demo:
    gr.Markdown("## Simple ChatBot")

    chatbot = gr.Chatbot(label="History", type="messages")
    msg = gr.Textbox(label="Asking a question")
    clear_btn = gr.Button("Clear")

    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=chatbot,
        queue=True
    ).then(
        lambda text: "",
        inputs=msg,
        outputs=msg,
        queue=False
    )

    clear_btn.click(lambda: [], outputs=chatbot, queue=False)

if __name__ == "__main__":
    demo.queue()
    demo.launch()