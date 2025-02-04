import queue
import os
import logging
from threading import Thread

import torch
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.peft_model import PeftModel
from peft.config import PeftConfig
from safetensors.torch import load_file, save_file

from src.utils import get_scaled_adapter


TOP_TOKENS = 50
RARITY_SCALING_FACTOR = 2.0


class TokenSelectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Token Selector")
        self.root.geometry("800x600")

        self.active_loras = []
        self.base_model = None

        # Define color scheme
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'select_bg': '#2d2d2d',
            'select_fg': '#ffffff',
            'button': '#3d3d3d',
            'button_hover': '#4d4d4d',
            'highlight': '#007acc',
            'text_bg': '#252526',
            'border': '#3d3d3d'
        }

        # Configure root window
        self.root.configure(bg=self.colors['bg'])

        # Create and configure style
        self.style = ttk.Style()
        self.style.configure('Dark.TFrame', background=self.colors['bg'])
        self.style.configure('Dark.TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('Dark.TButton', background=self.colors['button'], foreground=self.colors['fg'], borderwidth=0, relief='flat', padding=5)
        self.style.map('Dark.TButton', background=[('active', self.colors['button_hover'])])
        self.style.configure('Dark.Treeview', background=self.colors['text_bg'], foreground=self.colors['fg'], fieldbackground=self.colors['text_bg'], borderwidth=0)
        self.style.map('Dark.Treeview', background=[('selected', self.colors['highlight'])], foreground=[('selected', self.colors['fg'])])

        # Add LoRA management frame
        self.create_lora_widgets()

        # Initialize model loading status
        self.model_loaded = False
        self.loading_queue = queue.Queue()

        # Initialize model attributes
        self.tokenizer = None
        self.model = None
        self.context = None
        self.generation_active = False

        # Create and layout widgets
        self.create_widgets()
        self.layout_widgets()

        # Start loading the model in a separate thread
        self.load_model_thread()

        self.rarity_scaling_factor = RARITY_SCALING_FACTOR


    def create_widgets(self):
        # Create frames
        self.input_frame = ttk.Frame(self.root, style='Dark.TFrame')
        self.params_frame = ttk.Frame(self.root, style='Dark.TFrame')
        self.text_frame = ttk.Frame(self.root, style='Dark.TFrame')
        self.control_frame = ttk.Frame(self.root, style='Dark.TFrame')
        self.token_frame = ttk.Frame(self.root, style='Dark.TFrame')

        # Create widgets for input frame
        self.model_label = ttk.Label(self.input_frame, text="Loading model... Please wait", style='Dark.TLabel')
        self.start_button = ttk.Button(self.input_frame, text="Start Generation", command=self.toggle_generation, style='Dark.TButton')
        self.start_button["state"] = "disabled"

        # Create widgets for parameters frame
        # self.seq_length_label = ttk.Label(self.params_frame, text="Tokens per sequence:", style='Dark.TLabel')
        # self.seq_length_entry = ttk.Entry(self.params_frame, textvariable=self.sequence_length, width=5, style='Dark.TEntry')
        # self.num_seq_label = ttk.Label(self.params_frame, text="Number of sequences:", style='Dark.TLabel')
        # self.num_seq_entry = ttk.Entry(self.params_frame, textvariable=self.num_sequences, width=5, style='Dark.TEntry')
        # self.generate_sequences_button = ttk.Button(self.params_frame, text="Generate Sequences",
        #                                             command=self.generate_sequences, style='Dark.TButton')
        # self.generate_sequences_button["state"] = "disabled"

        # Create widgets for text frame
        self.text_area = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD, width=70, height=10)
        self.text_area.configure(
            bg=self.colors['text_bg'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg'],  # cursor color
            selectbackground=self.colors['highlight'],
            selectforeground=self.colors['fg'],
            font=('Consolas', 10),
            padx=10,
            pady=10,
            borderwidth=1,
            relief='flat')

        # Create widgets for token frame
        self.token_notebook = ttk.Notebook(self.token_frame)
        self.single_token_frame = ttk.Frame(self.token_notebook)
        self.token_list = ttk.Treeview(self.single_token_frame,
                                     columns=("Token", "Probability"),
                                     show="headings", height=15)
        self.token_list.heading("Token", text="Token")
        self.token_list.heading("Probability", text="Probability")

        # Add scrollbar for token list
        self.token_scrollbar = ttk.Scrollbar(self.token_frame, orient="vertical",
                                           command=self.token_list.yview)
        self.token_list.configure(yscrollcommand=self.token_scrollbar.set)

        # Create select button
        self.select_button = ttk.Button(self.single_token_frame, text="Select Token",
                                      command=self.select_token)
        self.select_button["state"] = "disabled"

        # # Sequence selection tab
        # self.sequence_frame = ttk.Frame(self.token_notebook)
        # self.sequence_list = ttk.Treeview(self.sequence_frame,
        #                                 columns=("Sequence", "Score"),
        #                                 show="headings", height=15)
        # self.sequence_list.heading("Sequence", text="Token Sequence")
        # self.sequence_list.heading("Score", text="Probability Score")

        # # Add scrollbar for sequence list
        # self.sequence_scrollbar = ttk.Scrollbar(self.sequence_frame, orient="vertical",
        #                                       command=self.sequence_list.yview)
        # self.sequence_list.configure(yscrollcommand=self.sequence_scrollbar.set)

        # # Create select sequence button
        # self.select_sequence_button = ttk.Button(self.sequence_frame, text="Select Sequence",
        #                                        command=self.select_sequence)
        # self.select_sequence_button["state"] = "disabled"

        # # Add tabs to notebook
        self.token_notebook.add(self.single_token_frame, text="Single Token")
        # self.token_notebook.add(self.sequence_frame, text="Token Sequences")

    def layout_widgets(self):
        # Layout input frame
        self.input_frame.pack(pady=10, padx=10, fill=tk.X)
        self.model_label.pack(pady=5)
        self.control_frame.pack(pady=5, padx=10, fill=tk.X)
        self.start_button.pack(pady=5)

        # Layout parameters frame
        # self.params_frame.pack(pady=5, padx=10, fill=tk.X)
        # self.seq_length_label.pack(side=tk.LEFT, padx=5)
        # self.seq_length_entry.pack(side=tk.LEFT, padx=5)
        # self.num_seq_label.pack(side=tk.LEFT, padx=5)
        # self.num_seq_entry.pack(side=tk.LEFT, padx=5)
        # self.generate_sequences_button.pack(side=tk.LEFT, padx=5)

        # Layout text frame
        self.text_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.text_area.pack(fill=tk.BOTH, expand=True)

        # Layout token frame
        self.token_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.token_notebook.pack(fill=tk.BOTH, expand=True)

        # Layout single token tab
        self.token_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.token_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.select_button.pack(pady=5)

        # # Layout sequence tab
        # self.sequence_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # self.sequence_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # self.select_sequence_button.pack(pady=5)

    def create_lora_widgets(self):
        # Create LoRA management frame
        self.lora_frame = ttk.Frame(self.root, style='Dark.TFrame')

        # Create LoRA list
        self.lora_list = ttk.Treeview(self.lora_frame,
                                    columns=("Name", "Status"),
                                    show="headings", height=3)
        self.lora_list.heading("Name", text="LoRA Adapter")
        self.lora_list.heading("Status", text="Status")

        # Create LoRA controls
        self.lora_controls = ttk.Frame(self.lora_frame, style='Dark.TFrame')
        self.load_lora_button = ttk.Button(self.lora_controls,
                                         text="Load LoRA",
                                         command=self.load_lora_adapter,
                                         style='Dark.TButton')
        self.remove_lora_button = ttk.Button(self.lora_controls,
                                           text="Remove LoRA",
                                           command=self.remove_lora_adapter,
                                           style='Dark.TButton')

        # Layout LoRA widgets
        self.lora_frame.pack(pady=5, padx=10, fill=tk.X)
        self.lora_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lora_controls.pack(side=tk.RIGHT, padx=5)
        self.load_lora_button.pack(side=tk.LEFT, padx=2)
        self.remove_lora_button.pack(side=tk.LEFT, padx=2)

        # Initially disable LoRA controls until model is loaded
        self.load_lora_button["state"] = "disabled"
        self.remove_lora_button["state"] = "disabled"

        # Add weight control slider and input
        self.weight_frame = ttk.Frame(self.lora_frame, style='Dark.TFrame')
        self.weight_label = ttk.Label(self.weight_frame,
                                    text="Adapter Weight:",
                                    style='Dark.TLabel')

        # Create a StringVar for the weight value
        self.weight_var = tk.StringVar(value="1.0")
        self.weight_entry = ttk.Entry(self.weight_frame,
                                    textvariable=self.weight_var,
                                    width=6)

        # self.apply_weight_button = ttk.Button(self.weight_frame,
        #                                     text="Apply Weight",
        #                                     command=self.apply_adapter_weight,
        #                                     style='Dark.TButton')

        # Layout weight controls
        self.weight_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.weight_label.pack(side=tk.LEFT, padx=5)
        self.weight_entry.pack(side=tk.LEFT, padx=5)
        # self.apply_weight_button.pack(side=tk.LEFT, padx=5)

        # Initially disable weight controls
        # self.weight_entry["state"] = "disabled"
        # self.apply_weight_button["state"] = "disabled"

    def load_model_thread(self):
        def load():
            model_name = "meta-llama/Llama-3.1-8B"
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                model.eval()
                self.loading_queue.put((tokenizer, model))
            except Exception as e:
                self.loading_queue.put(e)

        Thread(target=load, daemon=True).start()
        self.root.after(100, self.check_model_loading)

    def check_model_loading(self):
        try:
            result = self.loading_queue.get_nowait()
            if isinstance(result, Exception):
                self.model_label.config(text=f"Error loading model: {str(result)}")
                self.model_loaded = False
            else:
                self.model_label.config(text="Model loaded successfully!")
                self.start_button["state"] = "normal"
                self.load_lora_button["state"] = "normal"
                self.model_loaded = True
                self.tokenizer, self.model = result
                self.base_model = self.model
                self.token_frequencies = self._calculate_token_frequencies()
        except queue.Empty:
            self.root.after(100, self.check_model_loading)

    def load_lora_adapter(self):
        """Load a LoRA adapter from a local directory."""
        adapter_path = filedialog.askdirectory(
            title="Select LoRA Adapter Directory",
            initialdir=os.getcwd()
        )

        if adapter_path is None or self.base_model is None:
            return

        try:
            # Load the adapter configuration
            config = PeftConfig.from_pretrained(adapter_path)

            weight = float(self.weight_var.get())
            if weight < 0:
                raise ValueError("Weight must be non-negative")

            if weight != 1.0:
                adapter_path = get_scaled_adapter(adapter_path, weight)

            # Create a new model instance with the adapter
            adapted_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Update the model
            self.model = adapted_model

            # Add to active adapters list
            adapter_name = os.path.basename(adapter_path)
            self.active_loras.append({
                'name': adapter_name,
                'path': adapter_path,
                'weight': weight
            })

            # Update the LoRA list
            self.lora_list.insert("", tk.END,
                                values=(adapter_name, f"Active (weight: {weight})"),
                                tags=(adapter_path,))

            # Enable controls
            self.remove_lora_button["state"] = "normal"
            self.weight_entry["state"] = "normal"
            # self.apply_weight_button["state"] = "normal"

            self.model_label.config(text=f"LoRA adapter '{adapter_name}' loaded successfully!")

        except Exception as e:
            raise e #(text=f"Error loading LoRA adapter: {str(e)}")

    def remove_lora_adapter(self):
        """Remove the selected LoRA adapter."""
        selection = self.lora_list.selection()
        if not selection:
            return

        # Get adapter info
        adapter_item = self.lora_list.item(selection[0])
        adapter_name = adapter_item['values'][0]

        try:
            # Restore base model
            self.model = self.base_model

            # Remove from active adapters
            self.active_loras = [
                adapter for adapter in self.active_loras
                if adapter['name'] != adapter_name
            ]

            # Remove from list
            self.lora_list.delete(selection[0])

            # Disable controls if no adapters left
            if not self.active_loras:
                self.remove_lora_button["state"] = "disabled"
                self.weight_entry["state"] = "disabled"
                # self.apply_weight_button["state"] = "disabled"

            self.model_label.config(text=f"LoRA adapter '{adapter_name}' removed successfully!")

        except Exception as e:
            self.model_label.config(text=f"Error removing LoRA adapter: {str(e)}")

    def _calculate_token_frequencies(self) -> dict[int, float]:
        """
        Calculate token frequencies using the model's token embeddings.
        The intuition is that common tokens tend to have more central/typical embeddings,
        while rare tokens often have more extreme or unusual embedding patterns.

        Returns:
            dict: Mapping of token IDs to their estimated frequencies
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generating tokens")

        # Get token embeddings from the model
        embeddings = self.model.get_input_embeddings().weight.detach()

        # Calculate the centroid (mean point) of all embeddings
        centroid = embeddings.mean(dim=0)

        # Calculate Euclidean distances from each token's embedding to the centroid
        distances = torch.norm(embeddings - centroid, dim=1)

        # Convert distances to frequencies (closer to centroid = more frequent)
        # We normalize and invert the distances so that:
        # - Tokens close to the centroid get high frequencies
        # - Tokens far from the centroid get low frequencies
        max_distance = distances.max()
        frequencies = 1 - (distances / max_distance)

        # Apply softmax to get a proper probability distribution
        frequencies = torch.softmax(frequencies * 5, dim=0)  # Scale factor of 5 for more pronounced differences

        # Convert to dictionary
        return {i: freq.item() for i, freq in enumerate(frequencies)}

    def _calculate_rarity_score(self, token_id):
        """
        Calculate rarity score for a token based on its frequency.

        Args:
            token_id (int): Token ID

        Returns:
            float: Rarity score (higher for rarer tokens)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generating tokens")

        frequency = self.token_frequencies.get(token_id, 1/self.tokenizer.vocab_size)
        return 1 / (frequency ** self.rarity_scaling_factor)

    def get_next_token_probabilities(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generating tokens")

        with torch.no_grad():
            outputs = self.model(self.context)
            logits = outputs.logits[0, -1, :]
            probabilities = torch.softmax(logits, dim=0)

            top_k = 200  ### Param
            top_prob, top_indices = torch.topk(probabilities, top_k)

            # Calculate rarity scores for top tokens
            rarity_scores = np.array([self._calculate_rarity_score(idx.item()) for idx in top_indices])

            # Scale probabilities by rarity
            scaled_probabilities = top_prob.numpy() * rarity_scores

            # Normalize scaled probabilities
            scaled_probabilities = scaled_probabilities / np.sum(scaled_probabilities)

            # Sort by scaled probabilities and take top {TOP_TOKENS}
            sorted_indices = np.argsort(scaled_probabilities)[-TOP_TOKENS:][::-1]

            final_probs = scaled_probabilities[sorted_indices]
            final_token_ids = np.array(top_indices)[sorted_indices]
            decoded_tokens = [self.tokenizer.decode(idx.item()) for idx in final_token_ids]

            return final_probs, final_token_ids, decoded_tokens

    # def generate_sequences(self):
        # try:
        #     # Clear previous sequences
        #     for item in self.sequence_list.get_children():
        #         self.sequence_list.delete(item)

        #     num_sequences = self.num_sequences.get()
        #     seq_length = self.sequence_length.get()

        #     sequences = []
        #     scores = []

        #     # Generate multiple sequences
        #     for _ in range(num_sequences):
        #         if self.context is None:
        #             raise ValueError("Context cannot be None")

        #         current_context = self.context.clone()
        #         sequence_tokens = []
        #         sequence_score = 0.0

        #         # Generate each token in the sequence
        #         for _ in range(seq_length):
        #             with torch.no_grad():
        #                 if self.model is None:
        #                     raise ValueError("Model cannot be None")

        #                 outputs = self.model(current_context)
        #                 logits = outputs.logits[0, -1, :]
        #                 probabilities = torch.softmax(logits, dim=0)

        #                 # Sample from the distribution
        #                 token_id = torch.multinomial(probabilities, 1).item()
        #                 if self.tokenizer is None:
        #                     raise ValueError("Tokenizer cannot be None")

        #                 token = self.tokenizer.decode(token_id)
        #                 sequence_tokens.append(token)

        #                 # Update score (average log probability)
        #                 sequence_score += torch.log(probabilities[int(token_id)]).item()

        #                 # Update context
        #                 current_context = torch.cat([current_context, torch.tensor([[token_id]])], dim=1)

        #         # Calculate average score
        #         sequence_score /= seq_length
        #         sequence_text = ''.join(sequence_tokens)
        #         sequences.append(sequence_text)
        #         scores.append(sequence_score)

        #     # Sort sequences by score
        #     sorted_sequences = sorted(zip(sequences, scores), key=lambda x: x[1], reverse=True)

        #     # Update sequence list
        #     for i, (seq, score) in enumerate(sorted_sequences):
        #         self.sequence_list.insert("", tk.END,
        #                                 values=(seq, f"{score:.4f}"))

        #     self.select_sequence_button["state"] = "normal"

        # except Exception as e:
        #     self.model_label.config(text=f"Error generating sequences: {str(e)}")

    def toggle_generation(self):
        if not self.generation_active:
            self.start_generation()
        else:
            self.stop_generation()

    def start_generation(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generating tokens")

        try:
            # Get current text
            current_text = self.text_area.get(1.0, tk.END).strip()

            # Initialize context with current text
            self.context = self.tokenizer.encode(current_text, return_tensors='pt')

            # Update UI state
            self.generation_active = True
            self.start_button.config(text="Pause Generation")
            self.select_button["state"] = "normal"
            self.text_area["state"] = "disabled"

            # Get and display token options
            self.update_token_options()

        except Exception as e:
            self.model_label.config(text=f"Error: {str(e)}")
            self.start_button["state"] = "normal"
            raise e

    def stop_generation(self):
        # Update UI state
        self.generation_active = False
        self.start_button.config(text="Resume Generation")
        self.select_button["state"] = "disabled"
        self.text_area["state"] = "normal"

        # Clear token list
        for item in self.token_list.get_children():
            self.token_list.delete(item)

    def update_token_options(self):
        # Clear previous tokens
        for item in self.token_list.get_children():
            self.token_list.delete(item)

        # Get new token probabilities
        probabilities, self.token_ids, decoded_tokens = self.get_next_token_probabilities()

        # Update token list
        for token, prob in zip(decoded_tokens, probabilities):
            self.token_list.insert("", tk.END, values=(token, f"{prob*10:.4f}%"))

    def select_token(self):
        # Get selected token
        selection = self.token_list.selection()
        if not selection:
            return

        # Get token index
        index = self.token_list.index(selection[0])
        self._append_token(self.token_ids[index])

    # def select_sequence(self):
    #     # Get selected sequence
    #     selection = self.sequence_list.selection()
    #     if not selection:
    #         return

    #     # Get sequence text
    #     sequence = self.sequence_list.item(selection[0])['values'][0]

    #     # Update text area
    #     self.text_area["state"] = "normal"
    #     self.text_area.insert(tk.END, sequence)
    #     self.text_area["state"] = "disabled"

    #     # Update context
    #     if self.tokenizer is None:
    #         raise ValueError("Tokenizer cannot be None")

    #     token_ids = self.tokenizer.encode(sequence)
    #     if not isinstance(self.context, torch.Tensor):
    #         raise ValueError("Context must be a torch.Tensor")

    #     self.context = torch.cat([self.context, torch.tensor([token_ids]).to(self.context.device)], dim=1)

    #     # Get new token options
    #     self.update_token_options()

    def _append_token(self, token_id):
        # Update context with chosen token
        if not isinstance(self.context, torch.Tensor):
            raise ValueError("Context is not a tensor")

        self.context = torch.cat([self.context, torch.tensor([[token_id]])], dim=1)

        # Update generated text
        if self.tokenizer is None:
            raise ValueError("Tokenizer cannot be None")

        chosen_token = self.tokenizer.decode(token_id)
        self.text_area["state"] = "normal"
        self.text_area.insert(tk.END, chosen_token)
        self.text_area["state"] = "disabled"

        # Get new token options
        self.update_token_options()


def main():
    root = tk.Tk()
    app = TokenSelectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
