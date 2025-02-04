import tkinter as tk
from tkinter import ttk, scrolledtext
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from threading import Thread
import queue

class TokenSelectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Token Selector")
        self.root.geometry("800x600")

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
        self.style.configure('Dark.TLabel',
                           background=self.colors['bg'],
                           foreground=self.colors['fg'])
        self.style.configure('Dark.TButton',
                           background=self.colors['button'],
                           foreground=self.colors['fg'],
                           borderwidth=0,
                           relief='flat',
                           padding=5)
        self.style.map('Dark.TButton',
                      background=[('active', self.colors['button_hover'])])
        self.style.configure('Dark.Treeview',
                           background=self.colors['text_bg'],
                           foreground=self.colors['fg'],
                           fieldbackground=self.colors['text_bg'],
                           borderwidth=0)
        self.style.map('Dark.Treeview',
                      background=[('selected', self.colors['highlight'])],
                      foreground=[('selected', self.colors['fg'])])
        self.style.configure('Dark.TEntry',
                           fieldbackground=self.colors['text_bg'],
                           foreground=self.colors['fg'],
                           borderwidth=1,
                           relief='flat')

        # Initialize model loading status
        self.model_loaded = False
        self.loading_queue = queue.Queue()

        # Initialize model attributes
        self.tokenizer = None
        self.model = None
        self.context = None

        # Create and layout widgets
        self.create_widgets()
        self.layout_widgets()

        # Start loading the model in a separate thread
        self.load_model_thread()

        self.rarity_scaling_factor = 1.0


    def create_widgets(self):
        # Create frames
        self.input_frame = ttk.Frame(self.root, style='Dark.TFrame')
        self.text_frame = ttk.Frame(self.root, style='Dark.TFrame')
        self.token_frame = ttk.Frame(self.root, style='Dark.TFrame')

        # Create widgets for input frame
        self.model_label = ttk.Label(self.input_frame, text="Loading model... Please wait", style='Dark.TLabel')
        self.prompt_label = ttk.Label(self.input_frame, text="Enter initial prompt:", style='Dark.TLabel')
        self.prompt_entry = ttk.Entry(self.input_frame, width=50, style='Dark.TEntry')
        self.start_button = ttk.Button(self.input_frame, text="Start Generation", command=self.start_generation, style='Dark.TButton')
        self.start_button["state"] = "disabled"

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

        self.text_area.insert(tk.END, "Generated text will appear here...")
        self.text_area["state"] = "disabled"

        # Create widgets for token frame
        self.token_list = ttk.Treeview(self.token_frame, columns=("Token", "Probability"),
                                     show="headings", height=15, style='Dark.Treeview')
        # Configure column headings
        self.token_list.heading("Token", text="Token")
        self.token_list.heading("Probability", text="Probability")
        self.token_list.column("Token", width=200)
        self.token_list.column("Probability", width=100)

        # Add scrollbar for token list
        self.token_scrollbar = ttk.Scrollbar(self.token_frame, orient="vertical",
                                           command=self.token_list.yview,
                                           style='Dark.Vertical.TScrollbar')
        self.token_list.configure(yscrollcommand=self.token_scrollbar.set)

        # Configure scrollbar style
        self.style.configure('Dark.Vertical.TScrollbar',
                           background=self.colors['button'],
                           arrowcolor=self.colors['fg'],
                           borderwidth=0,
                           relief='flat')
        self.style.map('Dark.Vertical.TScrollbar',
                      background=[('active', self.colors['button_hover'])])

        # Bind click event to token selection
        self.token_list.bind('<<TreeviewSelect>>', self.on_token_select)

    def layout_widgets(self):
        # Layout input frame with modern spacing
        self.input_frame.pack(pady=20, padx=20, fill=tk.X)
        self.model_label.pack(pady=(0, 10))
        self.prompt_label.pack(pady=(0, 5))
        self.prompt_entry.pack(pady=(0, 10), fill=tk.X)
        self.start_button.pack(pady=(0, 10))

        # Layout text frame
        self.text_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.text_area.pack(fill=tk.BOTH, expand=True)

        # Layout token frame
        self.token_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.token_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.token_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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
                self.model_loaded = True
                self.tokenizer, self.model = result
                self.token_frequencies = self._calculate_token_frequencies()
        except queue.Empty:
            self.root.after(100, self.check_model_loading)

    def _calculate_token_frequencies(self):
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

            top_k = 50
            top_prob, top_indices = torch.topk(probabilities, top_k)

            # Calculate rarity scores for top tokens
            rarity_scores = np.array([self._calculate_rarity_score(idx.item()) for idx in top_indices])

            # Scale probabilities by rarity
            scaled_probabilities = top_prob.numpy() * rarity_scores

            # Normalize scaled probabilities
            scaled_probabilities = scaled_probabilities / np.sum(scaled_probabilities)

            # Sort by scaled probabilities and take top 20
            sorted_indices = np.argsort(scaled_probabilities)[-20:][::-1]

            final_probs = scaled_probabilities[sorted_indices]
            final_token_ids = np.array(top_indices)[sorted_indices]
            decoded_tokens = [self.tokenizer.decode(idx.item()) for idx in final_token_ids]

            return final_probs, final_token_ids, decoded_tokens

    def start_generation(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generating tokens")

        try:
            # Clear previous content
            self.text_area["state"] = "normal"
            self.text_area.delete(1.0, tk.END)

            # Get initial prompt
            initial_prompt = self.prompt_entry.get()
            self.text_area.insert(tk.END, initial_prompt)
            self.text_area["state"] = "disabled"

            # Initialize context
            self.context = self.tokenizer.encode(initial_prompt, return_tensors='pt')

            # Get and display first token options
            self.update_token_options()

        except Exception as e:
            self.model_label.config(text=f"Error: {str(e)}")
            self.start_button["state"] = "normal"
            raise e

    def update_token_options(self):
        # Clear previous tokens
        for item in self.token_list.get_children():
            self.token_list.delete(item)

        # Get new token probabilities
        probabilities, self.token_ids, decoded_tokens = self.get_next_token_probabilities()

        # Update token list
        for token, prob in zip(decoded_tokens, probabilities):
            self.token_list.insert("", tk.END, values=(token, f"{prob:.4f}"))

    def on_token_select(self, event):
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model and tokenizer must be loaded before selecting tokens")

            # Get selected token
            selection = self.token_list.selection()
            if selection is None or len(selection) == 0:
                return

            selected_item = selection[0]  # Get the selected item ID
            if not selected_item:
                return

            # Get token index
            index = self.token_list.index(selected_item)

            # Update context with chosen token
            chosen_token_id = self.token_ids[index]
            self.context = torch.cat([torch.tensor(self.context), torch.tensor([[chosen_token_id]])], dim=1)

            # Update generated text
            chosen_token = self.tokenizer.decode(chosen_token_id)
            self.text_area["state"] = "normal"
            self.text_area.insert(tk.END, chosen_token)
            self.text_area["state"] = "disabled"

            # Get new token options
            self.update_token_options()

            # Ensure the text area scrolls to show the latest content
            self.text_area.see(tk.END)
        except IndexError:
            # Silently handle the selection error - this can occur during normal operation
            # when the selection is cleared during the update process
            pass

def main():
    root = tk.Tk()
    app = TokenSelectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# To run this script, you'll need to install the following packages:
# pip install transformers torch tkinter