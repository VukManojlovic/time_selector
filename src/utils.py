import os

from peft.config import PeftConfig
from safetensors.torch import load_file, save_file


def get_scaled_adapter(adapter_path, weight):
        config = PeftConfig.from_pretrained(adapter_path)

        state_dict = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))

        scaled_state_dict = {
            k: v * weight for k, v in state_dict.items()
            if "lora_" in k  # Only scale LoRA weights
        }

        # Update the non-LoRA weights
        scaled_state_dict.update({
            k: v for k, v in state_dict.items()
            if "lora_" not in k
        })

        # Create a temporary directory for the scaled adapter
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        # Copy the original adapter files
        shutil.copytree(adapter_path,
                        os.path.join(temp_dir, "scaled_adapter"),
                        dirs_exist_ok=True)

        # Save the scaled weights
        save_file(
            scaled_state_dict,
            os.path.join(temp_dir, "scaled_adapter", "adapter_model.sefetensors")  #oldmessage# If this desn't work, try .safetensors instead of .bin
        )

        return os.path.join(temp_dir, "scaled_adapter")


# old func:
    # def apply_adapter_weight(self):
    #     """Apply the specified weight to the selected LoRA adapter."""
    #     selection = self.lora_list.selection()
    #     if not selection or self.base_model is None:
    #         return

    #     try:
    #         # Get the weight value
    #         weight = float(self.weight_var.get())
    #         if weight < 0:
    #             raise ValueError("Weight must be non-negative")

    #         adapter_item = self.lora_list.item(selection[0])
    #         adapter_name = adapter_item['values'][0]
    #         adapter_path = adapter_item['tags'][0]

    #         # We need to reload the adapter with the new weight
    #         # First, get the adapter config
    #         config = PeftConfig.from_pretrained(adapter_path)

    #         # Load state dict and adjust weights
    #         state_dict = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))

    #         # Scale the weights
    #         scaled_state_dict = {
    #             k: v * weight for k, v in state_dict.items()
    #             if "lora_" in k  # Only scale LoRA weights
    #         }

    #         # Update the non-LoRA weights
    #         scaled_state_dict.update({
    #             k: v for k, v in state_dict.items()
    #             if "lora_" not in k
    #         })

    #         # Create a temporary directory for the scaled adapter
    #         import tempfile
    #         import shutil

    #         with tempfile.TemporaryDirectory() as temp_dir:
    #             # Copy the original adapter files
    #             shutil.copytree(adapter_path,
    #                           os.path.join(temp_dir, "scaled_adapter"),
    #                           dirs_exist_ok=True)

    #             # Save the scaled weights
    #             save_file(
    #                 scaled_state_dict,
    #                 os.path.join(temp_dir, "scaled_adapter", "adapter_model.sefetensors")  #oldmessage# If this desn't work, try .safetensors instead of .bin
    #             )

    #             # Load the scaled adapter
    #             self.model = PeftModel.from_pretrained(
    #                 self.base_model,
    #                 os.path.join(temp_dir, "scaled_adapter"),
    #                 torch_dtype=torch.float16,
    #                 device_map="auto"
    #             )

    #         # Update the adapter info in our tracking list
    #         for adapter in self.active_loras:
    #             if adapter['name'] == adapter_name:
    #                 adapter['weight'] = weight
    #                 break

    #         # Update the status in the list
    #         self.lora_list.set(selection[0],
    #                          column="Status",
    #                          value=f"Active (weight: {weight:.2f})")

    #         self.model_label.config(
    #             text=f"Updated '{adapter_name}' weight to {weight:.2f}")

    #     except ValueError as ve:
    #         self.model_label.config(text=str(ve))
    #     except Exception as e:
    #         self.model_label.config(
    #             text=f"Error updating adapter weight: {str(e)}")
    #         raise Exception(e)
