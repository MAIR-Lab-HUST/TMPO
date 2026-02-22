import os

from torch.utils.data import Dataset

class TextPromptDataset(Dataset):
    def __init__(self, file_path, max_length=None):
        """
        Args:
            file_path (string): Path to the text file with prompts
            max_length (int, optional): Maximum number of prompts to load
        """
        self.prompts = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Prompt file not found: {file_path}. "
                "Provide a newline-delimited prompt file or use `prompts_sample.txt`."
            )

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                prompt = line.strip()
                if prompt:  # Skip empty lines
                    self.prompts.append(prompt)
                    if max_length and len(self.prompts) >= max_length:
                        break
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]