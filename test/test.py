import unittest
import torch
from torch.utils.data import DataLoader
from your_dataset_class import YourDataset
from transformers import YourTokenizer

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = YourDataset("path/to/your/data")
        self.tokenizer = YourTokenizer.from_pretrained("tokenizer_name")
        self.data_loader = DataLoader(self.dataset, batch_size=2, shuffle=True)

    def test_data_sample_output(self):
        # Check if DataLoader can iterate
        for sample in self.data_loader:
            self.assertTrue(isinstance(sample, dict))
            break  # Test one sample to ensure it's in the correct format

    def test_tokenizer_padding_truncation(self):
        sample_text = "This is a test."
        tokens = self.tokenizer.encode_plus(sample_text, max_length=512, truncation=True, padding='max_length')
        self.assertEqual(len(tokens['input_ids']), 512)  # Assuming max_length=512

    def test_dataset_and_tokenizer_integration(self):
        sample = next(iter(self.data_loader))
        # Assuming your dataset returns a batch with 'texts' key
        inputs = self.tokenizer(sample['texts'], padding=True, truncation=True, return_tensors='pt')
        self.assertIn('input_ids', inputs)
        self.assertIn('attention_mask', inputs)



from your_model_class import YourModel
import torch.nn as nn

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = YourModel()

    def test_forward_computation(self):
        input_ids = torch.randint(0, 2000, (2, 512))  # Example input
        attention_mask = torch.ones(2, 512)
        output = self.model(input_ids, attention_mask=attention_mask)
        self.assertIsInstance(output, torch.Tensor)  # Or more specific checks based on your model's output

    def test_model_layers(self):
        # Ensure all parts of the model are present, example for a simple model
        self.assertIsInstance(self.model.embedding, nn.Embedding)
        self.assertIsInstance(self.model.linear, nn.Linear)
        # Add checks for all layers you expect to be in your model

    def test_output_format(self):
        input_ids = torch.randint(0, 2000, (1, 512))  # Single sample input
        attention_mask = torch.ones(1, 512)
        output = self.model(input_ids, attention_mask=attention_mask)
        self.assertEqual(output.shape, torch.Size([1, your_model_output_size]))  # Replace your_model_output_size with the expected size
