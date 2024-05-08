import torch
from torch import Tensor
import lightning as L

from ..config.config import Config
from ..layers.networks import Generator, Discriminator
from ..layers.optimizer import AdamW

from transformers import AutoTokenizer
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
)

def get_BART(model_path:str='facebook/bart-large'):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


class LM(L.LightningModule):
    def __init__(self, config:Config):
        super().__init__()
        self.save_hyperparameters("config")
        self.config = config
        self.batch_size = config.batch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.automatic_optimization = False
        self._register_discriminator_fw_hook()

    def forward_generator(self, input_ids:Tensor, attention_mask:Tensor=None, labels:Tensor=None, **kwargs):
        return self.generator(input_ids, attention_mask, labels, **kwargs)
    
    def forward_discriminator(self, input_ids:Tensor, attention_mask:Tensor=None, labels:Tensor=None, **kwargs):
        return self.discriminator(input_ids, attention_mask, labels, **kwargs)

    def training_step(self, batch, batch_idx):
        # forward
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        output_gen, loss_gen = self.forward_generator(input_ids, attention_mask, labels)
        batch = self._get_discriminator_inputs(batch, output_gen, is_stochastic=True)
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        output_disc, loss_disc = self.forward_discriminator(input_ids, attention_mask, labels)
        loss_disc = self.config.lambda_discriminator * loss_disc

        self.manual_backward(loss_gen / self.gradient_accumulation_steps)
        self.manual_backward(loss_disc / self.gradient_accumulation_steps)
        # optimize
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # print('##### Optimizing #####')
            opt_gen, opt_disc = self.optimizers()
            self.clip_gradients(
                opt_gen,
                gradient_clip_val=self.config.gradient_clip_val,
                gradient_clip_algorithm=self.config.gradient_clip_algorithm)
            opt_gen.step()
            opt_gen.zero_grad()
            self.clip_gradients(
                opt_disc,
                gradient_clip_val=self.config.gradient_clip_val,
                gradient_clip_algorithm=self.config.gradient_clip_algorithm)
            opt_disc.step()
            opt_disc.zero_grad()

        # log
        self.log('loss_gen', loss_gen, on_step=True, prog_bar=True)
        self.log('loss_disc', loss_disc, on_step=True, prog_bar=True)
        # return [loss_gen, loss_disc]
    
    def configure_optimizers(self):
        gen_optimizer = AdamW(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta_1, self.config.beta_2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay)
        disc_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta_1, self.config.beta_2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay)
        return [gen_optimizer, disc_optimizer], []
    
    def _get_discriminator_inputs(self, masked_data, logits, is_stochastic=True, **kwargs):
        replaced_input_ids = self._replace_masked_tokens(masked_data, logits, is_stochastic, **kwargs)
        masked_data['input_ids'] = replaced_input_ids
        new_labels = self._check_labels(masked_data)
        masked_data['labels'] = new_labels
        return masked_data
    
    def _replace_masked_tokens(self, masked_data, logits, is_stochastic=True, **kwargs):
        masked_input_ids = masked_data['input_ids']
        if is_stochastic:
            probs = torch.softmax(logits, dim=-1)
            sampled_ids = torch.distributions.multinomial.Multinomial(1, probs).sample().topk(1).indices.squeeze(-1)
        else:
            sampled_ids = torch.argmax(logits, dim=-1)
        replaced_input_ids = torch.where(masked_data['labels'] > 0, sampled_ids, masked_input_ids)
        return replaced_input_ids

    def _check_labels(self, masked_data):
        replaced_input_ids = masked_data['input_ids']
        labels = masked_data['labels']
        new_labels = torch.zeros_like(labels)
        new_labels = torch.where(labels > 0, labels != replaced_input_ids, new_labels)
        return new_labels
    
    def _register_discriminator_fw_hook(self, *kwargs):
        word_bias = torch.zeros_like(self.discriminator.embedding.word_embedding_layer.weight)
        word_bias = torch.nn.Parameter(word_bias)
        position_bias = torch.zeros_like(self.discriminator.embedding.absolute_position_embedding_layer.weight)
        position_bias = torch.nn.Parameter(position_bias)
        delattr(self.discriminator.embedding.word_embedding_layer, 'weight')
        self.discriminator.embedding.word_embedding_layer.register_parameter('_weight', word_bias)
        delattr(self.discriminator.embedding.absolute_position_embedding_layer, 'weight')
        self.discriminator.embedding.absolute_position_embedding_layer.register_parameter('_weight', position_bias)

        def fw_hook(module, *inputs):
            if self.config.share_embedding == 'gdes': # Gradient-disentangled weight/embedding sharing
                g_w_ebd = self.generator.embedding.word_embedding_layer
                d_w_ebd = self.discriminator.embedding.word_embedding_layer
                self._set_param(d_w_ebd, 'weight', g_w_ebd.weight.detach() + d_w_ebd._weight)
                g_p_ebd = self.generator.embedding.absolute_position_embedding_layer
                d_p_ebd = self.discriminator.embedding.absolute_position_embedding_layer
                self._set_param(d_p_ebd, 'weight', g_p_ebd.weight.detach() + d_p_ebd._weight)

            elif self.config.share_embedding == 'es': # vallina embedding sharing
                g_w_ebd = self.generator.embedding.word_embedding_layer
                d_w_ebd = self.discriminator.embedding.word_embedding_layer
                self._set_param(d_w_ebd, 'weight', g_w_ebd.weight)
                g_p_ebd = self.generator.embedding.absolute_position_embedding_layer
                d_p_ebd = self.discriminator.embedding.absolute_position_embedding_layer
                self._set_param(d_p_ebd, 'weight', g_p_ebd.weight)
            return None
        self.discriminator.register_forward_pre_hook(fw_hook)

    @staticmethod
    def _set_param(module, param_name, value):
        if hasattr(module, param_name):
            delattr(module, param_name)
        module.register_buffer(param_name, value)
    
