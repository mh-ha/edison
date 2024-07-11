import yaml
import torch

from ...edison.layers.lm import LM
from ...edison.configs.config import Config

def load_state(path:str):
    return torch.load(path)

def load_pretrained_LM(
        model=None,
        config=None,
        pretrained_path:list=['weights/disc_xsmall.bin', 'weights/gen_xsmall.bin'],
        config_path:str='edison/config/config.yaml',):
    disc_layer_map = {
        'deberta.embeddings.word_embeddings._weight': 'embedding.word_embedding_layer._weight',
        # 'deberta.embeddings.word_embeddings.weight': 'embedding.word_embedding_layer.weight',
        'deberta.embeddings.position_embeddings._weight': 'embedding.absolute_position_embedding_layer._weight',
        # 'deberta.embeddings.position_embeddings.weight': 'embedding.absolute_position_embedding_layer.weight',
        'deberta.embeddings.LayerNorm.weight': 'embedding.layernorm.weight',
        'deberta.embeddings.LayerNorm.bias': 'embedding.layernorm.bias',

        'attention.self.query_proj.weight': 'attention.query_layer.weight',
        'attention.self.query_proj.bias': 'attention.query_layer.bias',
        'attention.self.key_proj.weight': 'attention.key_layer.weight',
        'attention.self.key_proj.bias': 'attention.key_layer.bias',
        'attention.self.value_proj.weight': 'attention.value_layer.weight',
        'attention.self.value_proj.bias': 'attention.value_layer.bias',
        'attention.output.dense.weight': 'attention.feedforward.dense.weight',
        'attention.output.dense.bias': 'attention.feedforward.dense.bias',
        'attention.output.LayerNorm.weight': 'attention.feedforward.layernorm.weight',
        'attention.output.LayerNorm.bias': 'attention.feedforward.layernorm.bias',
        'intermediate.dense.weight': 'feedforward.feedforward_1.weight',
        'intermediate.dense.bias': 'feedforward.feedforward_1.bias',
        'output.dense.weight': 'feedforward.feedforward_2.weight',
        'output.dense.bias': 'feedforward.feedforward_2.bias',
        'output.LayerNorm.weight': 'feedforward.layernorm.weight',
        'output.LayerNorm.bias': 'feedforward.layernorm.bias',

        'deberta.encoder.rel_embeddings.weight': 'relative_position_embedding.relative_position_embedding_layer.weight',
        'deberta.encoder.LayerNorm.weight': 'relative_position_embedding.layernorm.weight',
        'deberta.encoder.LayerNorm.bias': 'relative_position_embedding.layernorm.bias',
        'mask_predictions.dense.weight': 'head.dense.weight',
        'mask_predictions.dense.bias': 'head.dense.bias',
        'mask_predictions.LayerNorm.weight': 'head.layernorm.weight',
        'mask_predictions.LayerNorm.bias': 'head.layernorm.bias',
        'mask_predictions.classifier.weight': 'head.classifier.weight',
        'mask_predictions.classifier.bias': 'head.classifier.bias',
    }
    gen_layer_map = {
        'deberta.embeddings.word_embeddings.weight': 'embedding.word_embedding_layer.weight',
        'deberta.embeddings.position_embeddings.weight': 'embedding.absolute_position_embedding_layer.weight',
        'deberta.embeddings.LayerNorm.weight': 'embedding.layernorm.weight',
        'deberta.embeddings.LayerNorm.bias': 'embedding.layernorm.bias',

        'attention.self.query_proj.weight': 'attention.query_layer.weight',
        'attention.self.query_proj.bias': 'attention.query_layer.bias',
        'attention.self.key_proj.weight': 'attention.key_layer.weight',
        'attention.self.key_proj.bias': 'attention.key_layer.bias',
        'attention.self.value_proj.weight': 'attention.value_layer.weight',
        'attention.self.value_proj.bias': 'attention.value_layer.bias',
        'attention.output.dense.weight': 'attention.feedforward.dense.weight',
        'attention.output.dense.bias': 'attention.feedforward.dense.bias',
        'attention.output.LayerNorm.weight': 'attention.feedforward.layernorm.weight',
        'attention.output.LayerNorm.bias': 'attention.feedforward.layernorm.bias',
        'intermediate.dense.weight': 'feedforward.feedforward_1.weight',
        'intermediate.dense.bias': 'feedforward.feedforward_1.bias',
        'output.dense.weight': 'feedforward.feedforward_2.weight',
        'output.dense.bias': 'feedforward.feedforward_2.bias',
        'output.LayerNorm.weight': 'feedforward.layernorm.weight',
        'output.LayerNorm.bias': 'feedforward.layernorm.bias',

        'deberta.encoder.rel_embeddings.weight': 'relative_position_embedding.relative_position_embedding_layer.weight',
        'deberta.encoder.LayerNorm.weight': 'relative_position_embedding.layernorm.weight',
        'deberta.encoder.LayerNorm.bias': 'relative_position_embedding.layernorm.bias',
        'lm_predictions.lm_head.bias': 'head.bias',
        'lm_predictions.lm_head.dense.weight': 'head.dense.weight',
        'lm_predictions.lm_head.dense.bias': 'head.dense.bias',
        'lm_predictions.lm_head.LayerNorm.weight': 'head.layernorm.weight',
        'lm_predictions.lm_head.LayerNorm.bias': 'head.layernorm.bias',
    }

    if config is None:
        config = yaml.safe_load(open(config_path, 'r'))
        config = Config(**config)
    if model is None:
        model = LM(config)
    def get_state_dict(base_state_dict, layer_map):
        state_dict = {}
        idx = None
        for k, v in base_state_dict.items():
            if k.startswith('deberta.encoder.layer'):
                prefix = 'encoder.layers'
                idx = int(k.split('.')[3])
                k = '.'.join(k.split('.')[4:])
            if k in layer_map:
                k = layer_map[k]
                try:
                    key = f'{prefix}.{idx}.{k}' if idx is not None else k
                    state_dict[key] = v
                    idx = None
                except KeyError:
                    print(f'KeyError: {k}')
            else:
                print(f'not included: {k}')
        return state_dict
    disc_state_dict = get_state_dict(torch.load(pretrained_path[0]), disc_layer_map)
    gen_state_dict = get_state_dict(torch.load(pretrained_path[1]), gen_layer_map)

    model.discriminator.load_state_dict(disc_state_dict)
    model.generator.load_state_dict(gen_state_dict)
    return model, disc_state_dict, gen_state_dict

