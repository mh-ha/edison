

def MaskedLayerNorm(layerNorm, input, mask=None):
    """ Masked LayerNorm which will apply mask over the output of LayerNorm to avoid inaccurate updatings to the LayerNorm module.
    
    Args:
        layernorm (:obj:`~DeBERTa.deberta.LayerNorm`): LayerNorm module or function
        input (:obj:`torch.tensor`): The input tensor
        mask (:obj:`torch.IntTensor`): The mask to applied on the output of LayerNorm where `0` indicate the output of that element will be ignored, i.e. set to `0`

    Example::

        # Create a tensor b x n x d
        x = torch.randn([1,10,100])
        m = torch.tensor([[1,1,1,0,0,0,0,0,0,0]], dtype=torch.int)
        LayerNorm = DeBERTa.deberta.LayerNorm(100)
        y = MaskedLayerNorm(LayerNorm, x, m)

    """
    output = layerNorm(input).to(input)
    if mask is None:
        return output
    if mask.dim()!=input.dim():
        if mask.dim()==4:
            mask=mask.squeeze(1).squeeze(1)
        mask = mask.unsqueeze(2)
    mask = mask.to(output.dtype)
    return output*mask