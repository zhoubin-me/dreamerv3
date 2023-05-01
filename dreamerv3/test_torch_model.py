import torch

from torch_model import ImageEncoderResnet, ImageDecoderResnet, MultiEncoder, MultiDecoder, RSSM

def test_image_encoder():
    kw = {'act': 'silu', 'norm': 'layer', 'winit': 'normal', 'fan': 'avg'}
    cnn_kw = {**kw, "minres": 4, "name": "cnn"}
    model = ImageEncoderResnet(96, 0, 'stride', **cnn_kw)

    x = torch.rand(32, 64, 64, 3)
    y = model(x)
    assert y.shape == (32, 12288)

def test_image_decoder():
    kw = {'act': 'silu', 'norm': 'layer', 'winit': 'normal', 'fan': 'avg'}
    cnn_kw = {**kw, "minres": 4, "sigmoid": False}
    model = ImageDecoderResnet((64, 64, 3), 96, 0, 'stride', in_chan=5120, **cnn_kw)

    x = torch.rand(32, 5120)
    y = model(x)
    assert y.shape == (32, 3, 64, 64)

def test_image_encoder_decoder():
    kw = {'act': 'silu', 'norm': 'layer', 'winit': 'normal', 'fan': 'avg'}
    cnn_kw = {**kw, "minres": 4, "name": "cnn"}
    encoder = ImageEncoderResnet(96, 0, 'stride', **cnn_kw)

    kw = {'act': 'silu', 'norm': 'layer', 'winit': 'normal', 'fan': 'avg'}
    cnn_kw = {**kw, "minres": 4, "sigmoid": False}
    decoder = ImageDecoderResnet((64, 64, 3), 96, 0, 'stride', in_chan=5120, **cnn_kw)

    x = torch.rand(32, 64, 64, 3)
    x = encoder(x)
    print(x.shape)
    x = torch.rand(32, 5120)
    x = decoder(x)
    print(x.shape)

def test_multi_encoder_decoder():
    encoder_cfg = {'mlp_keys': '.*', 'cnn_keys': 'image*', 'act': 'silu', 'norm': 'layer', 'mlp_layers': 5, 
                   'mlp_units': 1024, 'cnn': 'resnet', 'cnn_depth': 32, 'cnn_blocks': 0, 'resize': 'stride', 
                   'winit': 'normal', 'fan': 'avg', 'symlog_inputs': True, 'minres': 4}
    decoder_cfg = {'mlp_keys': '.*', 'cnn_keys': 'image*', 'act': 'silu', 'norm': 'layer', 'mlp_layers': 5, 
                   'mlp_units': 1024, 'cnn': 'resnet', 'cnn_depth': 32, 'cnn_blocks': 0, 
                   'image_dist': 'mse', 'vector_dist': 'symlog_mse', 'inputs': ('deter', 'stoch'), 
                   'resize': 'stride', 'winit': 'normal', 'fan': 'avg', 'outscale': 1.0, 'minres': 4, 
                   'cnn_sigmoid': False}
    
    shapes = {'reward': (), 'is_first': (), 'is_last': (), 'is_terminal': (), 'position': (2,), 'to_target': (2,), 
              'velocity': (2,), 'image': (64, 64, 3), 'image_top': (64, 64, 3)}
    
    data = {
        'action': torch.rand(16, 64, 2),
        'image': torch.rand(16, 64, 64, 64, 3),
        'image_top': torch.rand(16, 64, 64, 64, 3),
        'reward': torch.rand(16, 64),
        'is_first': torch.rand(16, 64),
        'is_last': torch.rand(16, 64),
        'is_terminal': torch.rand(16, 64),
        'position': torch.rand(16, 64, 2),
        'to_target': torch.rand(16, 64, 2),
        'velocity': torch.rand(16 ,64, 2),
        'cont': torch.rand(16, 64),
    }

    multi_encoder = MultiEncoder(shapes, **encoder_cfg)
    out = multi_encoder(data)
    print("Multi Encoder data inputs:", {k: v.shape for k, v in data.items()})
    print("Multi Encoder data outputs:", out.shape) # [16, 64, 5120]


    data = {
        'deter': torch.rand(16, 64, 512),
        'logit': torch.rand(16, 64, 32, 32),
        'stoch': torch.rand(16, 64, 32, 32),
        'embed': torch.rand(16, 64, 5120)
    }
    multi_decoder = MultiDecoder(shapes, **decoder_cfg)
    out = multi_decoder(data)
    print("Multi Decoder data outputs:", out)


def test_rssm():
    rssm_cfg = {'deter': 512, 'units': 512, 'stoch': 32, 'classes': 32, 'act': 'silu', 
                'norm': 'layer', 'initial': 'learned', 'unimix': 0.01, 'unroll': False, 
                'action_clip': 1.0, 'winit': 'normal', 'fan': 'avg'}
    
    rssm = RSSM(**rssm_cfg)

    data = {
        'action': torch.rand(16, 64, 2),
        'image': torch.rand(16, 64, 64, 64, 3),
        'image_top': torch.rand(16, 64, 64, 64, 3),
        'reward': torch.rand(16, 64),
        'is_first': torch.randint(0, 2, (16, 64)),
        'is_last': torch.rand(16, 64),
        'is_terminal': torch.rand(16, 64),
        'position': torch.rand(16, 64, 2),
        'to_target': torch.rand(16, 64, 2),
        'velocity': torch.rand(16 ,64, 2),
        'cont': torch.rand(16, 64),
        'embed': torch.rand(16, 64, 5120)
    }
    
    state = {
        'deter': torch.rand(16, 64, 512),
        'logit': torch.rand(16, 64, 32, 32),
        'stoch': torch.rand(16, 64, 32, 32),
        'embed': torch.rand(16, 64, 5120)
    }

    rssm.observe(data['embed'], data['action'], data['is_first'], state=state)

if __name__ == '__main__':
    # test_image_encoder_decoder()
    # test_multi_encoder_decoder()

    test_rssm()