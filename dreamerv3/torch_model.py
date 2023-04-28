import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict
from einops import rearrange
import re

from torch_utils import symexp, symlog, MSEDist
import torch.distributions as tdist

class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        cnn_depth=48,
        cnn_blocks=2,
        resize="stride",
        symlog_inputs=False,
        minres=4,
        **kw,
    ):
        super().__init__()
        excluded = ("is_first", "is_last")
        shapes = {
            k: v for k, v in shapes.items() if (k not in excluded and not k.startswith("log_"))
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if (len(v) == 3 and re.match(cnn_keys, k))
        }
        self.mlp_shapes = {
            k: v for k, v in shapes.items() if (len(v) in (1, 2) and re.match(mlp_keys, k))
        }
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)
        cnn_kw = {**kw, "minres": minres, "name": "cnn"}
        mlp_kw = {**kw, "symlog_inputs": symlog_inputs, "name": "mlp"}
        if cnn == "resnet":
            in_hw = 64
            in_chan = sum([v[-1] for _, v in self.cnn_shapes.items()])
            self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, in_hw=in_hw, in_chan=in_chan, **cnn_kw)
        else:
            raise NotImplementedError(cnn)
        if self.mlp_shapes:
            in_feat = np.sum([np.prod(v) for _, v in self.mlp_shapes.items()])
            self._mlp = MLP(None, mlp_layers, mlp_units, in_feat=in_feat, dist="none", **mlp_kw)
    
    def forward(self, data):
        some_key, some_shape = list(self.shapes.items())[0]
        batch_dims = data[some_key].shape[: -len(some_shape)]
        data = {k: v.reshape((-1,) + v.shape[len(batch_dims) :]) for k, v in data.items()}
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([data[k] for k in self.cnn_shapes], -1)
            output = self._cnn(inputs)
            output = output.reshape((output.shape[0], -1))
            outputs.append(output)
        if self.mlp_shapes:
            inputs = [
                data[k][..., None] if len(self.shapes[k]) == 0 else data[k] for k in self.mlp_shapes
            ]
            inputs = torch.cat([x.float() for x in inputs], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        outputs = outputs.reshape(batch_dims + outputs.shape[1:])
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        shapes,
        inputs=["tensor"],
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=4,
        mlp_units=512,
        cnn="resize",
        cnn_depth=48,
        cnn_blocks=2,
        image_dist="mse",
        vector_dist="mse",
        resize="stride",
        bins=255,
        outscale=1.0,
        minres=4,
        cnn_sigmoid=False,
        **kw,
    ):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3}
        self.mlp_shapes = {k: v for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) in (1, 2)}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)
        cnn_kw = {**kw, "minres": minres, "sigmoid": cnn_sigmoid}
        mlp_kw = {**kw, "dist": vector_dist, "outscale": outscale, "bins": bins}
        if self.cnn_shapes:
            shapes = list(self.cnn_shapes.values())
            assert all(x[:-1] == shapes[0][:-1] for x in shapes)
            shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
            in_shapes = {'deter': 512, 'logit': 32 * 32, 'stoch': 32 * 32, 'embed': 5120}
            in_feat = sum([v for k, v in in_shapes.items() if k in inputs])
            last_chan = sum([v[-1] for _, v in self.cnn_shapes.items()])
            if cnn == "resnet":
                self._cnn = ImageDecoderResnet(
                        shape, cnn_depth, cnn_blocks, resize, 
                        in_feat=in_feat, 
                        last_chan=last_chan,
                        name="cnn",
                        **cnn_kw
                    )
            else:
                raise NotImplementedError(cnn)
        if self.mlp_shapes:
            in_shapes = {'deter': 512, 'logit': 32 * 32, 'stoch': 32 * 32, 'embed': 5120}
            in_feat = sum([v for k, v in in_shapes.items() if k in inputs])
            self._mlp = MLP(self.mlp_shapes, mlp_layers, mlp_units, in_feat=in_feat, **mlp_kw, name="mlp")
        self._inputs = Input(inputs, dims="deter")
        self._image_dist = image_dist

    def forward(self, inputs, drop_loss_indices=None):
        features = self._inputs(inputs)
        dists = {}
        if self.cnn_shapes:
            feat = features
            if drop_loss_indices is not None:
                feat = feat[:, drop_loss_indices]
            flat = feat.reshape([-1, feat.shape[-1]])
            output = self._cnn(flat)
            output = output.reshape(feat.shape[:-1] + output.shape[1:])
            if len(self.cnn_shapes) > 1:
                means = torch.split(output, 3, -3)
            else:
                means = [output]
            dists.update(
                {
                    key: self._make_image_dist(key, mean)
                    for (key, shape), mean in zip(self.cnn_shapes.items(), means)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, name, mean):
        mean = mean.float()
        if self._image_dist == "normal":
            return tdist.Independent(tdist.Normal(mean, 1), 3)
        if self._image_dist == "mse":
            return MSEDist(mean, 3, "sum")
        raise NotImplementedError(self._image_dist)


class ImageEncoderResnet(nn.Module):
    def __init__(self, depth, blocks, resize, minres, in_hw, in_chan, **kw):
        super().__init__()
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._kw = kw

        stages = int(np.log2(in_hw) - np.log2(self._minres))
        modules = OrderedDict()
        out_chan = depth
        for i in range(stages):
            kw = {**self._kw, "preact": False}
            if self._resize == "stride":
                modules[f's{i}resize'] = ConvBlock(in_chan, out_chan, 4, 2, **kw)
            else:
                raise NotImplementedError(self._resize)
            for j in range(self._blocks):
                modules[f's{i}block{j}'] = ResBlock(out_chan, out_chan, 3, 1, **kw)
            in_chan = out_chan
            out_chan = out_chan * 2

        if self._blocks:
            modules['last'] = nn.SiLU()
        self.blocks = nn.Sequential(modules)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        x = self.blocks(x).flatten(1)
        return x


class ImageDecoderResnet(nn.Module):
    def __init__(self, shape, depth, blocks, resize, minres, sigmoid, in_feat, last_chan, **kw):
        self._shape = shape
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._sigmoid = sigmoid
        self._kw = kw
        super().__init__()

        stages = int(np.log2(64) - np.log2(self._minres))
        depth = self._depth * 2 ** (stages - 1)
        self._new_depth = depth
        self.in_fc = nn.Linear(in_feat, depth * minres * minres)
        modules = OrderedDict()
        in_chan = out_chan = depth
        for i in range(stages):
            for j in range(self._blocks):
                modules[f's{i}block{j}'] = ResBlock(in_chan, out_chan, 3, 1, **kw)
            
            out_chan = last_chan if i == stages - 1 else out_chan // 2
            
            if self._resize == "stride":
                modules[f's{i}resize'] = ConvBlock(in_chan, out_chan, 4, 2, transpose=True, **kw)
            else:
                raise NotImplementedError(self._resize)
            in_chan = out_chan
        self.blocks = nn.Sequential(modules)
    def forward(self, x):
        x = self.in_fc(x)
        x = x.view(-1, self._new_depth, self._minres, self._minres)
        x = self.blocks(x)
        if self._sigmoid:
            x = F.sigmoid(x)
        else:
            x = x + 0.5
        return x
                
class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, stride, transpose=False, **kwargs) -> None:
        super().__init__()
        Conv2d = nn.Conv2d if not transpose else nn.ConvTranspose2d
        self.blocks = nn.Sequential(*[
            Conv2d(in_chan, out_chan, kernel, stride, padding=1),
            nn.InstanceNorm2d(out_chan),
            nn.SiLU()
        ])
    
    def forward(self, x):
        return self.blocks(x)

class ResBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, **kwargs) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            nn.InstanceNorm2d(in_chan),
            nn.SiLU(),
            nn.Conv2d(in_chan, in_chan, kernel, 1, padding='same'),
            nn.InstanceNorm2d(in_chan),
            nn.SiLU(),
            nn.Conv2d(in_chan, out_chan, kernel, 1, padding='same'),
        ])
    
    def forward(self, x):
        skip = x
        x = self.blocks(x)
        return skip + x

class MLP(nn.Module):
    def __init__(
        self,
        shape,
        layers,
        units,
        in_feat,
        inputs=["tensor"],
        dims=None,
        symlog_inputs=False,
        **kw,
    ):
        super().__init__()
        assert shape is None or isinstance(shape, (int, tuple, dict)), shape
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self._layers = layers
        self._units = units
        self._inputs = Input(inputs, dims=dims)
        self._symlog_inputs = symlog_inputs
        distkeys = ("dist", "outscale", "minstd", "maxstd", "outnorm", "unimix", "bins")
        self._dense = {k: v for k, v in kw.items() if k not in distkeys}
        self._dist = {k: v for k, v in kw.items() if k in distkeys}
        blocks = OrderedDict()
        for i in range(layers):
            in_feat = in_feat if i == 0 else units
            blocks[f"fc{i}"] = nn.Sequential(
                nn.Linear(in_feat, units),
                nn.InstanceNorm1d(units),
                nn.SiLU()
            )
        self.blocks = nn.Sequential(blocks)
        if isinstance(self._shape, dict):
            self.dists = {k: Dist(shape=v, **self._dist) for k, v in self._shape.items()}
        elif isinstance(self._shape, tuple):
            self.dist = Dist(shape=self._shape, **self._dist)
        elif self._shape is None:
            pass
        else:
            raise ValueError(f"No such shape {self._shape} for dist")

    def forward(self, inputs):
        feat = self._inputs(inputs)
        if self._symlog_inputs:
            feat = symlog(feat)
        x = feat.float()
        x = x.reshape(*[-1, x.shape[-1]])
        x = self.blocks(x)
        x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
        if self._shape is None:
            return x
        elif isinstance(self._shape, tuple):
            return self.dist(x)
        elif isinstance(self._shape, dict):
            return {k: self.dists[k](x) for k, _ in self._shape.items()}
        else:
            raise ValueError(self._shape)
    
class Input:
    def __init__(self, keys=["tensor"], dims=None):
        assert isinstance(keys, (list, tuple)), keys
        self._keys = tuple(keys)
        self._dims = dims or self._keys[0]

    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {"tensor": inputs}
        inputs = inputs.copy()
        for key in self._keys:
            if key.startswith("softmax_"):
                inputs[key] = F.softmax(inputs[key[len("softmax_") :]])
        if not all(k in inputs for k in self._keys):
            needs = f'{{{", ".join(self._keys)}}}'
            found = f'{{{", ".join(inputs.keys())}}}'
            raise KeyError(f"Cannot find keys {needs} among inputs {found}.")
        values = [inputs[k] for k in self._keys]
        dims = len(inputs[self._dims].shape)
        for i, value in enumerate(values):
            if len(value.shape) > dims:
                values[i] = value.reshape(
                    value.shape[: dims - 1] + (np.prod(value.shape[dims - 1 :]),)
                )
        values = [x.float() for x in values]
        return torch.cat(values, -1)
    

class Dist(nn.Module):
    def __init__(
        self,
        shape,
        dist="mse",
        outscale=0.1,
        outnorm=False,
        minstd=1.0,
        maxstd=1.0,
        unimix=0.0,
        bins=255,
    ):
        super().__init__()
        assert all(isinstance(dim, int) for dim in shape), shape
        self._shape = shape
        self._dist = dist
        self._minstd = minstd
        self._maxstd = maxstd
        self._unimix = unimix
        self._outscale = outscale
        self._outnorm = outnorm
        self._bins = bins

        kw = {}
        kw["outscale"] = self._outscale
        kw["outnorm"] = self._outnorm
        self.out = nn.Sequential([
            nn.Linear
        ])

    def __call__(self, inputs):
        dist = self.inner(inputs)
        assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
            dist.batch_shape,
            dist.event_shape,
            inputs.shape,
        )
        return dist

    def inner(self, inputs):
        kw = {}
        kw["outscale"] = self._outscale
        kw["outnorm"] = self._outnorm
        shape = self._shape
        if self._dist.endswith("_disc"):
            shape = (*self._shape, self._bins)
        out = self.get("out", Linear, int(np.prod(shape)), **kw)(inputs)
        out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
        if self._dist in ("normal", "trunc_normal"):
            std = self.get("std", Linear, int(np.prod(self._shape)), **kw)(inputs)
            std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
        if self._dist == "symlog_mse":
            return jaxutils.SymlogDist(out, len(self._shape), "mse", "sum")
        if self._dist == "symlog_disc":
            return jaxutils.DiscDist(
                out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp
            )
        if self._dist == "mse":
            return jaxutils.MSEDist(out, len(self._shape), "sum")
        if self._dist == "normal":
            lo, hi = self._minstd, self._maxstd
            std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
            dist = tfd.Normal(jnp.tanh(out), std)
            dist = tfd.Independent(dist, len(self._shape))
            dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
            dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
            return dist
        if self._dist == "binary":
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._shape))
        if self._dist == "onehot":
            if self._unimix:
                probs = jax.nn.softmax(out, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                out = jnp.log(probs)
            dist = jaxutils.OneHotDist(out)
            if len(self._shape) > 1:
                dist = tfd.Independent(dist, len(self._shape) - 1)
            dist.minent = 0.0
            dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
            return dist
        raise NotImplementedError(self._dist)