import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict
from einops import rearrange
from optree import tree_map
import re

from torch_utils import symexp, symlog, MSEDist, SymlogDist, DiscDist
import torch.distributions as tdist

class RSSM(nn.Module):
    def __init__(
        self,
        deter=1024,
        stoch=32,
        classes=32,
        unroll=False,
        initial="learned",
        unimix=0.01,
        action_clip=1.0,
        **kw,
    ):
        super().__init__()
        self._deter = deter
        self._stoch = stoch
        self._classes = classes
        self._unroll = unroll
        self._initial = initial
        self._unimix = unimix
        self._action_clip = action_clip
        self._kw = kw
        self.img_in = nn.Linear(**self._kw)

    def initial(self, bs):
        if self._classes:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                logit=jnp.zeros([bs, self._stoch, self._classes], f32),
                stoch=jnp.zeros([bs, self._stoch, self._classes], f32),
            )
        else:
            state = dict(
                deter=jnp.zeros([bs, self._deter], f32),
                mean=jnp.zeros([bs, self._stoch], f32),
                std=jnp.ones([bs, self._stoch], f32),
                stoch=jnp.zeros([bs, self._stoch], f32),
            )
        if self._initial == "zeros":
            return cast(state)
        elif self._initial == "learned":
            deter = self.get("initial", jnp.zeros, state["deter"][0].shape, f32)
            state["deter"] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
            state["stoch"] = self.get_stoch(cast(state["deter"]))
            return cast(state)
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
        inputs = swap(action), swap(embed), swap(is_first)
        start = state, state
        post, prior = jaxutils.scan(step, inputs, start, self._unroll)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        state = self.initial(action.shape[0]) if state is None else state
        assert isinstance(state, dict), state
        action = swap(action)
        prior = jaxutils.scan(self.img_step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_dist(self, state, argmax=False):
        if self._classes:
            logit = state["logit"].astype(f32)
            return tfd.Independent(jaxutils.OneHotDist(logit), 1)
        else:
            mean = state["mean"].astype(f32)
            std = state["std"].astype(f32)
            return tfp.MultivariateNormalDiag(mean, std)

    def obs_step(self, prev_state, prev_action, embed, is_first):
        # is_first = cast(is_first)
        # prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            with torch.no_grad():
                prev_action *= self._action_clip / torch.maximum(self._action_clip, torch.abs(prev_action))
        prev_state, prev_action = tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action)
        )
        prev_state = tree_map(
            lambda x, y: x + self._mask(y, is_first),
            prev_state,
            self.initial(len(is_first)),
        )
        prior = self.img_step(prev_state, prev_action)
        if len(embed.shape) > len(prior['deter'].shape):
            embed = embed.reshape(embed.shape[0], -1)
        x = jnp.concatenate([prior["deter"], embed], -1)
        x = self.get("obs_out", Linear, **self._kw)(x)
        stats = self._stats("obs_stats", x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return cast(post), cast(prior)

    def img_step(self, prev_state, prev_action):
        prev_stoch = prev_state["stoch"]
        if self._action_clip > 0.0:
            with torch.no_grad():
                prev_action *= self._action_clip / torch.maximum(self._action_clip, torch.abs(prev_action))
        if self._classes:
            shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch = prev_stoch.reshape(shape)
        if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)
        x = torch.concatenate([prev_stoch, prev_action], -1)
        x = self.get("img_in", Linear, **self._kw)(x)
        x, deter = self._gru(x, prev_state["deter"])
        x = self.get("img_out", Linear, **self._kw)(x)
        stats = self._stats("img_stats", x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())
        prior = {"stoch": stoch, "deter": deter, **stats}
        return cast(prior)

    def get_stoch(self, deter):
        x = self.get("img_out", Linear, **self._kw)(deter)
        stats = self._stats("img_stats", x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def _gru(self, x, deter):
        x = jnp.concatenate([deter, x], -1)
        kw = {**self._kw, "act": "none", "units": 3 * self._deter}
        x = self.get("gru", Linear, **kw)(x)
        reset, cand, update = jnp.split(x, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter, deter

    def _stats(self, name, x):
        if self._classes:
            x = self.get(name, Linear, self._stoch * self._classes)(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
            if self._unimix:
                probs = jax.nn.softmax(logit, -1)
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                logit = jnp.log(probs)
            stats = {"logit": logit}
            return stats
        else:
            x = self.get(name, Linear, 2 * self._stoch)(x)
            mean, std = jnp.split(x, 2, -1)
            std = 2 * jax.nn.sigmoid(std / 2) + 0.1
            return {"mean": mean, "std": std}

    def _mask(self, value, mask):
        return jnp.einsum("b...,b->b...", value, mask.astype(value.dtype))

    def dyn_loss(self, post, prior, impl="kl", free=1.0):
        if impl == "kl":
            loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
        elif impl == "logprob":
            loss = -self.get_dist(prior).log_prob(sg(post["stoch"]))
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss

    def rep_loss(self, post, prior, impl="kl", free=1.0):
        if impl == "kl":
            loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
        elif impl == "uniform":
            uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
            loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
        elif impl == "entropy":
            loss = -self.get_dist(post).entropy()
        elif impl == "none":
            loss = jnp.zeros(post["deter"].shape[:-1])
        else:
            raise NotImplementedError(impl)
        if free:
            loss = jnp.maximum(loss, free)
        return loss


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
            self.dists = {k: Dist(shape=v, in_feat=units, **self._dist) for k, v in self._shape.items()}
        elif isinstance(self._shape, tuple):
            self.dist = Dist(shape=self._shape, **self._dist)
        elif self._shape is None:
            pass
        else:
            raise ValueError(f"No such shape {self._shape} for dist")

    def forward(self, inputs):
        feat = self._inputs(inputs).float()
        if self._symlog_inputs:
            feat = symlog(feat)
        x = feat
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
        in_feat,
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
        out_feat = np.prod(self._shape)
        if self._dist.endswith('_disc'):
            out_feat = out_feat * self._bins
        self.out =  nn.Linear(in_feat, out_feat)


    def forward(self, inputs):
        shape = self._shape
        if self._dist.endswith("_disc"):
            shape = (*self._shape, self._bins)
        x = self.out(inputs)
        x = x.reshape(inputs.shape[:-1] + shape).float()
        if self._dist == 'symlog_mse':
            return SymlogDist(x, len(self._shape), "mse", "sum")
        elif self._dist == 'symlog_disc':
            return DiscDist(x, len(self._shape), -20, 20, symlog, symexp)
        else:
            raise NotImplementedError(f"Such dist {self._dist} is not implemented")
        

    # def inner(self, inputs):
    #     kw = {}
    #     kw["outscale"] = self._outscale
    #     kw["outnorm"] = self._outnorm
    #     shape = self._shape
    #     if self._dist.endswith("_disc"):
    #         shape = (*self._shape, self._bins)
    #     out = self.get("out", Linear, int(np.prod(shape)), **kw)(inputs)
    #     out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
    #     if self._dist in ("normal", "trunc_normal"):
    #         std = self.get("std", Linear, int(np.prod(self._shape)), **kw)(inputs)
    #         std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
    #     if self._dist == "symlog_mse":
    #         return jaxutils.SymlogDist(out, len(self._shape), "mse", "sum")
    #     if self._dist == "symlog_disc":
    #         return jaxutils.DiscDist(
    #             out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp
    #         )
    #     if self._dist == "mse":
    #         return jaxutils.MSEDist(out, len(self._shape), "sum")
    #     if self._dist == "normal":
    #         lo, hi = self._minstd, self._maxstd
    #         std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
    #         dist = tfd.Normal(jnp.tanh(out), std)
    #         dist = tfd.Independent(dist, len(self._shape))
    #         dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
    #         dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
    #         return dist
    #     if self._dist == "binary":
    #         dist = tfd.Bernoulli(out)
    #         return tfd.Independent(dist, len(self._shape))
    #     if self._dist == "onehot":
    #         if self._unimix:
    #             probs = jax.nn.softmax(out, -1)
    #             uniform = jnp.ones_like(probs) / probs.shape[-1]
    #             probs = (1 - self._unimix) * probs + self._unimix * uniform
    #             out = jnp.log(probs)
    #         dist = jaxutils.OneHotDist(out)
    #         if len(self._shape) > 1:
    #             dist = tfd.Independent(dist, len(self._shape) - 1)
    #         dist.minent = 0.0
    #         dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
    #         return dist
    #     raise NotImplementedError(self._dist)