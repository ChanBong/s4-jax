from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, uniform
from jax.numpy.linalg import eig, inv, matrix_power
from jax.scipy.signal import convolve

rng = jax.random.PRNGKey(1)


def random_SSM(rng, N):
    '''
    SSM are, in few words, projection of a one-D signal (u) onto another 1-D signal (y) using a latent matrix (A)
    in between. To achieve this functionality, Our SSMs will be defined by three matrices which
    we will learn. For now we begin with a random SSM
    '''

    # Jax unlike numpy doesn't update the PRNG automatically. We'll use this all along to have new keys
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return A, B, C


def discretize(A, B, C, step):
    '''
    For applying SSM to a discret sequence instead of a contious function, we discritize it using billinear method
    '''
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def scan_SSM(Ab, Bb, Cb, u, x0):
    '''
    Post discretization, we get to a reccurance relation in u_k to y_k that can be seen like an RNN. The "step" function
    does look a lot like RNN.
    '''
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    # Using jax.lax.scan saves us a ton of code. We can eliminate for-loops that have carry over with jax.scan()
    # It essentially takes a vector (here our incoming signal), "steps" it through the recursion using x0 as the staring point
    # ans keeps appending the results to a result array. Remember that step function has the signature [ c -> a -> (c, b) ]

    return jax.lax.scan(step, x0, u)

    '''
    For comparison a vanilla for loop implementation of this scan would be

    result = [x0]
    x_k, y_k = 0, 0
    for i in range(len(u)):
        x_k = Ab @ x[i] + Bb @ u[i]
        y_k = Cb @ x[i]
        result.append(x_k)

    final = x_k
    return final, result

    '''



def run_SSM(A, B, C, u):
    '''
    This utility lets us run a sample SSM
    '''
    L = u.shape[0]
    N = A.shape[0]
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)

    # Run recurrence
    return scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))[1]


def K_conv(Ab, Bb, Cb, L):
    '''
    The idea behind using this kernel is that we can essentially turn the "RNN" into "CNN"
    '''
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )



def non_circular_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


def make_HiPPO(N):
    '''
    Mdifying an SSM from a random matrix A to HiPPO improves SSM's performance.
    For all intents and purposes we only need to calculate it once and it has a nice, simple structure
    '''
    def v(n, k):
        if n > k:
            return np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
        elif n == k:
            return n + 1
        else:
            return 0

    # Do it slow so we don't mess it up :)
    mat = [[v(n, k) for k in range(1, N + 1)] for n in range(1, N + 1)]
    return -np.array(mat)


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


# About flax.linen #
# Its the base class for all nn modules.
# `setup` is called to initialize the module
# In flax, 'setup' is called each time the parameters are updated. Similar to [Torch parameterizations](https://pytorch.org/tutorials/intermediate/parametrizations.html).
# `__call()__` is a popular choice for implimenting forward pass method, as it allows using model instances as if they were functions


class SSMLayer(nn.Module):
    '''
    We will learn the parameters B, C, D use the Hippo Matrix for A.
    step_size is also learned
    For RNN style training, we cache the previous states in cache
    '''

    A: np.DeviceArray  # HiPPO
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # SSM parameters
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = np.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*self.ssm, self.l_max)

        # RNN cache for long sequences
        self.x_k_1 = self.variable("cache", "cache_x_k", np.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN Mode
            return non_circular_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


# For stacking copies of layer and vectorize them we use vmap.
# Jax can efficiently "vectorize" function inputs and outputs. By this, I mean that it can add extra axes to the inputs
# and then evaluate the function in parallel over those axes (independently of each other)
# In other words it returns a function which maps the function specified over using in_axes and stack them together using out_axes

def cloneLayer(layer):
    '''
    Here vmap stacks the `layer` wi
    '''
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


def SSMInit(N):
    '''
    We then initialize A with the HiPPO matrix, and pass it into the stack of modules
    '''
    # With this we effectively put together layers of SSMLayer and initialised all of them with a hippo matrix and N
    return partial(cloneLayer(SSMLayer), A=make_HiPPO(N), N=N)


class SequenceBlock(nn.Module):
    '''
    Put the SSM Layer into a statndard NN.
    Add dropout and linear projection to this SSM
    '''

    layer: nn.Module
    l_max: int
    dropout: float
    d_model: int
    training: bool = True
    decode: bool = False

    def setup(self):
        self.seq = self.layer(l_max=self.l_max, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        x2 = self.seq(x)
        z = self.drop(self.out(self.drop(nn.gelu(x2))))
        return self.norm(z + x)


# We can then stack a bunch of these blocks on top of each other
# to produce a stack of SSM layers. This can be used for
# classification or generation in the standard way as a Transformer.


class StackedModel(nn.Module):
    layer: nn.Module
    d_output: int
    d_model: int
    l_max: int
    n_layers: int
    dropout: float = 0.2
    training: bool = True
    classification: bool = False
    decode: bool = False

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer=self.layer,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                l_max=self.l_max,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]
        # TODO try this with vmap

    def __call__(self, x):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


# In Flax we add the batch dimension as a lifted transformation.
# We need to route through several variable collections which
# handle RNN and parameter caching (described below).


BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)
