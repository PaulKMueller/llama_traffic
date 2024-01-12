import tensorflow as tf
import numpy as np
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger
from tqdm.keras import TqdmCallback
import json
from sklearn.model_selection import train_test_split
import random


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model=2):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=101, depth=2)

    def call(self, x):
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + positional_encoding(101, 2)
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding()

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
        )

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del x._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return x


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {}


def train_transformer():
    direction_counter_dict = {
        "Left": 0,
        "Right": 0,
        "Stationary": 0,
        "Right-U-Turn": 0,
        "Left-U-Turn": 0,
        "Straight-Right": 0,
        "Straight-Left": 0,
        "Straight": 0,
    }
    # Load labeled trajectory data
    with open("datasets/labeled_ego_trajectories.json", "r") as file:
        trajectories_data = json.load(file)

    trajectories = []

    for value in trajectories_data.values():
        direction = value["Direction"]

        # Coordinates as Numpy array
        coordinates = np.array(value["Coordinates"])
        # coordinates = np.expand_dims(coordinates, axis=0).reshape(1, 101, 2)
        # coordinates = np.expand_dims(coordinates, axis=0).reshape(1, 101, 2)
        trajectories.append(coordinates)

    # dataset_x = tf.data.Dataset.from_tensor_slices(X)
    # dataset_y = tf.data.Dataset.from_tensor_slices(Y)
    # data_pairs = tf.data.Dataset.from_tensor_slices(data)
    # data_pairs = tf.data.Dataset.zip(dataset_x, dataset_y)
    # data_pairs = data_pairs.batch(1)

    print("Dataset finished")

    # X = tf.convert_to_tensor(X, dtype=tf.float64)
    # print(X.shape)
    # Y = tf.convert_to_tensor(Y, dtype=tf.float64)
    # print(Y.shape)

    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=42
    # )

    # model.fit(
    #     X_train,
    #     Y_train,
    #     epochs=200,
    #     batch_size=32,
    #     validation_split=0.1,
    #     verbose=0,
    #     callbacks=[TqdmCallback(verbose=2), WandbMetricsLogger()],
    # )

    # coordinates = np.array(trajectory.rotated_coordinates)
    # coordinates = np.expand_dims(coordinates, axis=0).reshape(1, 101, 2)

    num_layers = 2
    d_model = 2
    dff = 64
    num_heads = 3
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate,
    )

    def masked_loss(label, pred):
        print(f"Shape label: {label.shape}")
        print(f"Shape prediction: {pred.shape}")
        mask = label != 0
        loss_object = tf.keras.losses.MeanSquaredError()
        loss = loss_object(label, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer.compile(loss="mean_squared_error", optimizer="adam")

    # data = tf.data.Dataset.from_tensor_slices((coordinates, coordinates))

    def data_generator(coordinates_set, batch_size):
        # while True:
        #     random.shuffle(coordinates_set)
        #     coordinates_set_sample = random.sample(coordinates_set, 1000)

        #     for i in range(0, len(coordinates_set_sample), batch_size):
        #         batch = coordinates_set_sample[i : i + batch_size]
        #         batch_tensors = [
        #             tf.convert_to_tensor(coords, dtype=tf.float64) for coords in batch
        #         ]
        #         batch_tensor = tf.stack(batch_tensors)
        #         yield (batch_tensor, batch_tensor), batch_tensor

        # Shuffle coordinates_set if needed
        random.shuffle(coordinates_set)
        # coordinates_set = random.sample(coordinates_set, 1000)

        training_data = []

        # Yield batches
        for i in range(0, len(coordinates_set), batch_size):
            batch = coordinates_set[i : i + batch_size]
            batch_tensors = [
                tf.convert_to_tensor(coords, dtype=tf.float64) for coords in batch
            ]
            batch_tensor = tf.stack(batch_tensors)
            print(batch_tensor.shape)
            # training_data.append(((batch_tensor, batch_tensor), batch_tensor))
            yield (batch_tensor, batch_tensor), batch_tensor

        # return training_data

    print(len(trajectories))

    wandb.init(config={"bs": 12})
    transformer.fit(
        data_generator(trajectories, 4),
        epochs=1,
        callbacks=[WandbMetricsLogger()],
    )

    # output = transformer((coordinates, coordinates))
    # print(output)
    transformer.save("models/my_transformer_model", save_format="tf")
    # test_loss = transformer.evaluate(X_test, Y_test)
    # print(f"Test Loss: {test_loss}")
    wandb.finish()

    # print(coordinates + encoding)
