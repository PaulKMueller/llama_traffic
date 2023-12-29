import tensorflow as tf

def positional_encoding(positions, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    angle_rads = get_angles(
        tf.range(positions, dtype=tf.float32)[:, tf.newaxis],
        tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model,
    )

    # Apply sin to even indices in the array; 2i
    sines = tf.math.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    cosines = tf.math.cos(angle_rads[:, 1::2])

    angle_rads = tf.concat([sines, cosines], axis=-1)
    pos_encoding = angle_rads[tf.newaxis, ...]

    return pos_encoding