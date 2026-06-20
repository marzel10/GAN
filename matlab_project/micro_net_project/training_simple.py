import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(1)

def compute_weibull_health(N):
    min_stress        = -6.5
    max_stress        = -65
    ultimate_strength = -104

    amp_stress      = abs(max_stress - min_stress) / 2
    mean_stress     = (max_stress + min_stress) / 2
    nor_amp_stress  = amp_stress  / ultimate_strength
    nor_mean_stress = mean_stress / ultimate_strength

    k = -1.17 * nor_mean_stress - 19.9 * nor_amp_stress + 5.92
    lam = 0.0661 * nor_mean_stress - 1.05 * nor_amp_stress + 0.754

    weibull_health = np.zeros(N, dtype=float)

    for i in range(N):
        x = (i + 1) / N
        val = lam * (np.log(x) + 0j) ** (1 / k)   # allow complex intermediate
        weibull_health[i] = np.real(val)

    return weibull_health

if __name__ == "__main__":
    # Data
    N = 1000
    health_table = compute_weibull_health(N)
    target_tf = tf.linspace(0, 1, N)  # shape (N,1)
    weibul_distr = tf.convert_to_tensor(health_table.reshape(-1, 1), dtype=tf.float32)  # shape (N,1)
    for i in range(15):
        temp_in = weibul_distr + tf.random.normal(weibul_distr.shape, mean=0.0, stddev=0.0)  # Add noise to input
        if i == 0:
            input_tf = temp_in # shape (N,1)
        else:
            input_tf = tf.concat([input_tf, temp_in], axis=1)  # shape (N,15)
        
    plt.plot(input_tf[:,1].numpy(), 'o', color='blue', label='Health Table (Input)')
    plt.plot(target_tf.numpy(), 'x', color='red', label='Target x')
    plt.legend()
    plt.show()

    plt.plot(input_tf[:,0].numpy(), 'o', color='blue', label='Health Table (Input)')
    plt.plot(input_tf[:,1].numpy(), 'o', color='green', label='Health Table (Input)')
    plt.show()
    # Network architecture
    D_in = 15
    H = 30
    D_out = 1

    # Alternative model with kareas 
    new_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(D_in,)),
        tf.keras.layers.Dense(H, activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
        tf.keras.layers.Dense(D_out, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
    ])

    # Monotonicity loss function
    def monotonicity_loss(y_true, y_pred):
        # y_pred shape is (batch, 1); enforce monotonicity along batch order
        y_flat = tf.reshape(y_pred, [-1])
        start = y_true[0] * 10_000
        batch_size = tf.shape(y_flat)[0]
        diff = y_flat[1:] - y_flat[:-1]  # consecutive differences, shape (batch-1,)
        diff = diff - 10*tf.ones(batch_size-1, dtype=tf.float32) + tf.random.normal([batch_size-1], mean=0.0, stddev=0.1)  # Shift to ensure positive values for monotonic increase
        diff = tf.pow(diff, 2)  # Square the differences to penalize negative values more heavily
        # Penalize negative steps (violations of monotonic increase)
        #loss = tf.reduce_mean(tf.square(tf.nn.relu(-diff)))
        loss = tf.reduce_mean(diff)
        loss *= 100000
        loss += tf.reduce_mean(tf.square(y_flat[0] - start))  # Penalize deviation from start value
        return loss

    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=monotonicity_loss, run_eagerly=True)
    history = new_model.fit(input_tf, target_tf, epochs=200, batch_size=64, verbose=0)
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Keras Model Training Loss')
    plt.show()

    plt.plot(input_tf[:,1].numpy(),'x', color='red', label='Input')
    plt.plot(target_tf.numpy(),'o', color='blue', label='Target')
    pred_keras = new_model.predict(input_tf)
    plt.plot(pred_keras, 'x', color='green', label='Keras Prediction')
    plt.legend()
    plt.show()
