from sklearn.neighbors import KNeighborsClassifier
import time

input_shape = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE
dataset_name = "mnist"
null_augmenter = {"name": "null_augmenter",}
shuffle_buffer = 5000
steps_per_epoch = 200
unlabelled_images = 60000
unlabeled_batch_size = unlabelled_images // steps_per_epoch
train_dataset = (tfds.load(dataset_name, split="train[:2%]", as_supervised=True, shuffle_files=True)
      .shuffle(buffer_size=shuffle_buffer)
      .batch(unlabeled_batch_size, drop_remainder=True)
  )
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
width = 40
aiml_augmenter = {"name": "aiml_augmenter","rotation": True,}



samplesizes = [250, 500, 1000, 2000, 4000]
idxs = tf.range(tf.shape(x_train)[0])
knn = KNeighborsClassifier(n_neighbors=3)
results = []
for i in range(0, 5): 
  for t in range(0,10):
    sample_num = samplesizes[i]
    sample_num_test = int(0.2 * sample_num)
    ridxs = tf.random.shuffle(idxs)[:sample_num]
    rx_train = tf.gather(x_train, ridxs)
    ry_train = tf.gather(y_train, ridxs)
    train_dataset2 = tf.data.Dataset.from_tensor_slices((rx_train, ry_train))
    train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(20, drop_remainder=True)
    
    start = time.time()
    model = AIML(temperature=0.2, tuneparameter = (1,0,500), width=width)
    model.compile(aiml_optimizer=keras.optimizers.Adam())
    pretrain_history = model.fit(train_dataset2, epochs=25)
    end = time.time()
    time1 = end - start
    resulting_encoder = keras.Sequential(
      [
        n_augmenter(**null_augmenter),
        model.encoder,
      ],
      name="resulting_model",
    )
    rx_train_representation = resulting_encoder(rx_train)
    knn.fit(rx_train_representation[0:(sample_num-sample_num_test),], ry_train[0:(sample_num-sample_num_test)])
    accuracy = knn.score(rx_train_representation[(sample_num-sample_num_test):sample_num,], ry_train[(sample_num-sample_num_test):sample_num])
    results.append([sample_num,1,t,time1,accuracy])
    
    start = time.time()
    model = AIML(temperature=0.2, tuneparameter = (1,0,500), width=width)
    model.compile(aiml_optimizer=keras.optimizers.Adam())
    pretrain_history = model.fit(train_dataset2, epochs=50)
    end = time.time()
    time1 = end - start
    resulting_encoder = keras.Sequential(
      [
        n_augmenter(**null_augmenter),
        model.encoder,
      ],
      name="resulting_model",
    )
    rx_train_representation = resulting_encoder(rx_train)
    knn.fit(rx_train_representation[0:(sample_num-sample_num_test),], ry_train[0:(sample_num-sample_num_test)])
    accuracy = knn.score(rx_train_representation[(sample_num-sample_num_test):sample_num,], ry_train[(sample_num-sample_num_test):sample_num])
    results.append([sample_num,2,t,time1,accuracy])
    
    start = time.time()
    model = AIML(temperature=0.2, tuneparameter = (1,0,500), width=width)
    model.compile(aiml_optimizer=keras.optimizers.Adam())
    pretrain_history = model.fit(train_dataset2, epochs=100)
    end = time.time()
    time1 = end - start
    resulting_encoder = keras.Sequential(
      [
        n_augmenter(**null_augmenter),
        model.encoder,
      ],
      name="resulting_model",
    )
    rx_train_representation = resulting_encoder(rx_train)
    knn.fit(rx_train_representation[0:(sample_num-sample_num_test),], ry_train[0:(sample_num-sample_num_test)])
    accuracy = knn.score(rx_train_representation[(sample_num-sample_num_test):sample_num,], ry_train[(sample_num-sample_num_test):sample_num])
    results.append([sample_num,3,t,time1,accuracy])
    
    start = time.time()
    rx_train_representation=AILaplacian(rx_train, 0.2, 40, augmenter(**aiml_augmenter), 3)
    end = time.time()
    time1 = end - start
    knn.fit(rx_train_representation[0:(sample_num-sample_num_test),], ry_train[0:(sample_num-sample_num_test)])
    accuracy = knn.score(rx_train_representation[(sample_num-sample_num_test):sample_num,], ry_train[(sample_num-sample_num_test):sample_num])
    results.append([sample_num,4,t,time1,accuracy])
    
    start = time.time()
    rx_train_representation=AILaplacian(rx_train, 0.2, 40, augmenter(**aiml_augmenter), 5)
    end = time.time()
    time1 = end - start
    knn.fit(rx_train_representation[0:(sample_num-sample_num_test),], ry_train[0:(sample_num-sample_num_test)])
    accuracy = knn.score(rx_train_representation[(sample_num-sample_num_test):sample_num,], ry_train[(sample_num-sample_num_test):sample_num])
    results.append([sample_num,5,t,time1,accuracy])

np.savez('twoformulation.npz', results=results)




