from sklearn.neighbors import KNeighborsClassifier

input_shape = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE
dataset_name = "mnist"
null_augmenter = {"name": "null_augmenter",}
aiml_augmenter = {"name": "aiml_augmenter","rotation": True,}
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
width = 40

samplesizes = [5000, 10000, 20000, 40000]
tt = [0.1, 0.05, 0.01, 0.005, 0.001]
idxs = tf.range(tf.shape(x_train)[0])
knn = KNeighborsClassifier(n_neighbors=3)
results = []
for i in range(0,4):
  sample_num = samplesizes[i]
  sample_num_train = int(0.2 * sample_num)
  for j in range(0,5):
    for t in range(0,5):
      ridxs = tf.random.shuffle(idxs)[:sample_num]
      rx_train = tf.gather(x_train, ridxs)
      ry_train = tf.gather(y_train, ridxs)
      train_dataset2 = tf.data.Dataset.from_tensor_slices((rx_train, ry_train))
      train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(50, drop_remainder=True)
      model = AIML(temperature=tt[j], tuneparameter = (1,0,200), width=width)
      model.compile(aiml_optimizer=keras.optimizers.Adam())
      pretrain_history = model.fit(train_dataset2, epochs=50)
      resulting_encoder = keras.Sequential(
        [
          n_augmenter(**null_augmenter),
          model.encoder,
        ],
        name="resulting_model",
      )
      rx_train_representation = resulting_encoder(rx_train)
      knn.fit(rx_train_representation[0:sample_num_train,], ry_train[0:sample_num_train])
      accuracy = knn.score(rx_train_representation[sample_num_train:(sample_num_train*2),], ry_train[sample_num_train:(sample_num_train*2)])
      results.append([sample_num,tt[j],t,accuracy])

np.savez('temperature_tuning.npz', results=results)


wds = [20, 40, 60, 80]
sample_num = 10000
sample_num_train = int(0.2 * sample_num)
idxs = tf.range(tf.shape(x_train)[0])
knn = KNeighborsClassifier(n_neighbors=3)
results = []
for i in range(0,4):
  for t in range(0,5):
    ridxs = tf.random.shuffle(idxs)[:sample_num]
    rx_train = tf.gather(x_train, ridxs)
    ry_train = tf.gather(y_train, ridxs)
    train_dataset2 = tf.data.Dataset.from_tensor_slices((rx_train, ry_train))
    train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(50, drop_remainder=True)
    model = AIML(temperature=0.05, tuneparameter = (1,0,200), width=wds[i])
    model.compile(aiml_optimizer=keras.optimizers.Adam())
    pretrain_history = model.fit(train_dataset2, epochs=50)
    resulting_encoder = keras.Sequential(
      [
        n_augmenter(**null_augmenter),
        model.encoder,
      ],
      name="resulting_model",
    )
    rx_train_representation = resulting_encoder(rx_train)
    knn.fit(rx_train_representation[0:sample_num_train,], ry_train[0:sample_num_train])
    accuracy = knn.score(rx_train_representation[sample_num_train:(sample_num_train*2),], ry_train[sample_num_train:(sample_num_train*2)])
    results.append([wds[i],t,accuracy])

np.savez('width_tuning.npz', results=results)



lambda1s = [0.1, 1, 10, 100, 1000]
lambda2s = [0.1, 1, 10, 100, 1000]
sample_num = 10000
sample_num_train = int(0.2 * sample_num)
idxs = tf.range(tf.shape(x_train)[0])
knn = KNeighborsClassifier(n_neighbors=3)
results = []
for i in range(0,5):
  for j in range(0,5):
    for t in range(0,5):
      ridxs = tf.random.shuffle(idxs)[:sample_num]
      rx_train = tf.gather(x_train, ridxs)
      ry_train = tf.gather(y_train, ridxs)
      train_dataset2 = tf.data.Dataset.from_tensor_slices((rx_train, ry_train))
      train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(50, drop_remainder=True)
      model = AIML(temperature=0.05, tuneparameter = (1,lambda1s[i],lambda2s[j]), width=width)
      model.compile(aiml_optimizer=keras.optimizers.Adam())
      pretrain_history = model.fit(train_dataset2, epochs=50)
      resulting_encoder = keras.Sequential(
        [
          n_augmenter(**null_augmenter),
          model.encoder,
        ],
        name="resulting_model",
      )
      rx_train_representation = resulting_encoder(rx_train)
      knn.fit(rx_train_representation[0:sample_num_train,], ry_train[0:sample_num_train])
      accuracy = knn.score(rx_train_representation[sample_num_train:(sample_num_train*2),], ry_train[sample_num_train:(sample_num_train*2)])
      results.append([lambda1s[i],lambda2s[j],t,accuracy])

np.savez('lambda_tuning.npz', results=results)


epochs = [25, 50, 100]
batchsizes = [50, 100, 200, 400]
sample_num = 10000
sample_num_train = int(0.2 * sample_num)
idxs = tf.range(tf.shape(x_train)[0])
knn = KNeighborsClassifier(n_neighbors=3)
results = []
for i in range(0,4):
  for j in range(0,3):
    for t in range(0,5):
      ridxs = tf.random.shuffle(idxs)[:sample_num]
      rx_train = tf.gather(x_train, ridxs)
      ry_train = tf.gather(y_train, ridxs)
      train_dataset2 = tf.data.Dataset.from_tensor_slices((rx_train, ry_train))
      train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(batchsizes[i], drop_remainder=True)
      model = AIML(temperature=0.05, tuneparameter = (1,1,200), width=width)
      model.compile(aiml_optimizer=keras.optimizers.Adam())
      pretrain_history = model.fit(train_dataset2, epochs=epochs[j])
      resulting_encoder = keras.Sequential(
        [
          n_augmenter(**null_augmenter),
          model.encoder,
        ],
        name="resulting_model",
      )
      rx_train_representation = resulting_encoder(rx_train)
      knn.fit(rx_train_representation[0:sample_num_train,], ry_train[0:sample_num_train])
      accuracy = knn.score(rx_train_representation[sample_num_train:(sample_num_train*2),], ry_train[sample_num_train:(sample_num_train*2)])
      results.append([batchsizes[i],epochs[j],t,accuracy])

np.savez('batchepoch_tuning.npz', results=results)



