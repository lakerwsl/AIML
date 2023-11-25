from sklearn.neighbors import KNeighborsClassifier

input_shape = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE
dataset_name = "mnist"
null_augmenter = {"name": "null_augmenter",}
aiml_augmenter = {"name": "aiml_augmenter","rotation": True,}
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
width = 40
idxs = tf.range(tf.shape(x_train)[0])
knn = KNeighborsClassifier(n_neighbors=3)
results = []
sample_num = 60000
samplesize_downstream = [1000, 2000, 3000, 4000]
shuffle_buffer = 5000

ridxs = tf.random.shuffle(idxs)[:sample_num]
rx_train = tf.gather(x_train, ridxs)
ry_train = tf.gather(y_train, ridxs)
train_dataset2 = tf.data.Dataset.from_tensor_slices((rx_train, ry_train))
train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(100, drop_remainder=True)
model = AIML(temperature=0.01, tuneparameter = (1,1,200), width=width)
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
for i in range(0,4):
  sample_num_train = samplesize_downstream[i]
  for t in range(0, 10):
    ridxs = tf.random.shuffle(idxs)[:sample_num_train]
    knn.fit(tf.gather(rx_train_representation, ridxs), tf.gather(ry_train, ridxs))
    ridxs = tf.random.shuffle(idxs)[:sample_num_train]
    accuracy = knn.score(tf.gather(rx_train_representation, ridxs), tf.gather(ry_train, ridxs))
    results.append([sample_num_train,1,t,accuracy]) 
  
  
  
  



aiml_augmenter = {"name": "aiml_augmenter","rotation": False,}

ridxs = tf.random.shuffle(idxs)[:sample_num]
rx_train = tf.gather(x_train, ridxs)
ry_train = tf.gather(y_train, ridxs)
train_dataset2 = tf.data.Dataset.from_tensor_slices((rx_train, ry_train))
train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(100, drop_remainder=True)
model = AIML(temperature=0.01, tuneparameter = (1,1,200), width=width)
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
for i in range(0,4):
  sample_num_train = samplesize_downstream[i]
  for t in range(0, 10):
    ridxs = tf.random.shuffle(idxs)[:sample_num_train]
    knn.fit(tf.gather(rx_train_representation, ridxs), tf.gather(ry_train, ridxs))
    ridxs = tf.random.shuffle(idxs)[:sample_num_train]
    accuracy = knn.score(tf.gather(rx_train_representation, ridxs), tf.gather(ry_train, ridxs))
    results.append([sample_num_train,2,t,accuracy]) 

np.savez('augmentation_tuning.npz', results=results)
