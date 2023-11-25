input_shape = (28, 28, 1)
AUTOTUNE = tf.data.AUTOTUNE
dataset_name = "mnist"
null_augmenter = {"name": "null_augmenter",}
shuffle_buffer = 5000
steps_per_epoch = 600
unlabelled_images = 60000
unlabeled_batch_size = unlabelled_images // steps_per_epoch
train_dataset = (tfds.load(dataset_name, split="train", as_supervised=True, shuffle_files=True)
      .shuffle(buffer_size=shuffle_buffer)
      .batch(unlabeled_batch_size, drop_remainder=True)
  )
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
width = 40
aiml_augmenter = {"name": "aiml_augmenter","rotation": True,}
samplesize_downstream = [200, 400, 800, 1600]
results = []
sample_num = 60000
idxs = tf.range(tf.shape(x_train)[0])
idxs_t = tf.range(tf.shape(x_test)[0])
knn = KNeighborsClassifier(n_neighbors=3)

    
model = AIML(temperature=0.001, tuneparameter = (1,1,200), width=width)
model.compile(aiml_optimizer=keras.optimizers.Adam())
pretrain_history = model.fit(train_dataset, epochs=50)
resulting_encoder = keras.Sequential(
    [
        n_augmenter(**null_augmenter),
        model.encoder,
    ],
    name="resulting_model",
)
x_train_representation = resulting_encoder(x_train)
x_test_representation = resulting_encoder(x_test)
for i in range(0,4):
  sample_num_train = samplesize_downstream[i]
  for t in range(0, 10):
    ridxs = tf.random.shuffle(idxs)[:sample_num_train]
    knn.fit(tf.gather(x_train_representation, ridxs), tf.gather(y_train, ridxs))
    ridxs_t = tf.random.shuffle(idxs_t)[:sample_num_train]
    accuracy = knn.score(tf.gather(x_test_representation, ridxs_t), tf.gather(y_test, ridxs_t))
    results.append([1,sample_num_train,2,t,accuracy])
    train_dataset2 = tf.data.Dataset.from_tensor_slices((tf.gather(x_train, ridxs), tf.gather(y_train, ridxs)))
    train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(20, drop_remainder=True)
    test_dataset2 = tf.data.Dataset.from_tensor_slices((tf.gather(x_test, ridxs_t), tf.gather(y_test, ridxs_t)))
    test_dataset2 = test_dataset2.shuffle(buffer_size=shuffle_buffer).batch(20, drop_remainder=True)
    finetuning_model = keras.Sequential(
      [
        n_augmenter(**null_augmenter),
        model.encoder,
        layers.Dense(10, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.02)),
      ],
      name="finetuning_model",
    )
    finetuning_model.layers[0].trainable = False
    finetuning_model.layers[1].trainable = False
    linear_optimizer = tfa.optimizers.LAMB()
    finetuning_model.compile(
      optimizer=linear_optimizer,
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"],
    )
    finetuning_history = finetuning_model.fit(
      train_dataset2, epochs=50, validation_data=test_dataset2
    )
    accuracy = finetuning_model.evaluate(test_dataset2)
    results.append([2,sample_num_train,2,t,accuracy])


# hyperparameters
num_epochs = 50
steps_per_epoch = 600
width = 40

# hyperparameters corresponding to each algorithm
hyperparams = {
    SimCLR: {"temperature": 0.1},
    BarlowTwins: {"redundancy_reduction_weight": 0.1},
    MoCo: {"momentum_coeff": 0.99, "temperature": 0.1, "queue_size": 10000},
    NNCLR: {"temperature": 0.1, "queue_size": 10000},
}

Algorithms = [SimCLR, BarlowTwins, MoCo, NNCLR]

for j in range(0,4):
  Algorithm = Algorithms[j]
  model = Algorithm(
    augmenter(**aiml_augmenter),
    encoder=encoder(width),
    projection_head=keras.Sequential(
        [
            layers.Input(shape=(width,)),
            layers.Dense(width),
        ],
        name="projection_head",
    ),
    **hyperparams[Algorithm],
  )
  model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
  )
  history = model.fit(train_dataset, epochs=num_epochs)
  resulting_encoder = keras.Sequential(
    [
        n_augmenter(**null_augmenter),
        model.encoder,
    ],
    name="resulting_model",
  )
  x_train_representation = resulting_encoder(x_train)
  x_test_representation = resulting_encoder(x_test)
  for i in range(0,4):
    sample_num_train = samplesize_downstream[i]
    for t in range(0, 10):
      ridxs = tf.random.shuffle(idxs)[:sample_num_train]
      knn.fit(tf.gather(x_train_representation, ridxs), tf.gather(y_train, ridxs))
      ridxs_t = tf.random.shuffle(idxs_t)[:sample_num_train]
      accuracy = knn.score(tf.gather(x_test_representation, ridxs_t), tf.gather(y_test, ridxs_t))
      results.append([1,sample_num_train,j+3,t,accuracy])
      train_dataset2 = tf.data.Dataset.from_tensor_slices((tf.gather(x_train, ridxs), tf.gather(y_train, ridxs)))
      train_dataset2 = train_dataset2.shuffle(buffer_size=shuffle_buffer).batch(20, drop_remainder=True)
      test_dataset2 = tf.data.Dataset.from_tensor_slices((tf.gather(x_test, ridxs_t), tf.gather(y_test, ridxs_t)))
      test_dataset2 = test_dataset2.shuffle(buffer_size=shuffle_buffer).batch(20, drop_remainder=True)
      finetuning_model = keras.Sequential(
        [
          n_augmenter(**null_augmenter),
          model.encoder,
          layers.Dense(10, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.02)),
        ],
        name="finetuning_model",
      )
      finetuning_model.layers[0].trainable = False
      finetuning_model.layers[1].trainable = False
      linear_optimizer = tfa.optimizers.LAMB()
      finetuning_model.compile(
        optimizer=linear_optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
      )
      finetuning_history = finetuning_model.fit(
        train_dataset2, epochs=50, validation_data=test_dataset2
      )
      accuracy = finetuning_model.evaluate(test_dataset2)
      results.append([2,sample_num_train,j+3,t,accuracy])

results2=[]
for i in range(0,len(results)):
  if i % 2 == 0:
    results2.append(results[i])
  else:
    a=results[i]
    b=a[:4]+a[4]
    del b[4]
    results2.append(b)

np.savez('comparison_mnist.npz', results=results2)





