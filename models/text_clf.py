import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

print(model.bert)

# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# model.compile(optimizer=optimizer, loss=model.compute_loss) # can also use any keras loss fn
# model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)
