from preprocess import tokenize, create_topic_embeddings, get_data 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_TEXT_LENGTH = 1257
TEST_SIZE = 100 

def main():

		# Retrieve tokenized users
		print('~~~~~~~~~~~~~~~~~~~~~~~~~retrieving users...~~~~~~~~~~~~~~~~~~~~')
		users = tokenize(get_data("./DL_dataset/T1/DATA"))
		print('type of users:', type(users))
		print('number of users:', len(users))
		print('type of first user:', type(users[0]))
		print('first user:', users[0])
		print('~~~~~~~~~~~~~~~~~~~~~~~~~users retrieved!~~~~~~~~~~~~~~~~~~~~')

		# create topic embeddings for each user
		print('~~~~~~~~~~~~~~~~~~~~~~~~~creating topic embeddings...~~~~~~~~~~~~~~~~~~~~')
		topic_embeddings = create_topic_embeddings(users)
		print('type of topic embeddings:', type(topic_embeddings))
		print('number of topic embeddings:', len(topic_embeddings))
		print('~~~~~~~~~~~~~~~~~~~~~~~~~topic embeddings created!~~~~~~~~~~~~~~~~~~~~')

		# create one-hot vector based on user labels
		user_labels = []
		for user in users:
				user_labels.append(user.label)
		user_labels = tf.convert_to_tensor(user_labels)
		user_labels = tf.one_hot(user_labels, depth=2)

		topic_embeddings = tf.convert_to_tensor(topic_embeddings)
		print(topic_embeddings.shape)
		print("Topic embeddings processed")
		print(user_labels.shape)

    # How to add tf.keras.dropout?
    # model = keras.Sequential(
    #     [
    #         layers.Dense(
    #             512, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="layer1"
    #         ),
    #         layers.Dense(
    #             256, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="layer2"
    #         ),
    #         layers.Dense(
    #             256, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="layer3"
    #         ),
    #         layers.Dense(2, name="layer4"),
    #     ]
    # )
		model = keras.Sequential()
		model.add(layers.Flatten())
		model.add(layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="layer1"))
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="layer2"))
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="layer3"))
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(2, activation = "sigmoid", name="layer4"))

		#model.add(layers.Dense(2, activation = "sigmoid", name="layer4"))

		METRICS = [
		#keras.metrics.SparseCategoricalAccuracy(name='SparseCategoricalAccuracy'),
		keras.metrics.TruePositives(name='tp'),
		keras.metrics.FalsePositives(name='fp'),
		keras.metrics.TrueNegatives(name='tn'),
		keras.metrics.FalseNegatives(name='fn'), 
		keras.metrics.BinaryAccuracy(name='accuracy'),
		keras.metrics.Precision(name='precision'),
		keras.metrics.Recall(name='recall'),
		#keras.metrics.AUC(name='auc'),
		]

		# the compile() method: specifying a loss, metrics, and an optimizer
		model.compile(
				optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
				loss = keras.losses.BinaryCrossentropy(),
				#loss=keras.losses.SparseCategoricalCrossentropy(),
				metrics=[METRICS],
		)

		history = model.fit(
				topic_embeddings[:-TEST_SIZE],
				user_labels[:-TEST_SIZE],
				batch_size=50,
				epochs=1,
		)

		print("Now Testing Model")

		predictions = model.predict(
			topic_embeddings[-TEST_SIZE:],
		)
		print('predictions are:', predictions)
		argmax_predictions = tf.math.argmax(predictions, 1)
		print('argmax predictions are:', argmax_predictions)
		print('number of predictions:', len(predictions))

		model.evaluate(
			topic_embeddings[-TEST_SIZE:],
			user_labels[-TEST_SIZE:],
			batch_size=20
		)


if __name__ == "__main__":
    main()