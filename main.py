from preprocess import tokenize, create_topic_embeddings, get_data 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_TEXT_LENGTH = 1257

def main():

		# Retrieve tokenized users
		users = tokenize(get_data("./DL_dataset/T1/DATA"))

		# create topic embeddings for each user
		topic_embeddings = create_topic_embeddings(users)

		# create one-hot vector based on user labels
		user_labels = []
		for user in users:
				user_labels.append(user.label)
		user_labels = tf.one_hot(user_labels, depth=2)

		topic_embeddings = tf.convert_to_tensor(topic_embeddings)
		print(topic_embeddings.shape)
		print("Topic embeddings processed")

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
		model.add(layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="layer1"))
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="layer2"))
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.3), name="layer3"))
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(2, name="layer4"))

		# the compile() method: specifying a loss, metrics, and an optimizer
		model.compile(
				optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
				loss=keras.losses.SparseCategoricalCrossentropy(),
				metrics=[keras.metrics.SparseCategoricalAccuracy()],
		)

		history = model.fit(
				topic_embeddings,
				user_labels,
				batch_size=50,
				epochs=1,
		)


if __name__ == "__main__":
    main()