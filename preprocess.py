import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import digits
import xml.etree.ElementTree as ET
from user import User
from post import Post
import random
import gensim
from gensim import corpora
from gensim import models
from gensim.models import LsiModel
from gensim.matutils import Sparse2Corpus, corpus2dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 1257

def create_post(post_xml_tree_element):
    """
    Converts an XML tree element to a Post instance.

    param post_xml_tree_element: The parsed XML tree element for a post
    """
    # Extract date, text, title, and info of post

    date = post_xml_tree_element.find("DATE")
    if date is not None:
        date = date.text

    text = post_xml_tree_element.find("TEXT")
    if text is not None:
        text = text.text

    title = post_xml_tree_element.find("TITLE")
    if title is not None:
        title = title.text

    info = post_xml_tree_element.find("INFO")
    if info is not None:
        info = info.text

    return Post(date, text, title, info)


def create_user(user_xml_tree_root):
    """
    Converts an XML tree root to a User instance.

    param user_xml_tree_root: The parsed XML tree root for a user
    return: A User constructed using data from the XML file
    """
    user_id = user_xml_tree_root.find("ID")
    if user_id is not None:
        user_id = user_id.text

    user_posts = []

    # First child of tree is the ID tag, and following children are all posts
    for post_xml_element in user_xml_tree_root[1:]:
        user_posts.append(create_post(post_xml_element))

    return User(user_id, user_posts)


def create_user_label_map(label_file_path):
    """
    Creates a map from subject ID -> label.
    Used when assigning user labels when creating User objects
    """
    label_file = open(label_file_path, encoding="utf-8")
    lines = label_file.readlines()

    user_label_map = {}
    for line in lines:
        user_id, label = line.split()
        user_label_map[user_id] = int(label)

    return user_label_map


def get_data(directory_path):
    """
    Extracts all data for each user and their posts.
    Creates a User instance for each subject XML file and returns them as a list.

    param directory_path: The path to the directory containing the subject XML files
    return: a list of Users
    """
    user_label_map = create_user_label_map("./DL_dataset/T1/T1_erisk_golden_truth.txt")

    num_non_depressed = sum(x ==0 for x in user_label_map.values()) # 319
    num_depressed = sum(x ==1 for x in user_label_map.values()) # 104

    ideal_num_non_depressed = num_depressed*2
    count = 0
    users = []
    file_names = os.listdir(directory_path)
    random.shuffle(file_names)
    for user_filename in file_names:
        file_path = directory_path + "/" + user_filename
        # user_file = open(f"{directory_path}/{user_filename}", "rb")
        user_file = open(file_path, "rb")

        # Parse the XML file and create a User instance from the XML tree
        user_xml_tree = ET.parse(user_file)
        user = create_user(user_xml_tree.getroot())
        label = user_label_map[user.id]
        user.set_label(label)

        if label == 0:
            if count < ideal_num_non_depressed:
                users.append(user)
                count +=1
        else:
            users.append(user)

    assert(count <= ideal_num_non_depressed+1)    
    return users


def tokenize(users):
    """
    Extracts all data for each user and their posts.
    Creates a User instance for each subject XML file and returns them as a list.

    param users: A list of user objects with untokenized title and text
    return: A list of user objects with tokenized title and text and transformed to lowercase
    """

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words("english"))
    porter = PorterStemmer()

    text_tokens = []
    for user in users:
        for post in user.posts:
            if post.title is not None:
                # remove the digits
                remove_digits = str.maketrans("", "", digits)
                post.title = post.title.translate(remove_digits)

                # tokenize words, remove punctutations,transform to lower case
                post.title = tokenizer.tokenize(post.title.lower())

                # remove stopwords
                post.title = [t for t in post.title if not t in stop_words]

                # stemming of words
                post.title = [porter.stem(word) for word in post.title]

            if post.text is not None:
                # remove the digits
                remove_digits = str.maketrans("", "", digits)
                post.text = post.text.translate(remove_digits)

                # tokenize words, remove punctutations,transform to lower case
                post.text = tokenizer.tokenize(post.text.lower())

                # remove stopwords
                post.text = [t for t in post.text if not t in stop_words]

                # stemming of words
                post.text = [porter.stem(word) for word in post.text]

                # add tokenized text to lst

                # if post.text != []:
                #     text_tokens.append(post.text)

    # # Creating a transformation
    # dictionary = corpora.Dictionary(text_tokens)
    # num_terms = len(dictionary)
    # corpus = [dictionary.doc2bow(text) for text in text_tokens]

    # # step 1 -- initialize a model
    # tfidf = models.TfidfModel(corpus)

    # # step 2 -- use the model to transform vectors
    # corpus_tfidf = tfidf[corpus]

    # lsi_model = models.LsiModel(
    #     corpus_tfidf, id2word=dictionary, num_topics=128
    # )  # initialize an LSI transformation
    # corpus_lsi = lsi_model[
    #     corpus_tfidf
    # ]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    # # print(lsi_model.print_topic(15, topn=10)) prints topic weights for a given topic

    # embeddings = gensim.matutils.corpus2dense(corpus_lsi, num_terms)

    return users


def create_topic_embeddings(users):
		"""
		Receives a list of users with tokenized text and generates topic embeddings for each user.

		param users: A list of user objects with tokenized title and text
		return: A list of topic embeddings
		"""

		topic_embeddings = []
		users_with_empty_text = []

		for i, user in enumerate(users):
				text_tokens = []
				for post in user.posts:
						if post.text and post.text != []:
								text_tokens.append(post.text)

				# Creating a transformation
				print(len(text_tokens))
				if (len(text_tokens) == 0):
					users_with_empty_text.append(i)
					continue
				dictionary = corpora.Dictionary(text_tokens)
				num_terms = len(dictionary)

				corpus = [dictionary.doc2bow(text) for text in text_tokens]
				# step 1 -- initialize a model
				tfidf = models.TfidfModel(corpus)

				# step 2 -- use the model to transform vectors
				corpus_tfidf = tfidf[corpus]

				lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=128)

				corpus_lsi = lsi_model[corpus_tfidf]

				embedding = gensim.matutils.corpus2dense(corpus_lsi, 128)
				embedding = pad_sequences(embedding, maxlen=MAX_LENGTH, padding="post")
				print(embedding.shape)

				topic_embeddings.append(embedding)

		for i in users_with_empty_text:
			users.pop(i)

		return topic_embeddings


# users = tokenize(get_data("./DL_dataset/T1/DATA"))

# print("users tokenized")

# topic_embeddings = create_topic_embeddings(users)

# print("topic embeddings created")
