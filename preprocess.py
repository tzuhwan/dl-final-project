import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import digits
import xml.etree.ElementTree as ET
from user import User
from post import Post


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


def get_data(directory_path):
    """
    Extracts all data for each user and their posts.
    Creates a User instance for each subject XML file and returns them as a list.

    param directory_path: The path to the directory containing the subject XML files
    return: a list of Users
    """
    users = []
    for user_filename in os.listdir(directory_path):
        file_path = directory_path + "/" + user_filename
        # user_file = open(f"{directory_path}/{user_filename}", "rb")
        user_file = open(file_path, "rb")

        # Parse the XML file and create a User instance from the XML tree
        user_xml_tree = ET.parse(user_file)
        users.append(create_user(user_xml_tree.getroot()))

    return users


def tokenize(users):
    """
    Extracts all data for each user and their posts.
    Creates a User instance for each subject XML file and returns them as a list.

    param users: A list of user objects with untokenized title and text
    return: A list of user objects with tokenized title and text and transformed to lowercase
    """

    print("before cleaning")
    for x in range(10):
        print("post", x)
        print(users[50].posts[x].title)
        print(users[50].posts[x].text)

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words("english"))
    porter = PorterStemmer()

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

    print("after cleaning")
    for x in range(10):
        print("post", x)
        print(users[50].posts[x].title)
        print(users[50].posts[x].text)

    return users


tokenize(get_data("./DL_dataset/T1/DATA"))