

class User:
	"""
	Class representing a user with their posts.
	"""

	def __init__(self, user_id, user_posts):
		self.id = user_id
		self.posts = user_posts
	
	def set_label(self, label):
		self.label = label