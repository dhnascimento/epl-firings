from flask_frozen import Freezer
from lemmatizer import lemmaNLTK
from app import app

freezer = Freezer(app)

if __name__ == '__main__':
    freezer.freeze()