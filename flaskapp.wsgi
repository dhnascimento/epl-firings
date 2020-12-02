#flaskapp.wsgi
import sys
sys.path.insert(0, '/var/www/html/flaskapp')

from lemmatizer import lemmaNLTK
from app import app as application