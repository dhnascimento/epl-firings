#flaskapp.wsgi
import sys
sys.path.insert(0, '/var/www/html/flaskapp')

from lemmatizer import lemmaNLTK
from server import app as application