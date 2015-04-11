from bottle import route, run, template, request
import jsonpickle

@route('/')
def index():
  return "Hello World!"
 
@route('/hello')
def hello():
  return "HEY"
 
run(port=80)
