from flask import Flask, render_template
from flask import request, redirect
app = Flask(__name__)
email_addresses = []

@app.route("/")
def hello_world():
    author = "Argon Beer Games"
    name = "Daniel, Luis and Egor"
    return render_template('index.html')

@app.route('/signup', methods = ['POST'])
def signup():
    email = request.form['email']
    email_addresses.append(email)
    print(email_addresses) 
    return redirect('/')

if __name__ == "__main__":
  app.run()